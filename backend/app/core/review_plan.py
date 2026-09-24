"""Turn a scan's analysis units + evidence into LLM review prompts.

A *review unit* is a planner unit (``analysis_planner.plan_units``: one function,
or a whole file / snippet) plus the evidence gathered for it: the retrieved CVE
and team matches anchored to it, its Semgrep hits and guard_diff result
(``core.evidence``), and its prompt-ready code (``untrusted.sanitize_untrusted``,
same line count as the real code). Pure.

Token budget (``plan_batches``): units are ordered by evidence — guard_diff
alert, a Semgrep hit, guard_removed, then best retrieval similarity — and packed
first-fit into prompts of at most ``LLM_MAX_PROMPT_TOKENS`` (estimated) and
``LLM_MAX_UNITS_PER_PROMPT`` units, at most ``LLM_MAX_CALLS_PER_SCAN`` prompts,
so a scan that fits is still exactly one call. Each unit keeps at most
``LLM_MAX_CVES_PER_UNIT`` CVE matches (and as many team matches). A unit too big
for a prompt on its own loses its references, then code is elided around its
changed lines (head + tail without change information), with explicit markers
and real line numbers (``_shrink_to_fit``). Units that don't fit are returned,
never silently dropped.
"""
import math
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.markdown_renderer import (
    SYSTEM_PROMPT,
    build_unit_section,
    prompt_closing,
    prompt_preamble,
)
from backend.app.core.untrusted import sanitize_untrusted

# Tokens are estimated as characters / 4, times this safety factor (code and
# non-English text tokenize denser than prose; the estimate must not undershoot).
TOKEN_SAFETY = 1.25
# Placeholder nonce for sizing: same length as untrusted.new_nonce(), so a
# section's size doesn't depend on which nonce the real prompt gets.
SIZING_NONCE = "0" * 16

UnitKey = tuple[str | None, str | None, int]


def unit_key(unit: dict[str, Any]) -> UnitKey:
    return (unit.get("file_path"), unit.get("function_name"), int(unit.get("start_line") or 1))


def _anchor_key(match: dict[str, Any]) -> UnitKey:
    return (
        match.get("anchor_file_path"),
        match.get("anchor_function_name"),
        int(match.get("anchor_start_line") or 1),
    )


def snippet_unit(code: str, language: str | None) -> dict[str, Any]:
    """Snippet mode as one unit (no file, starts at line 1)."""
    return {
        "file_path": None,
        "function_name": None,
        "start_line": 1,
        "end_line": len(code.splitlines()) or 1,
        "code": code,
        "language": language,
    }


def build_review_units(
    units: list[dict[str, Any]],
    raw: dict[str, Any],
    semgrep: dict[UnitKey, list[dict]] | None = None,
    guard: dict[UnitKey, dict] | None = None,
) -> list[dict[str, Any]]:
    """One review unit per planner unit, with its retrieval matches, Semgrep
    hits (``semgrep``) and guard_diff result (``guard``, both keyed by
    ``unit_key``) attached.

    Files-mode matches carry ``anchor_*`` keys naming their unit; unanchored
    matches (snippet mode) belong to the single (first) unit. Matches stay in
    relevance order.
    """
    review = [
        {**u, "key": unit_key(u), "prompt_code": sanitize_untrusted(u.get("code") or ""),
         "cves": [], "team": [], "semgrep": (semgrep or {}).get(unit_key(u), []),
         "guard": (guard or {}).get(unit_key(u))}
        for u in units
    ]
    by_key = {r["key"]: r for r in review}
    for source, target in (("ghost_hunter_findings", "cves"), ("team_memory_findings", "team")):
        for match in raw.get(source, []):
            if match.get("anchor_file_path") is not None:
                owner = by_key.get(_anchor_key(match))
            else:
                owner = review[0] if review else None
            if owner is not None:
                owner[target].append(match)
    return review


def assign_uids(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Label the units of one prompt U1..Un (ids are per prompt)."""
    for i, unit in enumerate(batch, start=1):
        unit["uid"] = f"U{i}"
    return batch


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4 * TOKEN_SAFETY)


def unit_priority(unit: dict[str, Any]) -> tuple:
    """Higher first: alert-tier guard change, a (non-regex) Semgrep hit,
    guard_removed, then the best retrieved-CVE similarity."""
    guard = unit.get("guard") or {}
    return (
        bool(guard.get("alert")),
        any(not h.get("low_confidence") for h in unit.get("semgrep") or []),
        guard.get("risk") == "guard_removed",
        max((float(c.get("similarity_score") or 0.0) for c in unit.get("cves") or []),
            default=0.0),
    )


@dataclass
class Batch:
    units: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0


# Placeholder for the truncation note while sizing (the real note is shorter).
_NOTE_PLACEHOLDER = "x" * 120


def elide(lines: list[str], keep: set[int], start_line: int) -> tuple[str, list[int | None]]:
    """``lines`` with the ones not in ``keep`` (0-based) collapsed into
    ``[... N lines omitted ...]`` markers. Returns (text, line_numbers): the
    real file line of each output line, None for a marker."""
    out: list[str] = []
    numbers: list[int | None] = []
    i = 0
    while i < len(lines):
        if i in keep:
            out.append(lines[i])
            numbers.append(start_line + i)
            i += 1
            continue
        j = i
        while j < len(lines) and j not in keep:
            j += 1
        first, last = start_line + i, start_line + j - 1
        span = f"line {first}" if first == last else f"lines {first}-{last}"
        out.append(f"[... {j - i} line(s) omitted: {span} ...]")
        numbers.append(None)
        i = j
    return "\n".join(out), numbers


def _around_changes(n: int, focus: list[int], radius: int, limit: int | None = None) -> set[int]:
    """The first line (signature) plus ``radius`` lines either side of each of
    the first ``limit`` focus lines."""
    keep = {0}
    for c in focus[:limit]:
        keep.update(range(max(c - radius, 0), min(c + radius, n - 1) + 1))
    return keep


def _head_and_tail(n: int, k: int) -> set[int]:
    return set(range(min(k, n))) | set(range(max(n - k, 0), n))


def _largest_fitting(lo: int, hi: int, fits) -> int | None:
    """Largest x in [lo, hi] with fits(x) (fits is monotone), or None."""
    if hi < lo or not fits(lo):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _shrink_to_fit(unit: dict[str, Any], size_of, available: int) -> bool:
    """Make an oversized unit fit ``available`` tokens: drop team matches, then
    CVE matches (least relevant first), then elide code.

    Code is cut to a window, never just from the end (the offending line may be
    the last one): with change information (files mode, ``changed_lines``) the
    signature plus as many lines as fit around every changed line (or, if even
    that is too big, as many changed lines as fit); without it (snippet mode, or
    a file sent without a patch) the head and tail. Elided regions become
    explicit ``[... N lines omitted ...]`` markers and every shown line keeps its
    real number (``line_numbers``). ``truncated`` describes the cut;
    ``partial`` is True when changed lines (or, without change information, any
    lines) were cut, i.e. part of what the review is for went unseen. False
    when not even one line of code fits.
    """
    while size_of(unit) > available and (unit["team"] or unit["cves"]):
        (unit["team"] if unit["team"] else unit["cves"]).pop()
    if size_of(unit) <= available:
        return True
    lines = unit["prompt_code"].splitlines()
    n = len(lines)
    start = int(unit.get("start_line") or 1)
    focus = sorted({ln - start for ln in unit.get("changed_lines") or ()
                    if 0 <= ln - start < n})

    def fits(keep: set[int]) -> bool:
        text, numbers = elide(lines, keep, start)
        return size_of({**unit, "prompt_code": text, "line_numbers": numbers,
                        "truncated": _NOTE_PLACEHOLDER}) <= available

    keep = None
    if focus:
        radius = _largest_fitting(0, n, lambda r: fits(_around_changes(n, focus, r)))
        if radius is not None:
            keep = _around_changes(n, focus, radius)
        else:
            count = _largest_fitting(1, len(focus),
                                     lambda k: fits(_around_changes(n, focus, 0, k)))
            if count is not None:
                keep = _around_changes(n, focus, 0, count)
        how = "the signature and the lines around the changed lines"
    else:
        k = _largest_fitting(1, (n + 1) // 2, lambda k: fits(_head_and_tail(n, k)))
        keep = _head_and_tail(n, k) if k is not None else None
        how = "the first and last lines"
    if keep is None:
        return False
    unit["prompt_code"], unit["line_numbers"] = elide(lines, keep, start)
    shown = len(keep)
    unit["partial"] = not set(focus) <= keep if focus else shown < n
    unit["truncated"] = (
        f"{shown} of {n} lines ({how}); the rest exceeded the prompt budget and is "
        "replaced by '[... N line(s) omitted ...]' markers"
    )
    return True


def plan_batches(
    units: list[dict[str, Any]],
    *,
    render_section,
    overhead_tokens: int,
    max_prompt_tokens: int,
    max_units_per_prompt: int,
    max_calls: int,
    max_refs_per_unit: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Pack review units into LLM prompts under the budget (module docstring).

    ``render_section(unit) -> str`` renders one unit's prompt text (sized with
    ``SIZING_NONCE``); ``overhead_tokens`` is the system prompt plus the
    per-prompt preamble / closing. Returns (batches, not_reviewed); each
    not-reviewed unit carries ``not_reviewed_reason`` ("budget" or "too_large").
    Batches come in priority order, as do the units inside each.
    """
    available = max_prompt_tokens - overhead_tokens
    ordered = sorted(units, key=unit_priority, reverse=True)
    batches: list[Batch] = []
    not_reviewed: list[dict[str, Any]] = []

    def size_of(u):
        return estimate_tokens(render_section(u)) + 2  # + the blank line between units

    for unit in ordered:
        unit["cves"] = unit["cves"][:max_refs_per_unit]
        unit["team"] = unit["team"][:max_refs_per_unit]
        if size_of(unit) > available and not _shrink_to_fit(unit, size_of, available):
            not_reviewed.append({**unit, "not_reviewed_reason": "too_large"})
            continue
        size = size_of(unit)
        target = next(
            (b for b in batches
             if b.tokens + size <= available and len(b.units) < max_units_per_prompt),
            None,
        )
        if target is None and len(batches) < max_calls:
            target = Batch()
            batches.append(target)
        if target is None:
            not_reviewed.append({**unit, "not_reviewed_reason": "budget"})
            continue
        target.units.append(unit)
        target.tokens += size
    return [b.units for b in batches], not_reviewed


def plan_review_prompts(
    units: list[dict[str, Any]],
    *,
    max_prompt_tokens: int,
    max_units_per_prompt: int,
    max_calls: int,
    max_refs_per_unit: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """``plan_batches`` with the real prompt renderer (``build_unit_section``)
    and overhead (system prompt + preamble / closing for a full batch). Assign
    uids (``assign_uids``) per batch before building each prompt."""
    uids = [f"U{i}" for i in range(1, max(max_units_per_prompt, 1) + 1)]
    overhead = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(
        prompt_preamble(SIZING_NONCE, uids) + "\n" + prompt_closing(uids)
    )

    def render(unit):
        return build_unit_section({**unit, "uid": uids[-1]}, SIZING_NONCE)

    return plan_batches(
        units, render_section=render, overhead_tokens=overhead,
        max_prompt_tokens=max_prompt_tokens, max_units_per_prompt=max_units_per_prompt,
        max_calls=max_calls, max_refs_per_unit=max_refs_per_unit,
    )
