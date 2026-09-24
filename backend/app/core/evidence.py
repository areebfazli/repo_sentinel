"""Deterministic evidence for the LLM review: Semgrep hits and diff direction.

- **Semgrep** (``semgrep_scanner``): one engine run over the scan's units. Files
  mode passes the full file text (``sources``) so rules see imports; snippet
  mode scans the snippet. Only hits at/above ``SEMGREP_MIN_SEVERITY`` (default
  high) are kept, with the noisiest rules excluded (``SEMGREP_EXCLUDED_RULES``):
  on the eval sets a high/critical hit fired on 3.2% of vulnerable functions vs
  1.4% of fixed twins and 0.5% of ordinary ones — evidence, not a verdict.
  Regex / ReDoS rules are marked low-confidence in the prompt.
- **guard_diff** (files mode only; a snippet has no previous version): each
  unit's old code is rebuilt from the file's ``patch`` and compared with the new
  one. ``guard_removed`` changes go to the prompt as change-direction evidence
  (held-out TPR 0.227 / FPR 0.030); the ``alert`` tier (unsafe-API swap, flag
  flip or SQL interpolation; TPR 0.032 at 0/506 FPR) also becomes its own
  deterministic finding, reported even if the LLM says nothing.

Every function here is blocking and never raises on bad input; call from async
code via ``asyncio.to_thread``.
"""
import hashlib
import re
from typing import Any

from loguru import logger

from backend.app.core.guard_diff import (
    ALERT_MIN_CONFIDENCE,
    GuardDiffResult,
    guard_diff_for_file,
    patch_touched_lines,
)
from backend.app.core.review_plan import UnitKey, unit_key

SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
# Rule ids matching this are heuristics about regular expressions (ReDoS,
# non-literal RegExp): flagged as low-confidence evidence in the prompt.
LOW_CONFIDENCE_RULE_RE = re.compile(r"regex|redos|regexp", re.IGNORECASE)



# --- Semgrep -----------------------------------------------------------------


def filter_semgrep_hits(
    hits: dict[UnitKey, list[dict]], min_severity: str, excluded: set[str] | frozenset = frozenset()
) -> dict[UnitKey, list[dict]]:
    """Hits at/above ``min_severity`` and not excluded, each marked
    ``low_confidence`` for regex heuristics. Units left without hits drop out."""
    floor = SEVERITY_RANK.get((min_severity or "high").lower(), 3)
    out: dict[UnitKey, list[dict]] = {}
    for key, unit_hits in hits.items():
        kept = [
            {**h, "low_confidence": bool(LOW_CONFIDENCE_RULE_RE.search(h.get("rule_id", "")))}
            for h in unit_hits
            if SEVERITY_RANK.get(str(h.get("severity", "")).lower(), 0) >= floor
            and h.get("rule_id") not in excluded
        ]
        if kept:
            out[key] = kept
    return out


def semgrep_evidence(
    scanner, units: list[dict], sources: dict[str, str] | None, min_severity: str,
    excluded: set[str] | frozenset = frozenset(),
) -> dict[UnitKey, list[dict]]:
    """Run ``scanner`` over ``units`` and keep the evidence-grade hits.
    ``{}`` when disabled, unavailable or failed (the scanner logs why)."""
    if scanner is None or not units:
        return {}
    try:
        hits = scanner.scan_units(units, sources=sources)
    except Exception:  # the scanner already never raises; belt and braces
        logger.exception("semgrep evidence failed; continuing without it")
        return {}
    return filter_semgrep_hits(hits, min_severity, excluded)


_PY_HINT = re.compile(r"^\s*(def |class |import |from \S+ import |async def )", re.MULTILINE)
_JS_HINT = re.compile(
    r"\bfunction\b|=>|\b(const|let|var)\s+\w+\s*=|\brequire\(|\bmodule\.exports\b|;\s*$",
    re.MULTILINE,
)


def guess_language(code: str) -> str | None:
    """Best-effort python / javascript for a snippet whose language the client
    didn't send (only used to pick Semgrep rules and label the prompt)."""
    if _PY_HINT.search(code or ""):
        return "python"
    if _JS_HINT.search(code or ""):
        return "javascript"
    return None


# --- guard_diff ----------------------------------------------------------------


def plan_with_touched_lines(files: list) -> list:
    """Files for ``plan_units`` with deletion points counted as changed.

    ``plan_units`` derives changed lines from the patch's ADDED lines only, so a
    function whose only change is a deleted guard would never be analysed. For
    files without explicit ``changed_lines``, use ``patch_touched_lines`` (added
    lines plus the lines around each deletion) instead.
    """
    planned = []
    for f in files:
        if getattr(f, "changed_lines", None) is None and getattr(f, "patch", None):
            f = f.model_copy(update={"changed_lines": patch_touched_lines(f.patch)})
        planned.append(f)
    return planned


def _new_line(unit: dict, change) -> int | None:
    """Real-file line of a change on the NEW side (the unit's code), if any."""
    first = (change.new_text or "").strip().splitlines()[:1]
    if first:
        needle = " ".join(first[0].split())
        for i, ln in enumerate((unit.get("code") or "").splitlines()):
            if needle and needle in " ".join(ln.split()):
                return int(unit.get("start_line") or 1) + i
    return None


def guard_to_dict(unit: dict, result: GuardDiffResult) -> dict[str, Any]:
    return {
        "file_path": unit.get("file_path"),
        "function_name": unit.get("function_name"),
        "start_line": unit.get("start_line"),
        "risk": result.risk,
        "alert": result.alert,
        "removed_score": result.removed_score,
        "added_score": result.added_score,
        "note": result.note,
        "changes": [
            {
                "direction": c.direction,
                "kind": c.kind,
                "line": _new_line(unit, c),
                "old_text": c.old_text,
                "new_text": c.new_text,
                "confidence": c.confidence,
                "rationale": c.rationale,
            }
            for c in result.changes
        ],
    }


def guard_evidence(files: list, units: list[dict], parser) -> dict[UnitKey, dict[str, Any]]:
    """guard_diff per unit (files mode), keyed like ``unit_key``; units without a
    signal (``risk`` none) are left out. Missing / non-applying patches, new
    functions and unsupported languages simply give no signal."""
    by_file: dict[str, list[dict]] = {}
    for u in units:
        by_file.setdefault(u["file_path"], []).append(u)
    out: dict[UnitKey, dict[str, Any]] = {}
    for f in files:
        file_units = by_file.get(f.path)
        if not file_units:
            continue
        try:
            results = guard_diff_for_file(f, file_units, parser)
        except Exception:  # guard_diff never raises by design; don't let it break a scan
            logger.exception("guard_diff failed for {}; continuing without it", f.path)
            continue
        for unit, result in zip(file_units, results, strict=True):
            if result.risk != "none":
                out[unit_key(unit)] = guard_to_dict(unit, result)
    return out


def guard_alert_findings(guard: dict[UnitKey, dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic report findings for the ``alert`` tier: each weakened
    change with confidence >= ALERT_MIN_CONFIDENCE in a unit flagged alert."""
    findings = []
    for g in guard.values():
        if not g.get("alert"):
            continue
        for c in g["changes"]:
            if c["direction"] != "weakened" or c["confidence"] < ALERT_MIN_CONFIDENCE:
                continue
            kind = c["kind"].replace("_", " ")
            digest = hashlib.sha1(
                f"{c['kind']}|{' '.join((c['new_text'] or '').split())}".encode()
            ).hexdigest()[:16]
            findings.append(
                {
                    "severity": "high",
                    "cve_id": None,
                    "team_pr_id": None,
                    "cwe": None,
                    "title": f"This change weakens a security check ({kind})",
                    "explanation": c["rationale"],
                    "reasoning": (
                        f"Deterministic diff check: {c['direction']} {c['kind']} "
                        f"(confidence {c['confidence']:.2f}); before: {c['old_text'] or '-'}; "
                        f"after: {c['new_text'] or '-'}"
                    ),
                    "quoted_code": c["new_text"] or "",
                    "fix_snippet": "",
                    "file_path": g["file_path"],
                    "start_line": g["start_line"],
                    "function_name": g["function_name"],
                    "line": c["line"],
                    "end_line": c["line"],
                    "finding_id": None,
                    "point_id": None,
                    "source": "guard_diff",
                    "deterministic": True,
                    "dedupe_key": f"guard:{digest}",
                }
            )
    return findings
