"""Offline per-item facts the PR-level eval (``run_pr_eval``) scores with.

Everything here is computed from the dataset items alone (no network, no
model, no LLM): it is attached to each result record as ``rec["facts"]`` by a
live run and recomputed from the dataset by ``--rescore``.

- **Change anchors** (``anchors_by_path``): per file, the new-file lines the
  PR's diff touches *including deletion points* (``guard_diff.
  patch_touched_lines``, the production definition: every added line plus, for
  each run of removed lines, the new-file lines just before and after it). In a
  ``vuln_introducing`` item (a reversed fix) these are the lines the real fix
  modified plus where it inserted code (e.g. the guard it added, now deleted);
  in a ``vuln_fix`` item, the fix's own changes.
- **Touched functions** (``touched_functions_by_path``): per file, the
  innermost *named* function (tree-sitter ``CodeParser``: Python ``def``, JS/TS
  function declarations and methods; an anonymous arrow function only when no
  named one encloses the line) of the NEW file containing each anchor line.
- **Added files** (``added_files``): files the PR adds whole (``old_content``
  None). In a reversed fix, a file the real fix deleted (e.g. a vulnerable
  module the fix removed).
- **Scope** (``scope``): whether the item's vulnerability class is one the PR
  audit prompt tells the model not to report (``SCOPE_*`` below).
- **Leaky diff** (``deleted_security_terms``): ``SECURITY_VOCAB`` terms found
  in the PR's deleted lines. In a reversed fix the deleted lines are what the
  real fix added, so a comment like "# SECURITY: prevent XXE" there tells the
  reviewer exactly what is being removed.
- **Provenance** (``provenance``, benign items): the bystander source and
  whether the benign PR shares its commit with a vuln_introducing / vuln_fix
  item of the same selection (and whether its files are an identical-content
  subset of that fix item's files).
"""
from __future__ import annotations

from pathlib import Path

from backend.app.core.guard_diff import patch_touched_lines
from backend.app.core.pr_context import diff_lines

KIND_INTRO, KIND_FIX, KIND_BENIGN = "vuln_introducing", "vuln_fix", "benign"

# ---------------------------------------------------------------------------
# Scope: vulnerability classes the PR audit prompt excludes
# ---------------------------------------------------------------------------
#
# Derived from the "DO NOT REPORT" list of backend/app/core/prompts/pr_audit.py
# (AUDIT_SYSTEM_PROMPT) and the verifier's matching exclusions, as used by the
# dev200 run:
#
# - "Denial of service, resource exhaustion, memory or CPU consumption, or
#   missing rate limiting" and "Regular-expression DoS": always out of scope.
#   CWEs: 400 uncontrolled resource consumption, 405 asymmetric resource
#   consumption, 407 inefficient algorithmic complexity, 409 data amplification
#   (decompression bombs), 674 uncontrolled recursion, 770 allocation without
#   limits, 789 excessive memory allocation, 799 interaction frequency, 834
#   excessive iteration, 835 infinite loop, 920 power consumption, 1050
#   excessive platform resource consumption in a loop, 1333 ReDoS. The corpus
#   category "redos" is cwe_to_category's bucket for ANY of the advisory's CWEs
#   being 400 / 1333 (an item's ``cwe`` is only the first one, e.g. Django's
#   CVE-2024-45230 is "CWE-120" + "CWE-400"), so that category is out too.
# - "Theoretical race conditions or timing attacks": reported as its own
#   ``timing_race`` group (CWE-203 observable discrepancy, 208 timing
#   discrepancy, 362 race, 367 TOCTOU), not folded into either side: the dev200
#   prompt excluded timing attacks wholesale, while a later prompt revision
#   makes removing an existing constant-time comparison reportable, and a
#   reversed fix of a timing / race CVE removes a concrete control, not a
#   theoretical one.
# - "Memory-safety issues in memory-safe languages": NOT used as an exclusion.
#   Every item is Python or JavaScript, so a memory-safety CWE tag there is
#   either a mislabelled DoS (Keystone's CVE-2013-0270, corrected in
#   build_pr_eval.LABEL_OVERRIDES; Django's CVE-2024-45230, caught by the
#   "redos" category) or a bug in a compiler written in Python whose OUTPUT is
#   memory-unsafe (vyper's CWE-129 / CWE-683): a real, in-scope logic bug of
#   the reviewed code.
SCOPE_OUT_CWES = frozenset({400, 405, 407, 409, 674, 770, 789, 799, 834, 835, 920, 1050, 1333})
SCOPE_OUT_CATEGORIES = frozenset({"redos"})
SCOPE_TIMING_RACE_CWES = frozenset({203, 208, 362, 367})
SCOPE_IN, SCOPE_TIMING_RACE, SCOPE_DOS = "in", "timing_race", "dos"


def _cwe_number(cwe) -> int | None:
    try:
        return int(str(cwe).upper().replace("CWE-", "").strip())
    except (TypeError, ValueError):
        return None


def scope_of(category: str | None, cwe: str | None) -> str:
    """``"dos"`` (excluded by the audit prompt), ``"timing_race"`` (the prompt's
    "theoretical race conditions or timing attacks" rule) or ``"in"``."""
    n = _cwe_number(cwe)
    if category in SCOPE_OUT_CATEGORIES or n in SCOPE_OUT_CWES:
        return SCOPE_DOS
    if n in SCOPE_TIMING_RACE_CWES:
        return SCOPE_TIMING_RACE
    return SCOPE_IN


# ---------------------------------------------------------------------------
# Leaky diffs: security vocabulary in the deleted lines
# ---------------------------------------------------------------------------
#
# Deliberately narrow: words a reviewer reads as "this line is about security"
# whether they appear in a comment, a string or an identifier (a deleted call
# to ``sanitize_html`` is as much of a tip-off as a "# prevent XSS" comment).
# Matched as case-insensitive substrings of each deleted line. Generic control
# words (check, validate, escape, safe, allow) are left out on purpose: they
# are everyday code vocabulary.
SECURITY_VOCAB = (
    "security", "vulnerab", "exploit", "attack", "malicious", "untrusted",
    "injection", "traversal", "sanitiz", "sanitis", "xss", "xxe", "csrf", "ssrf",
    "cve-", "ghsa-",
)


def deleted_lines(files: list[dict]) -> list[str]:
    """The text of every removed line of the PR's patches."""
    out = []
    for f in files:
        for n, text in diff_lines(f.get("patch")):
            if n is None and text.startswith("-"):
                out.append(text[1:])
    return out


def security_terms(lines: list[str]) -> list[str]:
    """``SECURITY_VOCAB`` terms found in ``lines`` (sorted, unique)."""
    low = [ln.lower() for ln in lines]
    return sorted({t for t in SECURITY_VOCAB if any(t in ln for ln in low)})


# ---------------------------------------------------------------------------
# Change anchors and touched functions
# ---------------------------------------------------------------------------


def anchors_by_path(files: list[dict]) -> dict[str, list[int]]:
    """New-file lines each file's patch touches, deletion points included
    (``patch_touched_lines``); files without new content are skipped."""
    out = {}
    for f in files:
        if f.get("new_content") is None or not f.get("patch"):
            continue
        try:
            lines = patch_touched_lines(f["patch"])
        except Exception:  # noqa: BLE001 - a malformed patch: no anchors
            lines = []
        n = len(f["new_content"].splitlines()) or 1
        lines = sorted({min(max(x, 1), n) for x in lines})
        if lines:
            out[f["path"]] = lines
    return out


def _innermost(functions: list[dict], line: int) -> dict | None:
    """The innermost named function containing ``line``, else the innermost
    anonymous one, else None."""
    around = [fn for fn in functions if fn["start_line"] <= line <= fn["end_line"]]
    if not around:
        return None
    named = [fn for fn in around if fn.get("name") and fn["name"] != "<anonymous>"]
    pool = named or around
    return min(pool, key=lambda fn: (fn["end_line"] - fn["start_line"], -fn["start_line"]))


def touched_functions(files: list[dict], anchors: dict[str, list[int]],
                      parser) -> dict[str, list[list]]:
    """Per file: ``[start, end, name]`` of the innermost function (see
    ``_innermost``) of the NEW file around each anchor line, de-duplicated."""
    out: dict[str, list[list]] = {}
    for f in files:
        lines = anchors.get(f["path"])
        if not lines or f.get("new_content") is None:
            continue
        ext = Path(f["path"]).suffix.lower()
        if parser is None or not parser.supports(ext):
            continue
        try:
            functions = parser.extract_functions(f["new_content"], ext)
        except Exception:  # noqa: BLE001 - unparsable file: no spans
            continue
        spans = set()
        for line in lines:
            fn = _innermost(functions, line)
            if fn is not None:
                spans.add((fn["start_line"], fn["end_line"], fn["name"]))
        if spans:
            out[f["path"]] = [list(s) for s in sorted(spans)]
    return out


def added_files(files: list[dict]) -> list[str]:
    """Paths the PR adds whole (no old content, new content present)."""
    return sorted(f["path"] for f in files
                  if f.get("old_content") is None and f.get("new_content") is not None)


# ---------------------------------------------------------------------------
# Benign provenance
# ---------------------------------------------------------------------------


def _commit(item: dict) -> tuple[str, str | None]:
    return (str(item.get("repo") or "").lower(), (item.get("meta") or {}).get("commit"))


def _file_sig(f: dict) -> tuple:
    return (f["path"], f.get("old_content"), f.get("new_content"))


def benign_provenance(items: list[dict]) -> dict[str, dict]:
    """Per benign item of ``items`` (one selection): its source / bystander
    origin, the vuln_introducing / vuln_fix items of the selection sharing its
    commit, and whether every one of its files appears with identical content
    in one of those fix items."""
    by_commit: dict[tuple, list[dict]] = {}
    for it in items:
        if it["kind"] in (KIND_INTRO, KIND_FIX):
            by_commit.setdefault(_commit(it), []).append(it)
    out = {}
    for it in items:
        if it["kind"] != KIND_BENIGN:
            continue
        key = _commit(it)
        shared = by_commit.get(key, []) if key[1] else []
        sigs = {_file_sig(f) for f in it.get("files") or []}
        subset_of_fix = any(
            x["kind"] == KIND_FIX and sigs and sigs <= {_file_sig(f) for f in x["files"]}
            for x in shared)
        out[it["id"]] = {
            "source": it.get("source"),
            "bystander_of": (it.get("meta") or {}).get("bystander_of"),
            "commit": key[1],
            "shares_commit_with": sorted(x["id"] for x in shared),
            "shares_commit_with_intro": any(x["kind"] == KIND_INTRO for x in shared),
            "shares_commit_with_fix": any(x["kind"] == KIND_FIX for x in shared),
            "identical_subset_of_fix_files": subset_of_fix,
        }
    return out


# ---------------------------------------------------------------------------
# All facts of a selection
# ---------------------------------------------------------------------------


def item_facts(item: dict, parser=None) -> dict:
    """The facts of one item (provenance is added by ``selection_facts``)."""
    files = item.get("files") or []
    meta = item.get("meta") or {}
    facts = {
        "labels": {"category": item.get("category"), "cwe": item.get("cwe"),
                   "label_override": meta.get("label_override")},
        "scope": scope_of(item.get("category"), item.get("cwe"))
        if item["kind"] != KIND_BENIGN else None,
    }
    if item["kind"] in (KIND_INTRO, KIND_FIX):
        anchors = anchors_by_path(files)
        facts.update(
            anchors_by_path=anchors,
            touched_functions_by_path=touched_functions(files, anchors, parser),
            added_files=added_files(files),
            deleted_security_terms=security_terms(deleted_lines(files)),
        )
    return facts


def selection_facts(items: list[dict], parser=None) -> dict[str, dict]:
    """``{item id: facts}`` for one selection (provenance needs the whole
    selection: which vuln items share a benign item's commit)."""
    prov = benign_provenance(items)
    out = {}
    for it in items:
        facts = item_facts(it, parser)
        if it["id"] in prov:
            facts["provenance"] = prov[it["id"]]
        out[it["id"]] = facts
    return out
