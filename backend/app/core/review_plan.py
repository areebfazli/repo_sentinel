"""Turn a scan's analysis units + evidence into LLM review prompts.

A *review unit* is a planner unit (``analysis_planner.plan_units``: one function,
or a whole file / snippet) plus the evidence gathered for it: the retrieved CVE
and team matches anchored to it, and its prompt-ready code
(``untrusted.sanitize_untrusted``, same line count as the real code). Pure.
"""
from typing import Any

from backend.app.core.untrusted import sanitize_untrusted

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
    units: list[dict[str, Any]], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """One review unit per planner unit, with its retrieval matches attached.

    Files-mode matches carry ``anchor_*`` keys naming their unit; unanchored
    matches (snippet mode) belong to the single (first) unit. Matches stay in
    relevance order.
    """
    review = [
        {**u, "key": unit_key(u), "prompt_code": sanitize_untrusted(u.get("code") or ""),
         "cves": [], "team": []}
        for u in units
    ]
    by_key = {r["key"]: r for r in review}
    for field, target in (("ghost_hunter_findings", "cves"), ("team_memory_findings", "team")):
        for match in raw.get(field, []):
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
