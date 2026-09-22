"""Background scan execution.

Runs the retrieval + LLM pipeline for a queued Scan and persists the result.
Owns its own DB session (it runs outside the request lifecycle).
"""
import asyncio
import json
from datetime import UTC, datetime
from functools import lru_cache

from loguru import logger

from backend.app.core.markdown_renderer import (
    SYSTEM_PROMPT,
    build_user_prompt,
    render_markdown,
    severity_from_cvss,
    validate_findings,
)
from backend.app.db.models import Finding, Scan
from backend.app.db.session import SessionLocal


def _persist_findings(session, scan_id: str, raw: dict) -> list[Finding]:
    """Create a Finding row per retrieved match (CVE + team)."""
    rows: list[Finding] = []

    for match in raw.get("ghost_hunter_findings", []):
        rows.append(
            Finding(
                scan_id=scan_id,
                source="cve",
                collection=match.get("collection", "cve_corpus"),
                point_id=str(match.get("point_id", "")),
                cve_id=match.get("cve_id"),
                title=match.get("description") or match.get("cve_id") or "CVE match",
                severity=severity_from_cvss(match.get("severity")),
                file_path=match.get("anchor_file_path"),
                start_line=match.get("anchor_start_line"),
                end_line=match.get("anchor_end_line"),
                function_name=match.get("anchor_function_name"),
                similarity_score=float(match.get("similarity_score", 0.0)),
                rerank_score=float(match.get("rerank_score", 0.0)),
                rerank_prob=float(match.get("rerank_prob", 0.0)),
                adjusted_score=float(match.get("adjusted_score", match.get("rerank_prob", 0.0))),
                payload_json=json.dumps(
                    {k: match.get(k) for k in ("category", "severity", "language")}
                ),
            )
        )

    for match in raw.get("team_memory_findings", []):
        rows.append(
            Finding(
                scan_id=scan_id,
                source="team",
                collection=match.get("collection", "team_history"),
                point_id=str(match.get("point_id", "")),
                team_pr_id=match.get("pr_id"),
                title=match.get("title") or match.get("pr_id") or "Team memory match",
                file_path=match.get("anchor_file_path"),
                start_line=match.get("anchor_start_line"),
                end_line=match.get("anchor_end_line"),
                function_name=match.get("anchor_function_name"),
                similarity_score=float(match.get("similarity_score", 0.0)),
                rerank_score=float(match.get("rerank_score", 0.0)),
                rerank_prob=float(match.get("rerank_prob", 0.0)),
                adjusted_score=float(match.get("adjusted_score", match.get("rerank_prob", 0.0))),
                payload_json=json.dumps({"author": match.get("author"), "url": match.get("url")}),
            )
        )

    session.add_all(rows)
    session.flush()  # assign finding ids
    return rows


def _finding_out(row: Finding) -> dict:
    return {
        "finding_id": row.id,
        "point_id": row.point_id,
        "source": row.source,
        "title": row.title,
        "severity": row.severity,
        "cve_id": row.cve_id,
        "team_pr_id": row.team_pr_id,
        "file_path": row.file_path,
        "start_line": row.start_line,
        "similarity_score": row.similarity_score,
        "rerank_prob": row.rerank_prob,
    }


@lru_cache(maxsize=1)
def _get_parser():
    from backend.app.core.code_parser import CodeParser

    return CodeParser()


def _prompt_code_for_files(units: list[dict], raw: dict) -> str:
    """Build the LLM prompt code from only the functions that produced matches."""
    matched = {
        (f.get("anchor_file_path"), f.get("anchor_function_name"))
        for f in raw.get("ghost_hunter_findings", []) + raw.get("team_memory_findings", [])
    }
    parts = []
    for unit in units:
        if (unit["file_path"], unit["function_name"]) in matched:
            fn = unit["function_name"] or ""
            header = f"# {unit['file_path']}:{unit['start_line']} {fn}".rstrip()
            parts.append(f"{header}\n{unit['code']}")
    return "\n\n".join(parts)


def _row_snapshot(row: Finding) -> dict:
    """Capture the fields we need from a persisted row before the session commit
    expires its attributes."""
    return {
        "finding_id": row.id,
        "point_id": row.point_id,
        "source": row.source,
        "cve_id": row.cve_id,
        "team_pr_id": row.team_pr_id,
        "file_path": row.file_path,
        "start_line": row.start_line,
        "function_name": row.function_name,
    }


def _build_report_findings(validated: list[dict], row_snaps: list[dict]) -> list[dict]:
    """Join LLM-validated findings back to persisted retrieval rows so each carries
    a file/line anchor. A validated finding referencing an id retrieved for several
    functions yields one report finding per location; generic findings stay
    unanchored."""
    report: list[dict] = []
    for f in validated:
        base = {
            "severity": f.get("severity"),
            "cve_id": f.get("cve_id"),
            "team_pr_id": f.get("team_pr_id"),
            "title": f.get("title") or "Security finding",
            "explanation": f.get("explanation") or "",
            "fix_snippet": f.get("fix_snippet") or "",
        }
        matches = [
            r
            for r in row_snaps
            if (f.get("cve_id") and r["cve_id"] == f["cve_id"])
            or (f.get("team_pr_id") and str(r["team_pr_id"]) == str(f["team_pr_id"]))
        ]
        if matches:
            for r in matches:
                report.append(
                    {
                        **base,
                        "file_path": r["file_path"],
                        "start_line": r["start_line"],
                        "function_name": r["function_name"],
                        "finding_id": r["finding_id"],
                        "point_id": r["point_id"],
                        "source": r["source"],
                    }
                )
        else:
            report.append(
                {**base, "file_path": None, "start_line": None, "function_name": None,
                 "finding_id": None, "point_id": None, "source": None}
            )
    return report


async def _analyze_request(merger, request: dict) -> tuple[dict, str, str]:
    """Run the right retrieval mode. Returns (raw_findings, prompt_code, extra_note)."""
    if request.get("files"):
        from backend.app.config import settings
        from backend.app.core.analysis_planner import plan_units
        from backend.app.models.schemas import FileInput

        files = [FileInput(**f) for f in request["files"]]
        # tree-sitter parsing is CPU-bound; keep it off the event loop.
        units, dropped = await asyncio.to_thread(
            plan_units, files, _get_parser(), settings.MAX_UNITS_PER_SCAN
        )
        raw = await merger.analyze_units(units)
        note = (
            f"\n\n_Analysis capped at {settings.MAX_UNITS_PER_SCAN} functions; "
            f"{dropped} not scanned._"
            if dropped
            else ""
        )
        return raw, _prompt_code_for_files(units, raw), note

    code = request["code_snippet"]
    raw = await merger.analyze_code(code, request.get("language", "python"))
    return raw, code, ""


async def run_scan(scan_id: str, merger, router) -> None:
    """Execute a queued scan end-to-end and persist the outcome."""
    session = SessionLocal()
    try:
        scan = session.get(Scan, scan_id)
        if scan is None:
            logger.error("run_scan: scan {} not found", scan_id)
            return

        scan.status = "running"
        scan.started_at = datetime.now(UTC)
        session.commit()

        request = json.loads(scan.request_json)
        raw, prompt_code, extra_note = await _analyze_request(merger, request)
        cves = raw.get("ghost_hunter_findings", [])
        team = raw.get("team_memory_findings", [])

        # Persist retrieval findings and COMMIT before the (slow) LLM call so we
        # don't hold SQLite's write lock across the network round-trip. Snapshot
        # the rows first — commit expires the ORM attributes.
        rows = _persist_findings(session, scan_id, raw)
        findings_out = [_finding_out(r) for r in rows]
        row_snaps = [_row_snapshot(r) for r in rows]
        session.commit()

        provider_used = None
        report_findings: list[dict] = []
        if raw.get("is_vulnerable"):
            allowed_cves = {c.get("cve_id") for c in cves if c.get("cve_id")}
            allowed_prs = {str(t.get("pr_id")) for t in team if t.get("pr_id")}
            user_prompt = build_user_prompt(prompt_code, cves, team)
            llm_json, provider_used = await asyncio.to_thread(
                router.generate, SYSTEM_PROMPT, user_prompt
            )
            validated = validate_findings(llm_json.get("findings", []), allowed_cves, allowed_prs)
            report_findings = _build_report_findings(validated, row_snaps)
            report_markdown = render_markdown(validated, len(cves), len(team)) + extra_note
        else:
            report_markdown = render_markdown([], len(cves), len(team)) + extra_note

        # is_vulnerable reflects the LLM verdict (the precision filter), not the
        # raw high-recall retrieval — so it agrees with the report + report_findings.
        result = {
            "is_vulnerable": bool(report_findings),
            "report_markdown": report_markdown,
            "findings": findings_out,
            "report_findings": report_findings,
            "ghost_hunter_matches": len(cves),
            "team_memory_matches": len(team),
            "llm_provider_used": provider_used,
        }

        scan.is_vulnerable = result["is_vulnerable"]
        scan.report_markdown = report_markdown
        scan.result_json = json.dumps(result)
        scan.llm_provider_used = provider_used
        scan.status = "completed"
        scan.finished_at = datetime.now(UTC)
        session.commit()

    except Exception:
        logger.exception("run_scan failed for {}", scan_id)
        session.rollback()
        scan = session.get(Scan, scan_id)
        if scan is not None:
            scan.status = "failed"
            scan.error = "Analysis failed"  # sanitized; real traceback is logged
            scan.finished_at = datetime.now(UTC)
            session.commit()
    finally:
        session.close()
