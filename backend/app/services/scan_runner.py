"""Background scan execution.

Runs the retrieval + LLM review pipeline for a queued Scan and persists the
result. Owns its own DB session (it runs outside the request lifecycle).
"""
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from functools import lru_cache

from loguru import logger
from sqlalchemy import update

from backend.app.config import settings
from backend.app.core.evidence import (
    guard_alert_findings,
    guard_evidence,
    guess_language,
    plan_with_touched_lines,
    semgrep_evidence,
)
from backend.app.core.markdown_renderer import (
    SYSTEM_PROMPT,
    build_user_prompt,
    render_markdown,
    severity_from_cvss,
    validate_findings,
)
from backend.app.core.review_plan import assign_uids, build_review_units, snippet_unit
from backend.app.core.scoring import relevance
from backend.app.core.untrusted import new_nonce
from backend.app.db.models import Finding, Scan
from backend.app.db.session import SessionLocal

_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def get_scan_semaphore() -> asyncio.Semaphore:
    """The process-wide gate on concurrent model inference.

    Built lazily inside the running loop — a module-level Semaphore would bind to
    whatever loop existed at import time (none, under uvicorn). Rebuilt when the
    loop changes so tests, which get a fresh loop per case, never wait on a
    semaphore belonging to a closed one.
    """
    global _semaphore, _semaphore_loop

    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_SCANS)
        _semaphore_loop = loop
    return _semaphore


def reset_scan_semaphore() -> None:
    """Drop the cached gate so the next scan rebuilds it (tests re-read
    MAX_CONCURRENT_SCANS through this)."""
    global _semaphore, _semaphore_loop

    _semaphore = None
    _semaphore_loop = None


def _score_columns(match: dict) -> dict:
    """Score columns for a Finding row.

    With the reranker off (RERANKER_ENABLED=false) a match carries no
    rerank_score/rerank_prob. The columns are NOT NULL (create_all, no
    migrations), so they get a 0.0 placeholder and payload_json records
    ``"reranked": false`` — ``_finding_out`` then reports rerank_prob as None
    rather than a fake "0% relevant". adjusted_score is always set by
    finalize_matches (relevance-based either way).
    """
    reranked = match.get("rerank_prob") is not None
    return {
        "similarity_score": float(match.get("similarity_score", 0.0)),
        "rerank_score": float(match.get("rerank_score") or 0.0) if reranked else 0.0,
        "rerank_prob": float(match["rerank_prob"]) if reranked else 0.0,
        "adjusted_score": float(match.get("adjusted_score", relevance(match))),
    }


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
                **_score_columns(match),
                # Twin scores live in payload_json rather than new columns: tables
                # are made by create_all (no migrations), which would not add
                # columns to an existing findings table.
                payload_json=json.dumps(
                    {
                        **{
                            k: match.get(k)
                            for k in (
                                "category", "severity", "language", "sim_fixed", "twin_margin",
                            )
                        },
                        "reranked": match.get("rerank_prob") is not None,
                    }
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
                **_score_columns(match),
                payload_json=json.dumps(
                    {
                        "author": match.get("author"),
                        "url": match.get("url"),
                        "reranked": match.get("rerank_prob") is not None,
                    }
                ),
            )
        )

    session.add_all(rows)
    session.flush()  # assign finding ids
    return rows


def _finding_out(row: Finding) -> dict:
    payload = json.loads(row.payload_json) if row.payload_json else {}
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
        # None when no cross-encoder scored it (reranker off); rows persisted
        # before the flag existed were all reranked.
        "rerank_prob": row.rerank_prob if payload.get("reranked", True) else None,
        "sim_fixed": payload.get("sim_fixed"),
        "twin_margin": payload.get("twin_margin"),
    }


@lru_cache(maxsize=1)
def _get_parser():
    from backend.app.core.code_parser import CodeParser

    return CodeParser()


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


def _dedupe_key(prefix: str, *parts) -> str:
    """Stable id of a report finding within its function, for the Action's
    comment markers (which also hash file + function): survives line shifts and
    LLM rewording of the title, not a change of the quoted code."""
    text = "|".join(" ".join(str(p or "").split()) for p in parts)
    return f"{prefix}:{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def _build_report_findings(
    validated: list[dict], units_by_uid: dict[str, dict], row_snaps: list[dict]
) -> list[dict]:
    """Anchor LLM-validated findings (one prompt's, uids per prompt) to their
    unit's file / function and the exact quoted line. A finding citing a
    retrieved CVE / team match also links to that match's persisted row (same
    function), so feedback can target it."""
    report: list[dict] = []
    for f in validated:
        unit = units_by_uid[f["unit"]]
        where = (unit.get("file_path"), unit.get("function_name"))
        row = next(
            (
                r for r in row_snaps
                if (r["file_path"], r["function_name"]) == where
                and (
                    (f.get("cve_id") and r["cve_id"] == f["cve_id"])
                    or (f.get("team_pr_id") and str(r["team_pr_id"]) == str(f["team_pr_id"]))
                )
            ),
            None,
        )
        quote_first = (f.get("quoted_code") or "").strip().splitlines()[:1]
        report.append(
            {
                "severity": f.get("severity"),
                "cve_id": f.get("cve_id"),
                "team_pr_id": f.get("team_pr_id"),
                "cwe": f.get("cwe"),
                "title": f.get("title") or "Security finding",
                "explanation": f.get("explanation") or "",
                "reasoning": f.get("reasoning") or "",
                "quoted_code": f.get("quoted_code") or "",
                "fix_snippet": f.get("fix_snippet") or "",
                "file_path": unit.get("file_path"),
                "start_line": unit.get("start_line"),
                "function_name": unit.get("function_name"),
                "line": f.get("line"),
                "end_line": f.get("end_line"),
                "finding_id": row["finding_id"] if row else None,
                "point_id": row["point_id"] if row else None,
                "source": "llm",
                "deterministic": False,
                "dedupe_key": _dedupe_key("llm", quote_first[0] if quote_first else "",
                                          f.get("cwe") or f.get("title")),
            }
        )
    return report


@lru_cache(maxsize=1)
def _get_semgrep():
    """The shared Semgrep scanner, or None when SEMGREP_ENABLED is off (a
    missing engine is handled inside the scanner: a warning, no evidence)."""
    if not settings.SEMGREP_ENABLED:
        return None
    from backend.app.core.semgrep_scanner import SemgrepScanner

    return SemgrepScanner(
        timeout_s=settings.SEMGREP_TIMEOUT_S,
        exclude_rules=frozenset(settings.SEMGREP_EXCLUDED_RULES),
    )


def _semgrep_task(units: list[dict], sources: dict[str, str] | None) -> asyncio.Task:
    """Start Semgrep in a worker thread (a subprocess: runs alongside the
    embedding / retrieval work instead of after it)."""
    return asyncio.create_task(
        asyncio.to_thread(
            semgrep_evidence, _get_semgrep(), units, sources, settings.SEMGREP_MIN_SEVERITY,
            frozenset(settings.SEMGREP_EXCLUDED_RULES),
        )
    )


async def _with_semgrep(semgrep: asyncio.Task, work):
    """Await ``work`` then the Semgrep task; on failure don't leave the task
    un-awaited (its thread finishes on its own, bounded by SEMGREP_TIMEOUT_S)."""
    try:
        result = await work
    except BaseException:
        semgrep.cancel()
        raise
    return result, await semgrep


async def _analyze_request(merger, request: dict) -> dict:
    """Run the right retrieval mode plus the deterministic evidence. Returns
    {raw, units, semgrep, guard, notes}: the retrieval result, the analysis
    units the LLM reviews, Semgrep hits and guard_diff results per unit key,
    and trusted report notes."""
    if request.get("files"):
        from backend.app.core.analysis_planner import plan_units
        from backend.app.models.schemas import FileInput

        files = [FileInput(**f) for f in request["files"]]
        # Deletion points count as changes, so a function whose only change is
        # a removed guard is analysed. tree-sitter parsing is CPU-bound; keep it
        # off the event loop.
        units, dropped = await asyncio.to_thread(
            plan_units, plan_with_touched_lines(files), _get_parser(),
            settings.MAX_UNITS_PER_SCAN,
        )
        semgrep = _semgrep_task(units, {f.path: f.content for f in files})
        raw, semgrep_hits = await _with_semgrep(semgrep, merger.analyze_units(units))
        guard = await asyncio.to_thread(guard_evidence, files, units, _get_parser())
        notes = (
            [f"_Analysis capped at {settings.MAX_UNITS_PER_SCAN} functions; "
             f"{dropped} not scanned._"]
            if dropped
            else []
        )
        return {"raw": raw, "units": units, "semgrep": semgrep_hits, "guard": guard,
                "notes": notes}

    code = request["code_snippet"]
    language = request.get("language")
    units = [snippet_unit(code, language or guess_language(code))] if code.strip() else []
    semgrep = _semgrep_task(units, None)
    raw, semgrep_hits = await _with_semgrep(semgrep, merger.analyze_code(code, language))
    # guard_diff needs the previous version of the code: not applicable to a snippet.
    return {"raw": raw, "units": units, "semgrep": semgrep_hits, "guard": {}, "notes": []}


def _static_out(semgrep: dict) -> list[dict]:
    return [
        {"file_path": key[0], "function_name": key[1], "start_line": key[2],
         "rule_id": h["rule_id"], "severity": h.get("severity"), "cwe": h.get("cwe") or [],
         "line": h.get("line"), "message": h.get("message") or ""}
        for key, hits in semgrep.items()
        for h in hits
    ]


def _mark_failed(session, scan_id: str, error: str) -> None:
    """Flip a scan to failed with a sanitized message (the real cause is logged)."""
    session.rollback()
    scan = session.get(Scan, scan_id)
    if scan is not None:
        scan.status = "failed"
        scan.error = error
        scan.finished_at = datetime.now(UTC)
        session.commit()


async def run_scan(scan_id: str, merger, router) -> None:
    """Execute a queued scan end-to-end and persist the outcome.

    The scan stays ``queued`` until the concurrency gate admits it, so a burst of
    requests waits its turn instead of running N model inferences at once. An
    interrupted wait never held a permit, which leaves the scan queued — exactly
    the state restart recovery re-runs.
    """
    async with get_scan_semaphore():
        await _run_scan_guarded(scan_id, merger, router)


async def _run_scan_guarded(scan_id: str, merger, router) -> None:
    """The scan itself; runs only while holding a permit from the gate."""
    session = SessionLocal()
    try:
        # Atomic claim: only one runner (across processes too — every worker's
        # restart recovery sees the same queued rows) may take a scan.
        claimed = session.execute(
            update(Scan)
            .where(Scan.id == scan_id, Scan.status == "queued")
            .values(status="running", started_at=datetime.now(UTC))
        ).rowcount
        session.commit()
        if not claimed:
            logger.warning("run_scan: scan {} missing or already claimed; skipping", scan_id)
            return
        scan = session.get(Scan, scan_id)

        request = json.loads(scan.request_json)
        analysis = await _analyze_request(merger, request)
        raw, notes = analysis["raw"], analysis["notes"]
        cves = raw.get("ghost_hunter_findings", [])
        team = raw.get("team_memory_findings", [])

        # Persist retrieval findings and COMMIT before the (slow) LLM call so we
        # don't hold SQLite's write lock across the network round-trip. Snapshot
        # the rows first — commit expires the ORM attributes.
        rows = _persist_findings(session, scan_id, raw)
        findings_out = [_finding_out(r) for r in rows]
        row_snaps = [_row_snapshot(r) for r in rows]
        session.commit()

        # The LLM reviews every analysis unit on its own merits; retrieved
        # matches are reference context, not a precondition for the call.
        review_units = build_review_units(
            analysis["units"], raw, analysis["semgrep"], analysis["guard"]
        )
        provider_used = None
        # Deterministic guard_diff alerts are reported whatever the LLM says.
        report_findings: list[dict] = guard_alert_findings(analysis["guard"])
        if review_units:
            allowed_cves = {c.get("cve_id") for c in cves if c.get("cve_id")}
            allowed_prs = {str(t.get("pr_id")) for t in team if t.get("pr_id")}
            batch = assign_uids(review_units)
            user_prompt = build_user_prompt(batch, new_nonce())
            llm_json, provider_used = await router.generate(SYSTEM_PROMPT, user_prompt)
            raw_findings = llm_json.get("findings") if isinstance(llm_json, dict) else None
            validated = validate_findings(raw_findings, batch, allowed_cves, allowed_prs)
            report_findings += _build_report_findings(
                validated, {u["uid"]: u for u in batch}, row_snaps
            )
        if provider_used == "mock":
            notes.append("_LLM_PROVIDER=mock: no real review was performed._")
        static_out = _static_out(analysis["semgrep"])
        report_markdown = render_markdown(
            report_findings, len(cves), len(team), units_reviewed=len(review_units),
            static_hits=len(static_out), notes=notes,
        )

        # is_vulnerable reflects the reviewed findings (the precision filter), not
        # the raw high-recall retrieval — so it agrees with the report.
        result = {
            "is_vulnerable": bool(report_findings),
            "report_markdown": report_markdown,
            "findings": findings_out,
            "report_findings": report_findings,
            "ghost_hunter_matches": len(cves),
            "team_memory_matches": len(team),
            "llm_provider_used": provider_used,
            "static_analysis": static_out,
            "guard_diff": list(analysis["guard"].values()),
        }

        scan.is_vulnerable = result["is_vulnerable"]
        scan.report_markdown = report_markdown
        scan.result_json = json.dumps(result)
        scan.llm_provider_used = provider_used
        scan.status = "completed"
        scan.finished_at = datetime.now(UTC)
        session.commit()

    except asyncio.CancelledError:
        # Shutdown (or an explicit cancel) killed the scan mid-flight; record it
        # rather than leaving the row stuck in "running", then propagate.
        logger.warning("run_scan cancelled for {}", scan_id)
        _mark_failed(session, scan_id, "interrupted")
        raise
    except Exception:
        logger.exception("run_scan failed for {}", scan_id)
        _mark_failed(session, scan_id, "Analysis failed")
    finally:
        session.close()
