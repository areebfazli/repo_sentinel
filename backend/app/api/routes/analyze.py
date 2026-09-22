import json
import uuid
from datetime import UTC
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.app.api.deps import require_api_key
from backend.app.config import settings
from backend.app.core.llm_client import LLMRouter
from backend.app.core.rag_merger import RagMerger
from backend.app.db.models import Scan
from backend.app.db.session import SessionLocal
from backend.app.models.schemas import (
    AnalyzeAccepted,
    AnalyzeRequest,
    AnalyzeResult,
    JobStatusResponse,
)
from backend.app.services.scan_runner import run_scan

router = APIRouter()


@lru_cache(maxsize=1)
def get_merger() -> RagMerger:
    """Shared RagMerger singleton (loads gigabytes of models — construct once)."""
    return RagMerger()


@lru_cache(maxsize=1)
def get_llm_router() -> LLMRouter:
    """Shared LLM router singleton (validates provider keys at construction)."""
    return LLMRouter()


async def run_scan_job(job_id: str) -> None:
    """Run a scan against the shared singletons.

    The route hands ``run_scan`` its own ``Depends``-resolved instances (so tests
    can override them); callers outside the request lifecycle — restart recovery —
    go through here instead, resolving the singletons only once there is a scan
    to run, so a lazily-loaded merger stays unbuilt until it is really needed.
    """
    await run_scan(job_id, get_merger(), get_llm_router())


def _iso(dt) -> str | None:
    """ISO-8601 with a UTC designator. All writes are UTC, but SQLite can hand back
    naive datetimes, so coerce to UTC to avoid offset-less strings that clients
    misread as local time."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


@router.post(
    "/",
    status_code=202,
    response_model=AnalyzeAccepted,
    dependencies=[Depends(require_api_key)],
)
async def analyze_pr_code(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    merger: RagMerger = Depends(get_merger),
    llm_router: LLMRouter = Depends(get_llm_router),
):
    """Queue a scan and return a job id to poll. The heavy work runs in the background."""
    job_id = uuid.uuid4().hex
    session = SessionLocal()
    try:
        session.add(
            Scan(
                id=job_id,
                status="queued",
                mode="files" if request.files else "snippet",
                request_json=request.model_dump_json(),
                repo=str(request.repo_url) if request.repo_url else None,
                pr_number=request.pr_number,
                author=request.author,
            )
        )
        session.commit()
    finally:
        session.close()

    background_tasks.add_task(run_scan, job_id, merger, llm_router)

    return AnalyzeAccepted(
        job_id=job_id,
        status="queued",
        poll_url=f"{settings.API_V1_STR}/analyze/{job_id}",
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_scan_status(job_id: str):
    """Poll a scan job; includes the full result once completed."""
    session = SessionLocal()
    try:
        scan = session.get(Scan, job_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="Job not found")

        result = None
        if scan.status == "completed" and scan.result_json:
            result = AnalyzeResult(**json.loads(scan.result_json))

        return JobStatusResponse(
            job_id=scan.id,
            status=scan.status,
            created_at=_iso(scan.created_at),
            finished_at=_iso(scan.finished_at),
            error=scan.error,
            result=result,
        )
    finally:
        session.close()
