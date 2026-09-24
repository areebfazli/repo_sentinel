import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.app.api.routes import analyze, feedback
from backend.app.config import settings
from backend.app.db.session import init_db

# Strong references to the re-queued scans: asyncio only holds a weak one, so a
# task dropped here could be garbage-collected mid-scan.
_recovery_tasks: set[asyncio.Task] = set()
# Scans this process has already re-queued — cheap insurance against recovery
# being run twice (a second lifespan in the same process, say) scanning twice.
_recovered_scan_ids: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting RepoSentinel API Worker...")
    logger.info("Reranker: {}", _reranker_status())
    init_db()
    # Construct the LLM router eagerly so a missing provider key fails fast at
    # boot (cheap; no model load) rather than on the first scan.
    analyze.get_llm_router()
    if settings.PRELOAD_MODELS:
        logger.info("Loading ML models into memory (this may take a few seconds)...")
        analyze.get_merger()
        logger.info("ML Models loaded! API is ready to accept requests.")
    else:
        logger.info("PRELOAD_MODELS=False; skipping model warm-up (models load lazily).")
    # Only once the singletons live traffic uses are warm — and without awaiting
    # the scans, which would hold up startup.
    await recover_orphaned_scans()
    yield
    logger.info("Shutting down...")
    await _cancel_recovery_tasks()


def _reranker_status() -> str:
    if settings.RERANKER_ENABLED:
        return (
            f"enabled ({settings.RERANKER_MODEL}, max_tokens={settings.RERANKER_MAX_TOKENS})"
        )
    return "disabled (RERANKER_ENABLED=false; candidates ranked by similarity)"


async def recover_orphaned_scans() -> list[asyncio.Task]:
    """Reconcile the scans a restart orphaned; returns the tasks it scheduled.

    BackgroundTasks are in-process, so a restart leaves both states behind:
    "running" died mid-flight with unknown progress (fail it), while "queued"
    never started and can simply be run now. Re-queued scans go through the same
    concurrency gate as live traffic, so recovering a backlog can't stampede.
    """
    from sqlalchemy import select, update

    from backend.app.db.models import Scan
    from backend.app.db.session import SessionLocal

    with SessionLocal() as session:
        session.execute(
            update(Scan)
            .where(Scan.status == "running")
            .values(status="failed", error="interrupted by server restart")
        )
        queued = list(session.scalars(select(Scan.id).where(Scan.status == "queued")))
        session.commit()

    tasks = []
    for job_id in queued:
        if job_id in _recovered_scan_ids:
            continue
        _recovered_scan_ids.add(job_id)
        task = asyncio.create_task(analyze.run_scan_job(job_id), name=f"recover-scan-{job_id}")
        _recovery_tasks.add(task)
        task.add_done_callback(_on_recovery_done)
        tasks.append(task)

    if tasks:
        logger.info("Re-queued {} scan(s) orphaned by the last restart.", len(tasks))
    return tasks


def _on_recovery_done(task: asyncio.Task) -> None:
    """run_scan handles its own failures, so anything surfacing here is a bug —
    retrieve it rather than let it die as a never-retrieved task exception."""
    _recovery_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.opt(exception=task.exception()).error("Re-queued scan {} crashed", task.get_name())


async def _cancel_recovery_tasks() -> None:
    """Stop re-queued scans still in flight at shutdown (they mark themselves
    failed on cancellation) instead of letting the loop close over them."""
    pending = list(_recovery_tasks)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def create_app() -> FastAPI:
    # Fail fast: production must ship a shared secret for the analysis endpoints.
    if not settings.is_dev and not settings.REPOSENTINEL_API_KEY:
        raise RuntimeError(
            "REPOSENTINEL_API_KEY must be set when ENVIRONMENT=production"
        )

    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Your codebase's permanent security memory.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS is scoped to the configured dashboard origins (no wildcard + credentials).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Log the real traceback server-side; never leak internals to the client.
        logger.exception("Unhandled error on {} {}", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(
        analyze.router, prefix=f"{settings.API_V1_STR}/analyze", tags=["Analysis"]
    )
    app.include_router(
        feedback.router, prefix=f"{settings.API_V1_STR}/feedback", tags=["Feedback"]
    )

    @app.get("/health")
    def health_check():
        return {
            "status": "ok",
            "environment": settings.ENVIRONMENT,
            "reranker_enabled": settings.RERANKER_ENABLED,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Start the server without reload to protect local DB file locks
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
