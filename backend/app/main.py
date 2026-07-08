from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.app.api.routes import analyze
from backend.app.config import settings
from backend.app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting RepoSentinel API Worker...")
    init_db()
    if settings.PRELOAD_MODELS:
        logger.info("Loading ML models into memory (this may take a few seconds)...")
        analyze.get_merger()
        analyze.get_report_generator()
        logger.info("ML Models loaded! API is ready to accept requests.")
    else:
        logger.info("PRELOAD_MODELS=False; skipping model warm-up (models load lazily).")
    yield
    logger.info("Shutting down...")


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

    @app.get("/health")
    def health_check():
        return {"status": "ok", "environment": settings.ENVIRONMENT}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Start the server without reload to protect local DB file locks
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
