from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.routes import analyze

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Booting RepoSentinel API Worker...")
    print("Loading ML models into memory (this may take a few seconds)...")
    analyze.get_merger()
    analyze.get_report_generator()
    print("ML Models loaded! API is ready to accept requests.")
    yield
    print("Shutting down...")

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Your codebase's permanent security memory.",
        version="1.0.0",
        lifespan=lifespan
    )

    # Set up CORS for the frontend dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(analyze.router, prefix=f"{settings.API_V1_STR}/analyze", tags=["Analysis"])

    @app.get("/health")
    def health_check():
        return {"status": "ok", "environment": settings.ENVIRONMENT}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Start the server without reload to protect local DB file locks
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
