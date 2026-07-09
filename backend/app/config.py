import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "RepoSentinel"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"  # "development" or "production"

    # Pre-load the heavy ML models at boot. Tests set this to False so the
    # FastAPI lifespan doesn't download gigabytes of weights during collection.
    PRELOAD_MODELS: bool = True

    # Shared secret checked on the analysis/feedback endpoints. Optional in dev;
    # required in production (enforced at startup by create_app()).
    REPOSENTINEL_API_KEY: str | None = None

    # Browser origins allowed to call the API (the static dashboard).
    CORS_ORIGINS: list[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]

    # API Keys
    GITHUB_TOKEN: str | None = None

    # LLM providers (OpenAI-compatible chat-completions endpoints).
    # A configured provider with a missing key hard-errors at startup — never a silent mock.
    LLM_PROVIDER: str = "groq"          # groq | gemini | mock
    LLM_FALLBACK_PROVIDER: str | None = "gemini"
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    LLM_TIMEOUT_SECONDS: int = 60

    # Service URLs (for production)
    DATABASE_URL: str | None = None
    REDIS_URL: str | None = None
    QDRANT_HOST: str | None = None
    QDRANT_PORT: int | None = 6333

    # Absolute path override for the dev SQLite file (used by the test suite to
    # avoid clobbering the real repo_sentinel.sqlite).
    DEV_SQLITE_PATH: str | None = None

    # UniXcoder + mean pooling is the calibrated default (see ml/evaluation).
    # No CODEBERT_MODEL alias: a stale legacy env var must not silently downgrade
    # the model the thresholds/baseline were calibrated for.
    EMBEDDING_MODEL: str = "microsoft/unixcoder-base"
    EMBEDDING_POOLING: str = "mean"  # "cls" or "mean"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval thresholds (single source of truth; calibrated via ml/evaluation
    # for unixcoder-base + mean pooling — see ml/evaluation/baseline.json).
    # 0.25 gives recall ~0.96; precision is ~flat across thresholds (safe and
    # vulnerable near-misses embed similarly), so the LLM report is the real
    # precision filter and we favour recall. Mean pooling dilutes code with
    # comment tokens, so real commented code scores lower than clean eval code —
    # another reason to keep this gate low.
    SIM_THRESHOLD_CVE: float = 0.25
    SIM_THRESHOLD_TEAM: float = 0.25
    RERANK_THRESHOLD: float = 0.0        # gate on sigmoid(logit); 0.0 = keep all reranked
    RETRIEVAL_TOP_K: int = 3             # findings returned per collection
    ANN_CANDIDATES: int = 10             # broad ANN recall before reranking

    # Feedback loop tuning.
    FEEDBACK_SUPPRESS_NET: int = -2                # net vote at/below which a memory is dropped
    FEEDBACK_DOWNWEIGHT_PER_VOTE: float = 0.15     # per net-negative vote score penalty

    # Team Memory recency/seniority weighting.
    RECENCY_HALF_LIFE_DAYS: int = 180
    SENIORITY_WEIGHTS: dict[str, float] = {
        "OWNER": 1.2,
        "MEMBER": 1.1,
        "COLLABORATOR": 1.05,
    }

    # Cap on per-scan analysis units (functions) in files mode.
    MAX_UNITS_PER_SCAN: int = 50

    # Hybrid dense+sparse (BM25) retrieval with RRF fusion. Off by default; enabling
    # it requires re-ingesting BOTH collections (--recreate) so points carry sparse
    # vectors. HYBRID_SPARSE_MIN_SCORE gates the sparse prefetch so only strong
    # lexical matches are fused in.
    HYBRID_ENABLED: bool = False
    HYBRID_SPARSE_MIN_SCORE: float = 10.0  # best-observed in ml/evaluation (recall 1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def get_database_url(self) -> str:
        """Return SQLite URL for dev, Postgres for prod."""
        if self.is_dev or not self.DATABASE_URL:
            db_path = self.DEV_SQLITE_PATH or str(BASE_DIR / "repo_sentinel.sqlite")
            return f"sqlite:///{db_path}"
        return self.DATABASE_URL

    @property
    def qdrant_local_path(self) -> str | None:
        """Return local path for Qdrant if in dev mode."""
        if self.is_dev:
            path = BASE_DIR / "qdrant_data"
            os.makedirs(path, exist_ok=True)
            return str(path)
        return None

settings = Settings()
