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

    # LLM providers (OpenAI-compatible chat-completions endpoints). Routing and
    # model choice are code defaults; .env only needs the API keys (env vars
    # still override). A primary provider with a missing key hard-errors at
    # startup — never a silent mock; a fallback provider without a key is skipped.
    # Chain: openrouter:OPENROUTER_MODEL -> openrouter:OPENROUTER_FALLBACK_MODEL
    #        -> groq:GROQ_MODEL -> groq:GROQ_FALLBACK_MODEL.
    # OpenRouter's free models share an upstream pool and often 429 with
    # "upstream_provider_shared_pool" (seen for both qwen and gemma), so a
    # cross-provider fallback to our own Groq key is required, not optional.
    LLM_PROVIDER: str = "openrouter"    # groq | gemini | openrouter | mock
    LLM_FALLBACK_PROVIDER: str | None = "groq"
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    # Same-provider (Groq) fallback: a different model, so not hit by the same per-model rate limit.
    GROQ_FALLBACK_MODEL: str | None = "qwen/qwen3.8-27b"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    # OpenRouter (aimed at its free ":free" models, which have daily request caps).
    OPENROUTER_API_KEY: str | None = None
    # Supports structured outputs but rejects response_format json_object, so it
    # is always sent without it (see _NO_RESPONSE_FORMAT_MODELS in llm_client).
    OPENROUTER_MODEL: str = "qwen/qwen3.8-27b:free"
    # Same-provider (OpenRouter) fallback model on a different upstream; free
    # models rate-limit per model. Supports response_format, 262K context.
    # Set empty in .env to disable.
    OPENROUTER_FALLBACK_MODEL: str | None = "google/gemma-4-31b-it:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_RETRIES: int = 1                 # same-client retries on 429/5xx/timeout
    # Per-scan LLM budget. Units are ordered by evidence (guard_diff alert, Semgrep
    # hit, guard_removed, retrieval similarity) and packed into as few prompts as
    # fit; units beyond LLM_MAX_CALLS_PER_SCAN prompts are listed in the result as
    # not reviewed. Prompt tokens are estimated (chars / 4 x 1.25). 6000 keeps one
    # prompt plus a reasoning model's answer under Groq's free-tier 8K tokens/min
    # per model (the fallback provider); OpenRouter has no per-minute token cap.
    LLM_MAX_PROMPT_TOKENS: int = 6000
    LLM_MAX_UNITS_PER_PROMPT: int = 6
    LLM_MAX_CALLS_PER_SCAN: int = 6
    # Retrieved CVE matches shown per unit (team matches likewise), best first.
    LLM_MAX_CVES_PER_UNIT: int = 2
    # Pacing of the scan's (sequential) LLM calls, per model: estimated tokens
    # (prompt + LLM_OUTPUT_TOKENS_ESTIMATE) per rolling minute. Keys are a
    # provider or a "<provider>:<model>" label; null / missing = no limit. Groq's
    # free tier allows ~8K tokens/min per model, so 6 back-to-back ~6K-token
    # calls would 429 without it; OpenRouter's free models have request caps only.
    LLM_TPM_LIMITS: dict[str, int | None] = {"groq": 8000, "openrouter": None}
    LLM_OUTPUT_TOKENS_ESTIMATE: int = 1500
    # Longest single wait (rate budget or Retry-After) before trying the next
    # client instead; a daily-limit Retry-After is far longer.
    LLM_MAX_WAIT_S: float = 60.0
    # Wall-time budget for a scan's LLM stage; units whose call can't start in
    # time are listed as not reviewed (reason "time_budget"). The Action polls
    # for 15 minutes in total.
    LLM_SCAN_MAX_WALL_S: float = 480.0

    # Service URLs (for production)
    DATABASE_URL: str | None = None
    QDRANT_HOST: str | None = None
    QDRANT_PORT: int | None = 6333

    # Absolute path override for the dev SQLite file (used by the test suite to
    # avoid clobbering the real repo_sentinel.sqlite).
    DEV_SQLITE_PATH: str | None = None

    # jina-embeddings-v2-base-code + mean pooling is the calibrated default (see
    # ml/evaluation). No CODEBERT_MODEL alias: a stale legacy env var must not
    # silently downgrade the model the thresholds/baseline were calibrated for.
    EMBEDDING_MODEL: str = "jinaai/jina-embeddings-v2-base-code"
    EMBEDDING_POOLING: str = "mean"  # "cls" or "mean"
    EMBEDDING_DIM: int = 768
    EMBEDDING_MAX_TOKENS: int = 2048
    # jina v2 uses a custom BERT-ALiBi architecture that requires trust_remote_code=True
    # to load; kept as a setting so a plain HF model can turn it off later.
    EMBEDDING_TRUST_REMOTE_CODE: bool = True
    # Cross-encoder rerank stage. Off: on 150 held-out OSV items category hit was
    # 0.373 without it vs 0.387-0.413 with bge-v2-m3 (noise) at 25-49 s/item (ROADMAP 1d).
    # When off, the Reranker is never constructed and candidates keep similarity order.
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # Only used when RERANKER_ENABLED. 512 was the best-measured length for
    # bge-reranker-v2-m3 and half the cost of 1024; activation memory scales with it.
    RERANKER_MAX_TOKENS: int = 512
    RERANKER_BATCH_SIZE: int = 8

    # Retrieval thresholds (single source of truth; calibrated via ml/evaluation
    # for jina-embeddings-v2-base-code with the reranker off — see baseline.json).
    # 0.25 gives recall 1.0; precision is ~flat at 0.5 across thresholds (a safe
    # query and its vulnerable twin embed alike), so the LLM report is the real
    # precision filter and we favour recall. RERANK_THRESHOLD only applies when
    # RERANKER_ENABLED (it gates rerank_prob, which is absent otherwise); rerank
    # probability didn't separate vulnerable from fixed at any threshold in the
    # reranker eval (precision <= 0.5 at 0.1-0.9), so it stays 0.0 (keep all).
    SIM_THRESHOLD_CVE: float = 0.25
    SIM_THRESHOLD_TEAM: float = 0.25
    RERANK_THRESHOLD: float = 0.0        # gate on rerank_prob = sigmoid(logit); 0.0 = keep all
    RETRIEVAL_TOP_K: int = 3             # findings returned per collection
    ANN_CANDIDATES: int = 10             # broad ANN recall before the (optional) rerank
    # Patched-twin gate (ROADMAP 1b): drop a CVE candidate whose twin_margin
    # (cos(query, vulnerable) - cos(query, fixed)) is below this, i.e. the code
    # looks at least as much like the fix as like the bug. Candidates without a
    # stored fix always pass. None = off until ml/evaluation (--margin-sweep)
    # calibrates it.
    TWIN_MARGIN_MIN: float | None = None

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

    # Static-analysis evidence for the LLM review (Semgrep CE or Opengrep with the
    # vendored GitLab sast-rules, backend/app/rules/semgrep). Evidence, not a gate:
    # a high/critical hit fired on 3.2% of vulnerable functions vs 1.4% of their
    # fixed twins and 0.5% of ordinary ones on the eval sets. A missing engine only
    # logs a warning (the scan runs without it). Excluded: Bandit's assert rule
    # (the largest source of hits on fixed and ordinary code), Python `random` and
    # requests-without-timeout (not vulnerabilities in most code).
    SEMGREP_ENABLED: bool = True
    SEMGREP_MIN_SEVERITY: str = "high"      # info | low | medium | high | critical
    SEMGREP_TIMEOUT_S: float = 60.0         # whole engine run (~7-13 s fixed startup)
    SEMGREP_EXCLUDED_RULES: list[str] = [
        "python_assert_rule-assert-used",
        "python_random_rule-random",
        "python_requests_rule-request-without-timeout",
    ]

    # Cap on per-scan analysis units (functions) in files mode.
    MAX_UNITS_PER_SCAN: int = 50

    # Scans running at once in this process; the rest wait as "queued".
    MAX_CONCURRENT_SCANS: int = 2

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
