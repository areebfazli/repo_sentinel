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
    # Sampling temperature of review calls to a model WITHOUT an LLM_SAMPLING
    # entry. 0.2 is what production has always used; the eval
    # (ml/evaluation/run_eval --llm-temperature) overrides it per client (0.0).
    LLM_TEMPERATURE: float = 0.2
    # Per-model sampling (code default, not .env), keyed by model id; a key
    # matches the model with or without OpenRouter's ":free" suffix (a
    # "<provider>:<model>" key matches only that provider). Used whenever the
    # client has no explicit temperature override (``LLMClient.temperature`` is
    # None); an override sends only that temperature, as before.
    # qwen/qwen3.8-27b: the model card's thinking-mode recommendation (also
    # OpenRouter's defaults for it); greedy / low temperature is discouraged for
    # thinking mode (repetition loops). OpenRouter's ":free" endpoint for this
    # model serves an fp4-quantised build. top_k is only sent to providers that
    # accept it (llm_client.PROVIDERS "supports_top_k": OpenRouter).
    LLM_SAMPLING: dict[str, dict[str, float | int]] = {
        "qwen/qwen3.8-27b": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    }
    # Cap on completion tokens per call, sent as max_tokens (None = not sent).
    # Reasoning models count their thinking against it: the JSON answer is a few
    # hundred to ~2K tokens, but thinking-mode models spend several thousand
    # tokens reasoning first, so 16K leaves room for a long reasoning pass while
    # bounding a runaway generation (one call ran to 131,072 completion tokens
    # with no cap). A client with a tokens-per-minute limit (LLM_TPM_LIMITS, i.e.
    # Groq) gets at most what is left of that limit after the prompt, since Groq
    # counts max_tokens against it. Output cut at the cap (finish_reason
    # "length") is a failed call that falls through, never an empty review.
    LLM_MAX_OUTPUT_TOKENS: int | None = 16000
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

    # Files-mode review (core/pr_review.py). "pr": one PR-level audit ("what
    # does this change newly introduce?") over the diff + changed functions
    # before/after + leads, a bounded context loop (the model names symbols it
    # needs, resolved from the PR's own files), then a separate verifier call per
    # candidate. "units": the older per-function review (review_plan), kept for
    # comparison. Snippet mode always uses the per-unit review.
    REVIEW_MODE: str = "pr"                   # pr | units
    # Audit (and verifier) prompt budget, estimated tokens like
    # LLM_MAX_PROMPT_TOKENS. 12000: the PR-level numbers were measured with it
    # (6000 left 29 of 200 eval PRs partial). The primary, OpenRouter's free
    # models, has no tokens-per-minute cap; Groq's free tier (~8K tokens/min per
    # model, the fallback) can never take a prompt this size, so the router skips
    # a Groq client for a prompt that leaves no output room under its
    # LLM_TPM_LIMITS entry (falls through) instead of sending a doomed request.
    # Smaller prompts still use Groq as before.
    PR_REVIEW_MAX_PROMPT_TOKENS: int = 12000
    # All audit calls of a scan, context rounds included. A PR too big for one
    # prompt is split by file into several audit prompts; files beyond this
    # budget are listed as not reviewed ("budget").
    PR_REVIEW_MAX_AUDIT_CALLS: int = 4
    # Extra audit calls per prompt answering the model's context requests.
    PR_REVIEW_CONTEXT_ROUNDS: int = 2
    # Estimated tokens of requested context appended per audit prompt (all
    # rounds together); reserved out of PR_REVIEW_MAX_PROMPT_TOKENS.
    PR_REVIEW_CONTEXT_MAX_TOKENS: int = 1500
    # One verifier call per candidate finding, at most this many per scan.
    PR_REVIEW_MAX_VERIFIER_CALLS: int = 8
    # A finding is reported only when the verifier confirms it with at least
    # this confidence (1-10). 7 was chosen to match the verifier prompt's own
    # scale (7-10 = "likely a real vulnerability"), so the cutoff means what the
    # model is told the numbers mean. The dev-set policy curve (confirmed >= k,
    # run_pr_eval) already existed when it was chosen, so the dev numbers are
    # not an independent check of it: the held-out test split is.
    PR_REVIEW_MIN_CONFIDENCE: int = 7
    # Candidates the audit itself rates below this (1-10) are not verified.
    PR_REVIEW_MIN_AUDIT_CONFIDENCE: int = 5
    # "Worth a look" (non-blocking, never a finding or a gate): a candidate the
    # verifier left "uncertain" (or confirmed below PR_REVIEW_MIN_CONFIDENCE)
    # with at least PR_REVIEW_SUGGEST_MIN_CONFIDENCE, AND deterministic evidence
    # that the change deleted code that looks like a security control at that
    # spot: a guard_diff removed / weakened change of a guard_removed unit
    # inside the candidate's function (30 lines around it in module-level
    # code), whose old code is in no new file; or the verifier's
    # removed_control_quote found in the old file only on deleted, non-comment
    # lines that guard_diff classifies as a control, and in no new file; never
    # where a guard alert finding already reports the removal. Python /
    # JavaScript only (guard_diff has no Go / Java grammar: the tier never
    # fires there). At most PR_REVIEW_MAX_SUGGESTIONS per scan; 0 turns the
    # tier off.
    PR_REVIEW_MAX_SUGGESTIONS: int = 5
    PR_REVIEW_SUGGEST_MIN_CONFIDENCE: int = 4
    # Semgrep hits at/above this severity are shown to the audit as LEADS
    # (marked with their severity; leads focus attention, they are not findings).
    # Evidence-grade hits (static_analysis, corroboration) still use
    # SEMGREP_MIN_SEVERITY.
    PR_REVIEW_SEMGREP_LEAD_MIN_SEVERITY: str = "low"
    # Diff-scoped heuristic leads: added lines touching sensitive sinks (exec,
    # eval, subprocess, pickle, yaml.load, SQL execute with formatting, file
    # paths, redirects, outbound requests, innerHTML, child_process, ...).
    PR_REVIEW_SINK_LEADS: bool = True
    # Regex hard exclusions (DoS, rate limiting, resource leaks, memory safety
    # in non-C code, findings in docs / tests) applied before verification.
    PR_REVIEW_HARD_EXCLUSIONS: bool = True
    # Retrieved similar CVE fixes shown in the audit prompt as "how a similar
    # bug was fixed" examples (0 = off; retrieval added nothing measurable).
    PR_REVIEW_FIX_EXAMPLES: int = 0
    # "<provider>:<model>" to try first for verifier calls (e.g.
    # "groq:openai/gpt-oss-120b"); the normal chain remains the fallback.
    # None = the same chain as the audit. The model that answered is recorded
    # per finding (``verifier``).
    VERIFIER_MODEL: str | None = None

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

    # Severity of a guard_diff alert-tier finding (a deterministic diff pattern:
    # unsafe-API swap, flag flip, SQL interpolation). Medium: it is evidence, not
    # a verdict, and the Action's gate ignores it unless the LLM or Semgrep
    # corroborates it (or INPUT_GATE_ON_DETERMINISTIC is set).
    GUARD_ALERT_SEVERITY: str = "medium"   # low | medium | high | critical

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
