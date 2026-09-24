"""Unit tests for Settings behavior."""
from backend.app.config import Settings


def test_dev_uses_sqlite_and_ignores_database_url():
    s = Settings(ENVIRONMENT="development", DATABASE_URL="postgresql://x/y")
    assert s.is_dev
    assert s.get_database_url.startswith("sqlite:///")


def test_dev_sqlite_path_override():
    s = Settings(ENVIRONMENT="development", DEV_SQLITE_PATH="/tmp/custom.sqlite")
    assert s.get_database_url == "sqlite:////tmp/custom.sqlite"


def test_prod_uses_database_url():
    s = Settings(ENVIRONMENT="production", DATABASE_URL="postgresql://u:p@h/db")
    assert not s.is_dev
    assert s.get_database_url == "postgresql://u:p@h/db"


def test_legacy_codebert_env_var_is_ignored(monkeypatch):
    # A stale CODEBERT_MODEL must NOT override the calibrated default.
    monkeypatch.setenv("CODEBERT_MODEL", "microsoft/codebert-base")
    s = Settings()
    assert s.EMBEDDING_MODEL == "jinaai/jina-embeddings-v2-base-code"


def test_embedding_model_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "microsoft/graphcodebert-base")
    s = Settings()
    assert s.EMBEDDING_MODEL == "microsoft/graphcodebert-base"


def test_seniority_weights_default():
    s = Settings()
    assert s.SENIORITY_WEIGHTS["OWNER"] > s.SENIORITY_WEIGHTS["MEMBER"]


def test_embedder_and_reranker_defaults():
    # jina-embeddings-v2-base-code + bge-reranker-v2-m3 (roadmap items 1c/1d).
    s = Settings()
    assert s.EMBEDDING_DIM == 768
    assert s.EMBEDDING_MAX_TOKENS == 2048
    assert s.EMBEDDING_TRUST_REMOTE_CODE is True
    assert s.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert s.RERANKER_BATCH_SIZE == 8


def test_reranker_off_by_default(monkeypatch):
    # ROADMAP 1d: no measured gain from the cross-encoder, so it is off by
    # default; 512 tokens (best-measured, half the cost of 1024) when enabled.
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RERANKER_MAX_TOKENS", raising=False)
    s = Settings(_env_file=None)
    assert s.RERANKER_ENABLED is False
    assert s.RERANKER_MAX_TOKENS == 512


def test_reranker_enabled_env_override(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    assert Settings(_env_file=None).RERANKER_ENABLED is True
