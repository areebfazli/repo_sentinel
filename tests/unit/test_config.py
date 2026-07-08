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


def test_embedding_model_accepts_legacy_alias(monkeypatch):
    monkeypatch.setenv("CODEBERT_MODEL", "microsoft/unixcoder-base")
    s = Settings()
    assert s.EMBEDDING_MODEL == "microsoft/unixcoder-base"


def test_seniority_weights_default():
    s = Settings()
    assert s.SENIORITY_WEIGHTS["OWNER"] > s.SENIORITY_WEIGHTS["MEMBER"]
