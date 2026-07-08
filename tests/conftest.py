"""Test configuration.

Sets environment variables BEFORE any ``backend.app`` module is imported, so the
``settings`` singleton (created at import time of backend.app.config) picks them
up: no model downloads, and a throwaway SQLite file instead of the dev DB.
"""
import os
import tempfile

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("PRELOAD_MODELS", "false")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("REPOSENTINEL_API_KEY", "")

# Isolated SQLite file for the whole test session.
_TEST_DB = os.path.join(tempfile.gettempdir(), "repo_sentinel_test.sqlite")
os.environ.setdefault("DEV_SQLITE_PATH", _TEST_DB)
