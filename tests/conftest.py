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
# The Semgrep engine is a multi-second subprocess; tests that need it stub the
# scanner (tests/unit/test_semgrep_scanner.py covers the real engine separately).
os.environ.setdefault("SEMGREP_ENABLED", "false")

# Fresh, isolated SQLite file per test run (so job/finding/feedback rows don't
# accumulate across invocations).
_TEST_DIR = tempfile.mkdtemp(prefix="reposentinel_test_")
os.environ.setdefault("DEV_SQLITE_PATH", os.path.join(_TEST_DIR, "test.sqlite"))
