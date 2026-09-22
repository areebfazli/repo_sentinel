"""Unit tests for the LLM router (mock mode, fail-fast, fallback selection)."""
import asyncio

import pytest

from backend.app.config import settings
from backend.app.core import llm_client
from backend.app.core.llm_client import LLMClient, LLMRouter


def test_mock_mode_returns_canned_response(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    router = LLMRouter()
    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "mock"
    assert "findings" in payload


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError):
        LLMClient("groq")


def test_router_builds_primary_and_fallback(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    router = LLMRouter()
    assert [c.provider for c in router.clients] == ["groq", "gemini"]


def test_placeholder_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "your_groq_api_key_here")
    with pytest.raises(RuntimeError):
        LLMClient("groq")


def test_unknown_fallback_provider_gives_clean_error(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "openai")  # not supported
    with pytest.raises(ValueError):  # clean message, not a raw KeyError
        LLMRouter()


def test_missing_fallback_key_degrades_to_primary_only(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)  # no Gemini account
    router = LLMRouter()
    assert [c.provider for c in router.clients] == ["groq"]  # boots, no crash


class _FakeResp:
    def __init__(self, status, data, text="", headers=None):
        self.status_code = status
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient: async context manager whose `post` is async."""

    def __init__(self, responder):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, *args, **kwargs):
        return self._responder(*args, **kwargs)


def _patch_async_client(monkeypatch, responder):
    """monkeypatch llm_client.httpx.AsyncClient to return a canned response."""
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient",
        lambda *a, **k: _FakeAsyncClient(responder),
    )


def test_complete_null_content_is_retriable(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    client = LLMClient("groq")
    _patch_async_client(
        monkeypatch,
        lambda *a, **k: _FakeResp(200, {"choices": [{"message": {"content": None}}]}),
    )
    with pytest.raises(llm_client._Retriable):
        asyncio.run(client.complete("s", "u"))


def test_complete_empty_choices_is_retriable(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    client = LLMClient("groq")
    _patch_async_client(monkeypatch, lambda *a, **k: _FakeResp(200, {"choices": []}))
    with pytest.raises(llm_client._Retriable):
        asyncio.run(client.complete("s", "u"))


def test_router_falls_back_on_primary_failure(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    # 0 retries: this test is about fallback selection, not retry/backoff timing.
    monkeypatch.setattr(settings, "LLM_RETRIES", 0)
    router = LLMRouter()

    async def groq_fails(self, system, user):
        if self.provider == "groq":
            raise llm_client._Retriable("boom")
        return '{"findings": [{"title": "from gemini"}]}'

    monkeypatch.setattr(LLMClient, "complete", groq_fails)
    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "gemini"
    assert payload["findings"][0]["title"] == "from gemini"
