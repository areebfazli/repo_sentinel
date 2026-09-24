"""Unit tests for the LLM router (mock mode, fail-fast, fallback selection)."""
import asyncio

import pytest

from backend.app.config import Settings, settings
from backend.app.core import llm_client
from backend.app.core.llm_client import LLMClient, LLMRouter


def _pin_models(monkeypatch, groq_fallback="qwen/qwen3.8-27b"):
    """Pin model settings so a local .env can't change what these tests assert."""
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", groq_fallback)
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")


def test_default_models():
    # llama-3.3-70b-versatile was retired on Groq (404); these were verified live.
    assert Settings.model_fields["GROQ_MODEL"].default == "openai/gpt-oss-120b"
    assert Settings.model_fields["GROQ_FALLBACK_MODEL"].default == "qwen/qwen3.8-27b"


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


def test_client_model_defaults_to_provider_setting_and_is_overridable(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    assert LLMClient("groq").model == "openai/gpt-oss-120b"
    assert LLMClient("groq").label == "groq:openai/gpt-oss-120b"
    override = LLMClient("groq", model="qwen/qwen3.8-27b")
    assert override.model == "qwen/qwen3.8-27b"
    assert override.label == "groq:qwen/qwen3.8-27b"


def test_router_chain_order_primary_groq_fallback_model_then_gemini(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    router = LLMRouter()
    assert [c.label for c in router.clients] == [
        "groq:openai/gpt-oss-120b",
        "groq:qwen/qwen3.8-27b",
        "gemini:gemini-2.0-flash",
    ]


@pytest.mark.parametrize("fb_model", [None, "", "openai/gpt-oss-120b"])
def test_groq_fallback_model_skipped_when_unset_or_same_as_primary(monkeypatch, fb_model):
    _pin_models(monkeypatch, groq_fallback=fb_model)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    router = LLMRouter()
    assert [c.label for c in router.clients] == [
        "groq:openai/gpt-oss-120b",
        "gemini:gemini-2.0-flash",
    ]


def test_groq_fallback_model_not_used_for_non_groq_primary(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    router = LLMRouter()
    assert [c.label for c in router.clients] == [
        "gemini:gemini-2.0-flash",
        "groq:openai/gpt-oss-120b",
    ]


def test_missing_primary_key_still_fails_fast_with_fallback_model(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError):
        LLMRouter()


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


def test_missing_fallback_key_degrades_to_groq_only(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)  # no Gemini account
    router = LLMRouter()
    # Boots, no crash; the same-provider model fallback is still in the chain.
    assert [c.label for c in router.clients] == [
        "groq:openai/gpt-oss-120b",
        "groq:qwen/qwen3.8-27b",
    ]


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
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    # 0 retries: this test is about fallback selection, not retry/backoff timing.
    monkeypatch.setattr(settings, "LLM_RETRIES", 0)
    router = LLMRouter()

    async def groq_fails(self, system, user):
        if self.provider == "groq":  # both Groq models
            raise llm_client._Retriable("boom")
        return '{"findings": [{"title": "from gemini"}]}'

    monkeypatch.setattr(LLMClient, "complete", groq_fails)
    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "gemini:gemini-2.0-flash"
    assert payload["findings"][0]["title"] == "from gemini"


def test_router_reports_groq_fallback_model_when_primary_model_fails(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    monkeypatch.setattr(settings, "LLM_RETRIES", 0)
    router = LLMRouter()

    async def primary_model_fails(self, system, user):
        if self.model == "openai/gpt-oss-120b":
            raise llm_client.LLMError("groq:openai/gpt-oss-120b HTTP 404: model_not_found")
        return '{"findings": [{"title": "from qwen"}]}'

    monkeypatch.setattr(LLMClient, "complete", primary_model_fails)
    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "groq:qwen/qwen3.8-27b"  # not just "groq"
    assert payload["findings"][0]["title"] == "from qwen"
