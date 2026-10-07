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


def test_default_routing_is_openrouter_nemotron_without_fallback():
    # Routing is a code default (.env only needs keys). Free qwen left
    # OpenRouter (404) and free gemma is constantly rate-limited, so the default
    # is one model, no same-provider fallback model, no fallback provider.
    fields = Settings.model_fields
    assert fields["LLM_PROVIDER"].default == "openrouter"
    assert fields["LLM_FALLBACK_PROVIDER"].default is None
    assert fields["OPENROUTER_MODEL"].default == "nvidia/nemotron-3-super-120b-a12b:free"
    assert fields["OPENROUTER_FALLBACK_MODEL"].default is None


def test_default_model_has_no_sampling_entry(monkeypatch):
    # The measured configuration ran Nemotron Super at LLM_TEMPERATURE (0.2).
    _pin_default_chain(monkeypatch)
    for name in ("LLM_SAMPLING", "LLM_TEMPERATURE"):
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)
    assert LLMClient("openrouter").sampling_params() == {"temperature": 0.2}


def test_routing_defaults_apply_without_env(monkeypatch):
    # conftest forces LLM_PROVIDER=mock for the suite; drop it to see the defaults.
    for var in ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "OPENROUTER_FALLBACK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert (s.LLM_PROVIDER, s.LLM_FALLBACK_PROVIDER) == ("openrouter", None)
    assert s.OPENROUTER_FALLBACK_MODEL is None


DEFAULT_CHAIN = ["openrouter:nvidia/nemotron-3-super-120b-a12b:free"]

# An explicitly configured fallback chain (env opt-in), for the fall-through tests.
OPT_IN_CHAIN = [
    "openrouter:nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter:google/gemma-4-31b-it:free",
    "groq:openai/gpt-oss-120b",
    "groq:qwen/qwen3.8-27b",
]


def _pin_default_chain(monkeypatch, groq_key="gk"):
    """Pin routing/model settings to the Settings code defaults (a local .env or
    the conftest LLM_PROVIDER=mock can't change them), plus fake keys."""
    for name in (
        "LLM_PROVIDER", "LLM_FALLBACK_PROVIDER",
        "OPENROUTER_MODEL", "OPENROUTER_FALLBACK_MODEL", "OPENROUTER_BASE_URL",
        "GROQ_MODEL", "GROQ_FALLBACK_MODEL",
    ):
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "ork")
    monkeypatch.setattr(settings, "GROQ_API_KEY", groq_key)
    monkeypatch.setattr(settings, "LLM_RETRIES", 0)


def _pin_opt_in_chain(monkeypatch):
    """The defaults plus an env-configured OpenRouter fallback model and Groq."""
    _pin_default_chain(monkeypatch)
    monkeypatch.setattr(settings, "OPENROUTER_FALLBACK_MODEL", "google/gemma-4-31b-it:free")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")


def _warnings():
    from loguru import logger
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    return messages, sink_id


@pytest.mark.parametrize("groq_key", ["gk", None])
def test_default_chain_is_one_client_and_never_mentions_groq(monkeypatch, groq_key):
    from loguru import logger
    _pin_default_chain(monkeypatch, groq_key=groq_key)
    messages, sink_id = _warnings()
    try:
        clients = LLMRouter().clients
    finally:
        logger.remove(sink_id)
    assert [c.label for c in clients] == DEFAULT_CHAIN
    assert not any("groq" in m.lower() or "fallback" in m.lower() for m in messages)


@pytest.mark.parametrize("disabled", [None, ""])
def test_empty_fallback_settings_mean_disabled(monkeypatch, disabled):
    _pin_default_chain(monkeypatch)
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", disabled)
    monkeypatch.setattr(settings, "OPENROUTER_FALLBACK_MODEL", disabled)
    assert [c.label for c in LLMRouter().clients] == DEFAULT_CHAIN


def test_opt_in_chain_is_four_steps(monkeypatch):
    _pin_opt_in_chain(monkeypatch)
    assert [c.label for c in LLMRouter().clients] == OPT_IN_CHAIN


def test_opt_in_chain_without_groq_key_is_openrouter_only(monkeypatch):
    _pin_opt_in_chain(monkeypatch)
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    assert [c.label for c in LLMRouter().clients] == OPT_IN_CHAIN[:2]


def test_default_chain_missing_openrouter_key_fails_fast(monkeypatch):
    _pin_default_chain(monkeypatch)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMRouter()


def test_default_chain_429_fails_the_call(monkeypatch):
    """No fallback: a 429 on the only model is an LLMError (not bad output)."""
    _pin_default_chain(monkeypatch)
    monkeypatch.setattr(settings, "LLM_MAX_WAIT_S", 0.0)
    router = LLMRouter()
    calls: list[str] = []

    def responder(url, **kwargs):
        calls.append(kwargs["json"]["model"])
        return _FakeResp(429, {"error": {"code": 429, "message": "rate limited"}})

    _patch_async_client(monkeypatch, responder)
    with pytest.raises(llm_client.LLMError) as exc:
        asyncio.run(router.generate("sys", "user"))
    assert not exc.value.bad_output
    assert calls == ["nvidia/nemotron-3-super-120b-a12b:free"]


def test_opt_in_chain_falls_through_all_four_links(monkeypatch):
    """Stubbed HTTP: every link but the last fails, in chain order."""
    _pin_opt_in_chain(monkeypatch)
    router = LLMRouter()
    calls: list[str] = []
    shared_pool_429 = _FakeResp(
        429, {"error": {"code": 429, "message": "upstream_provider_shared_pool"}},
    )

    def responder(url, **kwargs):
        model = kwargs["json"]["model"]
        provider = "openrouter" if "openrouter.ai" in url else "groq"
        label = f"{provider}:{model}"
        calls.append(label)
        if label == OPT_IN_CHAIN[0] or label == OPT_IN_CHAIN[1]:
            return shared_pool_429
        if label == OPT_IN_CHAIN[2]:
            return _FakeResp(503, {}, text="unavailable")
        return _FakeResp(200, {"choices": [{"message": {"content": '{"findings": []}'}}]})

    _patch_async_client(monkeypatch, responder)
    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "groq:qwen/qwen3.8-27b"
    assert payload == {"findings": []}
    assert calls == OPT_IN_CHAIN


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


def test_fallback_provider_also_gets_its_same_provider_fallback_model(monkeypatch):
    _pin_models(monkeypatch)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    router = LLMRouter()
    assert [c.label for c in router.clients] == [
        "gemini:gemini-2.0-flash",
        "groq:openai/gpt-oss-120b",
        "groq:qwen/qwen3.8-27b",
    ]


def test_fallback_provider_model_fallback_skipped_when_unset(monkeypatch):
    _pin_models(monkeypatch, groq_fallback=None)
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

    async def groq_fails(self, system, user, max_tokens=None):
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

    async def primary_model_fails(self, system, user, max_tokens=None):
        if self.model == "openai/gpt-oss-120b":
            raise llm_client.LLMError("groq:openai/gpt-oss-120b HTTP 404: model_not_found")
        return '{"findings": [{"title": "from qwen"}]}'

    monkeypatch.setattr(LLMClient, "complete", primary_model_fails)
    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "groq:qwen/qwen3.8-27b"  # not just "groq"
    assert payload["findings"][0]["title"] == "from qwen"
