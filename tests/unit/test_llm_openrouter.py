"""Unit tests for the OpenRouter provider and the provider-agnostic JSON extractor.

Same scripted fake httpx.AsyncClient as test_llm_retry.py (per-(URL, model) queues
of canned responses, recording each request's headers/body) and a no-op
asyncio.sleep, so there is no network and no real waiting.
"""
import asyncio
import json

import pytest
from loguru import logger

from backend.app.config import Settings, settings
from backend.app.core import llm_client
from backend.app.core.llm_client import PROVIDERS, LLMClient, LLMError, LLMRouter, extract_json

OR_BASE = "https://openrouter.ai/api/v1"
OR_URL = f"{OR_BASE}/chat/completions"
GEMINI_URL = PROVIDERS["gemini"]["url"]

OR_PRIMARY = "qwen/qwen3.8-27b:free"  # listed in _NO_RESPONSE_FORMAT_MODELS
OR_FB = "google/gemma-4-31b-it:free"  # supports response_format
# A hypothetical model NOT in _NO_RESPONSE_FORMAT_MODELS, for the reactive 400 path.
UNLISTED = "example/unlisted-model:free"
GEMINI_MODEL = "gemini-2.0-flash"

PRIMARY = (OR_URL, OR_PRIMARY)
OR_FALLBACK = (OR_URL, OR_FB)
UNLISTED_KEY = (OR_URL, UNLISTED)
GEMINI = (GEMINI_URL, GEMINI_MODEL)

OK_JSON = '{"findings": [{"title": "ok"}]}'


class _FakeResp:
    def __init__(self, status, data=None, text="", headers=None):
        self.status_code = status
        self._data = data
        self.text = text or (json.dumps(data) if data is not None else "")
        self.headers = headers or {}

    def json(self):
        return self._data


def _ok(content):
    return _FakeResp(200, {"choices": [{"message": {"content": content}}]})


class _FakeAsyncClient:
    def __init__(self, responder):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, **kwargs):
        return self._responder(url, **kwargs)


class _Script:
    """Per-(URL, model) queue of scripted responses; records every request."""

    def __init__(self):
        self.queues: dict[tuple[str, str], list] = {}
        self.order: list[tuple[str, str]] = []
        self.requests: list[dict] = []

    def set(self, key, responses):
        self.queues[key] = list(responses)

    def __call__(self, url, **kwargs):
        key = (url, kwargs["json"]["model"])
        self.order.append(key)
        self.requests.append({"url": url, **kwargs})
        queue = self.queues.get(key)
        if not queue:
            raise AssertionError(f"unexpected call to {key} (no scripted response left)")
        return queue.pop(0)

    def call_count(self, key):
        return self.order.count(key)


def _pin_openrouter(monkeypatch, fallback_model=None, retries=1, fallback_provider="gemini"):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", fallback_provider)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "ork")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", OR_PRIMARY)
    monkeypatch.setattr(settings, "OPENROUTER_FALLBACK_MODEL", fallback_model)
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", OR_BASE)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    monkeypatch.setattr(settings, "GEMINI_MODEL", GEMINI_MODEL)
    monkeypatch.setattr(settings, "LLM_RETRIES", retries)


def _patch_sleep(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    return sleeps


def _patch_script(monkeypatch, script):
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(script))


def _capture_warnings():
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    return messages, sink_id


def _generate(router):
    return asyncio.run(router.generate("sys", "user"))


# --- settings / construction ------------------------------------------------------


def test_openrouter_setting_defaults():
    fields = Settings.model_fields
    assert fields["OPENROUTER_API_KEY"].default is None
    assert fields["OPENROUTER_MODEL"].default == "qwen/qwen3.8-27b:free"
    assert fields["OPENROUTER_FALLBACK_MODEL"].default == "google/gemma-4-31b-it:free"
    assert fields["OPENROUTER_BASE_URL"].default == OR_BASE


def test_openrouter_fallback_model_can_be_disabled_via_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "")
    assert not Settings().OPENROUTER_FALLBACK_MODEL


@pytest.mark.parametrize("key", [None, "", "your_openrouter_api_key_here"])
def test_missing_openrouter_key_fails_fast(monkeypatch, key):
    _pin_openrouter(monkeypatch)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", key)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMClient("openrouter")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMRouter()


def test_url_and_headers(monkeypatch):
    _pin_openrouter(monkeypatch)
    _patch_sleep(monkeypatch)
    client = LLMClient("openrouter")
    assert client.url == OR_URL
    assert client.label == f"openrouter:{OR_PRIMARY}"

    script = _Script()
    script.set(PRIMARY, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    assert asyncio.run(client.complete("s", "u")) == OK_JSON

    req = script.requests[0]
    assert req["url"] == OR_URL
    assert req["headers"]["Authorization"] == "Bearer ork"
    assert req["headers"]["HTTP-Referer"] == "https://github.com/areebfazli/repo_sentinel"
    assert req["headers"]["X-Title"] == "RepoSentinel"
    assert req["json"]["model"] == OR_PRIMARY
    assert "response_format" not in req["json"]  # qwen rejects it

    fb = LLMClient("openrouter", model=OR_FB)
    script.set(OR_FALLBACK, [_ok(OK_JSON)])
    asyncio.run(fb.complete("s", "u"))
    assert script.requests[1]["headers"]["X-Title"] == "RepoSentinel"
    assert script.requests[1]["json"]["response_format"] == {"type": "json_object"}


def test_base_url_setting_is_honoured(monkeypatch):
    _pin_openrouter(monkeypatch)
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://proxy.test/v1/")
    assert LLMClient("openrouter").url == "https://proxy.test/v1/chat/completions"


def test_groq_requests_carry_no_openrouter_headers(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    client = LLMClient("groq")
    script = _Script()
    script.set((PROVIDERS["groq"]["url"], "openai/gpt-oss-120b"), [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    asyncio.run(client.complete("s", "u"))
    assert script.requests[0]["headers"] == {"Authorization": "Bearer gk"}


# --- chain / per-provider fallback model -----------------------------------------


def test_chain_includes_openrouter_fallback_model(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=OR_FB)
    assert [c.label for c in LLMRouter().clients] == [
        f"openrouter:{OR_PRIMARY}",
        f"openrouter:{OR_FB}",
        f"gemini:{GEMINI_MODEL}",
    ]


@pytest.mark.parametrize("fb_model", [None, "", OR_PRIMARY])
def test_openrouter_fallback_model_skipped_when_unset_or_same(monkeypatch, fb_model):
    _pin_openrouter(monkeypatch, fallback_model=fb_model)
    assert [c.label for c in LLMRouter().clients] == [
        f"openrouter:{OR_PRIMARY}",
        f"gemini:{GEMINI_MODEL}",
    ]


def test_groq_fallback_model_not_used_for_openrouter_primary(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=None, fallback_provider=None)
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")
    assert [c.label for c in LLMRouter().clients] == [f"openrouter:{OR_PRIMARY}"]


def test_openrouter_as_fallback_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", None)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "ork")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", OR_PRIMARY)
    assert [c.label for c in LLMRouter().clients] == [
        "groq:openai/gpt-oss-120b",
        f"openrouter:{OR_PRIMARY}",
    ]


def test_primary_failure_falls_to_openrouter_fallback_model(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=OR_FB)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_FakeResp(404, {"error": {"message": "No endpoints found"}})])
    script.set(OR_FALLBACK, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)

    payload, provider = _generate(router)
    assert provider == f"openrouter:{OR_FB}"  # the model that actually answered
    assert payload == json.loads(OK_JSON)
    assert script.order == [PRIMARY, OR_FALLBACK]
    assert sleeps == []


# --- 429 --------------------------------------------------------------------------


def test_429_retries_then_succeeds(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=OR_FB, retries=1)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [
        _FakeResp(429, {"error": {"message": "Rate limit exceeded: free-models-per-min."}},
                  headers={"Retry-After": "4"}),
        _ok(OK_JSON),
    ])
    _patch_script(monkeypatch, script)

    _, provider = _generate(router)
    assert provider == f"openrouter:{OR_PRIMARY}"
    assert script.order == [PRIMARY, PRIMARY]
    assert sleeps == [4.0]  # Retry-After honoured


def test_429_exhausts_retries_then_openrouter_fallback_model(monkeypatch):
    retries = 2
    _pin_openrouter(monkeypatch, fallback_model=OR_FB, retries=retries)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_FakeResp(429, {}, text="rate limited") for _ in range(retries + 1)])
    script.set(OR_FALLBACK, [_ok(OK_JSON)])
    script.set(GEMINI, [])
    _patch_script(monkeypatch, script)

    _, provider = _generate(router)
    assert provider == f"openrouter:{OR_FB}"
    assert script.call_count(PRIMARY) == retries + 1
    assert script.call_count(GEMINI) == 0
    assert len(sleeps) == retries


# --- 402 --------------------------------------------------------------------------


def test_402_is_not_retried_and_skips_the_provider(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=OR_FB, retries=3)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    warnings, sink_id = _capture_warnings()
    script = _Script()
    credits_402 = {"error": {"code": 402, "message": "Insufficient credits"}}
    script.set(PRIMARY, [_FakeResp(402, credits_402)])
    script.set(OR_FALLBACK, [])  # same account: must not be tried
    script.set(GEMINI, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)

    try:
        _, provider = _generate(router)
    finally:
        logger.remove(sink_id)

    assert provider == f"gemini:{GEMINI_MODEL}"
    assert script.order == [PRIMARY, GEMINI]  # one call despite LLM_RETRIES=3
    assert sleeps == []
    credit = [m for m in warnings if "insufficient credits" in m and "not retrying" in m]
    assert len(credit) == 1 and "openrouter" in credit[0] and OR_PRIMARY in credit[0]
    assert any("Skipping LLM client" in m and OR_FB in m for m in warnings)


def test_402_with_no_other_provider_fails_immediately(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=OR_FB, retries=3, fallback_provider=None)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_FakeResp(402, {}, text="Insufficient credits")])
    _patch_script(monkeypatch, script)

    with pytest.raises(LLMError, match="402"):
        _generate(router)
    assert script.order == [PRIMARY]
    assert sleeps == []


# --- error inside an HTTP 200 -------------------------------------------------------


def test_error_in_200_is_bad_output_not_a_crash(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_model=OR_FB, retries=2)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_FakeResp(200, {"error": {"message": "Provider returned error"}})])
    script.set(OR_FALLBACK, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)

    _, provider = _generate(router)
    assert provider == f"openrouter:{OR_FB}"
    assert script.order == [PRIMARY, OR_FALLBACK]  # bad-output: no same-client retry
    assert sleeps == []


def test_upstream_429_in_200_is_retried_on_same_client(monkeypatch):
    _pin_openrouter(monkeypatch, retries=1)
    router = LLMRouter()
    sleeps = _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [
        _FakeResp(200, {"error": {"code": 429, "message": "upstream rate-limited"}}),
        _ok(OK_JSON),
    ])
    _patch_script(monkeypatch, script)

    _, provider = _generate(router)
    assert provider == f"openrouter:{OR_PRIMARY}"
    assert script.order == [PRIMARY, PRIMARY]
    assert len(sleeps) == 1


def test_complete_raises_retriable_on_error_in_200(monkeypatch):
    _pin_openrouter(monkeypatch)
    client = LLMClient("openrouter")
    script = _Script()
    script.set(PRIMARY, [_FakeResp(200, {"error": {"message": "boom"}, "choices": []})])
    _patch_script(monkeypatch, script)
    with pytest.raises(llm_client._Retriable) as info:
        asyncio.run(client.complete("s", "u"))
    assert info.value.transient is False and "boom" in str(info.value)


# --- response_format rejected ------------------------------------------------------


def _rf_400():
    return _FakeResp(400, {"error": {
        "code": 400, "message": "This model does not support response_format json_object"}})


def _pin_unlisted_primary(monkeypatch, retries):
    _pin_openrouter(monkeypatch, retries=retries, fallback_provider=None)
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", UNLISTED)


def test_response_format_400_retries_once_without_it(monkeypatch):
    _pin_unlisted_primary(monkeypatch, retries=0)
    router = LLMRouter()
    _patch_sleep(monkeypatch)
    warnings, sink_id = _capture_warnings()
    script = _Script()
    script.set(UNLISTED_KEY, [_rf_400(), _ok(OK_JSON), _ok(OK_JSON)])
    _patch_script(monkeypatch, script)

    try:
        payload, provider = _generate(router)
    finally:
        logger.remove(sink_id)

    assert provider == f"openrouter:{UNLISTED}"
    assert payload == json.loads(OK_JSON)
    assert script.order == [UNLISTED_KEY, UNLISTED_KEY]  # exactly one extra call, same model
    assert "response_format" in script.requests[0]["json"]
    assert "response_format" not in script.requests[1]["json"]
    assert any("rejected response_format" in m for m in warnings)

    # Remembered for this client: the next scan doesn't re-send it and re-fail.
    _generate(router)
    assert "response_format" not in script.requests[2]["json"]


def test_response_format_400_twice_is_not_retried_again(monkeypatch):
    _pin_unlisted_primary(monkeypatch, retries=2)
    router = LLMRouter()
    _patch_sleep(monkeypatch)
    script = _Script()
    script.set(UNLISTED_KEY, [_rf_400(), _rf_400()])
    _patch_script(monkeypatch, script)

    with pytest.raises(LLMError):
        _generate(router)
    assert script.order == [UNLISTED_KEY, UNLISTED_KEY]  # the 400 is non-retriable


def test_known_non_supporting_primary_never_gets_response_format(monkeypatch):
    # The qwen primary's FIRST request already omits response_format, so there is
    # no wasted 400 round-trip; the gemma fallback still sends it normally.
    _pin_openrouter(monkeypatch, fallback_model=OR_FB, retries=0, fallback_provider=None)
    router = LLMRouter()
    assert OR_PRIMARY in llm_client._NO_RESPONSE_FORMAT_MODELS
    assert OR_FB not in llm_client._NO_RESPONSE_FORMAT_MODELS
    assert [c.use_response_format for c in router.clients] == [False, True]
    _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_ok(OK_JSON), _FakeResp(503, {}, text="upstream down")])
    script.set(OR_FALLBACK, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)

    payload, provider = _generate(router)
    assert provider == f"openrouter:{OR_PRIMARY}"
    assert payload == json.loads(OK_JSON)
    assert script.order == [PRIMARY]  # a single request to qwen, no 400 retry
    assert "response_format" not in script.requests[0]["json"]

    # Primary down -> gemma answers, with response_format.
    _, provider = _generate(router)
    assert provider == f"openrouter:{OR_FB}"
    assert script.order == [PRIMARY, PRIMARY, OR_FALLBACK]
    assert "response_format" not in script.requests[1]["json"]
    assert script.requests[2]["json"]["response_format"] == {"type": "json_object"}


def test_no_response_format_set_is_scoped_to_openrouter(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    assert LLMClient("groq", model=OR_PRIMARY).use_response_format is True


def test_unrelated_400_does_not_drop_response_format(monkeypatch):
    _pin_unlisted_primary(monkeypatch, retries=0)
    router = LLMRouter()
    script = _Script()
    script.set(UNLISTED_KEY, [_FakeResp(400, {"error": {"message": "context length exceeded"}})])
    _patch_script(monkeypatch, script)

    with pytest.raises(LLMError):
        _generate(router)
    assert script.order == [UNLISTED_KEY]
    assert router.clients[0].use_response_format is True


def test_groq_never_drops_response_format(monkeypatch):
    # The fallback is OpenRouter-only; Groq's behavior is unchanged.
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    client = LLMClient("groq")
    key = (PROVIDERS["groq"]["url"], "openai/gpt-oss-120b")
    script = _Script()
    script.set(key, [_FakeResp(400, {"error": {"message": "response_format unsupported"}})])
    _patch_script(monkeypatch, script)
    with pytest.raises(LLMError):
        asyncio.run(client.complete("s", "u"))
    assert script.order == [key]


# --- tolerant JSON extraction -------------------------------------------------------


EXPECTED = {"findings": [{"title": "t", "severity": "high"}]}
BODY = json.dumps(EXPECTED)


@pytest.mark.parametrize("content", [
    BODY,                                                      # plain (Groq-style)
    f"  {BODY}\n",                                             # whitespace
    f"```json\n{BODY}\n```",                                   # fenced
    f"```\n{BODY}\n```",                                       # fenced, no language
    f"Let me analyze the code first.\n\n{BODY}",               # leading reasoning
    f"Reasoning: the {{set}} here is fine.\n```json\n{BODY}\n```\nDone.",
    f"<think>maybe {{\"findings\": []}} works?</think>\n{BODY}",
    f"Here is the report: {BODY} Hope that helps!",            # trailing prose
])
def test_extract_json_variants(content):
    assert extract_json(content) == EXPECTED


def test_extract_json_plain_json_is_json_loads():
    for content in ('{"findings": []}', "[1, 2]", '"x"'):
        assert extract_json(content) == json.loads(content)


@pytest.mark.parametrize("content", ["not json", "", "{broken", "```json\n{nope\n```"])
def test_extract_json_still_rejects_garbage(content):
    with pytest.raises(json.JSONDecodeError):
        extract_json(content)


def test_router_parses_fenced_reasoning_content(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_provider=None)
    router = LLMRouter()
    script = _Script()
    script.set(PRIMARY, [_ok(f"I'll check the snippet.\n```json\n{BODY}\n```")])
    _patch_script(monkeypatch, script)
    payload, provider = _generate(router)
    assert payload == EXPECTED and provider == f"openrouter:{OR_PRIMARY}"
