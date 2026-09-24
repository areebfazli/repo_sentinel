"""Unit tests for LLMRouter's same-client retry/backoff and the fallthrough chain.

Uses a scripted fake httpx.AsyncClient (per-client queues of canned responses,
keyed by (URL, model) since both Groq clients share a URL) and monkeypatches
llm_client.asyncio.sleep so no test actually sleeps; sleep durations are recorded
instead of waited on.
"""
import asyncio

import pytest
from loguru import logger

from backend.app.config import settings
from backend.app.core import llm_client
from backend.app.core.llm_client import PROVIDERS, LLMError, LLMRouter

GROQ_URL = PROVIDERS["groq"]["url"]
GEMINI_URL = PROVIDERS["gemini"]["url"]

GROQ_PRIMARY = "openai/gpt-oss-120b"
GROQ_FB = "qwen/qwen3.8-27b"
GEMINI_MODEL = "gemini-2.0-flash"

# Scripted-response keys: one per (endpoint, model) client in the chain.
PRIMARY = (GROQ_URL, GROQ_PRIMARY)
GROQ_FALLBACK = (GROQ_URL, GROQ_FB)
GEMINI = (GEMINI_URL, GEMINI_MODEL)


class _FakeResp:
    def __init__(self, status, data=None, text="", headers=None):
        self.status_code = status
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data


def _ok(content):
    return _FakeResp(200, {"choices": [{"message": {"content": content}}]})


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient: async context manager whose `post` is async."""

    def __init__(self, responder):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, **kwargs):
        return self._responder(url, **kwargs)


class _Script:
    """Per-client (by (URL, model)) queue of scripted responses; records call counts."""

    def __init__(self):
        self.queues: dict[tuple[str, str], list] = {}
        self.calls: dict[tuple[str, str], int] = {}
        self.order: list[tuple[str, str]] = []

    def set(self, key, responses):
        self.queues[key] = list(responses)

    def __call__(self, url, **kwargs):
        key = (url, kwargs["json"]["model"])
        self.calls[key] = self.calls.get(key, 0) + 1
        self.order.append(key)
        queue = self.queues.get(key)
        if not queue:
            raise AssertionError(f"unexpected call to {key} (no scripted response left)")
        return queue.pop(0)

    def call_count(self, key):
        return self.calls.get(key, 0)


def _setup_router(monkeypatch, retries, groq_fallback_model=None):
    """Groq primary + Gemini fallback provider; the Groq fallback model only if given."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    monkeypatch.setattr(settings, "GROQ_MODEL", GROQ_PRIMARY)
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", groq_fallback_model)
    monkeypatch.setattr(settings, "GEMINI_MODEL", GEMINI_MODEL)
    monkeypatch.setattr(settings, "LLM_RETRIES", retries)
    return LLMRouter()


def _capture_warnings():
    """Attach a loguru sink collecting WARNING+ messages; returns (messages, sink_id)."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    return messages, sink_id


def _patch_sleep(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    return sleeps


def _patch_script(monkeypatch, script):
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(script))


def test_primary_retries_then_succeeds_fallback_never_called(monkeypatch):
    router = _setup_router(monkeypatch, retries=1)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [
        _FakeResp(429, {}, text="rate limited"),
        _ok('{"findings": [{"title": "groq ok"}]}'),
    ])
    script.set(GEMINI, [])  # must never be called
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"groq:{GROQ_PRIMARY}"
    assert payload["findings"][0]["title"] == "groq ok"
    assert script.call_count(GEMINI) == 0
    assert len(sleeps) == 1


def test_primary_exhausts_retries_then_falls_back(monkeypatch):
    retries = 2
    router = _setup_router(monkeypatch, retries=retries)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_FakeResp(500, {}, text="server error") for _ in range(retries + 1)])
    script.set(GEMINI, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"gemini:{GEMINI_MODEL}"
    assert payload["findings"][0]["title"] == "gemini ok"
    assert script.call_count(PRIMARY) == retries + 1
    assert len(sleeps) == retries


def test_malformed_json_no_retry_immediate_fallback(monkeypatch):
    router = _setup_router(monkeypatch, retries=1)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_ok("not json")])  # 200, but content isn't valid JSON text
    script.set(GEMINI, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"gemini:{GEMINI_MODEL}"
    assert payload["findings"][0]["title"] == "gemini ok"
    assert script.call_count(PRIMARY) == 1  # no same-client retry
    assert sleeps == []


def test_retry_after_header_is_honoured_up_to_max_wait(monkeypatch):
    router = _setup_router(monkeypatch, retries=1)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [
        _FakeResp(429, {}, text="rate limited", headers={"Retry-After": "60"}),
        _ok('{"findings": [{"title": "groq ok"}]}'),
    ])
    script.set(GEMINI, [])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"groq:{GROQ_PRIMARY}"
    assert sleeps == [pytest.approx(60.0, abs=0.5)]  # honoured in full (LLM_MAX_WAIT_S=60)


def test_retry_after_beyond_max_wait_goes_to_the_next_client(monkeypatch):
    router = _setup_router(monkeypatch, retries=1)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_FakeResp(429, {}, text="daily limit", headers={"Retry-After": "3600"})])
    script.set(GEMINI, [_ok('{"findings": []}')])
    _patch_script(monkeypatch, script)

    _, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"gemini:{GEMINI_MODEL}"
    assert sleeps == [] and script.call_count(PRIMARY) == 1
    # The block is remembered: the next call skips the blocked model at once.
    script.set(GEMINI, [_ok('{"findings": []}')])
    _, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"gemini:{GEMINI_MODEL}" and script.call_count(PRIMARY) == 1


# --- Per-model token pacing (fake clock) ---------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_token_pacer_sliding_window():
    clock = FakeClock()
    pacer = llm_client.TokenPacer({"groq": 8000, "openrouter": None}, clock=clock,
                                  sleep=clock.sleep)

    def take(key, provider, n, **kw):
        return asyncio.run(pacer.acquire(key, provider, n, **kw))

    assert take("groq:a", "groq", 6000) == llm_client.PACE_OK and clock.sleeps == []
    assert take("groq:a", "groq", 1500) == llm_client.PACE_OK and clock.sleeps == []  # 7500
    assert take("groq:a", "groq", 6000) == llm_client.PACE_OK
    assert clock.sleeps == [60.0]  # both earlier calls had to leave the window
    assert take("groq:b", "groq", 6000) == llm_client.PACE_OK  # own bucket per model
    assert take("openrouter:x", "openrouter", 10**6) == llm_client.PACE_OK
    assert clock.sleeps == [60.0]
    # Budget / deadline checks don't reserve or wait.
    assert take("groq:a", "groq", 6000, max_wait=30) == llm_client.PACE_TOO_LONG
    assert take("groq:a", "groq", 6000, deadline=clock.now + 30) == llm_client.PACE_DEADLINE
    assert clock.sleeps == [60.0]
    assert take("groq:a", "groq", 6000, deadline=clock.now + 61) == llm_client.PACE_OK
    assert clock.sleeps == [60.0, 60.0]
    # A per-model limit overrides the provider's.
    pacer.limits["groq:c"] = 1000
    assert take("groq:c", "groq", 900) == llm_client.PACE_OK
    assert take("groq:c", "groq", 900) == llm_client.PACE_OK and clock.sleeps[-1] == 60.0


def test_router_paces_back_to_back_groq_calls(monkeypatch):
    clock = FakeClock()
    _setup_router(monkeypatch, retries=0)
    monkeypatch.setattr(settings, "LLM_OUTPUT_TOKENS_ESTIMATE", 1500)
    router = LLMRouter(pacer=llm_client.TokenPacer({"groq": 8000}, clock=clock,
                                                   sleep=clock.sleep))
    script = _Script()
    script.set(PRIMARY, [_ok('{"findings": []}') for _ in range(3)])
    _patch_script(monkeypatch, script)
    prompt = "x" * 19000  # ~5.9K estimated tokens + 1.5K output allowance

    for _ in range(3):
        _, provider = asyncio.run(router.generate("sys", prompt))
        assert provider == f"groq:{GROQ_PRIMARY}"
    assert clock.sleeps == [60.0, 60.0]  # one ~7.5K-token call per minute


def test_router_skips_clients_that_would_outlast_the_deadline(monkeypatch):
    clock = FakeClock()
    _setup_router(monkeypatch, retries=1)
    router = LLMRouter(pacer=llm_client.TokenPacer({"groq": 8000, "gemini": 8000},
                                                   clock=clock, sleep=clock.sleep))
    script = _Script()
    script.set(PRIMARY, [_ok('{"findings": []}')])
    script.set(GEMINI, [])
    _patch_script(monkeypatch, script)
    prompt = "x" * 19000

    asyncio.run(router.generate("sys", prompt, deadline=clock.now + 300))
    # Groq and Gemini would both need a ~60 s wait; the deadline is 10 s away.
    router.pacer.block(f"gemini:{GEMINI_MODEL}", 120)
    with pytest.raises(LLMError) as err:
        asyncio.run(router.generate("sys", prompt, deadline=clock.now + 10))
    assert err.value.deadline_exceeded is True
    assert clock.sleeps == [] and script.call_count(PRIMARY) == 1


def test_zero_retries_falls_through_immediately(monkeypatch):
    router = _setup_router(monkeypatch, retries=0)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_FakeResp(429, {}, text="rate limited")])
    script.set(GEMINI, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"gemini:{GEMINI_MODEL}"
    assert script.call_count(PRIMARY) == 1
    assert sleeps == []


# --- Same-provider (Groq) model fallback chain -------------------------------


def test_primary_model_success_reports_primary_model(monkeypatch):
    router = _setup_router(monkeypatch, retries=1, groq_fallback_model=GROQ_FB)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_ok('{"findings": [{"title": "primary ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"groq:{GROQ_PRIMARY}"
    assert payload["findings"][0]["title"] == "primary ok"
    assert script.order == [PRIMARY]
    assert sleeps == []


def test_404_on_primary_model_goes_straight_to_fallback_model(monkeypatch):
    router = _setup_router(monkeypatch, retries=2, groq_fallback_model=GROQ_FB)
    sleeps = _patch_sleep(monkeypatch)
    warnings, sink_id = _capture_warnings()

    script = _Script()
    script.set(PRIMARY, [_FakeResp(
        404, {}, text='{"error": {"message": "The model does not exist", '
                      '"code": "model_not_found"}}',
    )])
    script.set(GROQ_FALLBACK, [_ok('{"findings": [{"title": "fallback ok"}]}')])
    script.set(GEMINI, [])  # must never be called
    _patch_script(monkeypatch, script)

    try:
        payload, provider = asyncio.run(router.generate("sys", "user"))
    finally:
        logger.remove(sink_id)

    assert provider == f"groq:{GROQ_FB}"
    assert payload["findings"][0]["title"] == "fallback ok"
    # Non-retriable: one call on the primary despite LLM_RETRIES=2, no backoff.
    assert script.order == [PRIMARY, GROQ_FALLBACK]
    assert sleeps == []
    # A greppable warning names the retired model.
    gone = [m for m in warnings if "not found or decommissioned" in m]
    assert len(gone) == 1 and GROQ_PRIMARY in gone[0]


def test_decommissioned_400_is_logged_and_falls_through(monkeypatch):
    # Groq reports a decommissioned model as a 400 with code model_decommissioned.
    router = _setup_router(monkeypatch, retries=1, groq_fallback_model=GROQ_FB)
    sleeps = _patch_sleep(monkeypatch)
    warnings, sink_id = _capture_warnings()

    script = _Script()
    script.set(PRIMARY, [_FakeResp(
        400, {}, text='{"error": {"code": "model_decommissioned"}}',
    )])
    script.set(GROQ_FALLBACK, [_ok('{"findings": []}')])
    _patch_script(monkeypatch, script)

    try:
        _, provider = asyncio.run(router.generate("sys", "user"))
    finally:
        logger.remove(sink_id)

    assert provider == f"groq:{GROQ_FB}"
    assert script.order == [PRIMARY, GROQ_FALLBACK]
    assert sleeps == []
    assert any("decommissioned" in m and GROQ_PRIMARY in m for m in warnings)


def test_other_4xx_is_not_reported_as_retired_model(monkeypatch):
    router = _setup_router(monkeypatch, retries=1, groq_fallback_model=GROQ_FB)
    _patch_sleep(monkeypatch)
    warnings, sink_id = _capture_warnings()

    script = _Script()
    script.set(PRIMARY, [_FakeResp(401, {}, text='{"error": "invalid api key"}')])
    script.set(GROQ_FALLBACK, [_FakeResp(401, {}, text='{"error": "invalid api key"}')])
    script.set(GEMINI, [_ok('{"findings": []}')])
    _patch_script(monkeypatch, script)

    try:
        _, provider = asyncio.run(router.generate("sys", "user"))
    finally:
        logger.remove(sink_id)

    assert provider == f"gemini:{GEMINI_MODEL}"
    assert not any("decommissioned" in m for m in warnings)


def test_429_on_primary_retries_then_falls_to_fallback_model(monkeypatch):
    retries = 2
    router = _setup_router(monkeypatch, retries=retries, groq_fallback_model=GROQ_FB)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_FakeResp(429, {}, text="rate limited") for _ in range(retries + 1)])
    script.set(GROQ_FALLBACK, [_ok('{"findings": [{"title": "fallback ok"}]}')])
    script.set(GEMINI, [])  # must never be called
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"groq:{GROQ_FB}"
    assert payload["findings"][0]["title"] == "fallback ok"
    assert script.call_count(PRIMARY) == retries + 1
    assert script.call_count(GROQ_FALLBACK) == 1
    assert script.call_count(GEMINI) == 0
    assert len(sleeps) == retries


def test_both_groq_models_fail_then_gemini_answers(monkeypatch):
    router = _setup_router(monkeypatch, retries=1, groq_fallback_model=GROQ_FB)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_FakeResp(404, {}, text="model_not_found")])
    script.set(GROQ_FALLBACK, [_FakeResp(429, {}, text="rate limited") for _ in range(2)])
    script.set(GEMINI, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == f"gemini:{GEMINI_MODEL}"
    assert payload["findings"][0]["title"] == "gemini ok"
    assert script.order == [PRIMARY, GROQ_FALLBACK, GROQ_FALLBACK, GEMINI]
    assert len(sleeps) == 1  # only the fallback model's 429 was retried


def test_all_clients_fail_raises_llm_error(monkeypatch):
    router = _setup_router(monkeypatch, retries=0, groq_fallback_model=GROQ_FB)
    _patch_sleep(monkeypatch)

    script = _Script()
    script.set(PRIMARY, [_FakeResp(500, {}, text="down")])
    script.set(GROQ_FALLBACK, [_FakeResp(500, {}, text="down")])
    script.set(GEMINI, [_FakeResp(500, {}, text="down")])
    _patch_script(monkeypatch, script)

    with pytest.raises(LLMError, match="All LLM clients failed"):
        asyncio.run(router.generate("sys", "user"))
    assert script.order == [PRIMARY, GROQ_FALLBACK, GEMINI]
