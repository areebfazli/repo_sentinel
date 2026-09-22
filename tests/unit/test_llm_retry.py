"""Unit tests for LLMRouter's same-provider retry/backoff before fallthrough.

Uses a scripted fake httpx.AsyncClient (per-provider queues of canned responses,
keyed by URL) and monkeypatches llm_client.asyncio.sleep so no test actually
sleeps; sleep durations are recorded instead of waited on.
"""
import asyncio

from backend.app.config import settings
from backend.app.core import llm_client
from backend.app.core.llm_client import PROVIDERS, LLMRouter

GROQ_URL = PROVIDERS["groq"]["url"]
GEMINI_URL = PROVIDERS["gemini"]["url"]


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
    """Per-provider (by URL) queue of scripted responses; records call counts."""

    def __init__(self):
        self.queues: dict[str, list] = {}
        self.calls: dict[str, int] = {}

    def set(self, url, responses):
        self.queues[url] = list(responses)

    def __call__(self, url, **kwargs):
        self.calls[url] = self.calls.get(url, 0) + 1
        queue = self.queues.get(url)
        if not queue:
            raise AssertionError(f"unexpected call to {url} (no scripted response left)")
        return queue.pop(0)

    def call_count(self, url):
        return self.calls.get(url, 0)


def _setup_router(monkeypatch, retries):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    monkeypatch.setattr(settings, "LLM_RETRIES", retries)
    return LLMRouter()


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
    script.set(GROQ_URL, [
        _FakeResp(429, {}, text="rate limited"),
        _ok('{"findings": [{"title": "groq ok"}]}'),
    ])
    script.set(GEMINI_URL, [])  # must never be called
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "groq"
    assert payload["findings"][0]["title"] == "groq ok"
    assert script.call_count(GEMINI_URL) == 0
    assert len(sleeps) == 1


def test_primary_exhausts_retries_then_falls_back(monkeypatch):
    retries = 2
    router = _setup_router(monkeypatch, retries=retries)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(GROQ_URL, [_FakeResp(500, {}, text="server error") for _ in range(retries + 1)])
    script.set(GEMINI_URL, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "gemini"
    assert payload["findings"][0]["title"] == "gemini ok"
    assert script.call_count(GROQ_URL) == retries + 1
    assert len(sleeps) == retries


def test_malformed_json_no_retry_immediate_fallback(monkeypatch):
    router = _setup_router(monkeypatch, retries=1)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(GROQ_URL, [_ok("not json")])  # 200, but content isn't valid JSON text
    script.set(GEMINI_URL, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "gemini"
    assert payload["findings"][0]["title"] == "gemini ok"
    assert script.call_count(GROQ_URL) == 1  # no same-provider retry
    assert sleeps == []


def test_retry_after_header_capped_at_ten_seconds(monkeypatch):
    router = _setup_router(monkeypatch, retries=1)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(GROQ_URL, [
        _FakeResp(429, {}, text="rate limited", headers={"Retry-After": "60"}),
        _ok('{"findings": [{"title": "groq ok"}]}'),
    ])
    script.set(GEMINI_URL, [])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "groq"
    assert sleeps == [10.0]  # honoured but capped, not 60


def test_zero_retries_falls_through_immediately(monkeypatch):
    router = _setup_router(monkeypatch, retries=0)
    sleeps = _patch_sleep(monkeypatch)

    script = _Script()
    script.set(GROQ_URL, [_FakeResp(429, {}, text="rate limited")])
    script.set(GEMINI_URL, [_ok('{"findings": [{"title": "gemini ok"}]}')])
    _patch_script(monkeypatch, script)

    payload, provider = asyncio.run(router.generate("sys", "user"))
    assert provider == "gemini"
    assert script.call_count(GROQ_URL) == 1
    assert sleeps == []
