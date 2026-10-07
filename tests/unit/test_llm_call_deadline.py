"""The hard wall-clock deadline per LLM HTTP attempt (settings.LLM_CALL_DEADLINE_S).

httpx's LLM_TIMEOUT_SECONDS is a per-read timeout, which an upstream's
keep-alive bytes keep resetting; the call deadline bounds the whole request.
These tests use a real httpx.AsyncClient over an httpx.MockTransport whose
async handler sleeps past a tiny deadline (no network): the request is
cancelled, the client closed, the failure is a transient timeout (same-client
retry, then the next client, "failed" outcome, never bad output), and a
caller's cancellation (Ctrl-C in the eval) stays a CancelledError.
"""
import asyncio
import json
import time

import httpx
import pytest
from loguru import logger

from backend.app.config import settings
from backend.app.core import llm_client
from backend.app.core.llm_client import (
    PROVIDERS,
    CallDeadlineExceeded,
    LLMError,
    LLMRouter,
    TokenPacer,
)
from backend.app.core.pr_review import _failure_reason
from ml.evaluation import run_pr_eval as R
from ml.evaluation.llm_eval_common import HttpUsageTap, LLMCache, classify_llm_error

GROQ_URL = PROVIDERS["groq"]["url"]
GEMINI_URL = PROVIDERS["gemini"]["url"]
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.0-flash"
ANSWER = {"findings": []}

_real_sleep = asyncio.sleep
_RealAsyncClient = httpx.AsyncClient


class SlowUpstream:
    """MockTransport handler: models in ``slow`` hang (``hang_s``), others answer
    at once. Records each request's model, and which hung requests were
    cancelled (the deadline or the caller cancelling the request)."""

    def __init__(self, slow=(), hang_s=30.0, on_slow=None):
        self.slow = set(slow)
        self.hang_s = hang_s
        self.on_slow = on_slow
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.clients: list[httpx.AsyncClient] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        self.calls.append(model)
        if model in self.slow:
            if self.on_slow:
                self.on_slow()
            try:
                await _real_sleep(self.hang_s)
            except asyncio.CancelledError:
                self.cancelled.append(model)
                raise
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(ANSWER)}}]})

    def install(self, monkeypatch):
        """Every httpx.AsyncClient (what LLMClient._post opens) gets this
        transport. A subclass, so the eval's HttpUsageTap can still patch
        ``httpx.AsyncClient.post``."""
        transport = httpx.MockTransport(self)
        upstream = self

        class MockedAsyncClient(_RealAsyncClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, transport=transport, **kwargs)
                upstream.clients.append(self)

        monkeypatch.setattr(llm_client.httpx, "AsyncClient", MockedAsyncClient)
        return self


def _router(monkeypatch, *, deadline_s, retries=1, clock=None):
    """Groq primary (no same-provider fallback) + Gemini fallback; pacing
    without limits and without real sleeps (retry backoff is recorded)."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    monkeypatch.setattr(settings, "GROQ_MODEL", GROQ_MODEL)
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", None)
    monkeypatch.setattr(settings, "GEMINI_MODEL", GEMINI_MODEL)
    monkeypatch.setattr(settings, "LLM_RETRIES", retries)
    monkeypatch.setattr(settings, "LLM_CALL_DEADLINE_S", deadline_s)
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    pacer = TokenPacer({}, sleep=fake_sleep, **({"clock": clock} if clock else {}))
    router = LLMRouter(pacer=pacer)
    router.sleeps = sleeps
    return router


def _warnings():
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    return messages, sink


def test_deadline_cancels_a_hung_call_retries_it_then_falls_through(monkeypatch):
    up = SlowUpstream(slow={GROQ_MODEL}).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=0.05, retries=1)
    messages, sink = _warnings()
    t0 = time.monotonic()
    try:
        data, label = asyncio.run(router.generate("s", "u"))
    finally:
        logger.remove(sink)
    assert time.monotonic() - t0 < 5  # bounded by the deadline, not the 30 s hang
    assert (data, label) == (ANSWER, f"gemini:{GEMINI_MODEL}")
    # First try + one same-client retry (transient, like a timeout), then Gemini.
    assert up.calls == [GROQ_MODEL, GROQ_MODEL, GEMINI_MODEL]
    assert up.cancelled == [GROQ_MODEL, GROQ_MODEL]  # each hung request was cancelled
    assert len(router.sleeps) == 1  # the retry's backoff
    assert all(c.is_closed for c in up.clients)  # every client (and its pool) closed
    deadline_logs = [m for m in messages if "did not answer within LLM_CALL_DEADLINE_S=0.05s" in m]
    assert len(deadline_logs) == 2 and all(GROQ_MODEL in m for m in deadline_logs)


def test_deadline_on_every_client_is_llm_error_not_bad_output(monkeypatch):
    SlowUpstream(slow={GROQ_MODEL, GEMINI_MODEL}).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=0.05, retries=0)
    with pytest.raises(LLMError) as info:
        asyncio.run(router.generate("s", "u"))
    exc = info.value
    assert exc.bad_output is False and exc.deadline_exceeded is False
    assert exc.client_outcomes == [(f"groq:{GROQ_MODEL}", "failed"),
                                   (f"gemini:{GEMINI_MODEL}", "failed")]
    assert "call deadline" in str(exc)
    # Callers: an LLM error (not bad output, not the scan's time budget); the
    # eval: like a timeout - neither a rate limit nor a daily limit.
    assert _failure_reason(exc) == "llm_error"
    assert classify_llm_error(exc, []) == "other"


def test_deadline_error_is_a_timeout():
    exc = CallDeadlineExceeded("groq:m exceeded the call deadline")
    assert isinstance(exc, httpx.TimeoutException)
    assert not isinstance(exc, asyncio.CancelledError)


@pytest.mark.parametrize("deadline_s", [None, 0])
def test_deadline_off_waits_for_a_slow_answer(monkeypatch, deadline_s):
    up = SlowUpstream(slow={GROQ_MODEL}, hang_s=0.2).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=deadline_s, retries=0)
    data, label = asyncio.run(router.generate("s", "u"))
    assert (data, label) == (ANSWER, f"groq:{GROQ_MODEL}")
    assert up.calls == [GROQ_MODEL] and up.cancelled == []


def test_answer_within_the_deadline_is_used(monkeypatch):
    up = SlowUpstream(slow={GROQ_MODEL}, hang_s=0.05).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=5.0, retries=0)
    data, label = asyncio.run(router.generate("s", "u"))
    assert label == f"groq:{GROQ_MODEL}" and up.cancelled == []


def test_caller_cancel_propagates_as_cancelled_error_not_a_deadline(monkeypatch):
    up = SlowUpstream(slow={GROQ_MODEL, GEMINI_MODEL}).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=30.0, retries=1)
    messages, sink = _warnings()

    async def go():
        task = asyncio.create_task(router.generate("s", "u"))
        while not up.calls:
            await _real_sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(go())
    finally:
        logger.remove(sink)
    assert up.calls == [GROQ_MODEL]  # no retry, no fallthrough
    assert up.cancelled == [GROQ_MODEL]
    assert all(c.is_closed for c in up.clients)
    assert not any("LLM_CALL_DEADLINE_S" in m for m in messages)


def test_retry_after_a_deadline_hit_past_the_scan_budget_is_time_budget(monkeypatch):
    # The scan's LLM_SCAN_MAX_WALL_S runs out while the call hangs: the
    # deadline hit isn't retried (the pacer refuses a start past the scan's
    # deadline), so the call ends as "time_budget" like any other overrun.
    now = [0.0]
    up = SlowUpstream(slow={GROQ_MODEL}, on_slow=lambda: now.__setitem__(0, 1000.0))
    up.install(monkeypatch)
    router = _router(monkeypatch, deadline_s=0.05, retries=1, clock=lambda: now[0])
    with pytest.raises(LLMError) as info:
        asyncio.run(router.generate("s", "u", deadline=480.0))
    assert up.calls == [GROQ_MODEL]
    assert info.value.deadline_exceeded is True
    assert _failure_reason(info.value) == "time_budget"


# --- the eval's gate ----------------------------------------------------------------


def _gate(cache, **kw):
    base = dict(cache=cache, max_calls=100, token_budget=None, sleep_s=0.0, tpm=0,
                max_rate_limit_errors=3, temperature=0.0)
    base.update(kw)
    gate = R.EvalGate(**base)
    gate.begin_item("item-1")
    return gate


def test_eval_gate_deadline_is_an_uncached_other_error(monkeypatch, tmp_path):
    SlowUpstream(slow={GROQ_MODEL, GEMINI_MODEL}).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=0.05, retries=0)
    cache = LLMCache(tmp_path / "cache.jsonl")
    with HttpUsageTap() as tap:
        gate = _gate(cache, tap=tap)
        with pytest.raises(LLMError) as info:
            asyncio.run(gate.call(router, "s", "u"))
    assert not isinstance(info.value, R.EvalStopped)
    assert httpx.AsyncClient.post is _RealAsyncClient.post  # the tap was restored
    [entry] = gate.log
    assert entry["error_kind"] == "other" and "call deadline" in entry["error"]
    assert gate.item_error_kind == "other" and gate.stopped is None
    assert gate.consecutive_rl == 0
    assert len(cache) == 0 and not (tmp_path / "cache.jsonl").exists()
    # The next call still runs (a deadline hit is no stop condition).
    up = SlowUpstream().install(monkeypatch)
    data, _ = asyncio.run(gate.call(router, "s", "u"))
    assert data == ANSWER and up.calls == [GROQ_MODEL] and len(cache) == 1


def test_eval_gate_ctrl_c_still_stops_the_eval(monkeypatch, tmp_path):
    up = SlowUpstream(slow={GROQ_MODEL, GEMINI_MODEL}).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=30.0, retries=1)
    cache = LLMCache(tmp_path / "cache.jsonl")
    gate = _gate(cache)

    async def go():
        task = asyncio.create_task(gate.call(router, "s", "u"))
        while not up.calls:
            await _real_sleep(0.01)
        task.cancel()  # what asyncio.run does to the main task on Ctrl-C
        with pytest.raises(R.EvalStopped, match="interrupted"):
            await task

    asyncio.run(go())
    assert gate.stopped == "interrupted" and gate.item_not_run
    assert up.calls == [GROQ_MODEL] and up.cancelled == [GROQ_MODEL]
    assert len(cache) == 0


def test_negative_deadline_is_off_not_an_instant_timeout(monkeypatch):
    up = SlowUpstream(slow={GROQ_MODEL}, hang_s=0.05).install(monkeypatch)
    router = _router(monkeypatch, deadline_s=-1, retries=0)
    data, label = asyncio.run(router.generate("s", "u"))
    assert (data, label) == (ANSWER, f"groq:{GROQ_MODEL}") and up.cancelled == []
