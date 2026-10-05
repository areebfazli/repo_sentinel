"""LLM client: output cap (max_tokens), truncated output, per-model sampling,
and JSON extraction that prefers the final top-level answer. No network: a
scripted fake httpx client.
"""
import asyncio
import json

import pytest

from backend.app.config import Settings, settings
from backend.app.core import llm_client
from backend.app.core.llm_client import (
    MIN_OUTPUT_TOKENS,
    PROVIDERS,
    LLMClient,
    LLMError,
    LLMRouter,
    TokenPacer,
    extract_json,
    model_sampling,
)

# Same scripted fake httpx.AsyncClient as test_llm_openrouter.py / test_llm_retry.py.
OR_BASE = "https://openrouter.ai/api/v1"
OR_URL = f"{OR_BASE}/chat/completions"
OR_PRIMARY = "qwen/qwen3.8-27b:free"
PRIMARY = (OR_URL, OR_PRIMARY)
GEMINI = (PROVIDERS["gemini"]["url"], "gemini-2.0-flash")
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


def _pin_openrouter(monkeypatch, retries=1, fallback_provider="gemini"):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", fallback_provider)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "ork")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", OR_PRIMARY)
    monkeypatch.setattr(settings, "OPENROUTER_FALLBACK_MODEL", None)
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", OR_BASE)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gm")
    monkeypatch.setattr(settings, "GEMINI_MODEL", GEMINI[1])
    monkeypatch.setattr(settings, "LLM_RETRIES", retries)


def _patch_sleep(monkeypatch):
    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)


def _patch_script(monkeypatch, script):
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(script))


GROQ_URL = PROVIDERS["groq"]["url"]
QWEN = "qwen/qwen3.8-27b"


def _choice(content, finish_reason="stop", completion_tokens=None):
    data = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    if completion_tokens is not None:
        data["usage"] = {"completion_tokens": completion_tokens}
    return _FakeResp(200, data)


# --- extract_json: the last top-level object ------------------------------------------


def test_extract_json_prefers_the_final_answer_over_a_draft():
    final = {"findings": [{"title": "SQL injection", "line": 4}]}
    content = f'Draft {{"findings": []}} then final {json.dumps(final)}'
    assert extract_json(content) == final


def test_extract_json_prefers_the_last_fenced_answer():
    content = ('First try:\n```json\n{"findings": []}\n```\nOn reflection:\n'
               '```json\n{"findings": [{"title": "x"}]}\n```')
    assert extract_json(content) == {"findings": [{"title": "x"}]}


def test_extract_json_never_returns_an_inner_object_of_truncated_output():
    content = ('Here is my answer: {"findings": [{"title": "Path traversal", '
               '"quoted_code": "open(p)"}, {"title": "XSS", "quoted_co')
    with pytest.raises(json.JSONDecodeError):
        extract_json(content)


def test_extract_json_skips_nested_objects_and_prose_braces():
    content = ('The {set} of checks looks fine. {"findings": [{"title": "a"}, '
               '{"title": "b", "meta": {"k": "}"}}]} done')
    assert extract_json(content) == {"findings": [{"title": "a"},
                                                  {"title": "b", "meta": {"k": "}"}}]}


def test_extract_json_complete_draft_before_truncated_final_returns_the_draft():
    # The extractor alone can't know the second object was the real answer; the
    # client rejects such output via finish_reason "length" (tested below).
    content = 'Draft {"findings": []} then final {"findings": [{"title": "cut'
    assert extract_json(content) == {"findings": []}


def _findings_list(obj):
    return None if isinstance(obj.get("findings"), list) else "no findings list"


def test_extract_json_validator_skips_a_trailing_example_object():
    answer = {"findings": [{"title": "SQL injection", "line": 4}]}
    content = (f"{json.dumps(answer)}\nFor reference, each finding looks like "
               '{"title": "...", "line": 0}.')
    assert extract_json(content, validate=_findings_list) == answer
    # Without a validator the last complete object still wins (unchanged).
    assert extract_json(content) == {"title": "...", "line": 0}


def test_extract_json_validator_takes_the_last_passing_candidate():
    content = 'Draft {"findings": []} final {"findings": [1]} note {"note": "x"}'
    assert extract_json(content, validate=_findings_list) == {"findings": [1]}


@pytest.mark.parametrize("prose", [
    'He said "hi {" and meant it. ',
    'An unclosed {"key in prose, ',
    "Braces {\"a\": 1, ...} in prose. ",
])
def test_extract_json_unclosed_brace_or_quote_in_prose_does_not_hide_the_answer(prose):
    content = f'{prose}Answer: {{"findings": [3]}}'
    assert extract_json(content) == {"findings": [3]}
    assert extract_json(content, validate=_findings_list) == {"findings": [3]}


def test_extract_json_validator_never_takes_an_inner_object_of_truncated_output():
    content = ('{"findings": [{"title": "Path traversal", "quoted_code": "open(p)"}, '
               '{"title": "XSS", "quoted_co')
    with pytest.raises(json.JSONDecodeError):
        extract_json(content, validate=lambda o: None)


@pytest.mark.parametrize("cut", ["tru", "fals", "nul", "-", "1e", "1.", '"x\\u12', "12  "])
def test_extract_json_output_cut_mid_token_never_yields_an_inner_object(cut):
    # Cut off inside a literal / number / escape: the decode error is not at the
    # very end of the text, but the rest of the text is one partial token.
    content = '{"findings": [{"title": "XSS", "line": 3}, {"line": ' + cut
    with pytest.raises(json.JSONDecodeError):
        extract_json(content)
    with pytest.raises(json.JSONDecodeError):
        extract_json(content, validate=lambda o: None)


def test_extract_json_many_unclosed_starts_stay_bounded():
    # A long reply full of unclosed '{"' starts: each one is scanned to the end
    # of the text, so the scan gives up after _MAX_UNCLOSED_STARTS of them
    # rather than going quadratic (this took ~80 s at 100 KB without the cap).
    import time

    content = '{"a" x ' * 14_000 + '{"findings": []}'
    t0 = time.perf_counter()
    with pytest.raises(json.JSONDecodeError):
        extract_json(content)
    assert time.perf_counter() - t0 < 5
    # A few unclosed starts in prose still don't hide the answer.
    few = '{"a" x ' * llm_client._MAX_UNCLOSED_STARTS + '{"findings": []}'
    assert extract_json(few) == {"findings": []}


@pytest.mark.parametrize("content", [
    '{"title": "a bare finding"}',                      # plain JSON
    'Answer: {"title": "a"} and {"note": "b"}',         # objects, none passes
    '```json\n[{"title": "a"}]\n```',                 # a fenced list
])
def test_extract_json_no_passing_candidate_is_an_invalid_answer(content):
    with pytest.raises(llm_client.InvalidAnswer, match="no findings list|not a list|"
                                                       "object"):
        extract_json(content, validate=lambda o: _findings_list(o) if isinstance(o, dict)
                     else "not an object")


def test_validator_failure_on_one_model_falls_through_to_the_next(monkeypatch):
    from backend.app.core.pr_review import audit_schema_problem

    _pin_openrouter(monkeypatch, retries=1)  # fallback: gemini
    _patch_sleep(monkeypatch)
    script = _Script()
    # A bare finding (no findings list): parsed fine, but not an audit answer.
    script.set(PRIMARY, [_ok('{"title": "Path traversal", "line": 7}')])
    script.set(GEMINI, [_ok('{"findings": [{"title": "from gemini"}]}')])
    _patch_script(monkeypatch, script)
    data, label = asyncio.run(LLMRouter().generate(
        "s", "u", validate=lambda d: audit_schema_problem(d, final=True)))
    assert label == "gemini:gemini-2.0-flash" and data["findings"][0]["title"] == "from gemini"
    assert script.call_count(PRIMARY) == 1  # not retried on the same client


def test_every_model_failing_the_validator_raises_bad_output(monkeypatch):
    from backend.app.core.pr_review import verifier_schema_problem

    _pin_openrouter(monkeypatch, retries=1)
    _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_ok('{"note": "no verdict"}')])
    script.set(GEMINI, [_ok('{"findings": []}')])
    _patch_script(monkeypatch, script)
    with pytest.raises(LLMError, match="unusable answer: no verdict") as info:
        asyncio.run(LLMRouter().generate("s", "u", validate=verifier_schema_problem))
    assert info.value.bad_output is True


def test_other_failures_are_not_bad_output(monkeypatch):
    _pin_openrouter(monkeypatch, retries=0, fallback_provider=None)
    script = _Script()
    script.set(PRIMARY, [_FakeResp(500, text="boom")])
    _patch_script(monkeypatch, script)
    with pytest.raises(LLMError) as info:
        asyncio.run(LLMRouter().generate("s", "u", validate=lambda d: None))
    assert info.value.bad_output is False


# --- max_tokens and truncated output ---------------------------------------------------


def test_max_output_tokens_default_and_request_body(monkeypatch):
    assert Settings.model_fields["LLM_MAX_OUTPUT_TOKENS"].default == 16000
    _pin_openrouter(monkeypatch, fallback_provider=None)
    script = _Script()
    script.set(PRIMARY, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    assert asyncio.run(LLMRouter().generate("s", "u"))[0] == json.loads(OK_JSON)
    assert script.requests[0]["json"]["max_tokens"] == 16000


def test_max_output_tokens_none_is_not_sent(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_provider=None)
    monkeypatch.setattr(settings, "LLM_MAX_OUTPUT_TOKENS", None)
    script = _Script()
    script.set(PRIMARY, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    asyncio.run(LLMRouter().generate("s", "u"))
    assert "max_tokens" not in script.requests[0]["json"]


@pytest.mark.parametrize("content", [
    None,
    '{"findings": [{"title": "cut',
    'Draft {"findings": []} then final {"findings": [{"title": "cut',
    '<think>long reasoning that was cut</think>{"findings": [',
])
def test_length_cut_output_falls_through_never_an_empty_review(monkeypatch, content):
    _pin_openrouter(monkeypatch, retries=1)  # fallback: gemini
    _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_choice(content, "length", completion_tokens=16000)])
    script.set(GEMINI, [_ok('{"findings": [{"title": "from gemini"}]}')])
    _patch_script(monkeypatch, script)
    data, label = asyncio.run(LLMRouter().generate("s", "u"))
    assert label == "gemini:gemini-2.0-flash"
    assert data["findings"][0]["title"] == "from gemini"
    assert script.call_count(PRIMARY) == 1  # bad output: not retried on the same client


def test_length_with_a_complete_json_document_is_accepted(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_provider=None)
    script = _Script()
    script.set(PRIMARY, [_choice('{"findings": []}', "length")])
    _patch_script(monkeypatch, script)
    assert asyncio.run(LLMRouter().generate("s", "u")) == (
        {"findings": []}, f"openrouter:{OR_PRIMARY}")


def test_all_clients_truncated_raises(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_provider=None)
    script = _Script()
    script.set(PRIMARY, [_choice('{"findings": [', "length")])
    _patch_script(monkeypatch, script)
    with pytest.raises(LLMError, match="truncated"):
        asyncio.run(LLMRouter().generate("s", "u"))


@pytest.mark.parametrize("content", ["[1, 2]", '"findings"', "42"])
def test_json_that_is_not_an_object_falls_through(monkeypatch, content):
    _pin_openrouter(monkeypatch)
    _patch_sleep(monkeypatch)
    script = _Script()
    script.set(PRIMARY, [_ok(content)])
    script.set(GEMINI, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    assert asyncio.run(LLMRouter().generate("s", "u"))[1] == "gemini:gemini-2.0-flash"


def _groq_router(monkeypatch, limit=8000):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_FALLBACK_PROVIDER", None)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", None)
    return LLMRouter(pacer=TokenPacer({"groq": limit}))


def test_tpm_limited_client_gets_max_tokens_within_its_limit(monkeypatch):
    router = _groq_router(monkeypatch)
    script = _Script()
    script.set((GROQ_URL, "openai/gpt-oss-120b"), [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    user = "x" * 16000  # ~5000 estimated tokens
    asyncio.run(router.generate("s", user))
    sent = script.requests[0]["json"]["max_tokens"]
    assert MIN_OUTPUT_TOKENS <= sent <= 8000 - 5000


def test_prompt_too_large_for_the_tpm_limit_skips_the_client(monkeypatch):
    router = _groq_router(monkeypatch)
    script = _Script()  # no scripted response: any request would fail the test
    _patch_script(monkeypatch, script)
    user = "x" * 38400  # ~12000 estimated tokens: a full PR-review prompt
    with pytest.raises(LLMError, match="tokens/min"):
        asyncio.run(router.generate("s", user))
    assert script.requests == []


def test_unlimited_client_takes_a_large_prompt_with_the_full_cap(monkeypatch):
    _pin_openrouter(monkeypatch, fallback_provider=None)
    script = _Script()
    script.set(PRIMARY, [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    asyncio.run(LLMRouter().generate("s", "x" * 38400))
    assert script.requests[0]["json"]["max_tokens"] == 16000


# --- per-model sampling -------------------------------------------------------------------


def test_sampling_table_default_for_qwen():
    assert Settings.model_fields["LLM_SAMPLING"].default[QWEN] == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20}


@pytest.mark.parametrize("provider,model,found", [
    ("openrouter", "qwen/qwen3.8-27b:free", True),
    ("openrouter", QWEN, True),
    ("groq", QWEN, True),
    ("groq", "openai/gpt-oss-120b", False),
    ("openrouter", "google/gemma-4-31b-it:free", False),
])
def test_model_sampling_matches_with_and_without_free_suffix(provider, model, found):
    assert bool(model_sampling(provider, model)) is found


def test_provider_qualified_sampling_key_wins(monkeypatch):
    monkeypatch.setattr(settings, "LLM_SAMPLING", {
        QWEN: {"temperature": 1.0}, f"groq:{QWEN}": {"temperature": 0.6}})
    assert model_sampling("groq", QWEN) == {"temperature": 0.6}
    assert model_sampling("openrouter", f"{QWEN}:free") == {"temperature": 1.0}


def _sent_body(monkeypatch, client):
    script = _Script()
    script.set((client.url, client.model), [_ok(OK_JSON)])
    _patch_script(monkeypatch, script)
    asyncio.run(client.complete("s", "u"))
    return script.requests[0]["json"]


def test_openrouter_qwen_sends_the_model_card_sampling_with_top_k(monkeypatch):
    _pin_openrouter(monkeypatch)
    client = LLMClient("openrouter")
    assert client.temperature is None and client.effective_temperature == 1.0
    body = _sent_body(monkeypatch, client)
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 20)


def test_groq_qwen_gets_the_entry_but_never_top_k(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    body = _sent_body(monkeypatch, LLMClient("groq", model=QWEN))
    assert (body["temperature"], body["top_p"]) == (1.0, 0.95)
    assert "top_k" not in body


def test_model_without_an_entry_sends_llm_temperature_only(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gk")
    monkeypatch.setattr(settings, "LLM_TEMPERATURE", 0.3)
    body = _sent_body(monkeypatch, LLMClient("groq", model="openai/gpt-oss-120b"))
    assert body["temperature"] == 0.3
    assert "top_p" not in body and "top_k" not in body


@pytest.mark.parametrize("how", ["constructor", "attribute"])
def test_explicit_temperature_override_wins_and_sends_only_temperature(monkeypatch, how):
    _pin_openrouter(monkeypatch)
    if how == "constructor":
        client = LLMClient("openrouter", temperature=0.0)
    else:
        client = LLMClient("openrouter")
        client.temperature = 0.0  # what the eval's set_router_temperature does
    assert client.sampling_params() == {"temperature": 0.0}
    assert client.effective_temperature == 0.0
    body = _sent_body(monkeypatch, client)
    assert body["temperature"] == 0.0
    assert "top_p" not in body and "top_k" not in body
    client.temperature = None  # back to the model's entry
    assert client.sampling_params() == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
