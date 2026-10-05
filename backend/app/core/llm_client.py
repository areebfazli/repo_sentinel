"""LLM provider abstraction.

Groq, Gemini and OpenRouter all expose OpenAI-compatible chat-completions
endpoints, so one thin async client handles them. Each client is one
(provider, model) pair. LLMRouter's chain is: primary provider/model -> the
primary provider's same-provider fallback model (groq -> GROQ_FALLBACK_MODEL,
openrouter -> OPENROUTER_FALLBACK_MODEL) -> LLM_FALLBACK_PROVIDER's model -> that
provider's own same-provider fallback model. With the defaults:
openrouter:qwen/qwen3.8-27b:free -> openrouter:google/gemma-4-31b-it:free ->
groq:openai/gpt-oss-120b -> groq:qwen/qwen3.8-27b. Groq and OpenRouter's free
models rate-limit per model, so the second model is a genuine fallback, not a
redundant one; OpenRouter's free models also share an upstream pool (429
"upstream_provider_shared_pool"), hence the cross-provider step.

Calls are paced per model by estimated tokens per minute (``TokenPacer``,
settings.LLM_TPM_LIMITS: Groq's free tier allows ~8K tokens/min per model).
A transient failure (429 / 5xx / timeout) is retried on the SAME client up to
settings.LLM_RETRIES times after its Retry-After (else a short backoff); a wait
over settings.LLM_MAX_WAIT_S, or past the caller's deadline, skips to the next
client instead. Then it falls through to the next client; a
bad-output failure (malformed/empty response, null content, output cut at
max_tokens, unparseable JSON, no object passing ``validate``) or a non-retriable
HTTP error (e.g. 404 for a retired model, 401) skips straight to the next client.
When every client fails, ``LLMError.bad_output`` says whether it was only bad
output (callers report "bad_output") or not ("llm_error"), keyed on how each
client ENDED (``LLMError.client_outcomes``): a 429 that was retried and then
answered unusably ends that client as bad output. See ``_all_bad_output``.
HTTP 402 (insufficient credits) is account-wide: no retry, and the provider's
remaining models are skipped too.

OpenRouter quirks handled here: an error object inside an HTTP 200 body
(`{"error": {...}}`; its integer or numeric-string `code` is handled like an
HTTP status of that code: 429 / 5xx transient, 402 account-wide, other 4xx a
non-retriable error; an error object without such a code is bad output), and
free models that reject `response_format` (HTTP 400 ->
one retry without it, remembered for the client's lifetime). All providers'
content goes through a tolerant JSON extractor (```json fences, <think> blocks
or reasoning text before the object; the LAST complete top-level object wins,
or with the caller's ``validate`` format check the last one that passes it);
plain JSON parses exactly as before. A reply that isn't a JSON object, that has
no object passing ``validate``, or that was cut at max_tokens (finish_reason
"length") without a complete JSON document, is a bad-output failure: the next
client is tried.

Every request carries max_tokens (settings.LLM_MAX_OUTPUT_TOKENS, lowered for a
client with a tokens-per-minute limit to what the prompt leaves of it; a prompt
that leaves too little skips that client) and the model's sampling parameters
(settings.LLM_SAMPLING, else LLM_TEMPERATURE; ``LLMClient.temperature`` set by a
caller overrides the temperature, see ``LLMClient.sampling_params``).

`provider_used` is "<provider>:<model>" (e.g. "groq:openai/gpt-oss-120b"), or "mock".

Mock mode is ONLY entered via LLM_PROVIDER=mock — a configured real provider with
a missing key raises at construction (fail-fast at startup), never a silent mock.
"""
import asyncio
import json
import math
import random
import re
import time

import httpx
from loguru import logger

from backend.app.config import settings

# Per provider: a fixed "url", or a "base_url_setting" (+ "/chat/completions")
# resolved when the client is built; "fallback_model_setting" names the
# same-provider fallback model; "extra_headers" are sent on every request;
# "response_format_fallback" enables the retry-without-response_format path;
# "supports_top_k" allows sending top_k (not an OpenAI parameter: Groq and
# Gemini's OpenAI-compatible endpoints don't document it, so it isn't sent there).
PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_setting": "GROQ_API_KEY",
        "model_setting": "GROQ_MODEL",
        "fallback_model_setting": "GROQ_FALLBACK_MODEL",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_setting": "GEMINI_API_KEY",
        "model_setting": "GEMINI_MODEL",
    },
    "openrouter": {
        "base_url_setting": "OPENROUTER_BASE_URL",
        "key_setting": "OPENROUTER_API_KEY",
        "model_setting": "OPENROUTER_MODEL",
        "fallback_model_setting": "OPENROUTER_FALLBACK_MODEL",
        # Optional app-attribution headers (https://openrouter.ai/docs).
        "extra_headers": {
            "HTTP-Referer": "https://github.com/areebfazli/repo_sentinel",
            "X-Title": "RepoSentinel",
        },
        # Some free models reject response_format with HTTP 400.
        "response_format_fallback": True,
        "supports_top_k": True,
    },
}

# Least completion room (max_tokens) worth sending to a client with a
# tokens-per-minute limit; a prompt leaving less than this skips the client.
MIN_OUTPUT_TOKENS = 512

# Model ids (on providers with "response_format_fallback") known to reject
# response_format json_object: it is never sent to them, saving the 400
# round-trip. Unlisted models that reject it still hit the reactive
# 400 -> retry-once-without-it path in LLMClient.complete.
_NO_RESPONSE_FORMAT_MODELS = frozenset({
    "qwen/qwen3.8-27b:free",
})

# Canned structured response used only when LLM_PROVIDER=mock.
MOCK_RESPONSE = {
    "findings": [
        {
            "severity": "high",
            "cve_id": None,
            "team_pr_id": None,
            "title": "Mock finding (LLM_PROVIDER=mock)",
            "explanation": "LLM is in mock mode; set a real provider + key to get a full report.",
            "fix_snippet": "",
        }
    ]
}


class LLMError(Exception):
    """Raised when all configured clients fail (or on a non-retriable error).

    `provider_wide` marks an account-level failure (HTTP 402, insufficient
    credits): the router then skips the provider's remaining models as well.
    """

    def __init__(self, message: str, provider_wide: bool = False,
                 deadline_exceeded: bool = False, bad_output: bool = False,
                 client_outcomes: list[tuple[str, str]] | None = None):
        super().__init__(message)
        self.provider_wide = provider_wide
        # Some client was skipped because its rate budget would outlast the
        # caller's deadline (the scan's LLM time budget), not because it failed.
        self.deadline_exceeded = deadline_exceeded
        # Every client that was called ENDED answering unusably: no object of
        # the reply passed the caller's format check (``LLMRouter.generate``'s
        # ``validate``), not JSON, cut off at max_tokens, null content, or a
        # malformed HTTP 200 (incl. an error object without a usable code).
        # Callers report the call as "bad_output" rather than "llm_error". A
        # 429 / 5xx that was retried and then answered unusably still counts
        # as bad output (only each client's final outcome matters); a client
        # that ended on a transient / HTTP failure leaves it False
        # (``_all_bad_output``).
        self.bad_output = bad_output
        # Set by ``LLMRouter.generate`` when every client failed: one
        # (client label, final outcome) per client that was called (or skipped
        # for its rate budget), in chain order. Outcomes: "bad_output" (it
        # answered, unusably), "rate_limit" (its last attempt got a 429 -
        # HTTP or in an HTTP 200 body - or, after one, it was skipped for its
        # rate budget), "failed" (anything else: 5xx, timeout, transport,
        # non-retriable HTTP error, rate budget). Clients never called are
        # absent. None on an LLMError not raised by the router as a whole.
        self.client_outcomes = client_outcomes


class _Retriable(Exception):
    """Internal marker: this provider failed but the next one should be tried.

    `transient` distinguishes a same-provider-retriable failure (429/5xx from the
    HTTP layer) from a bad-output failure (malformed shape / null content on a 200)
    which goes straight to the next provider. `retry_after`, when the provider sent
    one (429/5xx only), overrides the exponential backoff for this attempt.
    `status` is the HTTP status (or in-body error code) of a transient failure,
    so the router can tell a client that ended on a 429 (rate limit) apart.
    """

    def __init__(self, message: str, transient: bool = True, retry_after: float | None = None,
                 status: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after
        self.status = status


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Parse a numeric (seconds-form) Retry-After header; ignore HTTP-date form."""
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _key_configured(api_key: str | None) -> bool:
    """A key counts as configured only if it's set and not a placeholder.

    .env.example ships values like `your_groq_api_key_here`; treating those as
    real keys would defeat the fail-fast-at-startup invariant.
    """
    return bool(api_key) and not api_key.endswith("_here")


# Provider error codes (OpenAI-compatible error body) meaning the model id itself
# is gone, even when the status isn't 404 (Groq uses 400 for decommissioned models).
_MODEL_GONE_CODES = ("model_not_found", "model_decommissioned")


def _model_gone(status: int, text: str) -> bool:
    """HTTP status (or in-body error code) ``status`` with body ``text``: is the
    model id itself gone?"""
    return status == 404 or any(code in text for code in _MODEL_GONE_CODES)


def error_object_code(err) -> int | None:
    """The status-like code of an OpenAI-style error object (``{"code": 429,
    "message": ...}``, as OpenRouter puts inside an HTTP 200 body): an int, or
    a numeric string such as "429". None when there is none usable: ``err`` not
    a dict, no code, a bool, or a non-numeric code (e.g. "model_not_found")."""
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str) and re.fullmatch(r"[0-9]+", code.strip()):
        return int(code.strip())
    return None


def _error_message(resp: httpx.Response) -> str:
    """The OpenAI-style `error.message` of a non-200 body, else the raw text."""
    try:
        err = resp.json().get("error")
    except (ValueError, AttributeError):
        err = None
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"]
    return resp.text


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*\s*\n?(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _object_end(text: str, start: int) -> int | None:
    """Index just past the ``}`` closing the ``{`` at ``start`` (JSON string
    aware), or None if it is never closed (e.g. output cut off mid-object)."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


# Where a JSON object can start: "{" then a key or "}" (prose braces like
# "{set}" are not candidates).
_OBJECT_START_RE = re.compile(r'\{\s*["}]')
_DECODER = json.JSONDecoder()
# What is left of a value cut off mid-token: "tru", "-", "1e", "1.", "\u12".
_PARTIAL_TOKEN_RE = re.compile(r"[A-Za-z0-9+\-.\\]*")
# Unclosed object starts (each one scanned to the end of the text) tried
# before the scan gives up, so a pathological reply full of unclosed '{"'
# costs at most this many linear scans instead of one per start (O(n^2): ~80 s
# for 100 KB, blocking the event loop).
_MAX_UNCLOSED_STARTS = 32


def _truncated(text: str, exc: json.JSONDecodeError) -> bool:
    """The decode failed because the text ENDED inside the value (output cut
    off, also mid-token such as ``tru`` or ``1e``), not because of a syntax
    error part-way (prose that only looks like the start of an object)."""
    if exc.msg.startswith("Unterminated string") or exc.pos >= len(text.rstrip()):
        return True
    return _PARTIAL_TOKEN_RE.fullmatch(text[exc.pos:].rstrip()) is not None


def _top_level_objects(text: str) -> list[dict]:
    """The complete top-level JSON objects in ``text``, in order.

    Scanned from each possible object start: a complete object is a
    candidate and the scan resumes after it (objects nested in it are not
    top-level); an object that runs to the end of the text (truncated output)
    ends the scan, so nothing nested in it is returned. A start that is not
    valid JSON is skipped: a closed span (``{"a": 1, ...}``) as a whole, an
    unclosed one (prose such as ``He said "hi {"``, whose quote never closes)
    by one character, so it can't hide an answer that follows it (after
    _MAX_UNCLOSED_STARTS such starts the scan ends, as for truncated output)."""
    out: list[dict] = []
    pos = 0
    unclosed = 0
    while (m := _OBJECT_START_RE.search(text, pos)) is not None:
        try:
            obj, end = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError as exc:
            if _truncated(text, exc):
                break
            end = _object_end(text, m.start())
            if end is None:
                unclosed += 1
                if unclosed > _MAX_UNCLOSED_STARTS:
                    break
            pos = end if end is not None else m.start() + 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        pos = end
    return out


class InvalidAnswer(ValueError):
    """The reply holds JSON, but no candidate object passes the caller's
    format check (``extract_json``'s ``validate``); ``problem`` says why the
    last candidate failed."""

    def __init__(self, problem: str):
        super().__init__(problem)
        self.problem = problem


def _first_valid(candidates, validate):
    """(the first of ``candidates`` that ``validate`` accepts or None, the
    problem of the first rejected one or None)."""
    first_problem = None
    for obj in candidates:
        problem = validate(obj)
        if problem is None:
            return obj, None
        first_problem = first_problem or problem
    return None, first_problem


def extract_json(content: str, validate=None):
    """Parse the model's JSON, tolerating reasoning-model wrappers.

    Plain JSON (Groq/Gemini with response_format) is parsed by the first
    json.loads, unchanged. Otherwise: drop <think>...</think> blocks, then take
    the LAST complete top-level JSON object in the text (inside a ``` fence or
    not): a model that drafts an answer and then gives its final one means the
    final one, and an inner object of a truncated answer (e.g. one finding of a
    cut-off findings list) is never mistaken for the answer. Failing that, the
    last Markdown code fence whose contents parse (e.g. a fenced list).

    ``validate(obj) -> str | None`` (a problem description, or None when
    ``obj`` is an acceptable answer): the LAST candidate it accepts is
    returned, so a trailing example / note object after the answer, or a
    draft after it, doesn't replace it. Candidates are the whole reply (plain
    JSON), else the top-level objects, then the parsable fences, each group
    from last to first. JSON was found but no candidate passes ->
    ``InvalidAnswer``.

    Raises json.JSONDecodeError (from the plain parse) if nothing parses.
    """
    check = validate or (lambda _obj: None)
    try:
        whole = json.loads(content)
    except json.JSONDecodeError as exc:
        original = exc
    else:
        problem = check(whole)
        if problem is None:
            return whole
        raise InvalidAnswer(problem)
    text = _THINK_RE.sub("", content)
    try:
        whole = json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    else:
        problem = check(whole)
        if problem is None:
            return whole
        raise InvalidAnswer(problem)
    objects = _top_level_objects(text)
    found, problem = _first_valid(reversed(objects), check)
    if found is not None:
        return found
    fenced = []
    for candidate in reversed(_FENCE_RE.findall(text)):
        try:
            fenced.append(json.loads(candidate.strip()))
        except json.JSONDecodeError:
            pass
    found, fence_problem = _first_valid(fenced, check)
    if found is not None:
        return found
    if objects or fenced:
        raise InvalidAnswer(problem or fence_problem or "no acceptable JSON answer")
    raise original


def _is_json_object(content: str) -> bool:
    """The whole content (whitespace aside) is one JSON object."""
    try:
        return isinstance(json.loads(content.strip()), dict)
    except (json.JSONDecodeError, AttributeError):
        return False


def model_sampling(provider: str, model: str) -> dict:
    """settings.LLM_SAMPLING's entry for this model ({} if none): a
    "<provider>:<model>" key first, then the model id, then the id without
    OpenRouter's ":free" suffix."""
    table = settings.LLM_SAMPLING or {}
    base = model.removesuffix(":free")
    for key in (f"{provider}:{model}", f"{provider}:{base}", model, base):
        if key in table:
            return dict(table[key])
    return {}


def _provider_key(provider: str) -> str | None:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return getattr(settings, PROVIDERS[provider]["key_setting"])


class LLMClient:
    def __init__(self, provider: str, model: str | None = None,
                 temperature: float | None = None):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider}")
        cfg = PROVIDERS[provider]
        self.provider = provider
        if "url" in cfg:
            self.url = cfg["url"]
        else:
            base = getattr(settings, cfg["base_url_setting"]).rstrip("/")
            self.url = f"{base}/chat/completions"
        self.extra_headers = dict(cfg.get("extra_headers", {}))
        self._response_format_fallback = bool(cfg.get("response_format_fallback"))
        self.api_key = getattr(settings, cfg["key_setting"])
        # None -> the provider's configured default model (GROQ_MODEL / GEMINI_MODEL / ...).
        self.model = model or getattr(settings, cfg["model_setting"])
        # Off from the start for known non-supporting models; otherwise flipped off
        # (for this client's lifetime) once the provider rejects it.
        self.use_response_format = not (
            self._response_format_fallback and self.model in _NO_RESPONSE_FORMAT_MODELS
        )
        # Explicit temperature override (the eval sets it per client, e.g. 0.0
        # for reproducible runs). None = the model's LLM_SAMPLING entry, else
        # settings.LLM_TEMPERATURE (see sampling_params).
        self.temperature = temperature
        self._supports_top_k = bool(cfg.get("supports_top_k"))
        # Reported as provider_used and used in logs: says exactly which model answered.
        self.label = f"{provider}:{self.model}"
        if not _key_configured(self.api_key):
            raise RuntimeError(
                f"{cfg['key_setting']} is not set (or still a placeholder) but LLM "
                f"provider '{provider}' is configured"
            )

    def _rejects_response_format(self, resp: httpx.Response) -> bool:
        return (
            self._response_format_fallback
            and self.use_response_format
            and resp.status_code == 400
            and "response_format" in _error_message(resp)
        )

    def sampling_params(self) -> dict:
        """The sampling fields sent with each request.

        With ``self.temperature`` set (an explicit override): only that
        temperature, exactly as before per-model sampling existed. With None:
        the model's settings.LLM_SAMPLING entry (temperature, top_p, and top_k
        on providers that accept it), its temperature defaulting to
        settings.LLM_TEMPERATURE; a model without an entry sends just
        LLM_TEMPERATURE. Read at request time, so settings changes apply."""
        if self.temperature is not None:
            return {"temperature": float(self.temperature)}
        entry = model_sampling(self.provider, self.model)
        params = {"temperature": float(entry.get("temperature", settings.LLM_TEMPERATURE))}
        if entry.get("top_p") is not None:
            params["top_p"] = float(entry["top_p"])
        if entry.get("top_k") is not None and self._supports_top_k:
            params["top_k"] = int(entry["top_k"])
        return params

    @property
    def effective_temperature(self) -> float:
        """The temperature this client actually sends."""
        return self.sampling_params()["temperature"]

    async def _post(self, system: str, user: str, max_tokens: int | None = None
                    ) -> httpx.Response:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.sampling_params(),
        }
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        if self.use_response_format:
            body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as http:
            return await http.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}", **self.extra_headers},
                json=body,
            )

    def _raise_http_error(self, status: int, what: str, detail: str, log_detail: str,
                          gone_text: str | None = None):
        """Raise the non-transient failure of a request that failed with HTTP
        status ``status`` (not 429 / 5xx), or of an error object with that code
        in an HTTP 200 body: ``what`` names it ("HTTP 404", "error in HTTP 200
        body (code 404)"), ``detail`` goes in the error, ``log_detail`` in the
        402 log line, ``gone_text`` (default ``detail``) is searched for a
        model-gone code. 402 -> provider-wide LLMError; anything else -> a
        plain LLMError (the router's "failed" outcome, next client), with a
        WARNING when the model id itself is gone."""
        if status == 402:
            # Account-level (insufficient credits): retrying, or trying another
            # model on the same account, can't help.
            logger.warning(
                "LLM provider '{}' returned {} (insufficient credits) for "
                "model '{}'; not retrying, skipping this provider's remaining "
                "models: {}",
                self.provider, what, self.model, log_detail[:200],
            )
            raise LLMError(
                f"{self.label} {what} (insufficient credits): {detail[:200]}",
                provider_wide=True,
            )
        if _model_gone(status, detail if gone_text is None else gone_text):
            # Non-retriable, falls through to the next client like any other
            # 4xx — but a retired/renamed model id must be obvious in the logs.
            logger.warning(
                "LLM model '{}' on provider '{}' not found or decommissioned "
                "({}); update the model setting. Falling through to the next "
                "client.",
                self.model, self.provider, what,
            )
        raise LLMError(f"{self.label} {what}: {detail[:200]}")

    async def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return the raw assistant message content (expected to be JSON).
        ``max_tokens`` defaults to settings.LLM_MAX_OUTPUT_TOKENS."""
        if max_tokens is None:
            max_tokens = settings.LLM_MAX_OUTPUT_TOKENS
        resp = await self._post(system, user, max_tokens)
        if self._rejects_response_format(resp):
            # The system prompt already asks for JSON and the router still
            # validates it, so dropping response_format loses no safety.
            logger.warning(
                "LLM model '{}' on provider '{}' rejected response_format (HTTP 400); "
                "retrying once without it (and not sending it again from this client).",
                self.model, self.provider,
            )
            self.use_response_format = False
            resp = await self._post(system, user, max_tokens)
        if resp.status_code != 200:
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retriable(
                    f"{self.label} HTTP {resp.status_code}",
                    retry_after=_retry_after_seconds(resp),
                    status=resp.status_code,
                )
            self._raise_http_error(resp.status_code, f"HTTP {resp.status_code}",
                                   resp.text, _error_message(resp))
        data = resp.json()
        # OpenRouter can report an (often upstream) error inside an HTTP 200 body.
        # Its code is handled like an HTTP status of that code: 429 / 5xx are
        # transient (same-client retry), 402 is account-wide and other 4xx are
        # non-retriable errors (next client), exactly as the HTTP path above. An
        # error object without a usable code (missing, non-numeric, outside
        # 400-599) says nothing about retrying: bad output (next client, no
        # same-client retry). Never a crash.
        err = data.get("error") if isinstance(data, dict) else None
        if err:
            code = error_object_code(err)
            message = str(err.get("message", err) if isinstance(err, dict) else err)
            raw_code = err.get("code") if isinstance(err, dict) else None
            what = f"error in HTTP 200 body (code {raw_code})"
            if code is not None and (code == 429 or code >= 500):
                raise _Retriable(f"{self.label} {what}: {message[:200]}", status=code)
            if code is not None and 400 <= code < 500:
                self._raise_http_error(code, what, message, message, gone_text=resp.text)
            raise _Retriable(f"{self.label} {what}: {message[:200]}", transient=False)
        # Malformed 200s (null content, empty choices) are bad-output, not transient —
        # try the fallback rather than crashing the whole scan, but no point retrying
        # the same provider for the same bad shape.
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _Retriable(f"{self.label} malformed response: {exc}", transient=False) from exc
        if content is None:
            # Say why: a reasoning model that hit its output cap mid-thought
            # (finish_reason "length", reasoning but no content) looks different
            # from a provider that returned nothing.
            message = choice.get("message") or {}
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            usage = data.get("usage") or {}
            raise _Retriable(
                f"{self.label} returned null content (finish_reason="
                f"{choice.get('finish_reason')!r}, completion_tokens="
                f"{usage.get('completion_tokens')}, reasoning_chars={len(reasoning)})",
                transient=False,
            )
        if choice.get("finish_reason") == "length" and not _is_json_object(content):
            # Cut off at max_tokens: whatever JSON could be salvaged from it (a
            # draft, one finding of a cut-off list) is not the answer.
            usage = data.get("usage") or {}
            raise _Retriable(
                f"{self.label} output truncated (finish_reason='length', completion_tokens="
                f"{usage.get('completion_tokens')}, max_tokens={max_tokens}) without a "
                "complete JSON answer",
                transient=False,
            )
        return content


def estimate_tokens(text: str) -> int:
    """Same estimate as review_plan.estimate_tokens: chars / 4 x 1.25."""
    return math.ceil(len(text or "") / 4 * 1.25)


PACE_OK = "ok"
PACE_DEADLINE = "deadline"
PACE_TOO_LONG = "too_long"


class TokenPacer:
    """Per-model tokens-per-minute budget for LLM calls (ROADMAP: Groq's free
    tier allows ~8K tokens/min per model, and a scan makes up to
    LLM_MAX_CALLS_PER_SCAN ~6K-token calls back to back).

    Each call reserves its estimated tokens on its client's key (the
    "<provider>:<model>" label) in a 60 s sliding window; the limit is
    ``limits[label]``, else ``limits[provider]`` (None / missing = unlimited).
    Reservations are made before waiting and in FIFO order, so concurrent scans
    sharing the router can't both take the same room. ``block`` makes a key
    wait (Retry-After, retry backoff). ``clock`` / ``sleep`` are injectable
    (tests use a fake clock); ``sleep`` defaults to ``asyncio.sleep`` looked up
    at call time.
    """

    WINDOW_S = 60.0

    def __init__(self, limits: dict | None = None, clock=time.monotonic, sleep=None):
        self.limits = dict(limits or {})
        self.clock = clock
        self._sleep = sleep
        self._reserved: dict[str, list[tuple[float, int]]] = {}
        self._blocked_until: dict[str, float] = {}

    def limit_for(self, key: str, provider: str) -> int | None:
        limit = self.limits.get(key, self.limits.get(provider))
        return int(limit) if limit else None

    def block(self, key: str, seconds: float) -> None:
        """No call on ``key`` starts before ``seconds`` from now."""
        until = self.clock() + max(float(seconds), 0.0)
        self._blocked_until[key] = max(self._blocked_until.get(key, 0.0), until)

    def earliest_start(self, key: str, provider: str, tokens: int, now: float) -> float:
        reserved = self._reserved.setdefault(key, [])
        reserved[:] = [(t, n) for t, n in reserved if t > now - self.WINDOW_S]
        start = max(now, self._blocked_until.get(key, 0.0))
        limit = self.limit_for(key, provider)
        if not limit:
            return start
        if reserved:
            start = max(start, reserved[-1][0])  # FIFO
        while True:
            window = [(t, n) for t, n in reserved if t > start - self.WINDOW_S]
            if not window or sum(n for _, n in window) + tokens <= limit:
                return start
            start = window[0][0] + self.WINDOW_S  # when the oldest one leaves

    async def acquire(
        self, key: str, provider: str, tokens: int, *,
        deadline: float | None = None, max_wait: float | None = None,
    ) -> str:
        """Reserve ``tokens`` on ``key`` and wait until they may be used.
        Returns PACE_OK, or (without reserving or waiting) PACE_DEADLINE when
        the call couldn't start before ``deadline``, PACE_TOO_LONG when the
        wait would exceed ``max_wait``."""
        now = self.clock()
        start = self.earliest_start(key, provider, tokens, now)
        if deadline is not None and start > deadline:
            return PACE_DEADLINE
        if max_wait is not None and start - now > max_wait:
            return PACE_TOO_LONG
        if self.limit_for(key, provider):
            self._reserved[key].append((start, tokens))
        if start > now:
            await (self._sleep or asyncio.sleep)(start - now)
        return PACE_OK


class LLMRouter:
    def __init__(self, pacer: TokenPacer | None = None):
        self.mock = settings.LLM_PROVIDER == "mock"
        self.pacer = pacer or TokenPacer(settings.LLM_TPM_LIMITS)
        self.clients: list[LLMClient] = []
        if not self.mock:
            # Primary must be fully configured (fail fast). The fallback is
            # best-effort: skip it if its key is absent so a valid single-provider
            # setup still boots.
            self.clients.extend(self._provider_clients(settings.LLM_PROVIDER))
            fb = settings.LLM_FALLBACK_PROVIDER
            if fb and fb not in ("mock", settings.LLM_PROVIDER):
                if _key_configured(_provider_key(fb)):
                    self.clients.extend(self._provider_clients(fb))
                else:
                    logger.warning(
                        "Fallback provider '{}' has no key configured; "
                        "running primary-only.", fb
                    )

    @staticmethod
    def _provider_clients(provider: str) -> list[LLMClient]:
        """The provider's default model, then its same-provider fallback model.

        groq -> GROQ_FALLBACK_MODEL, openrouter -> OPENROUTER_FALLBACK_MODEL
        (skipped when unset or equal to the default model). Built for the
        primary and the fallback provider alike. The first LLMClient raises on a
        missing key; the second shares that key, so it can't fail differently.
        """
        first = LLMClient(provider)
        clients = [first]
        fb_setting = PROVIDERS[provider].get("fallback_model_setting")
        fb_model = getattr(settings, fb_setting) if fb_setting else None
        if fb_model and fb_model != first.model:
            clients.append(LLMClient(provider, model=fb_model))
        return clients

    def max_output_tokens(self, client: LLMClient, prompt_tokens: int) -> int | None:
        """max_tokens for one call: settings.LLM_MAX_OUTPUT_TOKENS, capped for a
        client with a tokens-per-minute limit (Groq counts max_tokens against
        it, so prompt + max_tokens over the limit is rejected outright) to what
        the estimated prompt leaves of the limit. 0 = the prompt leaves under
        MIN_OUTPUT_TOKENS: skip the client. None = no cap sent."""
        cap = settings.LLM_MAX_OUTPUT_TOKENS
        limit = self.pacer.limit_for(client.label, client.provider)
        if not limit:
            return cap
        room = limit - prompt_tokens
        if room < MIN_OUTPUT_TOKENS:
            return 0
        return min(cap, room) if cap else room

    @property
    def clock(self):
        """The pacer's monotonic clock (scan deadlines are measured on it)."""
        return self.pacer.clock

    async def generate(
        self, system: str, user: str, *, deadline: float | None = None, validate=None
    ) -> tuple[dict, str]:
        """Return (parsed_json, provider_used). Raises LLMError if all fail.

        provider_used is the answering client's "<provider>:<model>" label
        (e.g. "groq:qwen/qwen3.8-27b"), or "mock" in mock mode.

        ``validate(obj) -> str | None`` is the caller's format check (a problem
        description, or None for an acceptable answer), e.g. the PR review's
        audit / verifier schema checks. The answer is the last JSON object of
        the reply that passes it (``extract_json``); a reply with none is a
        bad-output failure of that client (WARNING with the problem), and the
        next client is tried. If every client fails, the LLMError carries how
        each client ended (``client_outcomes``) and has ``bad_output`` set when
        every client that was called ended with bad output (this, a cut-off /
        null / malformed reply, non-JSON) - a 429 / 5xx retried before that
        doesn't count - and none ended on a 429 / 5xx / timeout / HTTP error
        (``_all_bad_output``). Mock mode returns its canned answer unchecked.

        Every attempt first takes its estimated tokens (prompt + an output
        allowance) from the client's per-minute budget (``self.pacer``), waiting
        if needed; a wait longer than LLM_MAX_WAIT_S, or one that would end
        after ``deadline`` (a ``self.clock`` time), skips to the next client
        instead. Retry-After (and backoff) waits go through the same pacer, so
        a rate-limited model is also avoided by later calls. If every client is
        skipped or fails and at least one was skipped for ``deadline``, the
        LLMError has ``deadline_exceeded`` set.
        """
        if self.mock:
            return MOCK_RESPONSE, "mock"

        prompt_tokens = estimate_tokens(system) + estimate_tokens(user)
        tokens = prompt_tokens + settings.LLM_OUTPUT_TOKENS_ESTIMATE
        last_error: Exception | None = None
        deadline_hit = False
        # How each client ENDED, (label, outcome): "bad_output" (it answered,
        # unusably), "rate_limit" (its last attempt got a 429, or it was skipped
        # for its rate budget after one), "failed" (5xx / timeout / transport /
        # HTTP error / rate budget), or no entry for one never called (provider
        # dead after a 402, prompt too large for its tokens/min limit). Only
        # the final outcome counts: a retried 429 followed by bad output is
        # "bad_output". See ``_all_bad_output`` / ``LLMError.client_outcomes``.
        outcomes: list[tuple[str, str]] = []
        dead_providers: set[str] = set()  # account-level failure (HTTP 402)
        for client in self.clients:
            if client.provider in dead_providers:
                logger.warning(
                    "Skipping LLM client {}: provider '{}' failed account-wide.",
                    client.label, client.provider,
                )
                continue
            max_tokens = self.max_output_tokens(client, prompt_tokens)
            if max_tokens == 0:
                limit = self.pacer.limit_for(client.label, client.provider)
                logger.warning(
                    "Skipping LLM client {}: the prompt (~{} est. tokens) leaves under {} "
                    "output tokens of its {} tokens/min limit.",
                    client.label, prompt_tokens, MIN_OUTPUT_TOKENS, limit,
                )
                last_error = last_error or LLMError(
                    f"{client.label}: prompt too large for its tokens/min limit")
                continue
            # attempt 0 is the first try; attempts 1..LLM_RETRIES are same-client
            # retries of a transient failure. LLM_RETRIES=0 -> exactly today's
            # behavior (one try, immediate fallthrough on any failure).
            rate_limited = False  # this client's latest attempt got a 429
            for attempt in range(settings.LLM_RETRIES + 1):
                slot = await self.pacer.acquire(
                    client.label, client.provider, tokens,
                    deadline=deadline, max_wait=settings.LLM_MAX_WAIT_S,
                )
                if slot != PACE_OK:
                    deadline_hit = deadline_hit or slot == PACE_DEADLINE
                    why = ("would end after the scan's LLM time budget"
                           if slot == PACE_DEADLINE else
                           f"needs a wait over LLM_MAX_WAIT_S={settings.LLM_MAX_WAIT_S:g}s")
                    logger.warning("Skipping LLM client {}: rate budget {}.", client.label, why)
                    last_error = last_error or LLMError(f"{client.label}: rate budget {why}")
                    # Its retry after a 429 can't be waited for: it ended rate-limited.
                    outcomes.append((client.label, "rate_limit" if rate_limited else "failed"))
                    break
                try:
                    content = await client.complete(system, user, max_tokens=max_tokens)
                    data = extract_json(content, validate=_object_check(validate))
                    return data, client.label
                except InvalidAnswer as exc:
                    # Parsed, but not an answer the caller can use: no same-client
                    # retry (the same prompt tends to give the same shape), next client.
                    logger.warning("LLM client {} answer failed the format check: {}",
                                   client.label, exc.problem)
                    last_error = LLMError(f"{client.label} returned an unusable answer: "
                                          f"{exc.problem}")
                    outcomes.append((client.label, "bad_output"))
                    break
                except (_Retriable, httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    transient = getattr(exc, "transient", True)
                    retry_after = getattr(exc, "retry_after", None)
                    rate_limited = getattr(exc, "status", None) == 429
                    if retry_after is not None:
                        # Honoured for every later call to this model too.
                        self.pacer.block(client.label, retry_after)
                    if transient and attempt < settings.LLM_RETRIES:
                        if retry_after is None:
                            self.pacer.block(client.label,
                                             min(2 * 2**attempt + random.uniform(0, 0.5), 10.0))
                        logger.warning(
                            "LLM client {} attempt {} failed (retriable), retrying: {}",
                            client.label, attempt + 1, exc,
                        )
                        continue  # the next acquire() waits out the block
                    logger.warning("LLM client {} failed (retriable): {}", client.label, exc)
                    # A non-transient _Retriable is a bad answer in an HTTP 200
                    # (cut off at max_tokens, null content, malformed body, an
                    # error object without a usable code); a transient one is
                    # a 429 (rate_limit) or a 5xx / timeout / transport failure.
                    # This is the client's final outcome, whatever came before.
                    outcomes.append((client.label,
                                     "bad_output" if not transient
                                     else "rate_limit" if rate_limited else "failed"))
                    break
                except json.JSONDecodeError as exc:
                    logger.warning("LLM client {} returned non-JSON: {}", client.label, exc)
                    last_error = exc
                    outcomes.append((client.label, "bad_output"))
                    break
                except LLMError as exc:
                    # Non-retriable (e.g. 4xx auth, 404 retired model, also as an
                    # error code in an HTTP 200 body) — no same-client retry, but
                    # still try the next client in the chain.
                    logger.warning("LLM client {} error: {}", client.label, exc)
                    last_error = exc
                    outcomes.append((client.label, "failed"))
                    if exc.provider_wide:
                        dead_providers.add(client.provider)
                    break
        raise LLMError(
            f"All LLM clients failed ({', '.join(c.label for c in self.clients)}): {last_error}",
            deadline_exceeded=deadline_hit, bad_output=_all_bad_output(outcomes),
            client_outcomes=outcomes,
        )


def _all_bad_output(outcomes: list[tuple[str, str]]) -> bool:
    """``LLMError.bad_output`` of a call where every client failed, from each
    client's FINAL outcome ((label, outcome) pairs): at least one client
    answered, and every client that was called (or skipped for its rate
    budget) ended by answering unusably - none ended on a 429 / 5xx / timeout
    / transport or HTTP error, which a later retry might not hit. A 429 / 5xx
    that was retried on the same client and then answered unusably counts as
    bad output (the eval's ``classify_llm_error`` follows the same rule).
    Clients never called (provider dead after a 402, prompt too large for the
    client's tokens/min limit) count neither way."""
    return bool(outcomes) and all(o == "bad_output" for _, o in outcomes)


def _object_check(validate):
    """``validate`` behind the router's own requirement: a JSON object."""
    def check(obj):
        if not isinstance(obj, dict):
            return "JSON that is not an object"
        return validate(obj) if validate is not None else None
    return check


def verifier_router_for(router):
    """The router for the PR review's verifier calls.

    With ``VERIFIER_MODEL`` ("<provider>:<model>") set, a router that tries
    that model first and then ``router``'s own chain, sharing its pacer (one
    token budget per model across both). Otherwise, for a mock router, or for
    anything that isn't an ``LLMRouter`` (test stubs), ``router`` itself. A
    malformed spec or a provider without a key logs a warning and falls back
    to ``router``. Cached on ``router``."""
    spec = (settings.VERIFIER_MODEL or "").strip()
    if not spec or not isinstance(router, LLMRouter) or router.mock:
        return router
    cached = getattr(router, "_verifier_router", None)
    if cached is not None and cached[0] == spec:
        return cached[1]
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        logger.warning("VERIFIER_MODEL={!r} is not '<provider>:<model>'; verifying with the "
                       "normal chain.", spec)
        return router
    if not _key_configured(_provider_key(provider)):
        logger.warning("VERIFIER_MODEL provider '{}' has no key configured; verifying with the "
                       "normal chain.", provider)
        return router
    client = LLMClient(provider, model=model)
    derived = LLMRouter.__new__(LLMRouter)
    derived.mock = False
    derived.pacer = router.pacer
    derived.clients = [client] + [c for c in router.clients if c.label != client.label]
    router._verifier_router = (spec, derived)
    return derived
