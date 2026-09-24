"""LLM provider abstraction.

Groq, Gemini and OpenRouter all expose OpenAI-compatible chat-completions
endpoints, so one thin async client handles them. Each client is one
(provider, model) pair. LLMRouter's chain is: primary provider/model -> the
primary provider's same-provider fallback model (groq -> GROQ_FALLBACK_MODEL,
openrouter -> OPENROUTER_FALLBACK_MODEL) -> LLM_FALLBACK_PROVIDER. Groq and
OpenRouter's free models rate-limit per model, so the second model is a genuine
fallback, not a redundant one.

A transient failure (429 / 5xx / timeout) is retried on the SAME client up to
settings.LLM_RETRIES times with backoff, then falls through to the next client; a
bad-output failure (malformed/empty response, unparseable JSON) or a non-retriable
HTTP error (e.g. 404 for a retired model, 401) skips straight to the next client.
HTTP 402 (insufficient credits) is account-wide: no retry, and the provider's
remaining models are skipped too.

OpenRouter quirks handled here: an error object inside an HTTP 200 body
(`{"error": {...}}`), and free models that reject `response_format` (HTTP 400 ->
one retry without it, remembered for the client's lifetime). All providers'
content goes through a tolerant JSON extractor (```json fences, <think> blocks
or reasoning text before the object); plain JSON parses exactly as before.

`provider_used` is "<provider>:<model>" (e.g. "groq:openai/gpt-oss-120b"), or "mock".

Mock mode is ONLY entered via LLM_PROVIDER=mock — a configured real provider with
a missing key raises at construction (fail-fast at startup), never a silent mock.
"""
import asyncio
import json
import random
import re

import httpx
from loguru import logger

from backend.app.config import settings

# Per provider: a fixed "url", or a "base_url_setting" (+ "/chat/completions")
# resolved when the client is built; "fallback_model_setting" names the
# same-provider fallback model; "extra_headers" are sent on every request;
# "response_format_fallback" enables the retry-without-response_format path.
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
    },
}

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

    def __init__(self, message: str, provider_wide: bool = False):
        super().__init__(message)
        self.provider_wide = provider_wide


class _Retriable(Exception):
    """Internal marker: this provider failed but the next one should be tried.

    `transient` distinguishes a same-provider-retriable failure (429/5xx from the
    HTTP layer) from a bad-output failure (malformed shape / null content on a 200)
    which goes straight to the next provider. `retry_after`, when the provider sent
    one (429/5xx only), overrides the exponential backoff for this attempt.
    """

    def __init__(self, message: str, transient: bool = True, retry_after: float | None = None):
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after


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


def _model_gone(resp: httpx.Response) -> bool:
    return resp.status_code == 404 or any(code in resp.text for code in _MODEL_GONE_CODES)


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


def extract_json(content: str):
    """Parse the model's JSON, tolerating reasoning-model wrappers.

    Plain JSON (Groq/Gemini with response_format) is parsed by the first
    json.loads, unchanged. Otherwise: drop <think>...</think> blocks, try the
    contents of each Markdown code fence (```json ... ```), then the first
    top-level JSON object in the text (e.g. after leading reasoning prose).
    Raises json.JSONDecodeError (from the plain parse) if nothing parses.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        original = exc
    text = _THINK_RE.sub("", content)
    for candidate in [text, *_FENCE_RE.findall(text)]:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise original


def _provider_key(provider: str) -> str | None:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return getattr(settings, PROVIDERS[provider]["key_setting"])


class LLMClient:
    def __init__(self, provider: str, model: str | None = None):
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

    async def _post(self, system: str, user: str) -> httpx.Response:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if self.use_response_format:
            body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as http:
            return await http.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}", **self.extra_headers},
                json=body,
            )

    async def complete(self, system: str, user: str) -> str:
        """Return the raw assistant message content (expected to be JSON)."""
        resp = await self._post(system, user)
        if self._rejects_response_format(resp):
            # The system prompt already asks for JSON and the router still
            # validates it, so dropping response_format loses no safety.
            logger.warning(
                "LLM model '{}' on provider '{}' rejected response_format (HTTP 400); "
                "retrying once without it (and not sending it again from this client).",
                self.model, self.provider,
            )
            self.use_response_format = False
            resp = await self._post(system, user)
        if resp.status_code != 200:
            if resp.status_code == 402:
                # Account-level (insufficient credits): retrying, or trying another
                # model on the same account, can't help.
                logger.warning(
                    "LLM provider '{}' returned HTTP 402 (insufficient credits) for "
                    "model '{}'; not retrying, skipping this provider's remaining "
                    "models: {}",
                    self.provider, self.model, _error_message(resp)[:200],
                )
                raise LLMError(
                    f"{self.label} HTTP 402 (insufficient credits): {resp.text[:200]}",
                    provider_wide=True,
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retriable(
                    f"{self.label} HTTP {resp.status_code}",
                    retry_after=_retry_after_seconds(resp),
                )
            if _model_gone(resp):
                # Non-retriable, falls through to the next client like any other
                # 4xx — but a retired/renamed model id must be obvious in the logs.
                logger.warning(
                    "LLM model '{}' on provider '{}' not found or decommissioned "
                    "(HTTP {}); update the model setting. Falling through to the next "
                    "client.",
                    self.model, self.provider, resp.status_code,
                )
            raise LLMError(f"{self.label} HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        # OpenRouter can report an (often upstream) error inside an HTTP 200 body.
        # Upstream 429/5xx codes are transient (same-client retry); anything else is
        # bad-output (straight to the next client). Never a crash.
        err = data.get("error") if isinstance(data, dict) else None
        if err:
            code = err.get("code") if isinstance(err, dict) else None
            message = err.get("message", err) if isinstance(err, dict) else err
            raise _Retriable(
                f"{self.label} error in HTTP 200 body (code {code}): {str(message)[:200]}",
                transient=isinstance(code, int) and (code == 429 or code >= 500),
            )
        # Malformed 200s (null content, empty choices) are bad-output, not transient —
        # try the fallback rather than crashing the whole scan, but no point retrying
        # the same provider for the same bad shape.
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _Retriable(f"{self.label} malformed response: {exc}", transient=False) from exc
        if content is None:
            raise _Retriable(f"{self.label} returned null content", transient=False)
        return content


class LLMRouter:
    def __init__(self):
        self.mock = settings.LLM_PROVIDER == "mock"
        self.clients: list[LLMClient] = []
        if not self.mock:
            # Primary must be fully configured (fail fast). The fallback is
            # best-effort: skip it if its key is absent so a valid single-provider
            # setup still boots.
            primary = LLMClient(settings.LLM_PROVIDER)
            self.clients.append(primary)
            # Same-provider model fallback (groq -> GROQ_FALLBACK_MODEL, openrouter
            # -> OPENROUTER_FALLBACK_MODEL). Shares the primary's key, which was
            # just validated, so it can't fail the fail-fast check.
            fb_setting = PROVIDERS[primary.provider].get("fallback_model_setting")
            fb_model = getattr(settings, fb_setting) if fb_setting else None
            if fb_model and fb_model != primary.model:
                self.clients.append(LLMClient(primary.provider, model=fb_model))
            fb = settings.LLM_FALLBACK_PROVIDER
            if fb and fb not in ("mock", settings.LLM_PROVIDER):
                if _key_configured(_provider_key(fb)):
                    self.clients.append(LLMClient(fb))
                else:
                    logger.warning(
                        "Fallback provider '{}' has no key configured; "
                        "running primary-only.", fb
                    )

    async def generate(self, system: str, user: str) -> tuple[dict, str]:
        """Return (parsed_json, provider_used). Raises LLMError if all fail.

        provider_used is the answering client's "<provider>:<model>" label
        (e.g. "groq:qwen/qwen3.8-27b"), or "mock" in mock mode.
        """
        if self.mock:
            return MOCK_RESPONSE, "mock"

        last_error: Exception | None = None
        dead_providers: set[str] = set()  # account-level failure (HTTP 402)
        for client in self.clients:
            if client.provider in dead_providers:
                logger.warning(
                    "Skipping LLM client {}: provider '{}' failed account-wide.",
                    client.label, client.provider,
                )
                continue
            # attempt 0 is the first try; attempts 1..LLM_RETRIES are same-client
            # retries of a transient failure. LLM_RETRIES=0 -> exactly today's
            # behavior (one try, immediate fallthrough on any failure).
            for attempt in range(settings.LLM_RETRIES + 1):
                try:
                    content = await client.complete(system, user)
                    return extract_json(content), client.label
                except (_Retriable, httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    transient = getattr(exc, "transient", True)
                    if transient and attempt < settings.LLM_RETRIES:
                        retry_after = getattr(exc, "retry_after", None)
                        backoff = 2 * 2**attempt + random.uniform(0, 0.5)
                        wait = min(retry_after if retry_after is not None else backoff, 10.0)
                        logger.warning(
                            "LLM client {} attempt {} failed (retriable), retrying in "
                            "{:.1f}s: {}",
                            client.label, attempt + 1, wait, exc,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.warning("LLM client {} failed (retriable): {}", client.label, exc)
                    break
                except json.JSONDecodeError as exc:
                    logger.warning("LLM client {} returned non-JSON: {}", client.label, exc)
                    last_error = exc
                    break
                except LLMError as exc:
                    # Non-retriable (e.g. 4xx auth, 404 retired model) — no same-client
                    # retry, but still try the next client in the chain.
                    logger.warning("LLM client {} error: {}", client.label, exc)
                    last_error = exc
                    if exc.provider_wide:
                        dead_providers.add(client.provider)
                    break
        raise LLMError(
            f"All LLM clients failed ({', '.join(c.label for c in self.clients)}): {last_error}"
        )
