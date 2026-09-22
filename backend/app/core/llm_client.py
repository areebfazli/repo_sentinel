"""LLM provider abstraction.

Groq and Gemini both expose OpenAI-compatible chat-completions endpoints, so one
thin client handles both, both async. LLMRouter retries a transient failure
(429 / 5xx / timeout) on the SAME provider up to settings.LLM_RETRIES times with
backoff, then falls through to the fallback provider; a bad-output failure
(malformed/empty response, unparseable JSON) skips straight to the fallback.

Mock mode is ONLY entered via LLM_PROVIDER=mock — a configured real provider with
a missing key raises at construction (fail-fast at startup), never a silent mock.
"""
import asyncio
import json
import random

import httpx
from loguru import logger

from backend.app.config import settings

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_setting": "GROQ_API_KEY",
        "model_setting": "GROQ_MODEL",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_setting": "GEMINI_API_KEY",
        "model_setting": "GEMINI_MODEL",
    },
}

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
    """Raised when all configured providers fail (or on a non-retriable error)."""


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


def _provider_key(provider: str) -> str | None:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return getattr(settings, PROVIDERS[provider]["key_setting"])


class LLMClient:
    def __init__(self, provider: str):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider}")
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.url = cfg["url"]
        self.api_key = getattr(settings, cfg["key_setting"])
        self.model = getattr(settings, cfg["model_setting"])
        if not _key_configured(self.api_key):
            raise RuntimeError(
                f"{cfg['key_setting']} is not set (or still a placeholder) but LLM "
                f"provider '{provider}' is configured"
            )

    async def complete(self, system: str, user: str) -> str:
        """Return the raw assistant message content (expected to be JSON)."""
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as http:
            resp = await http.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
        if resp.status_code != 200:
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retriable(
                    f"{self.provider} HTTP {resp.status_code}",
                    retry_after=_retry_after_seconds(resp),
                )
            raise LLMError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:200]}")
        # Malformed 200s (null content, empty choices) are bad-output, not transient —
        # try the fallback rather than crashing the whole scan, but no point retrying
        # the same provider for the same bad shape.
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _Retriable(f"{self.provider} malformed response: {exc}", transient=False) from exc
        if content is None:
            raise _Retriable(f"{self.provider} returned null content", transient=False)
        return content


class LLMRouter:
    def __init__(self):
        self.mock = settings.LLM_PROVIDER == "mock"
        self.clients: list[LLMClient] = []
        if not self.mock:
            # Primary must be fully configured (fail fast). The fallback is
            # best-effort: skip it if its key is absent so a valid single-provider
            # setup still boots.
            self.clients.append(LLMClient(settings.LLM_PROVIDER))
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
        """Return (parsed_json, provider_used). Raises LLMError if all fail."""
        if self.mock:
            return MOCK_RESPONSE, "mock"

        last_error: Exception | None = None
        for client in self.clients:
            # attempt 0 is the first try; attempts 1..LLM_RETRIES are same-provider
            # retries of a transient failure. LLM_RETRIES=0 -> exactly today's
            # behavior (one try, immediate fallthrough on any failure).
            for attempt in range(settings.LLM_RETRIES + 1):
                try:
                    content = await client.complete(system, user)
                    return json.loads(content), client.provider
                except (_Retriable, httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    transient = getattr(exc, "transient", True)
                    if transient and attempt < settings.LLM_RETRIES:
                        retry_after = getattr(exc, "retry_after", None)
                        backoff = 2 * 2**attempt + random.uniform(0, 0.5)
                        wait = min(retry_after if retry_after is not None else backoff, 10.0)
                        logger.warning(
                            "LLM provider {} attempt {} failed (retriable), retrying in "
                            "{:.1f}s: {}",
                            client.provider, attempt + 1, wait, exc,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.warning("LLM provider {} failed (retriable): {}", client.provider, exc)
                    break
                except json.JSONDecodeError as exc:
                    logger.warning("LLM provider {} returned non-JSON: {}", client.provider, exc)
                    last_error = exc
                    break
                except LLMError as exc:
                    # Non-retriable (e.g. 4xx auth) — still try the fallback provider.
                    logger.warning("LLM provider {} error: {}", client.provider, exc)
                    last_error = exc
                    break
        raise LLMError(f"All LLM providers failed: {last_error}")
