"""LLM provider abstraction.

Groq and Gemini both expose OpenAI-compatible chat-completions endpoints, so one
thin client handles both. LLMRouter tries the primary provider, then the fallback,
on retriable failures (429 / 5xx / timeout / unparseable JSON).

Mock mode is ONLY entered via LLM_PROVIDER=mock — a configured real provider with
a missing key raises at construction (fail-fast at startup), never a silent mock.
"""
import json

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
    """Internal marker: this provider failed but the next one should be tried."""


def _key_configured(api_key: str | None) -> bool:
    """A key counts as configured only if it's set and not a placeholder.

    .env.example ships values like `your_groq_api_key_here`; treating those as
    real keys would defeat the fail-fast-at-startup invariant.
    """
    return bool(api_key) and not api_key.endswith("_here")


def _provider_key(provider: str) -> str | None:
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

    def complete(self, system: str, user: str) -> str:
        """Return the raw assistant message content (expected to be JSON)."""
        resp = httpx.post(
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
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retriable(f"{self.provider} HTTP {resp.status_code}")
            raise LLMError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:200]}")
        # Malformed 200s (null content, empty choices) are retriable — try the
        # fallback rather than crashing the whole scan.
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _Retriable(f"{self.provider} malformed response: {exc}") from exc
        if content is None:
            raise _Retriable(f"{self.provider} returned null content")
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

    def generate(self, system: str, user: str) -> tuple[dict, str]:
        """Return (parsed_json, provider_used). Raises LLMError if all fail."""
        if self.mock:
            return MOCK_RESPONSE, "mock"

        last_error: Exception | None = None
        for client in self.clients:
            try:
                content = client.complete(system, user)
                return json.loads(content), client.provider
            except (_Retriable, httpx.TimeoutException, httpx.TransportError) as exc:
                logger.warning("LLM provider {} failed (retriable): {}", client.provider, exc)
                last_error = exc
            except json.JSONDecodeError as exc:
                logger.warning("LLM provider {} returned non-JSON: {}", client.provider, exc)
                last_error = exc
            except LLMError as exc:
                # Non-retriable (e.g. 4xx auth) — still try the fallback provider.
                logger.warning("LLM provider {} error: {}", client.provider, exc)
                last_error = exc
        raise LLMError(f"All LLM providers failed: {last_error}")
