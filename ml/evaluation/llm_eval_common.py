"""LLM-eval plumbing shared by ``run_eval`` (function-level) and
``run_pr_eval`` (PR-level): provider routers (pinned model, primary-only,
temperature), the per-call HTTP usage tap (real token counts, daily-limit
detection, OpenRouter upstream pinning), the JSONL result cache, deterministic
nonces, and the statistics (Wilson / Clopper-Pearson intervals, exact McNemar,
precision at a base rate).

Deliberately light: no embedder / reranker / vector-store imports (``run_eval``
pulls in torch; this module does not), so the PR eval's offline modes stay small.
"""
import hashlib
import json
import math
from pathlib import Path

import httpx

from backend.app.config import settings
from backend.app.core.llm_client import PROVIDERS, LLMClient, LLMRouter, TokenPacer

# Precision is reported at these vulnerable base rates.
BASE_RATES = (0.01, 0.02, 0.05)
# Guard against tuning on the held-out test set.
TEST_SPLIT_FLAG = "--i-know-this-is-the-test-set"

DEFAULT_LLM_SLEEP = 2.5
DEFAULT_LLM_TPM = 8000  # Groq free tier tokens/minute per model
DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS = 3
# Groq names the exhausted window in its 429 body ("... tokens per day (TPD)");
# OpenRouter's free tier says "Rate limit exceeded: free-models-per-day" (its
# per-minute throttle is "free-models-per-min", which matches none of these).
DAILY_LIMIT_MARKERS = ("per day", "(TPD)", "(RPD)", "free-models-per-day")
# A Retry-After this long only happens once a daily (not per-minute) window is spent.
DAILY_LIMIT_RETRY_AFTER_S = 300.0
# The PR eval's default: None = no temperature override, every call uses the
# model's recommended sampling (settings.LLM_SAMPLING via
# LLMClient.sampling_params, e.g. qwen/qwen3.8-27b: temperature 1.0, top_p 0.95,
# top_k 20; LLM_TEMPERATURE for a model without an entry), what production
# sends. A float (--llm-temperature T) forces that temperature.
DEFAULT_EVAL_TEMPERATURE = None
# The function-level eval (run_eval) keeps greedy decoding by default.
GREEDY_EVAL_TEMPERATURE = 0.0
# Every LLM cache entry written before the temperature was recorded used this.
LEGACY_CACHE_TEMPERATURE = 0.2


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for k successes out of n (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _rate(preds: list[bool]) -> dict:
    k, n = sum(preds), len(preds)
    return {
        "k": k, "n": n,
        "rate": round(k / n, 4) if n else None,
        "ci95": wilson_interval(k, n),
    }


def precision_at_base_rate(tpr: float | None, fpr: float | None, pi: float) -> float | None:
    """Expected precision when a fraction ``pi`` of scanned functions is
    vulnerable: TPR*pi / (TPR*pi + FPR*(1-pi)). None if undefined."""
    if tpr is None or fpr is None:
        return None
    denom = tpr * pi + fpr * (1 - pi)
    return round(tpr * pi / denom, 4) if denom else None


def precision_at_observed_fpr(tpr: float | None, fp_k: int, fp_n: int,
                              pi: float) -> float | None:
    """``precision_at_base_rate`` at the observed FPR ``fp_k / fp_n``, but None
    when no false positive was observed (``fp_k == 0``): an FPR of exactly 0
    gives a "precision" of 1.0 that only says the sample was too small to see
    one; report the bound from ``fpr_upper95_exact`` instead."""
    if not fp_n or not fp_k:
        return None
    return precision_at_base_rate(tpr, fp_k / fp_n, pi)


def fpr_upper95_exact(k: int, n: int) -> float | None:
    """Upper end of the exact (Clopper-Pearson) 95% interval of k / n."""
    ci = clopper_pearson(k, n)
    return ci[1] if ci else None


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided exact Fisher p-value of the table [[a, b], [c, d]]: the summed
    probability (hypergeometric, fixed margins) of every table no more likely
    than the observed one."""
    r1, c1, n = a + b, a + c, a + b + c + d
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    observed = prob(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(p for x in range(lo, hi + 1)
                        if (p := prob(x)) <= observed * (1 + 1e-9)))


def _fmt_rate(r: dict) -> str:
    if r["rate"] is None:
        return "n/a (n=0)"
    lo, hi = r["ci95"]
    return f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}] ({r['k']}/{r['n']})"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts ``b`` (only A
    positive) and ``c`` (only B positive): 2 * P(X <= min(b, c)), X ~
    Binomial(b + c, 1/2), capped at 1. 1.0 when there is no discordant pair."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p), summed in log space (no overflow)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq, lg = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    return min(1.0, sum(
        math.exp(lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
        for i in range(k + 1)
    ))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Exact (Clopper-Pearson) 1 - ``alpha`` interval for k of n (None if n == 0),
    by bisection on the binomial CDF."""
    if n == 0:
        return None

    def solve(f) -> float:  # f increasing in p, f(0) < 0 < f(1)
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return [round(lower, 4), round(upper, 4)]


def _exact_rate(preds: list[bool]) -> dict:
    k, n = sum(preds), len(preds)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci95_exact": clopper_pearson(k, n)}


def _fmt_exact(r: dict) -> str:
    if r["rate"] is None:
        return "n/a (n=0)"
    lo, hi = r["ci95_exact"]
    return f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}] ({r['k']}/{r['n']})"


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) — used for budgeting only when the
    provider response carries no ``usage``."""
    return math.ceil(len(text) / 4)


def eval_nonce(item_id: str) -> str:
    """Deterministic stand-in for untrusted.new_nonce(), so the prompt (and its
    cache key) is reproducible across runs."""
    return hashlib.sha256(f"eval-nonce:{item_id}".encode()).hexdigest()[:16]


def _cache_key(item_id: str, sha: str, model: str, temperature, repeat) -> tuple:
    """``temperature`` is a number (only a temperature was sent) or a
    ``sampling_key`` string (temperature plus other sampling fields)."""
    temp = temperature if isinstance(temperature, str) else round(float(temperature), 4)
    return (item_id, sha, model, temp, int(repeat))


class LLMCache:
    """Append-only JSONL of successful LLM results keyed by (item id, prompt
    sha256, model, temperature, repeat index); the temperature part is the
    ``sampling_key`` of what was sent (a plain number when only a temperature
    was, so pre-existing entries keep their keys). Entries written before the
    temperature / repeat were recorded count as LEGACY_CACHE_TEMPERATURE /
    repeat 0 (what they were). Errors are never cached, so a re-run retries
    them."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[tuple, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._entries[self._key(rec)] = rec
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue  # a torn last line from an interrupted run

    @staticmethod
    def _key(rec: dict) -> tuple:
        return _cache_key(rec["id"], rec["prompt_sha256"], rec["model"],
                          rec.get("temperature", LEGACY_CACHE_TEMPERATURE), rec.get("repeat", 0))

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, item_id: str, sha: str, model: str,
            temperature: float | str = LEGACY_CACHE_TEMPERATURE, repeat: int = 0) -> dict | None:
        return self._entries.get(_cache_key(item_id, sha, model, temperature, repeat))

    def put(self, rec: dict) -> None:
        rec = {"temperature": LEGACY_CACHE_TEMPERATURE, "repeat": 0, **rec}
        self._entries[self._key(rec)] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


def _http_event(resp: httpx.Response) -> dict:
    event = {"status": resp.status_code, "usage": None, "retry_after": None, "daily_limit": False}
    if resp.status_code == 200:
        try:
            data = resp.json()
            usage = data.get("usage")
            event["usage"] = usage if isinstance(usage, dict) else None
            # OpenRouter names the upstream that served the request (and the
            # exact model); logged so a silent upstream change is visible.
            if isinstance(data.get("provider"), str):
                event["upstream_provider"] = data["provider"]
            if isinstance(data.get("model"), str):
                event["response_model"] = data["model"]
            # OpenRouter can put an error (incl. an upstream 429) in a 200 body.
            err = data.get("error")
            if isinstance(err, dict):
                event["error_code"] = err.get("code")
                if err.get("code") == 429:
                    message = str(err.get("message", ""))
                    event["daily_limit"] = any(m in message for m in DAILY_LIMIT_MARKERS)
        except (ValueError, AttributeError):
            pass
    else:
        try:
            event["retry_after"] = float(resp.headers.get("Retry-After"))
        except (TypeError, ValueError):
            pass
        if resp.status_code == 429:
            body = resp.text
            event["daily_limit"] = any(m in body for m in DAILY_LIMIT_MARKERS) or (
                event["retry_after"] is not None
                and event["retry_after"] >= DAILY_LIMIT_RETRY_AFTER_S
            )
    return event


class HttpUsageTap:
    """While active, records every httpx.AsyncClient.post response's token
    ``usage`` and rate-limit signals. LLMClient returns only the message text and
    folds a 429 into "HTTP 429", so this is how the eval sees real token counts
    (including reasoning tokens) and tells a daily limit from a per-minute one.
    Only the LLM stage runs inside it (no other httpx traffic in the eval).

    ``inject`` fields are merged into the JSON body of every request whose URL
    starts with ``inject_url_prefix`` (the eval pins OpenRouter's upstream
    routing this way: ``{"provider": {"allow_fallbacks": false}}``) without
    changing the production client."""

    def __init__(self, inject: dict | None = None, inject_url_prefix: str | None = None):
        self.events: list[dict] = []
        self._orig = None
        self.inject = dict(inject or {})
        self.inject_url_prefix = inject_url_prefix

    def __enter__(self):
        self._orig = orig = httpx.AsyncClient.post
        tap = self

        async def post(client, *args, **kwargs):
            if tap.inject and tap.inject_url_prefix:
                url = str(args[0] if args else kwargs.get("url", ""))
                body = kwargs.get("json")
                if url.startswith(tap.inject_url_prefix) and isinstance(body, dict):
                    kwargs["json"] = {**body, **tap.inject}
            resp = await orig(client, *args, **kwargs)
            tap.events.append(_http_event(resp))
            return resp

        httpx.AsyncClient.post = post
        return self

    def __exit__(self, *exc):
        httpx.AsyncClient.post = self._orig
        return False

    def take(self) -> list[dict]:
        events, self.events = self.events, []
        return events


def _usage_totals(events: list[dict]) -> dict | None:
    usages = [e["usage"] for e in events if e.get("usage")]
    if not usages:
        return None
    return {
        k: sum(int(u.get(k) or 0) for u in usages)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def classify_llm_error(exc: BaseException, events: list[dict]) -> str:
    """'daily_limit' | 'rate_limit' | 'bad_output' | 'other' for a failed
    router.generate ('bad_output': ``LLMError.bad_output``, every model called
    answered unusably - failed the caller's format check, not JSON, cut off,
    null content or a malformed 200; a 429 among the failures is
    'rate_limit' first)."""
    text = str(exc)
    if any(e.get("daily_limit") for e in events) or any(m in text for m in DAILY_LIMIT_MARKERS):
        return "daily_limit"
    if (
        any(e.get("status") == 429 or e.get("error_code") == 429 for e in events)
        or "HTTP 429" in text
    ):
        return "rate_limit"
    if getattr(exc, "bad_output", False):
        return "bad_output"
    return "other"


def llm_model_key(router) -> str:
    """Cache/model key: the router's primary "<provider>:<model>" (or "mock")."""
    return "mock" if getattr(router, "mock", False) else router.clients[0].label


def client_sampling(client) -> dict:
    """The sampling fields ``client`` sends with each request: its
    ``sampling_params()`` (an explicit temperature override, else the model's
    settings.LLM_SAMPLING entry), or, for a client without that method, its
    ``temperature`` (settings.LLM_TEMPERATURE when unset)."""
    params = getattr(client, "sampling_params", None)
    if callable(params):
        return dict(params())
    value = getattr(client, "temperature", None)
    return {"temperature": float(settings.LLM_TEMPERATURE if value is None else value)}


def sampling_key(sampling: dict) -> float | str:
    """Cache-key form of the sampling sent: the temperature (rounded) when it is
    the only field, so explicit-temperature runs keep their pre-existing keys
    (e.g. 0.0); otherwise ``"sampling:" + canonical JSON`` (e.g. the model's
    recommended temperature 1.0 + top_p + top_k), so a temperature-0 cache entry
    is never replayed for model-default sampling, nor the reverse."""
    if set(sampling) <= {"temperature"}:
        return round(float(sampling.get("temperature", settings.LLM_TEMPERATURE)), 4)
    return "sampling:" + json.dumps({k: sampling[k] for k in sorted(sampling)},
                                    separators=(",", ":"))


def router_temperature(router) -> float | str:
    """``sampling_key`` of what the router's primary client sends
    (settings.LLM_TEMPERATURE for a router without clients)."""
    clients = getattr(router, "clients", None) or []
    if not clients:
        return float(settings.LLM_TEMPERATURE)
    return sampling_key(client_sampling(clients[0]))


def build_llm_router(primary_only: bool = False) -> LLMRouter:
    """The production router (fails fast on a missing key), optionally trimmed
    to its primary client."""
    # No per-minute token pacing in the router: the eval paces its own calls
    # (--llm-sleep / --llm-tpm), and pacing twice would double every wait.
    # Retry-After is still honoured.
    router = LLMRouter(pacer=TokenPacer({}))
    if primary_only and not router.mock:
        router.clients = router.clients[:1]
    return router


def parse_llm_model(spec: str) -> tuple[str, str]:
    """'groq:qwen/qwen3.8-27b' -> ('groq', 'qwen/qwen3.8-27b'); the model part
    may itself contain ':' (OpenRouter's ':free'). ValueError otherwise."""
    provider, sep, model = spec.partition(":")
    if not sep or not model.strip() or provider not in PROVIDERS:
        raise ValueError(f"expected PROVIDER:MODEL with PROVIDER in {', '.join(PROVIDERS)}; "
                         f"got {spec!r}")
    return provider, model.strip()


class PinnedRouter(LLMRouter):
    """A router of exactly one client (``--llm-model``): no fallback model or
    provider, whatever LLM_PROVIDER / *_FALLBACK_* say, and no key needed for
    any provider but the pinned one."""

    def __init__(self, client: LLMClient, pacer: TokenPacer):
        self.mock = False
        self.pacer = pacer
        self.clients = [client]


def build_pinned_router(spec: str) -> LLMRouter:
    """``PinnedRouter`` for "PROVIDER:MODEL" (fails fast on a missing key);
    no per-minute pacing, as in ``build_llm_router``."""
    provider, model = parse_llm_model(spec)
    return PinnedRouter(LLMClient(provider, model=model), TokenPacer({}))


def set_router_temperature(router, temperature: float | None) -> None:
    """Every client of ``router`` sends ``temperature`` (the eval's choice);
    None clears the override, so each client uses its model's recommended
    sampling (``LLMClient.sampling_params``)."""
    for client in getattr(router, "clients", None) or []:
        client.temperature = None if temperature is None else float(temperature)


def openrouter_routing(args, router) -> dict | None:
    """OpenRouter ``provider`` routing sent with a primary-only run whose
    client is on OpenRouter: no fallback to another upstream, and
    ``--llm-upstream`` pins which one (without it OpenRouter still
    load-balances across upstreams, which the per-item upstream log shows)."""
    if not args.llm_primary_only or getattr(router, "mock", False):
        return None
    if not any(getattr(c, "provider", None) == "openrouter" for c in router.clients):
        return None
    routing: dict = {"allow_fallbacks": False}
    if args.llm_upstream:
        routing["order"] = [args.llm_upstream]
    return routing


def call_sha256(system_prompt: str, user_prompt: str) -> str:
    """sha256 of one call's prompts (``run_eval.prompt_sha256`` with the
    system prompt explicit): the cache key's prompt part."""
    return hashlib.sha256(f"{system_prompt}\x00{user_prompt}".encode()).hexdigest()
