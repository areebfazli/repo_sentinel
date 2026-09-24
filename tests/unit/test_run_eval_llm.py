"""Unit tests for the eval's LLM report stage, realistic metrics and stratified
sampling (stub router / embedder / store: no network, no models)."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from backend.app.core.cve_retriever import CVERetriever
from backend.app.core.llm_client import LLMError
from backend.app.core.markdown_renderer import SYSTEM_PROMPT, build_user_prompt
from backend.app.core.reranker import _sigmoid
from ml.evaluation import run_eval

# --- stubs ------------------------------------------------------------------


class StubEmbedder:
    """The "vector" is the code itself, so the stub store can key on it."""

    def embed_text(self, text):
        return text


class StubVectorStore:
    cve_schema_error = None
    cve_collection = "cve_corpus"

    def __init__(self, by_code):
        self._by_code = by_code

    def search_cves(self, query_vector, limit=5, language=None, **_):
        return [dict(c) for c in self._by_code.get(query_vector, [])][:limit]

    def count(self, collection):
        return 1


class StubReranker:
    def __init__(self, logits):
        self._logits = logits

    def rerank(self, query_code, candidates, top_k=1):
        for c in candidates:
            c["rerank_score"] = self._logits.get(c["cve_id"], 0.0)
            c["rerank_prob"] = _sigmoid(c["rerank_score"])
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


class StubRouter:
    """``respond(user_prompt) -> dict`` or raises; records every call."""

    def __init__(self, respond, label="groq:test-model"):
        self.mock = False
        self.clients = [SimpleNamespace(label=label)]
        self._respond = respond
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system, user):
        self.calls.append((system, user))
        return self._respond(user), self.clients[0].label


def _payload(cve_id, sim, category="sqli", fixed=True, margin=None):
    return {
        "cve_id": cve_id,
        "category": category,
        "description": f"{cve_id} description",
        "severity": 9.8,
        "language": "python",
        "vulnerable_code": f"def f_{cve_id[-1]}(q):\n    db.execute('SELECT ' + q)\n",
        "fixed_code": (
            f"def f_{cve_id[-1]}(q):\n    db.execute('SELECT %s', (q,))\n" if fixed else None
        ),
        "source": "osv",
        "embedding_model": "stub",
        "similarity_score": sim,
        "point_id": f"pid-{cve_id}",
        "collection": "cve_corpus",
        "sim_fixed": None if margin is None else sim - margin,
        "twin_margin": margin,
    }


# ANN order deliberately not sorted; C5 is below the 0.25 similarity gate.
CODE = "def lookup(db, q):\n    db.execute('SELECT * FROM t WHERE x=' + q)"
CANDIDATES = [
    _payload("CVE-3", 0.50, category="xss"),
    _payload("CVE-1", 0.90, margin=0.02),
    _payload("CVE-5", 0.20),
    _payload("CVE-2", 0.70, fixed=False),
    _payload("CVE-4", 0.30),
]


@pytest.fixture
def op_settings(monkeypatch):
    s = run_eval.settings
    monkeypatch.setattr(s, "SIM_THRESHOLD_CVE", 0.25)
    monkeypatch.setattr(s, "RETRIEVAL_TOP_K", 3)
    monkeypatch.setattr(s, "TWIN_MARGIN_MIN", None)
    monkeypatch.setattr(s, "HYBRID_ENABLED", False)
    monkeypatch.setattr(s, "RERANK_THRESHOLD", 0.3)
    return s


def _entries_for(items, store, reranker, s):
    gathered, _ = run_eval.gather(items, StubEmbedder(), store, reranker, s.ANN_CANDIDATES)
    rerank_t = run_eval.NO_RERANK_THRESHOLD if reranker is None else s.RERANK_THRESHOLD
    return run_eval.build_llm_entries(
        gathered, s.SIM_THRESHOLD_CVE, rerank_t, s.TWIN_MARGIN_MIN, s.RETRIEVAL_TOP_K
    )


def _item(item_id, label, code, category="sqli", **kw):
    return {"id": item_id, "label": label, "category": category, "code": code, **kw}


def _run(entries, router, **kw):
    kw.setdefault("cache", None)
    kw.setdefault("max_calls", 50)
    kw.setdefault("sleep_s", 0.0)
    kw.setdefault("tpm", 0)
    kw.setdefault("progress", False)
    return asyncio.run(run_eval.run_llm_stage(entries, router, **kw))


# --- prompt identical to production ------------------------------------------


@pytest.mark.parametrize("with_reranker", [False, True])
def test_prompt_is_built_exactly_as_production(op_settings, with_reranker):
    logits = {"CVE-1": -3.0, "CVE-2": 2.0, "CVE-3": 1.0, "CVE-4": 3.0, "CVE-5": 5.0}
    make_reranker = (lambda: StubReranker(logits)) if with_reranker else (lambda: None)
    store = StubVectorStore({CODE: CANDIDATES})

    # Production: the real CVERetriever (feedback + gates) -> build_user_prompt.
    prod_cves = CVERetriever(StubEmbedder(), store, make_reranker()).find_vulnerabilities(
        CODE, language="python"
    )
    expected = build_user_prompt(CODE, prod_cves, [])

    entries = _entries_for(
        [_item("x_vuln", "vulnerable", CODE)], store, make_reranker(), op_settings
    )
    assert [c["cve_id"] for c in entries[0]["cves"]] == [c["cve_id"] for c in prod_cves]
    assert build_user_prompt(entries[0]["code"], entries[0]["cves"], []) == expected
    if with_reranker:
        # CVE-1 falls under RERANK_THRESHOLD, CVE-5 under the similarity gate.
        assert [c["cve_id"] for c in prod_cves] == ["CVE-4", "CVE-2", "CVE-3"]
    else:
        assert [c["cve_id"] for c in prod_cves] == ["CVE-1", "CVE-2", "CVE-3"]
        assert "how this CVE was fixed" in expected  # the fix diff reaches the prompt

    router = StubRouter(lambda user: {"findings": []})
    _run(entries, router)
    assert router.calls == [(SYSTEM_PROMPT, expected)]


# --- prediction rule -----------------------------------------------------------


def test_prediction_rule_mirrors_validation():
    cves = [{"cve_id": "CVE-1", "category": "sqli"}, {"cve_id": "CVE-2", "category": "xss"}]

    def decide(*findings):
        return run_eval.llm_decision({"findings": list(findings)}, cves)

    empty = decide()
    assert not empty["prediction"] and not empty["any_finding"]
    assert empty["validated_count"] == 0
    # A finding naming a retrieved CVE -> vulnerable.
    hit = decide({"cve_id": "CVE-2", "title": "x"})
    assert hit["prediction"] and hit["flagged_cve_ids"] == ["CVE-2"]
    assert hit["flagged_categories"] == ["xss"]
    # A hallucinated CVE id is dropped by the allowlist (anti-hallucination).
    fake = decide({"cve_id": "CVE-9999", "title": "x"})
    assert not fake["prediction"] and fake["validated_count"] == 0
    assert fake["raw_finding_count"] == 1
    # A generic finding is a production report finding but not a CVE finding.
    generic = decide({"cve_id": None, "title": "style nit"})
    assert not generic["prediction"] and generic["any_finding"]
    # No team matches were retrieved, so any team_pr_id is outside the allowlist.
    team = decide({"team_pr_id": 1042, "title": "x"})
    assert team["validated_count"] == 0
    # Missing "findings" key -> no findings (production's .get default).
    assert run_eval.llm_decision({}, cves)["validated_count"] == 0
    for bad in ([], "x", {"findings": {"cve_id": "CVE-1"}}, {"findings": None}):
        with pytest.raises(ValueError):
            run_eval.llm_decision(bad, cves)


def test_items_without_candidates_are_safe_without_a_call():
    entries = [{"id": "a_vuln", "kind": "vulnerable", "label": "vulnerable",
                "category": "sqli", "code": "x", "cves": []}]
    router = StubRouter(lambda user: pytest.fail("must not be called"))
    stage = _run(entries, router)
    rec = stage["records"][0]
    assert rec["status"] == "no_candidates" and rec["prediction"] is False
    assert stage["calls"] == 0 and router.calls == []


# --- errors, cache, budget, rate limits ----------------------------------------


def _entry(item_id, kind="vulnerable", label="vulnerable", cve="CVE-1"):
    return {"id": item_id, "kind": kind, "label": label, "category": "sqli",
            "code": f"code of {item_id}", "cves": [{"cve_id": cve, "category": "sqli"}]}


def _flag_all(user):
    return {"findings": [{"cve_id": "CVE-1", "title": "sqli"}]}


def test_llm_error_is_recorded_excluded_and_counted():
    def respond(user):
        if "b_vuln" in user:
            raise LLMError("All LLM clients failed (groq:test-model): boom")
        if "c_vuln" in user:
            return ["not", "an", "object"]
        return _flag_all(user)

    entries = [_entry("a_vuln"), _entry("b_vuln"), _entry("c_vuln"), _entry("d_vuln")]
    stage = _run(entries, StubRouter(respond))
    status = {r["id"]: r["status"] for r in stage["records"]}
    assert status == {"a_vuln": "ok", "b_vuln": "error", "c_vuln": "error", "d_vuln": "ok"}
    assert stage["records"][1]["error_kind"] == "other"
    assert stage["stopped"] is None and stage["calls"] == 4
    summary = run_eval.summarize_llm(stage, {e["id"]: True for e in entries})
    m = summary["realistic"]
    assert (m["n"], m["n_scored"], m["n_excluded"]) == (4, 2, 2)
    assert m["tpr_vulnerable"]["n"] == 2 and m["tpr_vulnerable"]["rate"] == 1.0
    # The retrieval comparison uses the same scored items.
    assert summary["realistic_retrieval_same_items"]["n_scored"] == 2
    assert summary["category_hit_rate"] == 1.0


def test_cache_resume_skips_done_items(tmp_path):
    path = tmp_path / "cache.jsonl"
    fail_b = {"on": True}

    def respond(user):
        if fail_b["on"] and "b_vuln" in user:
            raise LLMError("transient")
        return _flag_all(user)

    entries = [_entry("a_vuln"), _entry("b_vuln"), _entry("c_vuln")]
    first = StubRouter(respond)
    _run(entries, first, cache=run_eval.LLMCache(path))
    assert len(first.calls) == 3
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert sorted(r["id"] for r in lines) == ["a_vuln", "c_vuln"]  # errors not cached
    assert {r["model"] for r in lines} == {"groq:test-model"}

    fail_b["on"] = False
    second = StubRouter(respond)
    stage = _run(entries, second, cache=run_eval.LLMCache(path))
    assert [u for _, u in second.calls] == [
        build_user_prompt("code of b_vuln", entries[1]["cves"], [])
    ]  # only the errored item is re-called
    assert [r["status"] for r in stage["records"]] == ["cached", "ok", "cached"]
    assert all(r["prediction"] for r in stage["records"])
    assert stage["calls"] == 1

    # A different model (or prompt) is a cache miss.
    other = StubRouter(respond, label="groq:other-model")
    _run(entries, other, cache=run_eval.LLMCache(path))
    assert len(other.calls) == 3
    # A torn last line (interrupted run) is ignored, not fatal.
    with open(path, "a") as f:
        f.write('{"id": "trunc')
    assert len(run_eval.LLMCache(path)) == 6


def test_budget_stop_and_pacing():
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    entries = [_entry(f"{i}_vuln") for i in range(5)]
    router = StubRouter(_flag_all)
    stage = _run(entries, router, max_calls=2, sleep_s=2.5, sleep=fake_sleep)
    assert [r["status"] for r in stage["records"]] == ["ok", "ok", "not_run", "not_run", "not_run"]
    assert stage["stopped"] == "max_calls" and stage["calls"] == 2 and len(router.calls) == 2
    assert sleeps == [2.5]  # between calls, not before the first
    assert run_eval.realistic_metrics(
        [{**r, "pred": r["prediction"]} for r in stage["records"]]
    )["n_excluded"] == 3

    # Token budget: every call costs ~prompt_tokens_est (no usage without a tap).
    per_call = stage["records"][0]["prompt_tokens_est"]
    budget = _run(entries, StubRouter(_flag_all), token_budget=2 * per_call + 1)
    assert budget["stopped"] == "token_budget" and budget["calls"] == 2

    # TPM pacing: wait max(sleep_s, 60 * tokens / tpm).
    sleeps.clear()
    _run(entries[:2], StubRouter(_flag_all), sleep_s=0.5, tpm=per_call, sleep=fake_sleep)
    assert sleeps == [pytest.approx(60.0)]


def test_repeated_rate_limits_stop_cleanly():
    def respond(user):
        raise LLMError(
            "All LLM clients failed (groq:test-model): groq:test-model HTTP 429"
        )

    entries = [_entry(f"{i}_vuln") for i in range(6)]
    stage = _run(entries, StubRouter(respond), max_rate_limit_errors=3)
    assert [r["status"] for r in stage["records"]] == ["error"] * 3 + ["not_run"] * 3
    assert {r.get("error_kind") for r in stage["records"][:3]} == {"rate_limit"}
    assert stage["stopped"] == "rate_limit" and stage["calls"] == 3

    # A success in between resets the streak.
    n = {"i": 0}

    def flaky(user):
        n["i"] += 1
        if n["i"] % 2:
            raise LLMError("HTTP 429")
        return _flag_all(user)

    assert _run(entries, StubRouter(flaky), max_rate_limit_errors=2)["stopped"] is None


def test_daily_limit_stops_at_once():
    def respond(user):
        raise LLMError("Rate limit reached ... on tokens per day (TPD): Limit 200000")

    stage = _run([_entry(f"{i}_vuln") for i in range(3)], StubRouter(respond))
    assert stage["stopped"] == "daily_limit" and stage["calls"] == 1
    assert stage["records"][0]["error_kind"] == "daily_limit"


def test_http_tap_reads_usage_and_daily_limit():
    def handler(request):
        if request.url.path == "/ok":
            return httpx.Response(200, json={"choices": [], "usage": {
                "prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}})
        if request.url.path == "/minute":
            return httpx.Response(429, headers={"Retry-After": "3"},
                                  text="Rate limit reached on tokens per minute (TPM)")
        return httpx.Response(429, headers={"Retry-After": "965"},
                              text="Rate limit reached on tokens per day (TPD)")

    async def go():
        orig = httpx.AsyncClient.post
        with run_eval.HttpUsageTap() as tap:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                for path in ("/ok", "/minute", "/day"):
                    await c.post(f"https://llm.test{path}", json={})
            events = tap.take()
        assert httpx.AsyncClient.post is orig  # restored
        return events

    ok, minute, day = asyncio.run(go())
    assert ok["usage"]["total_tokens"] == 1500
    assert run_eval._usage_totals([ok, minute])["total_tokens"] == 1500
    assert minute["retry_after"] == 3.0 and not minute["daily_limit"]
    assert day["daily_limit"]
    assert run_eval.classify_llm_error(LLMError("x"), [minute]) == "rate_limit"
    assert run_eval.classify_llm_error(LLMError("x"), [minute, day]) == "daily_limit"
    assert run_eval.classify_llm_error(LLMError("HTTP 401"), []) == "other"


def test_http_tap_recognizes_openrouter_daily_cap():
    minute_body = {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-min. "}}
    day_body = {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day. "
                          "Add 10 credits to unlock 1000 free model requests per day"}}
    terse_day = {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}

    def handler(request):
        path = request.url.path
        if path == "/minute":
            return httpx.Response(429, json=minute_body)
        if path == "/day":
            return httpx.Response(429, json=day_body)
        if path == "/terse":
            return httpx.Response(429, json=terse_day)
        if path == "/in200":  # OpenRouter can wrap an error in an HTTP 200
            return httpx.Response(200, json=terse_day)
        return httpx.Response(200, json={"error": {"code": 429, "message": "upstream busy"}})

    async def go():
        with run_eval.HttpUsageTap() as tap:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                for path in ("/minute", "/day", "/terse", "/in200", "/in200-minute"):
                    await c.post(f"https://openrouter.test{path}", json={})
            return tap.take()

    minute, day, terse, in200, in200_minute = asyncio.run(go())
    assert not minute["daily_limit"]
    assert day["daily_limit"] and terse["daily_limit"] and in200["daily_limit"]
    assert not in200_minute["daily_limit"]
    assert run_eval.classify_llm_error(LLMError("HTTP 429"), [minute]) == "rate_limit"
    assert run_eval.classify_llm_error(LLMError("x"), [in200_minute]) == "rate_limit"
    for event in (day, terse, in200):
        assert run_eval.classify_llm_error(LLMError("HTTP 429"), [minute, event]) == "daily_limit"


def test_openrouter_daily_cap_stops_the_stage():
    def respond(user):
        raise LLMError("Rate limit exceeded: free-models-per-day")

    stage = _run([_entry(f"{i}_vuln") for i in range(3)], StubRouter(respond))
    assert stage["stopped"] == "daily_limit" and stage["calls"] == 1


# --- realistic metrics -----------------------------------------------------------


def _rec(item_id, kind, label, pred):
    return {"id": item_id, "kind": kind, "label": label, "pred": pred}


def test_realistic_metric_formulas_hand_computed():
    recs = []
    # 4 OSV pairs: vuln flagged T,T,T,F; twin flagged T,F,F,T.
    for i, (v, t) in enumerate([(True, True), (True, False), (True, False), (False, True)]):
        recs.append(_rec(f"p{i}_vuln", "vulnerable", "vulnerable", v))
        recs.append(_rec(f"p{i}_safe", "fixed_twin", "safe", t))
    # 10 ordinary, 1 flagged, plus one excluded (LLM error).
    recs += [_rec(f"o{i}", "ordinary", "safe", i == 0) for i in range(10)]
    recs.append(_rec("o_err", "ordinary", "safe", None))
    recs += [_rec(f"hs{i}", "handwritten", "safe", False) for i in range(2)]
    recs += [_rec(f"hv{i}", "handwritten", "vulnerable", True) for i in range(2)]

    m = run_eval.realistic_metrics(recs)
    assert (m["n"], m["n_scored"], m["n_excluded"]) == (23, 22, 1)
    assert m["tpr_vulnerable"] == {"k": 3, "n": 4, "rate": 0.75, "ci95": [0.3006, 0.9544]}
    assert m["fpr_fixed_twin"]["rate"] == 0.5
    assert m["fpr_ordinary"]["rate"] == 0.1 and m["fpr_ordinary"]["n"] == 10
    assert m["fpr_handwritten_safe"]["rate"] == 0.0
    assert m["tpr_handwritten"]["rate"] == 1.0
    # TPR*pi / (TPR*pi + FPR_ord*(1-pi)) with TPR=0.75, FPR_ord=0.1.
    assert m["precision_at_base_rate"] == {
        "0.01": round(0.0075 / (0.0075 + 0.099), 4),  # 0.0704
        "0.02": round(0.015 / (0.015 + 0.098), 4),    # 0.1327
        "0.05": round(0.0375 / (0.0375 + 0.095), 4),  # 0.283
    }
    assert m["precision_at_base_rate"]["0.01"] == 0.0704
    assert m["precision_balanced_vs_fixed_twin"] == 0.6    # 0.75 / (0.75 + 0.5)
    assert m["precision_balanced_vs_ordinary"] == 0.8824   # 0.75 / (0.75 + 0.1)
    p = m["pairs"]
    assert p["n"] == 4
    assert (p["vuln_only"], p["twin_only"], p["both"], p["neither"]) == (0.5, 0.25, 0.25, 0.0)
    assert m["observed"] == {"tp": 5, "fp": 3, "fn": 1, "tn": 13,
                             "precision": 0.625, "recall": 0.8333}


def test_wilson_and_undefined_rates():
    assert run_eval.wilson_interval(0, 10) == [0.0, 0.2775]
    assert run_eval.wilson_interval(5, 10) == [0.2366, 0.7634]
    assert run_eval.wilson_interval(0, 0) is None
    m = run_eval.realistic_metrics([_rec("a_vuln", "vulnerable", "vulnerable", True)])
    assert m["fpr_ordinary"]["rate"] is None
    assert m["precision_at_base_rate"]["0.01"] is None
    assert m["pairs"]["n"] == 0 and m["pairs"]["vuln_only"] is None
    # No false positives at all: precision 1.0, not a division error.
    assert run_eval.precision_at_base_rate(0.5, 0.0, 0.01) == 1.0
    assert run_eval.precision_at_base_rate(0.0, 0.0, 0.01) is None


# --- kinds + stratified sampling ---------------------------------------------------


def test_kind_derivation_from_ids():
    k = run_eval.item_kind
    assert k({"id": "CVE-1_fn_vuln"}) == "vulnerable"
    assert k({"id": "CVE-1_fn_safe"}) == "fixed_twin"
    assert k({"id": "sqli_vuln_1"}) == "handwritten"
    assert k({"id": "sqli_safe_3"}) == "handwritten"
    # An explicit kind wins over the id (an ordinary id may end in _safe).
    assert k({"id": "requests_get_safe", "kind": "ordinary"}) == "ordinary"
    assert k({"id": "x_vuln", "kind": "bogus"}) == "vulnerable"


def _kind_pool():
    items = []
    for i in range(20):
        items.append({"id": f"CVE-{i}_fn_vuln", "label": "vulnerable"})
        items.append({"id": f"CVE-{i}_fn_safe", "label": "safe"})
    items.append({"id": "CVE-99_lone_vuln", "label": "vulnerable"})  # twin missing
    for i in range(7):
        items.append({"id": f"sqli_vuln_{i}", "label": "vulnerable"})
    for i in range(30):  # some ordinary ids look like pair members
        items.append({"id": f"ord_{i}_safe" if i % 2 else f"ord_{i}", "label": "safe",
                      "category": "none", "kind": "ordinary"})
    return items


def _kinds(items):
    out = {}
    for i in items:
        out[run_eval.item_kind(i)] = out.get(run_eval.item_kind(i), 0) + 1
    return out


def test_sample_kinds_is_stratified_deterministic_and_keeps_pairs():
    pool = _kind_pool()
    quotas = {"vulnerable": 5, "fixed_twin": 5, "ordinary": 8}
    a = run_eval.sample_items_by_kind(pool, quotas, seed=3)
    assert a == run_eval.sample_items_by_kind(pool, quotas, seed=3)
    assert a != run_eval.sample_items_by_kind(pool, quotas, seed=4)
    assert _kinds(a) == quotas  # handwritten not requested -> left out
    ids = {i["id"] for i in a}
    for item_id in ids:
        if item_id.startswith("CVE-"):
            prefix = item_id.rsplit("_", 1)[0]
            assert {f"{prefix}_vuln", f"{prefix}_safe"} <= ids  # complete pairs only
    # Input order is kept.
    order = [i["id"] for i in pool]
    assert [i["id"] for i in a] == sorted(ids, key=order.index)

    # Unequal quotas: pairs are never split; a lone vuln may fill a vuln slot.
    b = run_eval.sample_items_by_kind(pool, {"vulnerable": 5, "fixed_twin": 3}, seed=1)
    twins = [i["id"] for i in b if i["id"].endswith("_safe")]
    assert len(twins) == 3
    for t in twins:
        assert t[: -len("_safe")] + "_vuln" in {i["id"] for i in b}
    assert _kinds(b)["vulnerable"] in (3, 4)

    # Only one pair kind requested: its members are sampled on their own.
    c = run_eval.sample_items_by_kind(pool, {"vulnerable": 3}, seed=1)
    assert _kinds(c) == {"vulnerable": 3}

    # The pool can run short of a quota.
    d = run_eval.sample_items_by_kind(pool, {"handwritten": 50}, seed=1)
    assert _kinds(d) == {"handwritten": 7}


def test_parse_sample_kinds():
    assert run_eval.parse_sample_kinds("vulnerable=40, fixed_twin=40,ordinary=80") == {
        "vulnerable": 40, "fixed_twin": 40, "ordinary": 80,
    }
    for bad in ("vulnerable", "safe=3", "vulnerable=0", "vulnerable=x",
                "vulnerable=1,vulnerable=2", ""):
        with pytest.raises(ValueError):
            run_eval.parse_sample_kinds(bad)


# --- CLI ---------------------------------------------------------------------------


def test_llm_and_sampling_arg_validation(capsys):
    ok = run_eval.parse_args(["--llm", "--llm-max-calls", "50", "--out", "o.json",
                              "--sample-kinds", "vulnerable=2,ordinary=4"])
    assert ok.llm and ok.llm_max_calls == 50 and ok.llm_sleep == 2.5
    assert ok.sample_kinds == {"vulnerable": 2, "ordinary": 4}
    for bad, msg in (
        (["--llm", "--out", "o.json"], "--llm-max-calls"),
        (["--llm", "--llm-max-calls", "201", "--out", "o.json"], "between 1 and 200"),
        (["--llm", "--llm-max-calls", "0", "--out", "o.json"], "between 1 and 200"),
        (["--llm", "--llm-max-calls", "5"], "--out"),
        (["--llm", "--llm-max-calls", "5", "--out", "o", "--write-baseline"], "--llm"),
        (["--sample-kinds", "vulnerable=2", "--write-baseline"], "--sample-kinds"),
        (["--sample-kinds", "vulnerable=2", "--sample", "4"], "mutually exclusive"),
        (["--sample-kinds", "nope=2"], "--sample-kinds"),
        (["--llm-max-calls", "5"], "requires --llm"),
        (["--llm", "--llm-max-calls", "5", "--out", "o", "--llm-sleep", "-1"], ">= 0"),
    ):
        with pytest.raises(SystemExit):
            run_eval.parse_args(bad)
        assert msg in capsys.readouterr().err, bad


def test_main_llm_end_to_end(tmp_path, monkeypatch, op_settings):
    ds = tmp_path / "ds.jsonl"
    items = [
        _item("CVE-1_f_vuln", "vulnerable", "vuln code"),
        _item("CVE-1_f_safe", "safe", "fixed code"),
        _item("ord_1", "safe", "ordinary code", category="none", kind="ordinary"),
        _item("ord_2", "safe", "unmatched code", category="none", kind="ordinary"),
    ]
    ds.write_text("".join(json.dumps(i) + "\n" for i in items))
    store = StubVectorStore({
        "vuln code": [_payload("CVE-7", 0.9)],
        "fixed code": [_payload("CVE-7", 0.88)],
        "ordinary code": [_payload("CVE-8", 0.4)],
    })
    monkeypatch.setattr(
        run_eval, "build_components", lambda *a, **k: (StubEmbedder(), store, None)
    )

    def respond(user):  # the LLM flags only the pre-fix code
        if "vuln code" in user:
            return {"findings": [{"cve_id": "CVE-7", "title": "sqli"}]}
        return {"findings": []}

    router = StubRouter(respond)
    monkeypatch.setattr(run_eval, "build_llm_router", lambda primary_only=False: router)
    out = tmp_path / "out.json"
    cache = tmp_path / "cache.jsonl"
    argv = ["--dataset", str(ds), "--out", str(out), "--no-rerank", "--llm",
            "--llm-max-calls", "10", "--llm-cache", str(cache), "--llm-tpm", "0",
            "--llm-sleep", "0"]
    run_eval.main(argv)

    data = json.loads(out.read_text())
    assert {"operating", "sweep", "realistic", "llm"} <= set(data)
    r = data["realistic"]  # retrieval-only: everything with a match is flagged
    assert r["tpr_vulnerable"]["rate"] == 1.0 and r["fpr_fixed_twin"]["rate"] == 1.0
    assert r["fpr_ordinary"]["rate"] == 0.5
    # Same retrieval-only rule as score() at the operating point.
    op = data["operating"]
    assert (op["tp"], op["fp"]) == (r["observed"]["tp"], r["observed"]["fp"])
    llm = data["llm"]
    assert len(router.calls) == 3  # ord_2 had no candidates -> no call
    assert llm["status_counts"] == {"ok": 3, "no_candidates": 1}
    assert llm["realistic"]["tpr_vulnerable"]["rate"] == 1.0
    assert llm["realistic"]["fpr_fixed_twin"]["rate"] == 0.0
    assert llm["realistic"]["fpr_ordinary"]["rate"] == 0.0
    assert llm["realistic"]["pairs"]["vuln_only"] == 1.0
    assert llm["config"]["model"] == "groq:test-model" and llm["providers"]
    assert {i["id"]: i["retrieval_pred"] for i in llm["items"]}["ord_2"] is False

    # Re-run resumes entirely from the cache.
    run_eval.main(argv)
    assert len(router.calls) == 3
    assert json.loads(out.read_text())["llm"]["status_counts"] == {"cached": 3, "no_candidates": 1}

