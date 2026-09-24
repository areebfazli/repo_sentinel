"""Unit tests for the eval's localised scoring, offline rescore / compare, LLM
temperature, repeats, split manifests and pinned-provider options. Stub
routers / stores and fake HTTP only: no network, no model loads."""
import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from backend.app.core import llm_client
from ml.evaluation import run_eval

# --- fix lines (difflib over vulnerable vs fixed) ------------------------------

VULN = "def f(q):\n    x = 1\n    db.execute('SELECT ' + q)\n    return x\n"
SAFE = "def f(q):\n    x = 1\n    db.execute('SELECT %s', (q,))\n    return x\n"


def test_modified_and_deleted_lines():
    assert run_eval.fix_changed_lines(VULN, SAFE) == [3]
    assert run_eval.fix_changed_lines(SAFE, VULN) == [3]  # the twin's changed line
    # A deleted line is a fix line of the vulnerable version; in the twin it is
    # an insertion point (the gap between lines 2 and 3) +/- 2 lines.
    vuln = "a\nb\nos.system(cmd)\nc\nd\ne\n"
    safe = "a\nb\nc\nd\ne\n"
    assert run_eval.fix_changed_lines(vuln, safe) == [3]
    assert run_eval.fix_changed_lines(safe, vuln) == [1, 2, 3, 4]


def test_pure_insertion_uses_insertion_point_plus_minus_two():
    vuln = "\n".join(f"line{i}" for i in range(1, 11))  # line1..line10
    lines = vuln.splitlines()
    safe = "\n".join(lines[:5] + ["if not ok: raise", "check()"] + lines[5:])
    # Inserted between line 5 and line 6 -> lines 4, 5 | 6, 7.
    assert run_eval.fix_changed_lines(vuln, safe) == [4, 5, 6, 7]
    assert run_eval.fix_changed_lines(vuln, safe, context=1) == [5, 6]
    # The twin's own lines are the inserted ones.
    assert run_eval.fix_changed_lines(safe, vuln) == [6, 7]
    # Clipped at both ends of the code.
    assert run_eval.fix_changed_lines("a\nb\nc", "guard\na\nb\nc") == [1, 2]
    assert run_eval.fix_changed_lines("a\nb\nc", "a\nb\nc\nguard") == [2, 3]


def test_reindent_and_blank_lines_are_not_changes():
    vuln = "def f(x):\n    run(x)\n    log(x)\n"
    safe = "def f(x):\n    if ok(x):\n        run(x)\n        log(x)\n"
    # The body only moved under the new if: just the insertion point counts.
    assert run_eval.fix_changed_lines(vuln, safe) == [1, 2, 3]
    assert run_eval.fix_changed_lines("a\nb\n", "a\n\n\nb\n") is None
    assert run_eval.fix_changed_lines(VULN, VULN) is None
    assert run_eval.fix_changed_lines(VULN, "  " + VULN.replace("\n", "\n  ")) is None


def test_annotate_fix_targets_pairs_by_group():
    items = [
        {"id": "A_1234abcd_vuln", "label": "vulnerable", "code": VULN},
        {"id": "A_1234abcd_safe", "label": "safe", "code": SAFE},
        {"id": "B_vuln", "label": "vulnerable", "code": VULN},  # twin not loaded
        {"id": "o1", "label": "safe", "code": "x", "kind": "ordinary"},
    ]
    stats = run_eval.annotate_fix_targets(items)
    assert stats == {"pairs": 1, "pairs_with_fix_lines": 1, "unpaired_items": 1,
                     "insertion_context": 2}
    assert items[0]["fix_lines"] == [3] and items[1]["fix_lines"] == [3]
    assert items[2]["fix_lines"] is None and "fix_lines" not in items[3]


# --- localised hit ---------------------------------------------------------------


def test_overlap_tolerance_boundaries():
    def hit(line, end=None, tol=2):
        return run_eval.localised_hit([{"line": line, "end_line": end}], [10], tol)

    assert hit(10) and hit(8) and hit(12)  # exactly +/- 2
    assert hit(7) is False and hit(13) is False
    assert hit(3, 7, tol=3) and hit(3, 7, tol=2) is False  # a range: its end counts
    assert hit(12, tol=0) is False and hit(10, tol=0)
    assert run_eval.localised_hit([], [10]) is False  # no finding: not a TP
    assert run_eval.localised_hit([{"line": 10}], None) is None  # no fix lines


def test_quote_is_located_when_a_finding_has_no_line():
    f = {"quoted_code": "db.execute('SELECT ' + q)"}
    assert run_eval.finding_span(f, VULN) == (3, 3)
    assert run_eval.localised_hit([f], [3], 0, code=VULN) is True
    assert run_eval.localised_hit([f], [9], 0, code=VULN) is False
    # Can't be anchored (no line, quote not in the code): undecidable, not False.
    assert run_eval.localised_hit([{"quoted_code": "nowhere()"}], [3], code=VULN) is None
    assert run_eval.localised_hit([{"title": "legacy"}], [3]) is None


def test_cwe_match_counts_even_off_the_fix_lines():
    far = {"line": 40, "cwe": "cwe-89"}
    assert run_eval.localised_hit([far], [3], expected_cwes=["CWE-89"]) is True
    assert run_eval.localised_hit([far], [3], expected_cwes=["CWE-79"]) is False
    assert run_eval.localised_hit([far], None, expected_cwes=["CWE-89"]) is True
    assert run_eval.localised_hit([far], None, expected_cwes=["CWE-79"]) is None
    assert run_eval.item_expected_cwes({"cwe": "CWE-89"}) == ["CWE-89"]
    assert run_eval.item_expected_cwes({"cwe_ids": ["CWE-79", "cwe 89", "x"]}) == [
        "CWE-79", "CWE-89"]
    assert run_eval.item_expected_cwes({"category": "sqli"}) == []


def test_record_localised_by_kind_and_arm():
    base = {"any_finding": True, "fix_lines": [3], "findings": [{"line": 3, "cwe": "CWE-89"}]}
    assert run_eval.record_localised({**base, "kind": "vulnerable"}) is True
    assert run_eval.record_localised({**base, "kind": "fixed_twin"}) is True  # localised FP
    assert run_eval.record_localised({**base, "kind": "ordinary"}) is None
    assert run_eval.record_localised({**base, "kind": "vulnerable"}, anchored=False) is None
    # A twin's CWE never counts (only a vulnerable item's expected CWE does).
    twin = {**base, "kind": "fixed_twin", "findings": [{"line": 30, "cwe": "CWE-89"}],
            "expected_cwes": ["CWE-89"]}
    assert run_eval.record_localised(twin) is False
    # An old record: flagged, but no stored findings -> unknown; unflagged -> False.
    old = {"kind": "vulnerable", "fix_lines": [3], "any_finding": True}
    assert run_eval.record_localised(old) is None
    assert run_eval.record_localised({**old, "any_finding": False}) is False
    assert run_eval.record_localised({**old, "any_finding": None}) is None


def _r(item_id, kind, any_finding, localised, label=None):
    return {"id": item_id, "kind": kind, "label": label or (
        "vulnerable" if kind == "vulnerable" else "safe"),
        "any_finding": any_finding, "prediction": any_finding, "localised": localised}


def test_localised_metrics_hand_computed():
    recs = [
        _r("p1_vuln", "vulnerable", True, True), _r("p1_safe", "fixed_twin", True, False),
        _r("p2_vuln", "vulnerable", True, False), _r("p2_safe", "fixed_twin", False, False),
        _r("p3_vuln", "vulnerable", False, False), _r("p3_safe", "fixed_twin", True, True),
        _r("p4_vuln", "vulnerable", True, None),  # unlocalisable (no fix lines)
        _r("o1", "ordinary", True, None), _r("o2", "ordinary", False, None),
    ]
    m = run_eval.localised_metrics(recs)
    assert (m["tpr_vulnerable"]["k"], m["tpr_vulnerable"]["n"]) == (1, 3)
    assert m["n_vulnerable_unlocalisable"] == 1
    assert m["tpr_vulnerable_any_finding_same_items"]["k"] == 2  # p1, p2
    assert (m["fpr_fixed_twin"]["k"], m["fpr_fixed_twin"]["n"]) == (2, 3)  # any finding
    assert (m["fpr_fixed_twin_localised"]["k"], m["fpr_fixed_twin_localised"]["n"]) == (1, 3)
    assert m["fpr_ordinary"]["rate"] == 0.5
    anyf = run_eval.realistic_metrics([run_eval._row(r, r["any_finding"]) for r in recs])
    h = run_eval.llm_headline(anyf, m)
    assert h["primary"] == "tpr_localised"
    assert h["tpr_localised"]["rate"] == round(1 / 3, 4)
    assert h["tpr_any_finding"]["rate"] == 0.75  # 3 of the 4 vulnerable items
    assert run_eval.llm_headline(anyf, None, "why")["primary"] == "tpr_any_finding"


# --- stage: findings stored, localised per record --------------------------------


class StubRouter:
    """``respond(user, n) -> dict`` (n = 0-based call number); records calls."""

    def __init__(self, respond, label="groq:test-model", temperature=None):
        self.mock = False
        client = SimpleNamespace(label=label, provider=label.split(":")[0])
        if temperature is not None:
            client.temperature = temperature
        self.clients = [client]
        self._respond = respond
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system, user, **kwargs):
        self.calls.append((system, user))
        return self._respond(user, len(self.calls) - 1), self.clients[0].label


def _stage(entries, router, **kw):
    kw.setdefault("cache", None)
    kw.setdefault("max_calls", 50)
    kw.setdefault("sleep_s", 0.0)
    kw.setdefault("tpm", 0)
    kw.setdefault("progress", False)
    kw.setdefault("arm", "no_retrieval")
    return asyncio.run(run_eval.run_llm_stage(entries, router, **kw))


def _pair_entries():
    items = [
        {"id": "P_0000aaaa_vuln", "label": "vulnerable", "code": VULN, "category": "sqli",
         "language": "python"},
        {"id": "P_0000aaaa_safe", "label": "safe", "code": SAFE, "category": "sqli",
         "language": "python"},
        {"id": "o1", "label": "safe", "code": "def g(a):\n    return helper(a)\n",
         "category": "none", "kind": "ordinary", "language": "python"},
    ]
    run_eval.annotate_fix_targets(items)
    return [{**i, "kind": run_eval.item_kind(i), "cves": [],
             "expected_cwes": run_eval.item_expected_cwes(i)} for i in items], items


def _flag(quote):
    return {"findings": [{"unit": "U1", "severity": "high", "cwe": "CWE-89",
                          "title": "sqli", "quoted_code": quote}]}


def test_stage_stores_findings_and_localises():
    entries, _ = _pair_entries()

    def respond(user, n):
        if "'SELECT ' + q" in user:
            return _flag("db.execute('SELECT ' + q)")
        if "def g(a)" in user:
            return _flag("return helper(a)")
        return _flag("def f(q):")  # the twin: flagged on line 1, 2 lines off the fix

    stage = _stage(entries, StubRouter(respond), tolerance=0)
    recs = {r["id"]: r for r in stage["records"]}
    v = recs["P_0000aaaa_vuln"]
    assert v["findings"] == [{"line": 3, "end_line": 3, "cwe": "CWE-89", "severity": "high",
                              "cve_id": None, "title": "sqli",
                              "quoted_code": "db.execute('SELECT ' + q)"}]
    assert v["fix_lines"] == [3] and v["localised"] is True
    assert recs["P_0000aaaa_safe"]["localised"] is False  # line 1, fix line 3, tol 0
    assert recs["o1"]["localised"] is None and recs["o1"]["any_finding"] is True
    assert "repeat" not in v  # a single run's records are unchanged
    summary = run_eval.summarize_llm(stage, {r["id"]: False for r in stage["records"]})
    h = summary["headline"]
    assert h["tpr_localised"]["rate"] == 1.0 and h["fpr_fixed_twin"]["rate"] == 1.0
    assert h["fpr_fixed_twin_localised"]["rate"] == 0.0
    assert "repeats" not in summary
    # With the default tolerance (2) the twin's finding is on the changed lines.
    stage = _stage(entries, StubRouter(respond))
    assert {r["id"]: r["localised"] for r in stage["records"]}["P_0000aaaa_safe"] is True


def test_legacy_arm_has_no_localised_numbers():
    entries, _ = _pair_entries()
    for e in entries:
        e["cves"] = [{"cve_id": "CVE-1", "category": "sqli"}]
    stage = _stage(entries, StubRouter(lambda u, n: {"findings": [{"cve_id": "CVE-1"}]}),
                   arm="legacy")
    assert all(r["localised"] is None for r in stage["records"])
    assert stage["records"][0]["findings"] == [
        {"severity": None, "cve_id": "CVE-1", "title": "Security finding"}]
    summary = run_eval.summarize_llm(stage, {e["id"]: True for e in entries})
    assert summary["realistic_localised"] is None
    assert summary["headline"]["tpr_localised"] is None
    assert "legacy" in summary["headline"]["note"]


# --- repeats, temperature, cache keys ----------------------------------------------


def test_repeats_flip_rate_and_cache_keys(tmp_path):
    entries, _ = _pair_entries()
    flaky = {"n": 0}

    def respond(user, n):
        if "'SELECT ' + q" in user:  # vulnerable: flagged in repeats 0 and 2 only
            flaky["n"] += 1
            return _flag("db.execute('SELECT ' + q)") if flaky["n"] != 2 else {"findings": []}
        return {"findings": []}  # twin and ordinary: stable

    cache_path = tmp_path / "cache.jsonl"
    router = StubRouter(respond)
    stage = _stage(entries, router, repeats=3, cache=run_eval.LLMCache(cache_path),
                   temperature=0.0)
    assert len(router.calls) == 9  # each repeat is its own call
    assert [r["repeat"] for r in stage["records"]] == [0, 0, 0]
    assert [[r["repeat"] for r in run] for run in stage["repeat_records"]] == [[1] * 3, [2] * 3]
    summary = run_eval.summarize_llm(stage, {e["id"]: False for e in entries})
    rep = summary["repeats"]
    assert rep["k"] == 3
    fr = rep["flip_rate_prediction"]
    assert (fr["k"], fr["n"]) == (1, 3) and fr["ci95"] == run_eval.wilson_interval(1, 3)
    assert rep["flip_rate_prediction_by_kind"]["vulnerable"]["rate"] == 1.0
    assert (rep["flip_rate_localised"]["k"], rep["flip_rate_localised"]["n"]) == (1, 2)
    assert [p["tpr_localised"] for p in rep["per_repeat"]] == [1.0, 0.0, 1.0]

    lines = [json.loads(x) for x in cache_path.read_text().splitlines()]
    assert sorted((r["id"], r["repeat"]) for r in lines if "vuln" in r["id"]) == [
        ("P_0000aaaa_vuln", 0), ("P_0000aaaa_vuln", 1), ("P_0000aaaa_vuln", 2)]
    assert {r["temperature"] for r in lines} == {0.0}
    # Re-run: everything from the cache, same outcome.
    again = StubRouter(respond)
    stage2 = _stage(entries, again, repeats=3, cache=run_eval.LLMCache(cache_path),
                    temperature=0.0)
    assert again.calls == []
    assert [r["prediction"] for r in stage2["repeat_records"][0]] == [False, False, False]
    # Another temperature is a cache miss.
    other = StubRouter(respond)
    _stage(entries, other, cache=run_eval.LLMCache(cache_path), temperature=0.2)
    assert len(other.calls) == 3


def test_old_cache_entries_count_as_temperature_0_2_repeat_0(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt_sha256": "s", "model": "m",
                                "llm_json": {"findings": []}}) + "\n")
    cache = run_eval.LLMCache(path)
    assert cache.get("a", "s", "m") is not None
    assert cache.get("a", "s", "m", 0.2, 0) is not None
    assert cache.get("a", "s", "m", 0.0, 0) is None
    assert cache.get("a", "s", "m", 0.2, 1) is None


def test_stage_temperature_defaults_to_the_router_client():
    entries, _ = _pair_entries()
    router = StubRouter(lambda u, n: {"findings": []}, temperature=0.7)
    assert _stage(entries[:1], router)["temperature"] == 0.7
    assert run_eval.router_temperature(SimpleNamespace(clients=[])) == (
        run_eval.settings.LLM_TEMPERATURE)
    run_eval.set_router_temperature(router, 0.0)
    assert router.clients[0].temperature == 0.0


def test_llm_client_sends_the_configured_temperature(monkeypatch):
    sent = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kw):
            sent.append(kw["json"])
            return SimpleNamespace(status_code=200, headers={}, text="",
                                   json=lambda: {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(llm_client.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(llm_client.settings, "LLM_TEMPERATURE", 0.2)
    client = llm_client.LLMClient("groq", model="m")
    asyncio.run(client.complete("sys", "user"))
    monkeypatch.setattr(llm_client.settings, "LLM_TEMPERATURE", 0.5)
    asyncio.run(llm_client.LLMClient("groq", model="m").complete("sys", "user"))
    client.temperature = 0.0  # what the eval does per client
    asyncio.run(client.complete("sys", "user"))
    assert [b["temperature"] for b in sent] == [0.2, 0.5, 0.0]


# --- pinned provider / OpenRouter routing --------------------------------------------


def test_parse_llm_model():
    assert run_eval.parse_llm_model("groq:qwen/qwen3.8-27b") == ("groq", "qwen/qwen3.8-27b")
    assert run_eval.parse_llm_model("openrouter:qwen/qwen3.8-27b:free") == (
        "openrouter", "qwen/qwen3.8-27b:free")
    for bad in ("qwen", "nope:model", "groq:", ":m"):
        with pytest.raises(ValueError):
            run_eval.parse_llm_model(bad)


def test_pinned_router_is_one_client_and_needs_only_its_key(monkeypatch):
    s = llm_client.settings
    monkeypatch.setattr(s, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(s, "OPENROUTER_API_KEY", None)  # the default primary has no key
    monkeypatch.setattr(s, "GROQ_API_KEY", "gsk_test")
    router = run_eval.build_pinned_router("groq:some/model")
    assert [c.label for c in router.clients] == ["groq:some/model"]
    assert run_eval.llm_model_key(router) == "groq:some/model" and not router.mock
    monkeypatch.setattr(s, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError):
        run_eval.build_pinned_router("groq:some/model")


def test_pinned_openrouter_end_to_end_sends_routing_and_logs_upstream(monkeypatch):
    s = llm_client.settings
    monkeypatch.setattr(s, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(s, "LLM_RETRIES", 0)
    bodies = []
    upstreams = iter(["Chutes", "DeepInfra"])

    async def fake_transport(self, request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "provider": next(upstreams), "model": "vendor/model",
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_transport)
    args = run_eval.parse_args([
        "--llm", "--llm-max-calls", "5", "--out", "x.json", "--llm-primary-only",
        "--llm-model", "openrouter:vendor/model:free", "--llm-upstream", "Chutes"])
    router = run_eval.build_eval_router(args)
    routing = run_eval.openrouter_routing(args, router)
    assert routing == {"allow_fallbacks": False, "order": ["Chutes"]}
    entries, _ = _pair_entries()

    async def go():
        with run_eval.HttpUsageTap(
            inject={"provider": routing},
            inject_url_prefix=s.OPENROUTER_BASE_URL.rstrip("/"),
        ) as tap:
            return await run_eval.run_llm_stage(
                entries[:2], router, cache=None, max_calls=5, sleep_s=0, tpm=0,
                progress=False, arm="no_retrieval", tap=tap)

    stage = asyncio.run(go())
    assert [b["provider"] for b in bodies] == [routing, routing]
    assert [b["temperature"] for b in bodies] == [0.0, 0.0]  # the eval default
    assert [b["model"] for b in bodies] == ["vendor/model:free"] * 2
    assert [r["upstream_provider"] for r in stage["records"]] == ["Chutes", "DeepInfra"]
    assert stage["upstream_providers"] == {"Chutes": 1, "DeepInfra": 1}
    assert stage["records"][0]["provider_used"] == "openrouter:vendor/model:free"


def test_tap_injects_only_into_matching_urls():
    seen = {}

    def handler(request):
        seen[request.url.host] = json.loads(request.content)
        return httpx.Response(200, json={"choices": []})

    async def go():
        with run_eval.HttpUsageTap(inject={"provider": {"allow_fallbacks": False}},
                                   inject_url_prefix="https://openrouter.test/api"):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                await c.post("https://openrouter.test/api/chat", json={"model": "m"})
                await c.post("https://groq.test/chat", json={"model": "m"})

    asyncio.run(go())
    assert seen["openrouter.test"] == {"model": "m", "provider": {"allow_fallbacks": False}}
    assert seen["groq.test"] == {"model": "m"}


def test_openrouter_routing_only_for_primary_only_openrouter():
    def args(**kw):
        return SimpleNamespace(**{"llm_primary_only": True, "llm_upstream": None, **kw})

    orr = SimpleNamespace(mock=False, clients=[SimpleNamespace(provider="openrouter")])
    groq = SimpleNamespace(mock=False, clients=[SimpleNamespace(provider="groq")])
    assert run_eval.openrouter_routing(args(), orr) == {"allow_fallbacks": False}
    assert run_eval.openrouter_routing(args(llm_primary_only=False), orr) is None
    assert run_eval.openrouter_routing(args(), groq) is None


def test_llm_option_validation(capsys, monkeypatch):
    base = ["--llm", "--llm-max-calls", "5", "--out", "x.json"]
    for argv, msg in (
        (base + ["--llm-model", "groq:m"], "require --llm-primary-only"),
        (base + ["--llm-primary-only", "--llm-model", "bogus:m"], "--llm-model"),
        (base + ["--llm-primary-only", "--llm-model", "groq:m", "--llm-upstream", "X"],
         "only applies to an OpenRouter"),
        (base + ["--llm-temperature", "3"], "between 0 and 2"),
        (base + ["--llm-repeat", "0"], "--llm-repeat must be"),
        (["--llm-temperature", "0.5"], "requires --llm"),
        (["--llm-repeat", "3"], "requires --llm"),
        (["--rescore", "r.json", *base], "offline"),
        (["--compare", "a.json", "b.json", "--write-baseline"], "offline"),
        (["--localise-tolerance", "-1"], ">= 0"),
    ):
        with pytest.raises(SystemExit):
            run_eval.parse_args(argv)
        assert msg in capsys.readouterr().err, argv
    args = run_eval.parse_args(base)
    assert args.llm_temperature == 0.0 and args.llm_repeat == 1 and args.llm_model is None


# --- split manifests -----------------------------------------------------------------


def _manifest(tmp_path, dev=("a", "b"), test=("c",), version=1):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"version": version, "seed": 7, "dev": {"ids": list(dev)},
                                "test": {"ids": list(test)}, "meta": {}}))
    return path


def test_load_split_and_guards(tmp_path, capsys):
    ids, meta = run_eval.load_split(_manifest(tmp_path), "dev")
    assert ids == {"a", "b"} and meta["seed"] == 7 and meta["n_ids"] == 2
    assert len(meta["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="version"):
        run_eval.load_split(_manifest(tmp_path, version=2), "dev")
    with pytest.raises(ValueError, match="both"):
        run_eval.load_split(_manifest(tmp_path, dev=("a", "c")), "dev")
    items = [{"id": i} for i in ("x", "b", "a")]
    assert run_eval.apply_split(items, {"a", "b", "zz"}) == ([{"id": "b"}, {"id": "a"}], 1)

    path = str(_manifest(tmp_path))
    for argv, msg in (
        (["--split", path, "--split-name", "test"], "--i-know-this-is-the-test-set"),
        (["--split", path], "go together"),
        (["--split-name", "dev"], "go together"),
        (["--split", path, "--split-name", "dev", "--i-know-this-is-the-test-set"],
         "only goes with"),
        (["--split", path, "--split-name", "dev", "--write-baseline"], "full datasets"),
        (["--split", str(tmp_path / "missing.json"), "--split-name", "dev"], "--split"),
    ):
        with pytest.raises(SystemExit):
            run_eval.parse_args(argv)
        assert msg in capsys.readouterr().err, argv
    args = run_eval.parse_args(["--split", path, "--split-name", "test",
                                "--i-know-this-is-the-test-set"])
    assert args.split_ids == {"c"} and args.split_meta["name"] == "test"


class StubEmbedder:
    def embed_text(self, text):
        return text


class StubVectorStore:
    cve_schema_error = None
    cve_collection = "cve_corpus"

    def search_cves(self, query_vector, limit=5, language=None, **_):
        return []

    def count(self, collection):
        return 1


def test_main_split_restricts_items_and_records_it(tmp_path, monkeypatch):
    items = [
        {"id": "P_0000aaaa_vuln", "label": "vulnerable", "code": VULN, "category": "sqli",
         "language": "python"},
        {"id": "P_0000aaaa_safe", "label": "safe", "code": SAFE, "category": "sqli",
         "language": "python"},
        {"id": "o1", "label": "safe", "code": "def g():\n    pass\n", "category": "none",
         "kind": "ordinary"},
        {"id": "o2", "label": "safe", "code": "def h():\n    pass\n", "category": "none",
         "kind": "ordinary"},
    ]
    ds = tmp_path / "ds.jsonl"
    ds.write_text("".join(json.dumps(i) + "\n" for i in items))
    split = _manifest(tmp_path, dev=("P_0000aaaa_vuln", "o1", "not-in-data"),
                      test=("P_0000aaaa_safe", "o2"))
    monkeypatch.setattr(run_eval, "build_components",
                        lambda *a, **k: (StubEmbedder(), StubVectorStore(), None))
    monkeypatch.setattr(run_eval, "build_semgrep_scanner", lambda: None)
    router = StubRouter(lambda u, n: _flag("db.execute('SELECT ' + q)")
                        if "'SELECT ' + q" in u else {"findings": []})
    monkeypatch.setattr(run_eval, "build_llm_router", lambda primary_only=False: router)
    out = tmp_path / "out.json"
    run_eval.main(["--dataset", str(ds), "--out", str(out), "--no-rerank",
                   "--split", str(split), "--split-name", "dev", "--llm",
                   "--llm-max-calls", "10", "--llm-cache", str(tmp_path / "c.jsonl"),
                   "--llm-tpm", "0", "--llm-sleep", "0", "--llm-prompt", "no_retrieval"])
    data = json.loads(out.read_text())
    assert data["split"]["name"] == "dev" and data["split"]["n_items"] == 2
    assert data["split"]["n_ids_not_in_datasets"] == 1
    llm = data["llm"]
    assert [i["id"] for i in llm["items"]] == ["P_0000aaaa_vuln", "o1"]
    # The twin is outside the split, but its diff still localises the vuln item.
    assert llm["items"][0]["fix_lines"] == [3] and llm["items"][0]["localised"] is True
    assert llm["headline"]["tpr_localised"]["rate"] == 1.0
    assert llm["config"]["temperature"] == 0.0 and llm["config"]["localise_tolerance"] == 2
    assert router.clients[0].temperature == 0.0


# --- offline rescore ---------------------------------------------------------------


def _saved_run(items, arm="no_retrieval", model="groq:test-model", **config):
    return {"operating": {}, "llm": {
        "config": {"model": model, "prompt": arm, **config},
        "calls": 3, "realistic_any_finding": {}, "items": items}}


def test_rescore_new_format_needs_no_dataset_or_cache():
    items = [
        {"id": "p_vuln", "kind": "vulnerable", "label": "vulnerable", "prediction": True,
         "any_finding": True, "fix_lines": [5], "findings": [{"line": 6, "end_line": 6}],
         "retrieval_pred": True, "status": "ok"},
        {"id": "p_safe", "kind": "fixed_twin", "label": "safe", "prediction": True,
         "any_finding": True, "fix_lines": [5], "findings": [{"line": 20, "end_line": 21}],
         "retrieval_pred": True, "status": "ok"},
        {"id": "o1", "kind": "ordinary", "label": "safe", "prediction": False,
         "any_finding": False, "findings": [], "retrieval_pred": False, "status": "ok"},
    ]
    block = run_eval.rescore_llm_block(_saved_run(items), tolerance=2)
    assert block["rescore"]["backfilled"] == {} and block["rescore"]["problems"] == {}
    h = block["headline"]
    assert h["tpr_localised"]["rate"] == 1.0 and h["fpr_fixed_twin_localised"]["rate"] == 0.0
    assert h["fpr_fixed_twin"]["rate"] == 1.0
    tight = run_eval.rescore_llm_block(_saved_run(items), tolerance=0)
    assert tight["headline"]["tpr_localised"]["rate"] == 0.0
    assert tight["config"]["localise_tolerance"] == 0
    assert block["calls"] == 3  # untouched fields are kept


def test_rescore_backfills_an_old_run_from_cache_and_dataset(tmp_path):
    """An old-format result (no findings / fix_lines, pre-migration ids)
    rescored from the raw responses in the cache + the migrated dataset."""
    ds = tmp_path / "ds.jsonl"
    ds.write_text("".join(json.dumps(i) + "\n" for i in [
        {"id": "P_0000aaaa_vuln", "label": "vulnerable", "code": VULN, "category": "sqli",
         "language": "python"},
        {"id": "P_0000aaaa_safe", "label": "safe", "code": SAFE, "category": "sqli",
         "language": "python"},
        {"id": "Q_1111bbbb_vuln", "label": "vulnerable", "code": VULN + "# q\n",
         "category": "sqli"},
        {"id": "Q_1111bbbb_safe", "label": "safe", "code": SAFE, "category": "sqli"},
    ]))
    cache = run_eval.LLMCache(tmp_path / "cache.jsonl")
    cache.put({"id": "P_vuln", "prompt_sha256": "s1", "model": "groq:test-model",
               "llm_json": _flag("db.execute('SELECT ' + q)")})  # an old entry (t 0.2)
    cache.put({"id": "P_safe", "prompt_sha256": "s2", "model": "groq:test-model",
               "llm_json": _flag("return x")})
    old = [
        {"id": "P_vuln", "kind": "vulnerable", "label": "vulnerable", "prediction": True,
         "any_finding": True, "validated_count": 1, "prompt_sha256": "s1", "status": "ok",
         "retrieval_pred": True},
        {"id": "P_safe", "kind": "fixed_twin", "label": "safe", "prediction": True,
         "any_finding": True, "validated_count": 1, "prompt_sha256": "s2", "status": "ok",
         "retrieval_pred": True},
        {"id": "Q_vuln", "kind": "vulnerable", "label": "vulnerable", "prediction": True,
         "any_finding": True, "validated_count": 1, "prompt_sha256": "s3", "status": "ok",
         "retrieval_pred": True},  # no cache entry
        {"id": "R_vuln", "kind": "vulnerable", "label": "vulnerable", "prediction": False,
         "any_finding": False, "validated_count": 0, "prompt_sha256": "s4", "status": "ok",
         "retrieval_pred": False},  # not in the dataset
    ]
    data = _saved_run(old)
    block = run_eval.rescore_llm_block(data, dataset_path=[ds],
                                       cache_path=tmp_path / "cache.jsonl", tolerance=0)
    got = {r["id"]: r for r in block["items"]}
    assert got["P_vuln"]["fix_lines"] == [3] and got["P_vuln"]["localised"] is True
    assert got["P_vuln"]["findings"][0]["line"] == 3
    assert got["P_safe"]["localised"] is False  # "return x" is line 4, fix line 3, tol 0
    assert got["Q_vuln"]["localised"] is None  # flagged, response not cached
    assert got["R_vuln"]["localised"] is None  # no twin / fix lines known
    problems = block["rescore"]["problems"]
    assert problems["findings: no cache entry"] == ["Q_vuln"]
    assert problems["fix lines: item missing"] == ["R_vuln"]
    assert block["rescore"]["backfilled"] == {"fix_lines": 3, "findings": 2}
    assert any("2 scored" in n for n in block["rescore"]["not_recomputable"])
    h = block["headline"]
    assert (h["tpr_localised"]["k"], h["tpr_localised"]["n"]) == (1, 1)
    assert h["n_vulnerable_unlocalisable"] == 2
    # A legacy run: nothing localisable, and it says so.
    legacy = run_eval.rescore_llm_block(_saved_run(old, arm=None), dataset_path=[ds])
    assert legacy["config"]["prompt"] == "legacy"
    assert legacy["realistic_localised"] is None
    assert "legacy" in legacy["rescore"]["not_recomputable"][0]


def test_rescore_cli_writes_a_rescorable_file(tmp_path, capsys):
    items = [
        {"id": "p_vuln", "kind": "vulnerable", "label": "vulnerable", "prediction": True,
         "any_finding": True, "fix_lines": [5], "findings": [{"line": 5}],
         "retrieval_pred": True, "status": "ok"},
    ]
    src = tmp_path / "run.json"
    src.write_text(json.dumps(_saved_run(items)))
    out = tmp_path / "rescored.json"
    run_eval.main(["--rescore", str(src), "--out", str(out)])
    assert "TPR vulnerable, localised (primary)  1.000" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["llm"]["rescore"]["offline"] and data["operating"] == {}
    assert run_eval.rescore_llm_block(data)["headline"] == data["llm"]["headline"]


def test_ambiguous_old_ids_are_not_guessed():
    items = [
        {"id": "X_aaaaaaaa_vuln", "label": "vulnerable", "code": "a\nb\n"},
        {"id": "X_aaaaaaaa_safe", "label": "safe", "code": "a\nc\n"},
        {"id": "X_bbbbbbbb_vuln", "label": "vulnerable", "code": "a\nd\n"},
        {"id": "X_bbbbbbbb_safe", "label": "safe", "code": "a\ne\n"},
        {"id": "Y_cccccccc_vuln", "label": "vulnerable", "code": "z\n"},
    ]
    index = run_eval.DatasetIndex(items)
    assert index.find("X_vuln") == (None, "ambiguous")
    assert index.find("Y_vuln")[0]["id"] == "Y_cccccccc_vuln"
    assert index.find("X_aaaaaaaa_vuln")[0]["code"] == "a\nb\n"
    assert index.find("nope_vuln") == (None, "missing")


# --- exact statistics / --compare ---------------------------------------------------


def test_mcnemar_exact_known_tables():
    assert run_eval.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)  # 0.0215
    assert run_eval.mcnemar_exact(0, 5) == pytest.approx(2 / 32)  # 0.0625
    assert run_eval.mcnemar_exact(2, 8) == pytest.approx(2 * 56 / 1024)  # 0.109
    assert run_eval.mcnemar_exact(0, 6) == pytest.approx(2 / 64)  # 0.03125
    assert run_eval.mcnemar_exact(3, 3) == 1.0 and run_eval.mcnemar_exact(0, 0) == 1.0
    assert run_eval.mcnemar_exact(9, 1) == run_eval.mcnemar_exact(1, 9)


def test_clopper_pearson_known_values():
    assert run_eval.clopper_pearson(0, 10)[1] == pytest.approx(1 - 0.025 ** 0.1, abs=1e-4)
    assert run_eval.clopper_pearson(10, 10)[0] == pytest.approx(0.025 ** 0.1, abs=1e-4)
    assert run_eval.clopper_pearson(1, 8) == pytest.approx([0.0032, 0.5265], abs=1e-4)
    assert run_eval.clopper_pearson(0, 0) is None
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((3, 20), (17, 500), (250, 1000)):
        lo, hi = run_eval.clopper_pearson(k, n)
        assert lo == pytest.approx(stats.beta.ppf(0.025, k, n - k + 1), abs=1e-4)
        assert hi == pytest.approx(stats.beta.ppf(0.975, k + 1, n - k), abs=1e-4)
    assert math.isclose(run_eval._binom_cdf(2, 4, 0.5), 11 / 16)


def test_compare_items_paired_counts():
    a = [_r(f"v{i}_vuln", "vulnerable", True, loc) for i, loc in enumerate(
        [True, True, True, False, None])]
    b = [_r(f"v{i}_vuln", "vulnerable", True, loc) for i, loc in enumerate(
        [True, False, False, True, True])]
    a += [_r("t1_safe", "fixed_twin", True, None), _r("t2_safe", "fixed_twin", False, None)]
    b += [_r("t1_safe", "fixed_twin", False, None), _r("t2_safe", "fixed_twin", False, None)]
    a += [_r(f"o{i}", "ordinary", i < 2, None) for i in range(10)]
    b += [_r(f"o{i}", "ordinary", i < 1, None) for i in range(10)] + [
        _r("extra", "ordinary", True, None)]
    cmp = run_eval.compare_items(a, b)
    loc = cmp["vulnerable_localised_tp"]
    assert (loc["n"], loc["both"], loc["a_only"], loc["b_only"], loc["neither"]) == (
        4, 1, 2, 1, 0)
    assert loc["p_mcnemar_exact"] == pytest.approx(1.0)
    twin = cmp["fixed_twin_fp"]
    assert (twin["a_only"], twin["b_only"], twin["n"]) == (1, 0, 2)
    o = cmp["ordinary_fpr"]
    assert (o["a"]["k"], o["a"]["n"], o["b"]["k"], o["b"]["n"]) == (2, 10, 2, 11)
    assert o["a"]["ci95_exact"] == run_eval.clopper_pearson(2, 10)
    assert (o["paired"]["n"], o["paired"]["a_only"]) == (10, 1)
    assert (cmp["n_common_ids"], cmp["n_only_in_b"]) == (17, 1)


def test_compare_cli(tmp_path, capsys):
    def run(flags):
        return _saved_run([
            {"id": f"v{i}_vuln", "kind": "vulnerable", "label": "vulnerable",
             "prediction": f, "any_finding": f, "fix_lines": [1],
             "findings": [{"line": 1}] if f else [], "retrieval_pred": True, "status": "ok"}
            for i, f in enumerate(flags)])

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run([True] * 6 + [False] * 4)))
    b.write_text(json.dumps(run([False] * 10)))
    out = tmp_path / "cmp.json"
    run_eval.main(["--compare", str(a), str(b), "--out", str(out)])
    text = capsys.readouterr().out
    assert "A only 6, B only 0" in text and "p = 0.03125" in text
    data = json.loads(out.read_text())
    assert data["vulnerable_localised_tp"]["p_mcnemar_exact"] == pytest.approx(0.03125)
