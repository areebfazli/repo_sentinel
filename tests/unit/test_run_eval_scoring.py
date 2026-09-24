"""Unit tests for the eval harness's in-memory scoring (no models, no Qdrant)."""
import hashlib
import json

import pytest

from backend.app.core.reranker import _sigmoid
from ml.evaluation import run_eval


def _cand(category, sim, prob, margin=None):
    return {
        "similarity_score": sim,
        "rerank_prob": prob,
        "category": category,
        "cve_id": f"CVE-{category}",
        "sim_fixed": None if margin is None else sim - margin,
        "twin_margin": margin,
    }


def _item(is_vuln, category, candidates):
    return {"id": "x", "is_vulnerable": is_vuln, "category": category, "candidates": candidates}


GATHERED = [
    # Vulnerable, top candidate's twin margin is clearly positive.
    _item(True, "sqli", [_cand("sqli", 0.8, 0.7, margin=0.10)]),
    # Safe twin: looks more like the fix -> only the margin gate rejects it.
    _item(False, "sqli", [_cand("sqli", 0.8, 0.7, margin=-0.05)]),
    # Vulnerable, matched entry has no twin (handwritten) -> never margin-gated.
    _item(True, "xss", [_cand("xss", 0.6, 0.6)]),
    # Safe, no candidates at all.
    _item(False, "cmd", []),
]


def test_score_without_margin_matches_legacy_rule():
    m = run_eval.score(GATHERED, 0.5, 0.0)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 1)
    assert m["margin_threshold"] is None
    assert m["category_hit_rate"] == 1.0


def test_score_margin_gate_removes_fix_lookalike():
    m = run_eval.score(GATHERED, 0.5, 0.0, margin_t=0.0)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    # Too strict a margin starts costing recall on the twinned vulnerable item.
    strict = run_eval.score(GATHERED, 0.5, 0.0, margin_t=0.15)
    assert (strict["tp"], strict["fn"]) == (1, 1)


def test_margin_falls_back_to_next_best_candidate():
    # The best-reranked candidate is gated; the next survivor decides the prediction.
    g = [_item(True, "sqli", [
        _cand("xss", 0.8, 0.9, margin=-0.2),
        _cand("sqli", 0.8, 0.5, margin=0.1),
    ])]
    assert run_eval.score(g, 0.5, 0.0)["category_hit_rate"] == 0.0
    assert run_eval.score(g, 0.5, 0.0, margin_t=0.0)["category_hit_rate"] == 1.0


def test_twin_coverage_counts_items_whose_top_candidate_has_a_twin():
    m = run_eval.score(GATHERED, 0.5, 0.0)
    assert m["twin_coverage"] == 0.5  # items 1 and 2 of 4
    # Measured before the margin gate, so it doesn't move with margin_t.
    assert run_eval.score(GATHERED, 0.5, 0.0, margin_t=0.15)["twin_coverage"] == 0.5
    # ...but it does follow the similarity gate.
    assert run_eval.score(GATHERED, 0.7, 0.0)["twin_coverage"] == 0.5
    assert run_eval.score(GATHERED, 0.9, 0.0)["twin_coverage"] == 0.0


def test_datasets_combine_and_hash_backward_compatibly(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"id": "1"}\n\n{"id": "2"}\n')
    b.write_text('{"id": "3"}\n')

    assert [i["id"] for i in run_eval.load_datasets([a, b])] == ["1", "2", "3"]
    assert [i["id"] for i in run_eval.load_datasets(a)] == ["1", "2"]
    # One file hashes exactly as the old single-path baseline did.
    assert run_eval.dataset_sha256([a]) == hashlib.sha256(a.read_bytes()).hexdigest()
    assert run_eval.dataset_sha256(str(a)) == run_eval.dataset_sha256([a])
    assert run_eval.dataset_sha256([a, b]) != run_eval.dataset_sha256([b, a])


def test_frange_accepts_negative_start():
    assert run_eval.frange("-0.10:0.20:0.05") == [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]


# --- --no-rerank / --sample / timing / reranker overrides -------------------


class StubEmbedder:
    """The "vector" is the code itself, so the stub store can key on it."""

    def embed_text(self, text):
        return text


class StubVectorStore:
    """Returns fresh copies of preset candidates keyed by the query's code."""

    cve_schema_error = None
    cve_collection = "cve_corpus"

    def __init__(self, by_code):
        self._by_code = by_code

    def search_cves(self, query_vector, limit=5, language=None):
        return [dict(c) for c in self._by_code.get(query_vector, [])][:limit]

    def count(self, collection):
        return 1


class StubReranker:
    """Sets rerank_score (a raw logit) and rerank_prob = sigmoid(logit) from a
    cve_id lookup, like the real one."""

    def __init__(self, logits):
        self._logits = logits
        self.calls = 0

    def rerank(self, query_code, candidates, top_k=1):
        self.calls += 1
        for c in candidates:
            c["rerank_score"] = self._logits.get(c["cve_id"], 0.0)
            c["rerank_prob"] = _sigmoid(c["rerank_score"])
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


def _raw(cve_id, category, sim):
    return {"cve_id": cve_id, "category": category, "similarity_score": sim,
            "vulnerable_code": f"code-{cve_id}"}


def _ds_item(item_id, label, code, category="sqli"):
    return {"id": item_id, "label": label, "category": category, "code": code}


def test_no_rerank_scoring_ranks_by_similarity():
    # rerank_prob == similarity_score, as gather() produces with reranker=None.
    g = [
        # Right category has the higher similarity -> category hit.
        _item(True, "sqli", [_cand("xss", 0.6, 0.6), _cand("sqli", 0.9, 0.9)]),
        # Wrong category wins on similarity -> TP without a category hit.
        _item(True, "cmd", [_cand("xss", 0.8, 0.8), _cand("cmd", 0.7, 0.7)]),
        # Safe item whose only candidate is below the sim gate -> TN.
        _item(False, "sqli", [_cand("sqli", 0.3, 0.3)]),
        # Safe item above the gate -> FP.
        _item(False, "xss", [_cand("xss", 0.55, 0.55)]),
    ]
    m = run_eval.score(g, 0.5, run_eval.NO_RERANK_THRESHOLD)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 1)
    assert m["precision"] == round(2 / 3, 4) and m["recall"] == 1.0
    assert m["category_hit_rate"] == 0.5


def test_gather_without_reranker_uses_similarity_as_prob():
    store = StubVectorStore({"a": [_raw("CVE-1", "xss", 0.6), _raw("CVE-2", "sqli", 0.9)]})
    gathered, timing = run_eval.gather(
        [_ds_item("a_vuln", "vulnerable", "a")], StubEmbedder(), store, None, 10
    )
    cands = gathered[0]["candidates"]
    assert [c["rerank_prob"] for c in cands] == [c["similarity_score"] for c in cands]
    assert run_eval.score(gathered, 0.5, 0.0)["category_hit_rate"] == 1.0
    assert timing["rerank_seconds"] == 0.0


def test_build_components_no_rerank_never_constructs_reranker(monkeypatch):
    built = []

    class _Tracking:
        def __init__(self, **kwargs):
            built.append(kwargs)

    def _boom(**kwargs):
        raise AssertionError("Reranker must not be constructed with no_rerank=True")

    monkeypatch.setattr(run_eval, "Embedder", lambda cache=None: "embedder")
    monkeypatch.setattr(run_eval, "EmbeddingCache", lambda: None)
    monkeypatch.setattr(run_eval, "VectorStore", lambda: "store")

    monkeypatch.setattr(run_eval, "Reranker", _boom)
    assert run_eval.build_components(no_rerank=True) == ("embedder", "store", None)
    # Default follows RERANKER_ENABLED (off): still never constructed.
    monkeypatch.setattr(run_eval.settings, "RERANKER_ENABLED", False)
    assert run_eval.build_components() == ("embedder", "store", None)

    monkeypatch.setattr(run_eval, "Reranker", _Tracking)
    _, _, reranker = run_eval.build_components(no_rerank=False)
    assert isinstance(reranker, _Tracking) and len(built) == 1
    # ...and with the setting on, the default builds one.
    monkeypatch.setattr(run_eval.settings, "RERANKER_ENABLED", True)
    _, _, reranker = run_eval.build_components()
    assert isinstance(reranker, _Tracking) and len(built) == 2


def test_reranker_overrides_are_passed_through(monkeypatch):
    from backend.app.config import settings

    before = (settings.RERANKER_MODEL, settings.RERANKER_MAX_TOKENS)
    captured = []
    monkeypatch.setattr(run_eval, "Embedder", lambda cache=None: "embedder")
    monkeypatch.setattr(run_eval, "EmbeddingCache", lambda: None)
    monkeypatch.setattr(run_eval, "VectorStore", lambda: "store")
    monkeypatch.setattr(run_eval, "Reranker", lambda **kw: captured.append(kw) or "rr")
    monkeypatch.setattr(settings, "RERANKER_ENABLED", False)

    # An override implies a reranker even with RERANKER_ENABLED off.
    run_eval.build_components(reranker_model="BAAI/bge-reranker-base", reranker_max_tokens=512)
    run_eval.build_components(reranker_max_tokens=128)
    run_eval.build_components(no_rerank=False)
    assert captured == [
        {"model_name": "BAAI/bge-reranker-base", "max_tokens": 512},
        {"model_name": before[0], "max_tokens": 128},
        {"model_name": before[0], "max_tokens": before[1]},
    ]
    # What output JSON records matches what was constructed.
    assert run_eval.reranker_config(False, "BAAI/bge-reranker-base", 512) == {
        "model": "BAAI/bge-reranker-base", "max_tokens": 512,
    }
    assert run_eval.reranker_config(True) is None
    assert run_eval.reranker_config() is None  # follows RERANKER_ENABLED (off)
    # Overrides go through constructor args; the settings singleton is untouched.
    assert (settings.RERANKER_MODEL, settings.RERANKER_MAX_TOKENS) == before


def _sample_pool():
    items = []
    for i in range(20):  # OSV-style pairs
        items.append({"id": f"CVE-{i}_fn_vuln", "label": "vulnerable"})
        items.append({"id": f"CVE-{i}_fn_safe", "label": "safe"})
    for i in range(7):  # handwritten-style singletons (no pairing suffix)
        items.append({"id": f"sqli_vuln_{i}", "label": "vulnerable"})
    for i in range(5):
        items.append({"id": f"sqli_safe_{i}", "label": "safe"})
    return items


def test_sample_is_deterministic_balanced_and_keeps_pairs():
    pool = _sample_pool()
    for n in (1, 7, 10, 25, 30):
        a = [i["id"] for i in run_eval.sample_items(pool, n, seed=42)]
        assert a == [i["id"] for i in run_eval.sample_items(pool, n, seed=42)]
        assert n - 1 <= len(a) <= n
        ids = set(a)
        for item_id in ids:
            for mine, other in (("_vuln", "_safe"), ("_safe", "_vuln")):
                if item_id.endswith(mine):
                    assert item_id[: -len(mine)] + other in ids, (n, item_id)
        labels = {i["id"]: i["label"] for i in pool}
        n_vuln = sum(labels[x] == "vulnerable" for x in a)
        assert abs(n_vuln - (len(a) - n_vuln)) <= 1

    samples = {
        tuple(i["id"] for i in run_eval.sample_items(pool, 10, seed=s)) for s in range(5)
    }
    assert len(samples) > 1  # the seed matters
    # n >= pool -> whole pool, unchanged.
    assert run_eval.sample_items(pool, len(pool) + 5, seed=1) == pool


def test_write_baseline_with_sample_is_refused(capsys):
    with pytest.raises(SystemExit):
        run_eval.parse_args(["--sample", "150", "--write-baseline"])
    assert "--write-baseline cannot be combined with --sample" in capsys.readouterr().err


def test_no_rerank_arg_validation(capsys, monkeypatch):
    monkeypatch.setattr(run_eval.settings, "RERANKER_ENABLED", False)
    args = run_eval.parse_args(["--no-rerank"])
    assert args.no_rerank and args.rerank_sweep is None
    assert run_eval.parse_args(["--rerank"]).rerank_sweep == run_eval.DEFAULT_RERANK_SWEEP
    for bad in (
        ["--no-rerank", "--rerank-sweep", "0.0:0.5:0.1"],
        ["--no-rerank", "--reranker-model", "BAAI/bge-reranker-base"],
        ["--no-rerank", "--reranker-max-tokens", "512"],
        ["--no-rerank", "--rerank"],
    ):
        with pytest.raises(SystemExit):
            run_eval.parse_args(bad)
    assert "--no-rerank" in capsys.readouterr().err
    # With the setting off, a rerank sweep needs an explicit --rerank.
    with pytest.raises(SystemExit):
        run_eval.parse_args(["--rerank-sweep", "0.0:0.5:0.1"])
    assert "--rerank" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("enabled", "argv", "expect_rerank"),
    [
        (False, [], False),
        (True, [], True),
        (False, ["--rerank"], True),
        (True, ["--no-rerank"], False),
        (False, ["--reranker-model", "BAAI/bge-reranker-base"], True),
        (False, ["--reranker-max-tokens", "256"], True),
    ],
)
def test_default_reranker_follows_setting(monkeypatch, enabled, argv, expect_rerank):
    monkeypatch.setattr(run_eval.settings, "RERANKER_ENABLED", enabled)
    args = run_eval.parse_args(argv)
    assert args.no_rerank is (not expect_rerank)
    assert (args.rerank_sweep == run_eval.DEFAULT_RERANK_SWEEP) is expect_rerank
    cfg = run_eval.reranker_config(
        args.no_rerank, args.reranker_model, args.reranker_max_tokens
    )
    assert (cfg is not None) is expect_rerank


def test_default_sim_sweep_contains_operating_point():
    from backend.app.config import settings

    args = run_eval.parse_args([])
    assert args.sim_sweep == run_eval.DEFAULT_SIM_SWEEP
    values = run_eval.frange(args.sim_sweep)
    assert 0.25 in values
    assert settings.SIM_THRESHOLD_CVE in values


def test_main_sweep_always_includes_operating_point(tmp_path, monkeypatch):
    ds = tmp_path / "ds.jsonl"
    _write_jsonl(ds, [_ds_item("x_vuln", "vulnerable", "a")])
    monkeypatch.setattr(
        run_eval, "build_components",
        lambda *a, **k: (StubEmbedder(), StubVectorStore({"a": [_raw("C", "sqli", 0.9)]}), None),
    )
    out = tmp_path / "out.json"
    run_eval.main(["--dataset", str(ds), "--out", str(out), "--sim-sweep", "0.6:0.8:0.1"])
    sims = {r["sim_threshold"] for r in json.loads(out.read_text())["sweep"]}
    assert run_eval.settings.SIM_THRESHOLD_CVE in sims and 0.6 in sims


def test_evaluate_ignores_rerank_threshold_without_reranker(tmp_path, monkeypatch):
    # RERANK_THRESHOLD gates cross-encoder probabilities; without a reranker
    # rerank_prob is the similarity, so applying it would be a second sim gate.
    ds = tmp_path / "ds.jsonl"
    _write_jsonl(ds, [_ds_item("x_vuln", "vulnerable", "a")])
    store = StubVectorStore({"a": [_raw("C", "sqli", 0.4)]})
    monkeypatch.setattr(run_eval.settings, "SIM_THRESHOLD_CVE", 0.25)
    monkeypatch.setattr(run_eval.settings, "RERANK_THRESHOLD", 0.9)
    m = run_eval.evaluate(StubEmbedder(), store, None, dataset_path=ds, margin_t=None)
    assert m["rerank_threshold"] == run_eval.NO_RERANK_THRESHOLD
    assert m["tp"] == 1


def test_gather_reports_timing(capsys):
    store = StubVectorStore({
        "a": [_raw("CVE-1", "sqli", 0.9)],
        "b": [],
    })
    reranker = StubReranker({"CVE-1": 2.0})
    items = [_ds_item("a_vuln", "vulnerable", "a"), _ds_item("b_safe", "safe", "b")]
    gathered, timing = run_eval.gather(
        items, StubEmbedder(), store, reranker, 10, progress=True
    )
    assert set(timing) == {
        "n_items", "total_seconds", "seconds_per_item",
        "embed_seconds", "ann_seconds", "rerank_seconds",
    }
    assert timing["n_items"] == 2
    assert all(timing[k] >= 0 for k in timing)
    assert timing["total_seconds"] >= timing["embed_seconds"] + timing["ann_seconds"] - 1e-3
    assert reranker.calls == 1  # empty candidate list skips the reranker
    assert gathered[0]["candidates"][0]["rerank_prob"] == _sigmoid(2.0)
    out = capsys.readouterr().out
    assert "[1/2]" in out and "[2/2]" in out

    _, empty = run_eval.gather([], StubEmbedder(), store, None, 10)
    assert empty["n_items"] == 0 and empty["seconds_per_item"] == 0.0


def test_gather_does_not_double_apply_sigmoid(monkeypatch):
    """gather() must pass the Reranker's rerank_prob through unchanged: the real
    Reranker (stubbed CrossEncoder emitting logits) already applied sigmoid."""
    import torch

    from backend.app.core import reranker as reranker_mod

    logits = {"code-CVE-1": -5.0, "code-CVE-2": 3.0}

    class _CE:
        def __init__(self, *a, **kw):
            pass

        def predict(self, pairs, batch_size=32, activation_fn=None, **kw):
            raw = torch.tensor([logits[d] for _, d in pairs])
            return (activation_fn or torch.nn.Sigmoid())(raw).numpy()

    monkeypatch.setattr(reranker_mod, "CrossEncoder", _CE)
    real = reranker_mod.Reranker(model_name="stub", max_tokens=8)
    store = StubVectorStore({"a": [_raw("CVE-1", "xss", 0.9), _raw("CVE-2", "sqli", 0.8)]})
    gathered, _ = run_eval.gather(
        [_ds_item("a_vuln", "vulnerable", "a")], StubEmbedder(), store, real, 10
    )
    probs = {c["cve_id"]: c["rerank_prob"] for c in gathered[0]["candidates"]}
    assert probs["CVE-1"] == pytest.approx(_sigmoid(-5.0))
    assert probs["CVE-2"] == pytest.approx(_sigmoid(3.0))
    assert probs["CVE-1"] < 0.5  # the double sigmoid floored this at 0.5
    assert probs["CVE-2"] != pytest.approx(_sigmoid(_sigmoid(3.0)))
    assert not hasattr(run_eval, "sigmoid")


def _write_jsonl(path, items):
    path.write_text("".join(json.dumps(i) + "\n" for i in items))


@pytest.mark.parametrize("no_rerank", [True, False])
def test_main_out_json_is_self_describing(tmp_path, monkeypatch, no_rerank):
    ds = tmp_path / "ds.jsonl"
    _write_jsonl(ds, [
        _ds_item("CVE-1_f_vuln", "vulnerable", "a"),
        _ds_item("CVE-1_f_safe", "safe", "b"),
        _ds_item("sqli_vuln_1", "vulnerable", "c"),
        _ds_item("sqli_safe_1", "safe", "d"),
    ])
    store = StubVectorStore({"a": [_raw("CVE-1", "sqli", 0.9)], "c": [_raw("CVE-2", "sqli", 0.8)]})
    built = []

    def fake_build(no_rerank=False, reranker_model=None, reranker_max_tokens=None):
        built.append((no_rerank, reranker_model, reranker_max_tokens))
        return StubEmbedder(), store, None if no_rerank else StubReranker({"CVE-1": 1.0})

    monkeypatch.setattr(run_eval, "build_components", fake_build)
    out = tmp_path / "out.json"
    argv = ["--dataset", str(ds), "--out", str(out), "--sim-sweep", "0.5:0.5:0.1",
            "--sample", "2", "--seed", "7"]
    if no_rerank:
        argv += ["--no-rerank"]
    else:
        argv += ["--reranker-model", "m", "--reranker-max-tokens", "64"]
    run_eval.main(argv)

    data = json.loads(out.read_text())
    assert data["sample"] == 2 and data["seed"] == 7
    assert data["operating"]["n"] == 2
    assert {"total_seconds", "seconds_per_item", "embed_seconds", "ann_seconds",
            "rerank_seconds"} <= set(data["timing"])
    if no_rerank:
        assert built == [(True, None, None)]
        assert data["reranker"] is None
        assert {r["rerank_threshold"] for r in data["sweep"]} == {run_eval.NO_RERANK_THRESHOLD}
    else:
        assert data["reranker"] == {"model": "m", "max_tokens": 64}
        assert len({r["rerank_threshold"] for r in data["sweep"]}) == 10


def test_main_out_json_without_sample_records_null(tmp_path, monkeypatch):
    ds = tmp_path / "ds.jsonl"
    _write_jsonl(ds, [_ds_item("x", "vulnerable", "a")])
    monkeypatch.setattr(
        run_eval, "build_components",
        lambda *a, **k: (StubEmbedder(), StubVectorStore({}), None),
    )
    out = tmp_path / "out.json"
    run_eval.main(["--dataset", str(ds), "--out", str(out), "--no-rerank"])
    data = json.loads(out.read_text())
    assert data["sample"] is None and data["seed"] is None


def test_pair_group_keys_never_merge_pairs_that_share_an_id():
    items = [{"id": "A_f_vuln", "label": "vulnerable"}, {"id": "A_f_safe", "label": "safe"},
             {"id": "A_f_vuln", "label": "vulnerable"}, {"id": "A_f_safe", "label": "safe"},
             {"id": "ord_1", "label": "safe", "kind": "ordinary"}]
    assert run_eval.pair_group_keys(items) == ["A_f", "A_f", "A_f#2", "A_f#2", "ord_1"]
    recs = [{"id": i["id"], "kind": run_eval.item_kind(i), "label": i["label"], "pred": p}
            for i, p in zip(items[:4], [True, False, False, True], strict=True)]
    pairs = run_eval.realistic_metrics(recs)["pairs"]
    assert pairs["n"] == 2 and pairs["n_vuln_only"] == 1 and pairs["n_twin_only"] == 1
    # Sampling keeps each pair whole and never glues two of them together.
    picked = run_eval.sample_items(items[:4], 2, seed=0)
    assert [i["label"] for i in picked] == ["vulnerable", "safe"]


def test_eval_dataset_ids_are_unique():
    from pathlib import Path

    for name in ("detection_eval_osv_pypi.jsonl", "detection_eval_osv_npm.jsonl"):
        items = run_eval.load_dataset(Path(run_eval.__file__).parent / "datasets" / name)
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids)), name
        keys = run_eval.pair_group_keys(items)
        assert all(keys.count(k) == 2 for k in keys), name  # every item in exactly one pair
