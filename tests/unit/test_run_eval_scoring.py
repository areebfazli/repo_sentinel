"""Unit tests for the eval harness's in-memory scoring (no models, no Qdrant)."""
import hashlib

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
