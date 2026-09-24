"""Regression guard: detection F1 (and category hit rate) at the configured
thresholds must not drop below the committed baseline (minus a small tolerance).

Marked ``slow`` — it loads the embedding model (plus the reranker, if the
baseline was recorded with one) and needs the cve_corpus collection seeded (run
scripts/ingest_cve_corpus.py) with the API stopped (local Qdrant is
single-process). Skips cleanly otherwise.

The run mirrors what baseline.json recorded: the same dataset files (checked by
sha256) and the same reranker (``"reranker": null`` = off), so a baseline
written with the reranker off is never compared against a reranked run or vice
versa, whatever RERANKER_ENABLED says today.
"""
import json

import pytest

from backend.app.config import BASE_DIR
from ml.evaluation import run_eval

BASELINE_PATH = BASE_DIR / "ml" / "evaluation" / "baseline.json"
F1_TOLERANCE = 0.02
CATEGORY_HIT_TOLERANCE = 0.02


def _baseline_components(baseline: dict):
    """build_components() with the reranker the baseline was recorded with.
    Baselines older than the ``reranker`` key fall back to the settings."""
    if "reranker" not in baseline:
        return run_eval.build_components()
    cfg = baseline["reranker"]
    if cfg is None:
        return run_eval.build_components(no_rerank=True)
    return run_eval.build_components(
        no_rerank=False, reranker_model=cfg["model"], reranker_max_tokens=cfg["max_tokens"]
    )


@pytest.mark.slow
@pytest.mark.integration
def test_detection_f1_not_below_baseline():
    if not BASELINE_PATH.exists():
        pytest.skip("no baseline.json — run run_eval.py --write-baseline first")
    baseline = json.loads(BASELINE_PATH.read_text())

    datasets = [BASE_DIR / p for p in baseline.get("datasets", [])] or [run_eval.DEFAULT_DATASET]
    missing = [str(p) for p in datasets if not p.exists()]
    if missing:
        pytest.skip(f"baseline eval set(s) missing: {missing}")
    assert run_eval.dataset_sha256(datasets) == baseline["dataset_sha256"], (
        "eval set changed since the baseline was written — re-run run_eval.py "
        "--write-baseline on the same --dataset files"
    )

    try:
        embedder, store, reranker = _baseline_components(baseline)
    except Exception as exc:  # qdrant lock held (API running) or models missing
        pytest.skip(f"components unavailable: {exc}")

    if store.cve_schema_error:
        pytest.skip(store.cve_schema_error)
    if store.count(store.cve_collection) == 0:
        pytest.skip("cve_corpus not seeded — run scripts/ingest_cve_corpus.py")

    metrics = run_eval.evaluate(embedder, store, reranker, dataset_path=datasets)
    assert metrics["n"] == baseline["n"]
    assert metrics["f1"] >= baseline["f1"] - F1_TOLERANCE, (
        f"F1 regressed: {metrics['f1']} < baseline {baseline['f1']} - {F1_TOLERANCE}"
    )
    if "category_hit_rate" in baseline:
        assert metrics["category_hit_rate"] >= (
            baseline["category_hit_rate"] - CATEGORY_HIT_TOLERANCE
        ), (
            f"category hit rate regressed: {metrics['category_hit_rate']} < baseline "
            f"{baseline['category_hit_rate']} - {CATEGORY_HIT_TOLERANCE}"
        )
