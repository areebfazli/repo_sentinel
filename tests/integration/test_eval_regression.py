"""Regression guard: detection F1 at the configured thresholds must not drop
below the committed baseline (minus a small tolerance).

Marked ``slow`` — it loads the embedding + reranker models and needs the
cve_corpus collection seeded (run scripts/ingest_cve_corpus.py) with the API
stopped (local Qdrant is single-process). Skips cleanly otherwise.
"""
import json

import pytest

from backend.app.config import BASE_DIR
from ml.evaluation import run_eval

BASELINE_PATH = BASE_DIR / "ml" / "evaluation" / "baseline.json"
F1_TOLERANCE = 0.02


@pytest.mark.slow
@pytest.mark.integration
def test_detection_f1_not_below_baseline():
    if not BASELINE_PATH.exists():
        pytest.skip("no baseline.json — run run_eval.py --write-baseline first")
    baseline = json.loads(BASELINE_PATH.read_text())

    try:
        embedder, store, reranker = run_eval.build_components()
    except Exception as exc:  # qdrant lock held (API running) or models missing
        pytest.skip(f"components unavailable: {exc}")

    if store.count(store.cve_collection) == 0:
        pytest.skip("cve_corpus not seeded — run scripts/ingest_cve_corpus.py")

    metrics = run_eval.evaluate(embedder, store, reranker)
    assert metrics["f1"] >= baseline["f1"] - F1_TOLERANCE, (
        f"F1 regressed: {metrics['f1']} < baseline {baseline['f1']} - {F1_TOLERANCE}"
    )
