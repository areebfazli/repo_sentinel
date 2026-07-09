"""Unit tests for files-mode finding dedup."""
from backend.app.core.rag_merger import _dedupe


def _f(fn, score):
    return {
        "point_id": "p1",
        "anchor_file_path": "a.py",
        "anchor_function_name": fn,
        "adjusted_score": score,
    }


def test_dedupe_keeps_same_point_in_different_functions():
    # duplicate of f1 with a lower score -> dropped; f1 and f2 both kept.
    out = _dedupe([_f("f1", 0.9), _f("f2", 0.8), _f("f1", 0.5)])
    assert sorted(f["anchor_function_name"] for f in out) == ["f1", "f2"]
    f1 = next(f for f in out if f["anchor_function_name"] == "f1")
    assert f1["adjusted_score"] == 0.9  # higher-scoring f1 kept


def test_dedupe_collapses_true_duplicates():
    out = _dedupe([_f("f1", 0.4), _f("f1", 0.9)])
    assert len(out) == 1
    assert out[0]["adjusted_score"] == 0.9
