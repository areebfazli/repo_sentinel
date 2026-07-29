"""Unit tests for files-mode finding dedup and batched feedback lookup."""
from backend.app.core import rag_merger
from backend.app.core.rag_merger import RagMerger, _dedupe


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


class _StubCVE:
    """Returns preset reranked candidates keyed by unit code; no models."""

    def __init__(self, by_code):
        self._by_code = by_code

    def rerank_candidates(self, code, language=None, query_vector=None):
        return [dict(m) for m in self._by_code.get(code, [])]

    @staticmethod
    def base_score(m):
        return m.get("rerank_prob", 0.0)


class _StubTeam:
    def rerank_candidates(self, code, language=None, query_vector=None):
        return []

    def base_score_factory(self):
        return lambda m: m.get("rerank_prob", 0.0)


def _unit(i):
    return {
        "code": f"u{i}",
        "file_path": f"f{i}.py",
        "start_line": 1,
        "end_line": 2,
        "function_name": f"fn{i}",
        "language": "python",
    }


def test_analyze_units_batches_feedback_into_one_query(monkeypatch):
    # Feedback votes must be fetched ONCE for the whole scan, not per unit.
    calls = {"n": 0}

    def counting_get_net_votes(ids):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(rag_merger.feedback_store, "get_net_votes", counting_get_net_votes)
    monkeypatch.setattr(rag_merger.settings, "RERANK_THRESHOLD", 0.0)
    monkeypatch.setattr(rag_merger.settings, "FEEDBACK_SUPPRESS_NET", -2)
    monkeypatch.setattr(rag_merger.settings, "FEEDBACK_DOWNWEIGHT_PER_VOTE", 0.15)
    monkeypatch.setattr(rag_merger.settings, "RETRIEVAL_TOP_K", 5)

    merger = object.__new__(RagMerger)  # skip model loading
    merger.cve_retriever = _StubCVE({
        f"u{i}": [{"point_id": f"p{i}", "rerank_prob": 0.9, "cve_id": f"p{i}"}]
        for i in range(3)
    })
    merger.team_retriever = _StubTeam()

    units = [_unit(i) for i in range(3)]
    vectors = [[0.0] * 768 for _ in range(3)]
    out = merger._analyze_units_sync(units, vectors)

    assert calls["n"] == 1  # one feedback round-trip for all 3 units
    # Every finding survives and is anchored to its own unit's function.
    anchors = {f["cve_id"]: f["anchor_function_name"] for f in out["ghost_hunter_findings"]}
    assert anchors == {"p0": "fn0", "p1": "fn1", "p2": "fn2"}


def test_analyze_units_applies_batched_suppression(monkeypatch):
    # A point downvoted past the threshold is suppressed even via the shared map.
    monkeypatch.setattr(
        rag_merger.feedback_store, "get_net_votes", lambda ids: {"p1": -2}
    )
    monkeypatch.setattr(rag_merger.settings, "RERANK_THRESHOLD", 0.0)
    monkeypatch.setattr(rag_merger.settings, "FEEDBACK_SUPPRESS_NET", -2)
    monkeypatch.setattr(rag_merger.settings, "FEEDBACK_DOWNWEIGHT_PER_VOTE", 0.15)
    monkeypatch.setattr(rag_merger.settings, "RETRIEVAL_TOP_K", 5)

    merger = object.__new__(RagMerger)
    merger.cve_retriever = _StubCVE({
        f"u{i}": [{"point_id": f"p{i}", "rerank_prob": 0.9, "cve_id": f"p{i}"}]
        for i in range(3)
    })
    merger.team_retriever = _StubTeam()

    out = merger._analyze_units_sync([_unit(i) for i in range(3)], [[0.0] * 768] * 3)
    found = {f["cve_id"] for f in out["ghost_hunter_findings"]}
    assert found == {"p0", "p2"}  # p1 suppressed by the shared vote map
