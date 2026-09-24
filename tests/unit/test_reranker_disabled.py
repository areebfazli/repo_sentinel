"""RERANKER_ENABLED=false (the default): no cross-encoder is built, retrievers
keep similarity order, rerank_prob stays absent (never fabricated), scoring falls
back to similarity via ``scoring.relevance`` and RERANK_THRESHOLD is ignored.
Stub components only — no model loads."""
import json
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.config import settings
from backend.app.core import feedback_store, rag_merger
from backend.app.core.cve_retriever import CVERetriever
from backend.app.core.rag_merger import RagMerger
from backend.app.core.scoring import compute_adjusted_score, relevance
from backend.app.core.team_retriever import TeamRetriever


class StubEmbedder:
    def __init__(self, cache=None):
        pass

    def embed_text(self, text):
        return [0.0] * 4

    def embed_texts(self, texts):
        return [[0.0] * 4 for _ in texts]


class StubStore:
    """search_cves / search_team_history return fresh copies of preset hits."""

    def __init__(self, cves=(), team=()):
        self._cves = list(cves)
        self._team = list(team)
        self.sparse_queries = []

    def search_cves(self, query_vector, limit=5, language=None, sparse_query=None,
                    dense_threshold=None):
        self.sparse_queries.append(sparse_query)
        return [dict(c) for c in self._cves][:limit]

    def search_team_history(self, query_vector, limit=5, language=None, sparse_query=None,
                            dense_threshold=None):
        return [dict(c) for c in self._team][:limit]


def _cve(cve_id, sim, **extra):
    return {"cve_id": cve_id, "point_id": f"pt-{cve_id}", "similarity_score": sim,
            "vulnerable_code": f"code-{cve_id}", **extra}


def _team(pr_id, sim, **extra):
    return {"pr_id": pr_id, "point_id": f"pt-{pr_id}", "similarity_score": sim,
            "text": f"review-{pr_id}", **extra}


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(settings, "RERANKER_ENABLED", False)
    monkeypatch.setattr(settings, "HYBRID_ENABLED", False)
    monkeypatch.setattr(settings, "SIM_THRESHOLD_CVE", 0.25)
    monkeypatch.setattr(settings, "SIM_THRESHOLD_TEAM", 0.25)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    monkeypatch.setattr(settings, "TWIN_MARGIN_MIN", None)
    monkeypatch.setattr(settings, "FEEDBACK_SUPPRESS_NET", -2)
    monkeypatch.setattr(settings, "FEEDBACK_DOWNWEIGHT_PER_VOTE", 0.15)
    monkeypatch.setattr(feedback_store, "get_net_votes", lambda ids: {})


# --- RagMerger ---------------------------------------------------------------


def _patch_merger_components(monkeypatch, reranker_factory):
    monkeypatch.setattr(rag_merger, "Embedder", StubEmbedder)
    monkeypatch.setattr(rag_merger, "EmbeddingCache", lambda: None)
    monkeypatch.setattr(rag_merger, "VectorStore", lambda: StubStore())
    monkeypatch.setattr(rag_merger, "Reranker", reranker_factory)


def test_merger_does_not_construct_reranker_when_disabled(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("Reranker must not be constructed when RERANKER_ENABLED=false")

    _patch_merger_components(monkeypatch, _boom)
    merger = RagMerger()
    assert merger.reranker is None
    assert merger.cve_retriever.reranker is None
    assert merger.team_retriever.reranker is None


def test_merger_constructs_reranker_when_enabled(monkeypatch):
    built = []
    _patch_merger_components(monkeypatch, lambda: built.append(1) or "rr")
    monkeypatch.setattr(settings, "RERANKER_ENABLED", True)
    merger = RagMerger()
    assert built == [1]
    assert merger.cve_retriever.reranker == "rr" == merger.team_retriever.reranker


# --- Retrievers --------------------------------------------------------------


def test_cve_retriever_orders_by_similarity_without_reranker():
    store = StubStore(cves=[_cve("MID", 0.5), _cve("LOW", 0.1), _cve("HIGH", 0.9)])
    retriever = CVERetriever(StubEmbedder(), store, None)

    ranked = retriever.rerank_candidates("code")
    assert [m["cve_id"] for m in ranked] == ["HIGH", "MID"]  # LOW below the 0.25 gate
    for m in ranked:
        assert "rerank_prob" not in m and "rerank_score" not in m  # nothing fabricated
        assert "rerank_text" not in m
        assert relevance(m) == m["similarity_score"]

    findings = retriever.find_vulnerabilities("code")
    assert [f["cve_id"] for f in findings] == ["HIGH", "MID"]
    assert [f["adjusted_score"] for f in findings] == [0.9, 0.5]


def test_cve_retriever_hybrid_keeps_ann_order_without_reranker(monkeypatch):
    monkeypatch.setattr(settings, "HYBRID_ENABLED", True)
    # RRF scores aren't cosines; the store's (fused) order is the ranking.
    store = StubStore(cves=[_cve("FIRST", 0.016), _cve("SECOND", 0.033)])
    retriever = CVERetriever(StubEmbedder(), store, None)

    ranked = retriever.rerank_candidates("code")
    assert [m["cve_id"] for m in ranked] == ["FIRST", "SECOND"]
    assert store.sparse_queries[-1] is not None  # hybrid path taken


def test_rerank_threshold_ignored_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.9)
    store = StubStore(cves=[_cve("A", 0.5)], team=[_team("7", 0.4)])
    assert [f["cve_id"] for f in CVERetriever(StubEmbedder(), store, None)
            .find_vulnerabilities("code")] == ["A"]
    assert [f["pr_id"] for f in TeamRetriever(StubEmbedder(), store, None)
            .find_team_history("code")] == ["7"]


def test_rerank_threshold_still_gates_reranked_matches(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.5)
    matches = [{"point_id": "a", "rerank_prob": 0.4, "similarity_score": 0.9},
               {"point_id": "b", "rerank_prob": 0.6, "similarity_score": 0.3},
               {"point_id": "c", "similarity_score": 0.3}]  # not reranked: not gated
    out = feedback_store.finalize_matches(matches, 10, settings, base_score=relevance)
    assert [m["point_id"] for m in out] == ["b", "c"]
    assert [m["adjusted_score"] for m in out] == [0.6, 0.3]


def test_team_retriever_scores_similarity_with_recency_and_seniority():
    now = datetime.now(UTC)
    store = StubStore(team=[
        _team("old", 0.8, created_at=(now - timedelta(days=720)).isoformat(),
              author_association="NONE"),
        _team("new", 0.8, created_at=now.isoformat(), author_association="OWNER"),
    ])
    findings = TeamRetriever(StubEmbedder(), store, None).find_team_history("code")
    assert [f["pr_id"] for f in findings] == ["new", "old"]
    new = findings[0]
    assert "rerank_prob" not in new
    expected = compute_adjusted_score(0.8, now, "OWNER", now, settings)
    assert new["adjusted_score"] == pytest.approx(expected, rel=1e-4)


def test_relevance_prefers_rerank_prob():
    assert relevance({"similarity_score": 0.3}) == 0.3
    assert relevance({"similarity_score": 0.3, "rerank_prob": None}) == 0.3
    assert relevance({"similarity_score": 0.3, "rerank_prob": 0.8}) == 0.8
    assert relevance({}) == 0.0


def test_feedback_finalize_without_rerank_prob(monkeypatch):
    monkeypatch.setattr(feedback_store, "get_net_votes", lambda ids: {"a": -1, "b": -2})
    matches = [{"point_id": p, "similarity_score": 0.6} for p in ("a", "b", "c")]
    out = feedback_store.finalize_matches(matches, 10, settings, base_score=relevance)
    assert [m["point_id"] for m in out] == ["c", "a"]  # b suppressed, a downweighted
    assert out[0]["adjusted_score"] == pytest.approx(0.6)
    assert out[1]["adjusted_score"] == pytest.approx(0.6 * 0.85)


# --- Files mode (two-pass) -----------------------------------------------------


def test_analyze_units_two_pass_without_reranker(monkeypatch):
    calls = {"n": 0}

    def votes(ids):
        calls["n"] += 1
        return {"pt-B": -2}

    monkeypatch.setattr(feedback_store, "get_net_votes", votes)
    merger = object.__new__(RagMerger)
    store = StubStore(cves=[_cve("A", 0.7), _cve("B", 0.8)], team=[_team("9", 0.5)])
    merger.cve_retriever = CVERetriever(StubEmbedder(), store, None)
    merger.team_retriever = TeamRetriever(StubEmbedder(), store, None)
    units = [
        {"code": f"u{i}", "file_path": "a.py", "start_line": i, "end_line": i + 1,
         "function_name": f"fn{i}", "language": "python"}
        for i in range(2)
    ]
    out = merger._analyze_units_sync(units, [[0.0] * 4] * 2)

    assert calls["n"] == 1  # one batched feedback lookup for the whole scan
    cves = out["ghost_hunter_findings"]
    assert {(f["cve_id"], f["anchor_function_name"]) for f in cves} == {
        ("A", "fn0"), ("A", "fn1")}  # B suppressed by the shared vote map
    assert all(f["adjusted_score"] == 0.7 and "rerank_prob" not in f for f in cves)
    assert len(out["team_memory_findings"]) == 2 and out["is_vulnerable"]


# --- Persistence / API output ------------------------------------------------------


def test_persisted_findings_without_reranker():
    from backend.app.db.models import Finding
    from backend.app.models.schemas import FindingOut
    from backend.app.services.scan_runner import _finding_out, _persist_findings

    class _Session:
        def add_all(self, rows):
            for i, r in enumerate(rows, start=1):
                r.id = i

        def flush(self):
            pass

    raw = {
        "ghost_hunter_findings": [
            {"cve_id": "C", "point_id": "p1", "similarity_score": 0.8, "adjusted_score": 0.8},
            {"cve_id": "R", "point_id": "p2", "similarity_score": 0.7, "rerank_score": 2.0,
             "rerank_prob": 0.88, "adjusted_score": 0.88},
        ],
        "team_memory_findings": [
            {"pr_id": "9", "point_id": "p3", "similarity_score": 0.5, "adjusted_score": 0.55},
        ],
    }
    rows = _persist_findings(_Session(), "scan-1", raw)
    # NOT NULL columns get a 0.0 placeholder; the payload records it wasn't reranked.
    assert (rows[0].rerank_prob, rows[0].rerank_score, rows[0].adjusted_score) == (0.0, 0.0, 0.8)
    assert json.loads(rows[0].payload_json)["reranked"] is False
    assert json.loads(rows[1].payload_json)["reranked"] is True
    assert json.loads(rows[2].payload_json)["reranked"] is False

    outs = [FindingOut(**_finding_out(r)) for r in rows]
    assert [o.rerank_prob for o in outs] == [None, 0.88, None]
    assert [o.similarity_score for o in outs] == [0.8, 0.7, 0.5]
    # A pre-flag row (always reranked back then) keeps its stored probability.
    legacy = Finding(id=9, point_id="p9", source="team", title="t", similarity_score=0.5,
                     rerank_prob=0.4, payload_json=json.dumps({"author": "x", "url": None}))
    assert FindingOut(**_finding_out(legacy)).rerank_prob == 0.4
