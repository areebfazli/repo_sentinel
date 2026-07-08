"""Unit tests for retriever filtering logic using stub components (no ML models)."""
from backend.app.config import settings
from backend.app.core.cve_retriever import CVERetriever


class StubEmbedder:
    def embed_text(self, text):
        return [0.0] * 768


class StubVectorStore:
    """Returns preset candidates and records the language it was called with."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.last_language = "UNSET"
        self.last_limit = None

    def search_cves(self, query_vector, limit=5, language=None):
        self.last_language = language
        self.last_limit = limit
        return list(self._candidates)


class StubReranker:
    """Assigns rerank_prob from a lookup keyed by cve_id; sorts + slices like the real one."""

    def __init__(self, probs):
        self._probs = probs
        self.seen_pairs = []

    def rerank(self, query_code, candidates, top_k=1):
        for c in candidates:
            self.seen_pairs.append((c.get("cve_id"), c.get("rerank_text")))
            c["rerank_prob"] = self._probs.get(c["cve_id"], 0.0)
            c["rerank_score"] = c["rerank_prob"]
        ranked = sorted(candidates, key=lambda c: c["rerank_prob"], reverse=True)
        return ranked[:top_k]


def _candidate(cve_id, sim, **extra):
    return {"cve_id": cve_id, "similarity_score": sim, "description": f"desc-{cve_id}", **extra}


def test_similarity_gate_drops_below_threshold(monkeypatch):
    monkeypatch.setattr(settings, "SIM_THRESHOLD_CVE", 0.8)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
    store = StubVectorStore([_candidate("A", 0.9), _candidate("B", 0.5)])
    reranker = StubReranker({"A": 0.9, "B": 0.9})
    retriever = CVERetriever(StubEmbedder(), store, reranker)

    results = retriever.find_vulnerabilities("code", language="python")
    ids = {r["cve_id"] for r in results}
    assert ids == {"A"}  # B filtered by the 0.8 similarity gate before reranking


def test_rerank_probability_gate(monkeypatch):
    monkeypatch.setattr(settings, "SIM_THRESHOLD_CVE", 0.0)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.5)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    store = StubVectorStore([_candidate("A", 0.9), _candidate("B", 0.9)])
    reranker = StubReranker({"A": 0.8, "B": 0.2})  # B below rerank gate
    retriever = CVERetriever(StubEmbedder(), store, reranker)

    results = retriever.find_vulnerabilities("code")
    assert [r["cve_id"] for r in results] == ["A"]


def test_language_and_ann_candidates_passed_through(monkeypatch):
    monkeypatch.setattr(settings, "SIM_THRESHOLD_CVE", 0.0)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "ANN_CANDIDATES", 7)
    store = StubVectorStore([_candidate("A", 0.9)])
    retriever = CVERetriever(StubEmbedder(), store, StubReranker({"A": 0.9}))

    retriever.find_vulnerabilities("code", language="go")
    assert store.last_language == "go"
    assert store.last_limit == 7  # ANN_CANDIDATES, not top_k


def test_rerank_text_prefers_vulnerable_code(monkeypatch):
    monkeypatch.setattr(settings, "SIM_THRESHOLD_CVE", 0.0)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
    store = StubVectorStore([_candidate("A", 0.9, vulnerable_code="the vuln code")])
    reranker = StubReranker({"A": 0.9})
    retriever = CVERetriever(StubEmbedder(), store, reranker)

    retriever.find_vulnerabilities("code")
    assert reranker.seen_pairs == [("A", "the vuln code")]


def test_top_k_limit(monkeypatch):
    monkeypatch.setattr(settings, "SIM_THRESHOLD_CVE", 0.0)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 2)
    store = StubVectorStore([_candidate(x, 0.9) for x in ("A", "B", "C", "D")])
    reranker = StubReranker({"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6})
    retriever = CVERetriever(StubEmbedder(), store, reranker)

    results = retriever.find_vulnerabilities("code")
    assert [r["cve_id"] for r in results] == ["A", "B"]
