"""Unit tests for the cve_corpus vuln/fixed named-vector layout and twin scoring
(ROADMAP 1b), using a stub Qdrant client — no Qdrant, no models."""
from types import SimpleNamespace

import pytest

from backend.app.core.vector_store import (
    DENSE_VECTOR,
    FIXED_VECTOR,
    SPARSE_VECTOR,
    VULN_VECTOR,
    VectorStore,
    twin_scores,
)


class StubClient:
    """Records query_points/upsert calls and serves preset hits."""

    def __init__(self, points=None, vectors_config=None, sparse_config=None):
        self._points = points or []
        self.query_kwargs = None
        self.upserts = []
        self._params = SimpleNamespace(vectors=vectors_config, sparse_vectors=sparse_config)

    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(points=list(self._points))

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=self._params))


def _store(client, hybrid=False):
    store = object.__new__(VectorStore)  # skip the real client + collection setup
    store.client = client
    store.cve_collection = "cve_corpus"
    store.team_collection = "team_history"
    store.vector_size = 3
    store.hybrid = hybrid
    store.cve_schema_error = None
    return store


def _hit(pid, score, vector, **payload):
    return SimpleNamespace(id=pid, score=score, payload={"cve_id": pid, **payload}, vector=vector)


# --- twin_scores ------------------------------------------------------------

def test_twin_scores_margin_is_vuln_minus_fixed():
    q = [1.0, 0.0, 0.0]
    out = twin_scores(q, vuln_vector=[0.8, 0.6, 0.0], fixed_vector=[0.6, 0.8, 0.0])
    assert out["sim_fixed"] == pytest.approx(0.6)
    assert out["twin_margin"] == pytest.approx(0.8 - 0.6)


def test_twin_scores_negative_when_code_looks_like_the_fix():
    q = [0.6, 0.8, 0.0]
    out = twin_scores(q, vuln_vector=[1.0, 0.0, 0.0], fixed_vector=[0.6, 0.8, 0.0])
    assert out["sim_fixed"] == pytest.approx(1.0)
    assert out["twin_margin"] == pytest.approx(0.6 - 1.0)


def test_twin_scores_normalises_unnormalised_vectors():
    out = twin_scores([2.0, 0.0, 0.0], vuln_vector=[5.0, 0.0, 0.0], fixed_vector=[0.0, 3.0, 0.0])
    assert out["sim_fixed"] == pytest.approx(0.0)
    assert out["twin_margin"] == pytest.approx(1.0)


@pytest.mark.parametrize("fixed", [None, []])
def test_twin_scores_missing_twin_is_none(fixed):
    out = twin_scores([1.0, 0.0, 0.0], vuln_vector=[1.0, 0.0, 0.0], fixed_vector=fixed)
    assert out == {"sim_fixed": None, "twin_margin": None}


# --- search -----------------------------------------------------------------

def test_search_cves_uses_vuln_vector_and_scores_twins():
    client = StubClient(
        points=[
            _hit("A", 0.8, {VULN_VECTOR: [0.8, 0.6, 0.0], FIXED_VECTOR: [0.6, 0.8, 0.0]}),
            _hit("B", 0.7, {VULN_VECTOR: [0.7, 0.71, 0.0]}),  # handwritten: no twin
            _hit("C", 0.6, None),  # defensive: no vectors returned at all
        ]
    )
    results = _store(client).search_cves([1.0, 0.0, 0.0], limit=5)

    assert client.query_kwargs["using"] == VULN_VECTOR
    assert set(client.query_kwargs["with_vectors"]) == {VULN_VECTOR, FIXED_VECTOR}
    a, b, c = results
    assert a["point_id"] == "A" and a["collection"] == "cve_corpus"
    assert a["similarity_score"] == 0.8
    assert a["sim_fixed"] == pytest.approx(0.6)
    assert a["twin_margin"] == pytest.approx(0.2)
    for no_twin in (b, c):
        assert no_twin["sim_fixed"] is None and no_twin["twin_margin"] is None
    # The raw vectors are not leaked onto the match dicts.
    assert all(VULN_VECTOR not in r and FIXED_VECTOR not in r for r in results)


def test_hybrid_margin_uses_cosines_not_the_rrf_score():
    client = StubClient(
        points=[_hit("A", 0.5, {VULN_VECTOR: [0.8, 0.6, 0.0], FIXED_VECTOR: [0.6, 0.8, 0.0]})]
    )
    store = _store(client, hybrid=True)
    (a,) = store.search_cves(
        [1.0, 0.0, 0.0], limit=5, sparse_query={"indices": [1], "values": [1.0]}
    )
    prefetch = client.query_kwargs["prefetch"]
    assert prefetch[0].using == VULN_VECTOR and prefetch[1].using == SPARSE_VECTOR
    assert a["similarity_score"] == 0.5  # RRF fusion score, untouched
    assert a["twin_margin"] == pytest.approx(0.2)  # 0.8 - 0.6, not 0.5 - 0.6


def test_team_search_is_unchanged():
    client = StubClient(points=[_hit("T", 0.9, None)])
    (t,) = _store(client).search_team_history([1.0, 0.0, 0.0], limit=5)
    assert client.query_kwargs["using"] is None  # unnamed vector in dense mode
    assert client.query_kwargs["with_vectors"] is False
    assert "twin_margin" not in t

    client = StubClient(points=[_hit("T", 0.9, None)])
    _store(client, hybrid=True).search_team_history([1.0, 0.0, 0.0], limit=5)
    assert client.query_kwargs["using"] == DENSE_VECTOR


def test_search_cves_refuses_pre_twin_collection():
    store = _store(StubClient())
    store.cve_schema_error = "rebuild with --recreate"
    with pytest.raises(RuntimeError, match="--recreate"):
        store.search_cves([1.0, 0.0, 0.0])


# --- insert -----------------------------------------------------------------

def test_insert_cves_writes_fixed_only_when_present():
    client = StubClient()
    _store(client).insert_cves(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [{"cve_id": "A"}, {"cve_id": "B"}],
        ids=["id-a", "id-b"],
        fixed_vectors=[[0.0, 0.0, 1.0], None],
    )
    (collection, points), = client.upserts
    assert collection == "cve_corpus"
    assert points[0].vector == {VULN_VECTOR: [1.0, 0.0, 0.0], FIXED_VECTOR: [0.0, 0.0, 1.0]}
    assert points[1].vector == {VULN_VECTOR: [0.0, 1.0, 0.0]}  # no twin -> vector omitted


def test_insert_cves_hybrid_adds_sparse():
    client = StubClient()
    _store(client, hybrid=True).insert_cves(
        [[1.0, 0.0, 0.0]], [{"cve_id": "A"}], ids=["id-a"],
        sparse_vectors=[{"indices": [3], "values": [1.0]}],
    )
    vector = client.upserts[0][1][0].vector
    assert set(vector) == {VULN_VECTOR, SPARSE_VECTOR}
    assert vector[SPARSE_VECTOR].indices == [3]


def test_insert_team_history_keeps_unnamed_vector():
    client = StubClient()
    _store(client).insert_team_history([[1.0, 0.0, 0.0]], [{"pr_id": "1"}], ids=["t"])
    assert client.upserts[0][1][0].vector == [1.0, 0.0, 0.0]


# --- schema check -------------------------------------------------------------

def test_schema_check_flags_legacy_unnamed_vector():
    legacy = SimpleNamespace(size=3)  # a bare VectorParams, not a dict
    assert "--recreate" in _store(StubClient(vectors_config=legacy))._check_cve_schema()


def test_schema_check_accepts_twin_layout_and_hybrid_mismatch():
    named = {VULN_VECTOR: object(), FIXED_VECTOR: object()}
    assert _store(StubClient(vectors_config=named))._check_cve_schema() is None
    # Hybrid on, but the collection was built dense-only: sparse search would fail.
    assert "--recreate" in _store(StubClient(vectors_config=named), hybrid=True)._check_cve_schema()
    # Built hybrid, now dense-only: still searchable.
    sparse = {SPARSE_VECTOR: object()}
    store = _store(StubClient(vectors_config=named, sparse_config=sparse))
    assert store._check_cve_schema() is None
