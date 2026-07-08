"""Unit tests for feedback suppression/downweighting (get_net_votes stubbed)."""
from backend.app.config import settings
from backend.app.core import feedback_store


def test_suppress_and_downweight(monkeypatch):
    monkeypatch.setattr(settings, "FEEDBACK_SUPPRESS_NET", -2)
    monkeypatch.setattr(settings, "FEEDBACK_DOWNWEIGHT_PER_VOTE", 0.15)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)

    votes = {"A": -2, "B": -1, "C": 3}
    monkeypatch.setattr(
        feedback_store,
        "get_net_votes",
        lambda ids: {k: v for k, v in votes.items() if k in ids},
    )

    matches = [{"point_id": p, "rerank_prob": 0.9} for p in ("A", "B", "C", "D")]
    out = feedback_store.finalize_matches(
        matches, limit=10, settings=settings, base_score=lambda m: m["rerank_prob"]
    )
    ids = [m["point_id"] for m in out]

    assert "A" not in ids  # net -2 <= suppress threshold -> dropped
    assert set(ids) == {"B", "C", "D"}

    b = next(m for m in out if m["point_id"] == "B")
    d = next(m for m in out if m["point_id"] == "D")
    assert b["adjusted_score"] < d["adjusted_score"]  # -1 vote downweights B
    # Sorted by adjusted_score: the downweighted B comes last.
    assert ids[-1] == "B"


def test_positive_votes_do_not_boost(monkeypatch):
    monkeypatch.setattr(settings, "FEEDBACK_SUPPRESS_NET", -2)
    monkeypatch.setattr(settings, "FEEDBACK_DOWNWEIGHT_PER_VOTE", 0.15)
    monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
    monkeypatch.setattr(feedback_store, "get_net_votes", lambda ids: {"A": 5})

    matches = [{"point_id": "A", "rerank_prob": 0.6}]
    out = feedback_store.finalize_matches(
        matches, limit=10, settings=settings, base_score=lambda m: m["rerank_prob"]
    )
    assert out[0]["adjusted_score"] == 0.6  # multiplier capped at 1.0 for positive votes
