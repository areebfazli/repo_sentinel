"""Feedback aggregation for retrieval-time suppression/downweighting.

Votes are summed by point_id (across all scans), so a memory the team keeps
downvoting is progressively suppressed no matter which scan surfaced it.
"""
from sqlalchemy import func, select

from backend.app.db.models import Feedback
from backend.app.db.session import SessionLocal, engine

_ensured = False


def _ensure_table() -> None:
    global _ensured
    if not _ensured:
        Feedback.__table__.create(bind=engine, checkfirst=True)
        _ensured = True


def get_net_votes(point_ids: list[str]) -> dict[str, int]:
    """Return {point_id: net_vote} for the given points (missing -> absent)."""
    if not point_ids:
        return {}
    _ensure_table()
    with SessionLocal() as session:
        rows = session.execute(
            select(Feedback.point_id, func.sum(Feedback.vote))
            .where(Feedback.point_id.in_(point_ids))
            .group_by(Feedback.point_id)
        ).all()
    return {point_id: int(total) for point_id, total in rows}


def feedback_multiplier(net_vote: int, settings) -> float:
    """Score multiplier from a net vote: negative votes downweight, positives don't boost."""
    return max(0.3, 1.0 + settings.FEEDBACK_DOWNWEIGHT_PER_VOTE * min(net_vote, 0))


def finalize_matches(
    matches: list[dict], limit: int, settings, base_score, net_votes: dict[str, int] | None = None
) -> list[dict]:
    """Apply feedback suppression + downweighting, gate, and sort by adjusted_score.

    base_score(match) -> float is the pre-feedback score (rerank_prob for CVEs,
    recency/seniority-weighted for Team Memory).

    ``net_votes`` lets a caller pass a pre-fetched {point_id: net_vote} map so a
    batch of finalize calls (files mode, one per unit) shares a single DB query
    instead of one round-trip each. When None, the votes are fetched here.
    """
    if net_votes is None:
        net_votes = get_net_votes([m.get("point_id", "") for m in matches])
    net = net_votes
    kept = []
    for m in matches:
        vote = net.get(m.get("point_id", ""), 0)
        if vote <= settings.FEEDBACK_SUPPRESS_NET:
            continue  # team has repeatedly rejected this memory
        if m.get("rerank_prob", 0.0) < settings.RERANK_THRESHOLD:
            continue
        m["adjusted_score"] = base_score(m) * feedback_multiplier(vote, settings)
        kept.append(m)
    kept.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return kept[:limit]
