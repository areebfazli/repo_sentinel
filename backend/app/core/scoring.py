"""Team Memory relevance weighting (pure functions).

A retrieved memory's score is boosted toward recent PRs and senior reviewers:
a lead architect's review from last month should outrank a stale drive-by
comment. Missing metadata contributes a neutral factor.
"""
from datetime import datetime
from typing import Any


def relevance(match: dict[str, Any]) -> float:
    """Pre-feedback relevance of a retrieved match.

    ``rerank_prob`` is set only when a cross-encoder actually scored the match
    (RERANKER_ENABLED); with the reranker off it is absent — never fabricated —
    and relevance falls back to ``similarity_score`` (cosine in dense mode, the
    RRF fusion score in hybrid mode). Every scorer goes through here, so CVE and
    Team scoring work identically with or without the reranker.
    """
    prob = match.get("rerank_prob")
    return float(prob) if prob is not None else float(match.get("similarity_score", 0.0))


def compute_adjusted_score(
    relevance_score: float,
    created_at: datetime | None,
    author_association: str | None,
    now: datetime,
    settings,
) -> float:
    """Weight a match's relevance (see ``relevance``) by recency (exponential
    half-life) and reviewer seniority."""
    recency_weight = 1.0
    if created_at is not None:
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        recency_weight = 0.5 ** (age_days / settings.RECENCY_HALF_LIFE_DAYS)

    seniority_weight = settings.SENIORITY_WEIGHTS.get(author_association or "", 1.0)

    # Recency contributes a 0.7..1.0 multiplier (never fully suppresses an old
    # but strongly-matching memory); seniority scales around 1.0.
    return relevance_score * (0.7 + 0.3 * recency_weight) * seniority_weight
