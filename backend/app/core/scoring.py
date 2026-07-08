"""Team Memory relevance weighting (pure functions).

A retrieved memory's score is boosted toward recent PRs and senior reviewers:
a lead architect's review from last month should outrank a stale drive-by
comment. Missing metadata contributes a neutral factor.
"""
from datetime import datetime


def compute_adjusted_score(
    rerank_prob: float,
    created_at: datetime | None,
    author_association: str | None,
    now: datetime,
    settings,
) -> float:
    """Weight rerank_prob by recency (exponential half-life) and reviewer seniority."""
    recency_weight = 1.0
    if created_at is not None:
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        recency_weight = 0.5 ** (age_days / settings.RECENCY_HALF_LIFE_DAYS)

    seniority_weight = settings.SENIORITY_WEIGHTS.get(author_association or "", 1.0)

    # Recency contributes a 0.7..1.0 multiplier (never fully suppresses an old
    # but strongly-matching memory); seniority scales around 1.0.
    return rerank_prob * (0.7 + 0.3 * recency_weight) * seniority_weight
