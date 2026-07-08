"""Unit tests for Team Memory recency/seniority weighting."""
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.config import settings
from backend.app.core.scoring import compute_adjusted_score

NOW = datetime(2026, 7, 8, tzinfo=UTC)


def test_missing_metadata_is_neutral():
    # created_at None -> recency 1.0 -> factor 1.0; no association -> seniority 1.0.
    assert compute_adjusted_score(0.5, None, None, NOW, settings) == pytest.approx(0.5)


def test_recent_senior_beats_old_junior():
    recent_senior = compute_adjusted_score(0.5, NOW - timedelta(days=5), "OWNER", NOW, settings)
    old_junior = compute_adjusted_score(0.5, NOW - timedelta(days=720), "NONE", NOW, settings)
    assert recent_senior > old_junior


def test_recency_decays_with_age():
    fresh = compute_adjusted_score(1.0, NOW, "NONE", NOW, settings)
    aged = compute_adjusted_score(
        1.0, NOW - timedelta(days=settings.RECENCY_HALF_LIFE_DAYS), "NONE", NOW, settings
    )
    assert fresh > aged


def test_seniority_boost():
    owner = compute_adjusted_score(0.5, NOW, "OWNER", NOW, settings)
    outsider = compute_adjusted_score(0.5, NOW, "NONE", NOW, settings)
    assert owner > outsider
