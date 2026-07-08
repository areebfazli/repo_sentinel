"""Integration tests for the feedback endpoint (upsert + net votes + 404)."""
import pytest
from fastapi.testclient import TestClient

from backend.app.db.models import Finding, Scan
from backend.app.db.session import SessionLocal
from backend.app.main import app


def _make_finding(scan_id: str, point_id: str) -> int:
    with SessionLocal() as session:
        session.merge(Scan(id=scan_id, status="completed", mode="snippet", request_json="{}"))
        finding = Finding(
            scan_id=scan_id,
            source="team",
            collection="team_history",
            point_id=point_id,
            title="bare except discussion",
        )
        session.add(finding)
        session.commit()
        return finding.id


@pytest.mark.integration
def test_feedback_upsert_and_net_votes():
    with TestClient(app) as client:  # lifespan runs init_db()
        finding_id = _make_finding("scan-fb-upsert", "pt-fb-upsert")

        r1 = client.post("/api/v1/feedback/", json={"finding_id": finding_id, "vote": -1})
        assert r1.status_code == 200
        assert r1.json()["net_votes"] == -1

        # Re-voting the same finding overwrites (not appends).
        r2 = client.post("/api/v1/feedback/", json={"finding_id": finding_id, "vote": 1})
        assert r2.status_code == 200
        assert r2.json()["net_votes"] == 1


@pytest.mark.integration
def test_feedback_accumulates_across_scans_by_point():
    with TestClient(app) as client:
        pid = "pt-fb-shared"
        f1 = _make_finding("scan-fb-a", pid)
        f2 = _make_finding("scan-fb-b", pid)
        client.post("/api/v1/feedback/", json={"finding_id": f1, "vote": -1})
        r = client.post("/api/v1/feedback/", json={"finding_id": f2, "vote": -1})
        # Same point downvoted in two scans -> net -2.
        assert r.json()["net_votes"] == -2


@pytest.mark.integration
def test_feedback_unknown_finding_404():
    with TestClient(app) as client:
        r = client.post("/api/v1/feedback/", json={"finding_id": 999999, "vote": 1})
        assert r.status_code == 404


@pytest.mark.integration
def test_feedback_rejects_invalid_vote():
    with TestClient(app) as client:
        r = client.post("/api/v1/feedback/", json={"finding_id": 1, "vote": 5})
        assert r.status_code == 422  # Literal[-1, 1] validation
