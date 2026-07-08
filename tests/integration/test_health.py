"""Integration test: app boots (lifespan runs init_db, no model preload) and serves /health."""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.mark.integration
def test_health_endpoint():
    # TestClient as a context manager runs the lifespan (init_db + no preload).
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["environment"] == "development"
