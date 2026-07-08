"""Integration test: POST /analyze -> 202 -> poll -> completed, with a stub merger
and the mock LLM provider (no real models, no network)."""
import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import analyze
from backend.app.main import app


class StubMerger:
    async def analyze_code(self, code, language=None):
        return {
            "ghost_hunter_findings": [
                {
                    "cve_id": "CVE-2023-28450",
                    "description": "SQL Injection in login query",
                    "severity": 9.8,
                    "category": "sqli",
                    "language": "python",
                    "similarity_score": 0.71,
                    "rerank_score": -4.0,
                    "rerank_prob": 0.02,
                    "point_id": "pt-cve-1",
                    "collection": "cve_corpus",
                    "vulnerable_code": "query = f\"... {username} ...\"",
                }
            ],
            "team_memory_findings": [],
            "is_vulnerable": True,
        }


@pytest.mark.integration
def test_analyze_queues_and_completes():
    app.dependency_overrides[analyze.get_merger] = lambda: StubMerger()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/analyze/",
                json={"code_snippet": "def f(): pass", "language": "python"},
            )
            assert resp.status_code == 202
            body = resp.json()
            job_id = body["job_id"]
            assert body["status"] == "queued"
            assert body["poll_url"].endswith(job_id)

            # TestClient runs the background task before returning, so it's done.
            status = client.get(f"/api/v1/analyze/{job_id}")
            assert status.status_code == 200
            data = status.json()
            assert data["status"] == "completed"
            result = data["result"]
            assert result["is_vulnerable"] is True
            assert result["ghost_hunter_matches"] == 1
            assert result["team_memory_matches"] == 0
            assert result["llm_provider_used"] == "mock"
            assert len(result["findings"]) == 1
            finding = result["findings"][0]
            assert finding["cve_id"] == "CVE-2023-28450"
            assert finding["point_id"] == "pt-cve-1"
            assert finding["finding_id"] > 0
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_unknown_job_returns_404():
    with TestClient(app) as client:
        resp = client.get("/api/v1/analyze/does-not-exist")
        assert resp.status_code == 404
