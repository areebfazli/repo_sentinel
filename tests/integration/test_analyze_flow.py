"""Integration test: POST /analyze -> 202 -> poll -> completed, with a stub merger
and the mock LLM provider (no real models, no network)."""
import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import analyze
from backend.app.main import app


def _cve_match(**anchor):
    return {
        "cve_id": "CVE-2023-28450",
        "description": "SQL Injection in login query",
        "severity": 9.8,
        "category": "sqli",
        "language": "python",
        "similarity_score": 0.71,
        "rerank_score": -4.0,
        "rerank_prob": 0.02,
        "adjusted_score": 0.02,
        "point_id": "pt-cve-1",
        "collection": "cve_corpus",
        "vulnerable_code": "query = f\"... {username} ...\"",
        **anchor,
    }


class StubMerger:
    async def analyze_code(self, code, language=None):
        return {
            "ghost_hunter_findings": [_cve_match()],
            "team_memory_findings": [],
            "is_vulnerable": True,
        }

    async def analyze_units(self, units):
        if not units:
            return {"ghost_hunter_findings": [], "team_memory_findings": [], "is_vulnerable": False}
        u = units[0]
        return {
            "ghost_hunter_findings": [
                _cve_match(
                    file_path=u["file_path"],
                    start_line=u["start_line"],
                    end_line=u["end_line"],
                    function_name=u["function_name"],
                )
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


FILES_CONTENT = (
    "def safe():\n"
    "    return 1\n"
    "\n"
    "def risky(u):\n"
    "    q = 'SELECT ' + u\n"
    "    db.execute(q)\n"
)


@pytest.mark.integration
def test_analyze_files_mode_anchors_finding():
    app.dependency_overrides[analyze.get_merger] = lambda: StubMerger()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/analyze/",
                json={
                    "files": [
                        {"path": "app.py", "content": FILES_CONTENT, "changed_lines": [5, 6]}
                    ]
                },
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            data = client.get(f"/api/v1/analyze/{job_id}").json()
            assert data["status"] == "completed"
            findings = data["result"]["findings"]
            assert len(findings) == 1
            # Only the changed function (risky) is analyzed and anchored.
            assert findings[0]["file_path"] == "app.py"
            assert findings[0]["start_line"] == 4
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_rejects_both_snippet_and_files():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze/",
            json={"code_snippet": "x", "files": [{"path": "a.py", "content": "y"}]},
        )
        assert resp.status_code == 422  # validator: exactly one mode
