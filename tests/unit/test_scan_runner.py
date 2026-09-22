"""Unit tests for the pure report-building logic in scan_runner."""
from backend.app.services.scan_runner import _build_report_findings


def test_build_report_findings_carries_function_name_for_matched_row():
    validated = [{"cve_id": "CVE-1", "title": "SQLi", "explanation": "bad", "fix_snippet": ""}]
    row_snaps = [
        {
            "finding_id": 1,
            "point_id": "pt-1",
            "source": "cve",
            "cve_id": "CVE-1",
            "team_pr_id": None,
            "file_path": "app.py",
            "start_line": 10,
            "function_name": "handle_login",
        }
    ]
    report = _build_report_findings(validated, row_snaps)
    assert len(report) == 1
    assert report[0]["function_name"] == "handle_login"
    assert report[0]["file_path"] == "app.py"
    assert report[0]["start_line"] == 10


def test_build_report_findings_unanchored_finding_has_no_function_name():
    # A validated finding that doesn't match any retrieval row (generic finding)
    # falls back to the unanchored branch, which must set function_name to None.
    validated = [
        {"cve_id": "CVE-UNMATCHED", "title": "generic", "explanation": "", "fix_snippet": ""}
    ]
    row_snaps = [
        {
            "finding_id": 1,
            "point_id": "pt-1",
            "source": "cve",
            "cve_id": "CVE-OTHER",
            "team_pr_id": None,
            "file_path": "app.py",
            "start_line": 10,
            "function_name": "handle_login",
        }
    ]
    report = _build_report_findings(validated, row_snaps)
    assert len(report) == 1
    assert report[0]["function_name"] is None
    assert report[0]["file_path"] is None


def test_finding_out_exposes_twin_scores_from_payload():
    import json

    from backend.app.db.models import Finding
    from backend.app.models.schemas import FindingOut
    from backend.app.services.scan_runner import _finding_out, _persist_findings

    class _Session:
        def add_all(self, rows):
            for i, r in enumerate(rows, start=1):
                r.id = i

        def flush(self):
            pass

    raw = {
        "ghost_hunter_findings": [
            {"cve_id": "CVE-T", "point_id": "p1", "similarity_score": 0.8,
             "sim_fixed": 0.7, "twin_margin": 0.1},
            {"cve_id": "CVE-H", "point_id": "p2", "similarity_score": 0.8,
             "sim_fixed": None, "twin_margin": None},
        ],
        "team_memory_findings": [{"pr_id": "9", "point_id": "p3", "similarity_score": 0.5}],
    }
    rows = _persist_findings(_Session(), "scan-1", raw)
    assert json.loads(rows[0].payload_json)["twin_margin"] == 0.1

    outs = [FindingOut(**_finding_out(r)) for r in rows]
    assert (outs[0].sim_fixed, outs[0].twin_margin) == (0.7, 0.1)
    assert (outs[1].sim_fixed, outs[1].twin_margin) == (None, None)
    assert (outs[2].sim_fixed, outs[2].twin_margin) == (None, None)  # team rows
    # Rows persisted before this change (no twin keys / no payload) still render.
    legacy = Finding(id=4, point_id="p4", source="cve", title="t", similarity_score=0.5,
                     rerank_prob=0.5, payload_json=None)
    assert FindingOut(**_finding_out(legacy)).twin_margin is None
