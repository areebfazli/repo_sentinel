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
