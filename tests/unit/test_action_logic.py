"""Unit tests for the GitHub Action's pure decision logic."""
from github_action.scan_pr import (
    anchor_line,
    build_comment_body,
    build_summary,
    extract_marker,
    finding_marker,
    parse_changed_lines,
    plan_comment_ops,
    severity_gate,
)


def test_parse_changed_lines_skips_headers():
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n keep\n+added\n"
    assert parse_changed_lines(patch) == [2]


def test_severity_gate():
    findings = [{"severity": "medium"}, {"severity": "high"}]
    assert severity_gate(findings, "none") == 0
    assert severity_gate(findings, "high") == 1
    assert severity_gate(findings, "critical") == 0  # nothing critical
    assert severity_gate([{"severity": "low"}], "medium") == 0


def test_anchor_line_prefers_exact_then_after():
    assert anchor_line({"start_line": 5}, [3, 5, 9]) == 5  # exact
    assert anchor_line({"start_line": 6}, [3, 8, 12]) == 8  # first at/after start
    assert anchor_line({"start_line": 20}, [3, 8]) == 8  # nearest fallback
    assert anchor_line({"start_line": 5}, []) is None  # not in diff -> summary only


def test_finding_marker_is_stable_and_distinct():
    a = finding_marker("app.py", "pt-1")
    assert a == finding_marker("app.py", "pt-1")
    assert a != finding_marker("app.py", "pt-2")
    assert a != finding_marker("other.py", "pt-1")


def test_marker_roundtrip_in_body():
    body = build_comment_body(
        {"file_path": "app.py", "point_id": "pt-1", "cve_id": "CVE-1", "title": "SQLi",
         "severity": "high", "source": "cve"}
    )
    assert "CVE-1" in body and "SQLi" in body
    assert extract_marker(body) == finding_marker("app.py", "pt-1")


def test_plan_comment_ops_create_update_delete():
    m_keep = "reposentinel:f:keep"
    m_stale = "reposentinel:f:stale"
    existing = [
        {"id": 1, "body": f"old text\n<!-- {m_keep} -->"},
        {"id": 2, "body": f"gone\n<!-- {m_stale} -->"},
    ]
    desired = [
        {"marker": m_keep, "path": "a.py", "line": 3, "body": f"new text\n<!-- {m_keep} -->"},
        {"marker": "reposentinel:f:fresh", "path": "b.py", "line": 5,
         "body": "fresh\n<!-- reposentinel:f:fresh -->"},
    ]
    ops = plan_comment_ops(existing, desired)
    assert [c["marker"] for c in ops["create"]] == ["reposentinel:f:fresh"]
    assert ops["update"] == [{"id": 1, "body": f"new text\n<!-- {m_keep} -->"}]
    assert ops["delete"] == [2]


def test_plan_comment_ops_noop_when_unchanged():
    marker = "reposentinel:f:same"
    body = f"same\n<!-- {marker} -->"
    ops = plan_comment_ops(
        [{"id": 9, "body": body}],
        [{"marker": marker, "path": "a.py", "line": 1, "body": body}],
    )
    assert ops == {"create": [], "update": [], "delete": []}


def test_build_summary_wraps_report_markdown():
    md = build_summary("## 🔴 RepoSentinel Security Report\nbody", "high")
    assert "RepoSentinel Security Report" in md
    assert "Severity gate: `high`" in md
    assert extract_marker(md) == "reposentinel:summary"

    empty = build_summary("", "none")
    assert "✅" in empty
    assert extract_marker(empty) == "reposentinel:summary"
