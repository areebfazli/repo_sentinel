"""Unit tests for the GitHub Action's pure decision logic."""
from github_action.scan_pr import (
    anchor_line,
    build_comment_body,
    build_summary,
    desired_comments,
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
    a = finding_marker("app.py", "pt-1", "foo")
    assert a == finding_marker("app.py", "pt-1", "foo")
    assert a != finding_marker("app.py", "pt-2", "foo")
    assert a != finding_marker("other.py", "pt-1", "foo")
    assert a != finding_marker("app.py", "pt-1", "bar")


def test_finding_marker_stable_across_start_line_shifts():
    # start_line plays no role in the marker: the same function keeps the same
    # marker even when code above it shifts its line number across pushes,
    # preserving comment threads instead of delete+recreate.
    a = finding_marker("app.py", "pt-1", "foo")
    b = finding_marker("app.py", "pt-1", "foo")
    assert a == b


def test_finding_marker_distinguishes_by_function_name():
    # Two functions in the same file matching the same point_id must still get
    # distinct markers, keyed by function_name (the backend's actual dedupe key).
    a = finding_marker("app.py", "pt-1", "foo")
    b = finding_marker("app.py", "pt-1", "bar")
    assert a != b
    # Deterministic across repeated calls for the same triple.
    assert a == finding_marker("app.py", "pt-1", "foo")
    assert b == finding_marker("app.py", "pt-1", "bar")
    # None normalizes deterministically too.
    assert finding_marker("app.py", "pt-1", None) == finding_marker("app.py", "pt-1", None)


def test_desired_comments_keeps_distinct_functions_as_separate_creates():
    # Same file_path + point_id, different function_name -> different markers ->
    # both findings must survive plan_comment_ops as separate "create" ops rather
    # than one clobbering the other in desired_by_marker.
    findings = [
        {"file_path": "app.py", "point_id": "pt-1", "start_line": 10, "function_name": "foo",
         "title": "SQLi", "cve_id": "CVE-1", "severity": "high", "source": "cve"},
        {"file_path": "app.py", "point_id": "pt-1", "start_line": 42, "function_name": "bar",
         "title": "SQLi", "cve_id": "CVE-1", "severity": "high", "source": "cve"},
    ]
    changed_by_file = {"app.py": [10, 42]}
    desired, unanchored = desired_comments(findings, changed_by_file)
    assert unanchored == []
    assert len({d["marker"] for d in desired}) == 2

    ops = plan_comment_ops(existing=[], desired=desired)
    assert len(ops["create"]) == 2


def test_desired_comments_marker_survives_start_line_shift():
    # Same function, but start_line shifted between pushes (code above it changed)
    # -> the marker must stay the same so plan_comment_ops treats it as an
    # update/no-op rather than a delete+recreate that loses the reply thread.
    before = [
        {"file_path": "app.py", "point_id": "pt-1", "start_line": 10, "function_name": "foo",
         "title": "SQLi", "cve_id": "CVE-1", "severity": "high", "source": "cve"},
    ]
    after = [
        {"file_path": "app.py", "point_id": "pt-1", "start_line": 25, "function_name": "foo",
         "title": "SQLi", "cve_id": "CVE-1", "severity": "high", "source": "cve"},
    ]
    changed_by_file = {"app.py": [10, 25]}
    desired_before, _ = desired_comments(before, changed_by_file)
    desired_after, _ = desired_comments(after, changed_by_file)
    assert desired_before[0]["marker"] == desired_after[0]["marker"]


def test_marker_roundtrip_in_body():
    body = build_comment_body(
        {"file_path": "app.py", "point_id": "pt-1", "cve_id": "CVE-1", "title": "SQLi",
         "severity": "high", "source": "cve", "start_line": 10, "function_name": "foo"}
    )
    assert "CVE-1" in body and "SQLi" in body
    assert extract_marker(body) == finding_marker("app.py", "pt-1", "foo")


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
