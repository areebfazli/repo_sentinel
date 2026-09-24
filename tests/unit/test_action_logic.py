"""Unit tests for the GitHub Action's pure decision logic."""
from github_action import scan_pr
from github_action.scan_pr import (
    _apply_comment_ops,
    _check,
    anchor_line,
    build_comment_body,
    build_summary,
    desired_comments,
    extract_marker,
    finding_marker,
    parse_changed_lines,
    parse_commentable_lines,
    plan_comment_ops,
    severity_gate,
)


class _FakeResp:
    """Minimal stand-in for requests.Response, enough for _check/_apply_comment_ops."""

    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text
        self.ok = status_code < 400


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


def test_check_success_and_failure(capsys):
    assert _check(_FakeResp(200), "PATCH inline comment 1") is True
    out = capsys.readouterr().out
    assert out == ""  # no warning logged on success

    assert _check(_FakeResp(422, "validation failed"), "PATCH inline comment 1") is False
    out = capsys.readouterr().out
    assert "PATCH inline comment 1" in out
    assert "422" in out
    assert "validation failed" in out


def test_check_delete_404_counts_as_success(capsys):
    # The comment is already gone — that's the desired end state, not a failure.
    assert _check(_FakeResp(404, "Not Found"), "DELETE inline comment 5", ok_404=True) is True
    assert capsys.readouterr().out == ""
    # Without ok_404, a 404 is still a failure (e.g. PATCH on a vanished comment).
    assert _check(_FakeResp(404, "Not Found"), "PATCH inline comment 5") is False


def test_check_failure_count_with_mixed_results():
    results = [
        _check(_FakeResp(200), "op 1"),
        _check(_FakeResp(500, "boom"), "op 2"),
        _check(_FakeResp(204), "op 3"),
        _check(_FakeResp(403, "forbidden"), "op 4"),
    ]
    failures = sum(1 for ok in results if not ok)
    assert failures == 2


def test_apply_comment_ops_continues_after_failed_patch(monkeypatch, capsys):
    # First PATCH fails, second PATCH and the DELETE must still run afterward,
    # and no exception should propagate out of _apply_comment_ops.
    patch_calls, delete_calls = [], []

    def fake_patch(url, **kwargs):
        patch_calls.append(url)
        if len(patch_calls) == 1:
            return _FakeResp(500, "server error")
        return _FakeResp(200)

    def fake_delete(url, **kwargs):
        delete_calls.append(url)
        return _FakeResp(200)

    monkeypatch.setattr(scan_pr.requests, "patch", fake_patch)
    monkeypatch.setattr(scan_pr.requests, "delete", fake_delete)

    ops = {
        "create": [],
        "update": [{"id": 1, "body": "a"}, {"id": 2, "body": "b"}],
        "delete": [3],
    }
    posted, failures = _apply_comment_ops("o/r", 1, "sha", "tok", ops)

    assert posted is True
    assert len(patch_calls) == 2  # both PATCHes ran despite the first failing
    assert len(delete_calls) == 1  # the DELETE still ran afterward
    assert failures == 1
    assert "WARNING" in capsys.readouterr().out


def test_apply_comment_ops_delete_404_not_counted_as_failure(monkeypatch):
    monkeypatch.setattr(scan_pr.requests, "delete", lambda url, **kw: _FakeResp(404, "gone"))

    ops = {"create": [], "update": [], "delete": [42]}
    posted, failures = _apply_comment_ops("o/r", 1, "sha", "tok", ops)

    assert posted is True
    assert failures == 0


def test_anchor_prefers_exact_line_even_on_context_lines():
    patch = "@@ -10,4 +10,3 @@\n ctx10\n-removed guard\n ctx11\n+added12\n"
    changed = parse_changed_lines(patch)
    commentable = parse_commentable_lines(patch)
    assert changed == [12] and commentable == [10, 11, 12]
    # Exact offending line on an unchanged context line next to a deleted guard.
    assert anchor_line({"line": 11, "start_line": 10}, changed, commentable) == 11
    # Exact line outside the diff: nearest added line at/after it.
    assert anchor_line({"line": 5, "start_line": 1}, changed, commentable) == 12
    # Deletion-only file: fall back to the nearest commentable line.
    assert anchor_line({"line": 30}, [], [10, 11, 13]) == 13
    # Older servers (no "line") keep the old behaviour.
    assert anchor_line({"start_line": 12}, changed) == 12


def test_marker_uses_dedupe_key_so_findings_in_one_function_stay_distinct():
    base = {"file_path": "a.py", "function_name": "f", "start_line": 1, "title": "t"}
    findings = [{**base, "dedupe_key": "llm:1"}, {**base, "dedupe_key": "llm:2"},
                {**base, "point_id": "pt-legacy"}]
    desired, _ = desired_comments(findings, {"a.py": [1]})
    assert len({d["marker"] for d in desired}) == 3
    assert desired[2]["marker"] == finding_marker("a.py", "pt-legacy", "f")
