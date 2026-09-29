"""Tests for scripts/fetch_benign_commits.py with stubbed HTTP (never the network).

Commit filtering (known fix commits, merges, bots, security keywords in the
message and in linked PRs, file/line caps), item format and patch round trips,
response caching, rate-limit / budget handling, and ``--dry-run`` making no call.
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.build_corpus_from_osv import GithubClient, RateLimitExceeded, Stats  # noqa: E402
from scripts.build_pr_eval import apply_patch, unified_diff  # noqa: E402
from scripts.fetch_benign_commits import (  # noqa: E402
    API,
    BenignClient,
    BudgetExceeded,
    cache_path,
    detail_reject_reason,
    estimate_calls,
    is_security_text,
    known_fix_commits,
    known_prefixes,
    list_reject_reason,
    load_repo_splits,
    parse_args,
    prs_reject_reason,
    repo_order,
    run,
)

REPO = "acme/app"
PARENT = "0" * 40


class FakeResponse:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code = status
        self._data = data
        self.headers = headers or {}

    def json(self):
        return self._data


class FakeSession:
    """URL -> FakeResponse (or a list of them, served in order)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        r = self.routes.get(url)
        if r is None:
            return FakeResponse(404, {"message": "Not Found"})
        if isinstance(r, list):
            return r.pop(0)
        return r


class NoNetwork:
    def get(self, *a, **k):
        raise AssertionError("network call attempted")


def _sha(c):
    return c * 40


def _list_entry(sha, msg, parents=1, login="dev"):
    return {"sha": sha, "commit": {"message": msg},
            "parents": [{"sha": PARENT}] * parents, "author": {"login": login, "type": "User"}}


OLD = "def add(a, b):\n    return a + b\n"
NEW = "def add(a, b):\n    total = a + b\n    return total\n"


def _gh_patch(old, new):
    return unified_diff(old, new, "x").split("\n", 2)[2]  # GitHub's patch: no ---/+++ header


def _detail(sha, files):
    return {"sha": sha, "parents": [{"sha": PARENT}],
            "commit": {"message": "m", "committer": {"date": "2026-01-02T00:00:00Z"}},
            "files": files}


def _content(text):
    return FakeResponse(200, {"content": base64.b64encode(text.encode()).decode(),
                              "encoding": "base64"})


def _routes():
    good, xss, merge, fix, docs, pr_sec, bot = (_sha(c) for c in "abcdefg")
    listing = [
        _list_entry(good, "Refactor add() for readability"),
        _list_entry(xss, "Fix XSS in template rendering"),
        _list_entry(merge, "Merge branch main", parents=2),
        _list_entry(fix, "Tidy things"),
        _list_entry(docs, "Update docs"),
        _list_entry(pr_sec, "Small change"),
        _list_entry(bot, "Bump deps", login="dependabot[bot]"),
    ]
    code_file = {"filename": "app/calc.py", "status": "modified", "additions": 2,
                 "deletions": 1, "patch": _gh_patch(OLD, NEW)}
    return {
        f"{API}/repos/{REPO}/commits?per_page=100&page=1": FakeResponse(200, listing),
        f"{API}/repos/{REPO}/commits/{good}": FakeResponse(200, _detail(good, [
            code_file, {"filename": "README.md", "status": "modified", "patch": "@@"},
            {"filename": "tests/test_calc.py", "status": "modified", "patch": "@@"}])),
        f"{API}/repos/{REPO}/commits/{good}/pulls": FakeResponse(200, [
            {"title": "Readability", "body": "no functional change", "labels": []}]),
        f"{API}/repos/{REPO}/contents/{quote('app/calc.py')}?ref={good}": _content(NEW),
        f"{API}/repos/{REPO}/commits/{docs}": FakeResponse(200, _detail(docs, [
            {"filename": "docs/index.md", "status": "modified", "patch": "@@"}])),
        f"{API}/repos/{REPO}/commits/{pr_sec}": FakeResponse(200, _detail(pr_sec, [code_file])),
        f"{API}/repos/{REPO}/commits/{pr_sec}/pulls": FakeResponse(200, [
            {"title": "Small change", "body": "", "labels": [{"name": "security"}]}]),
    }


def _setup(tmp_path):
    pr_eval = tmp_path / "pr_eval.jsonl"
    pr_eval.write_text(json.dumps({"repo": REPO, "split": "dev", "meta": {
        "commit": _sha("d"), "repo_side": "dev"}}) + "\n", encoding="utf-8")
    return pr_eval


def _args(tmp_path, pr_eval, *extra):
    return parse_args(["--pr-eval", str(pr_eval), "--cache-dir", str(tmp_path / "cache"),
                       "--out", str(tmp_path / "out.jsonl"), "--ordinary", "",
                       "--allow-unauthenticated", "--min-interval", "0", *extra])


def test_fetch_end_to_end_with_stubbed_http(tmp_path, capsys):
    pr_eval = _setup(tmp_path)
    session = FakeSession(_routes())
    assert run(_args(tmp_path, pr_eval), session=session) == 0
    items = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(items) == 1
    it = items[0]
    assert it["id"] == f"pr_benign_acme_app_{_sha('a')[:10]}"
    assert (it["kind"], it["source"], it["split"], it["language"]) == (
        "benign", "benign_commit", "dev", "python")
    assert it["advisory_id"] is None and it["target"]["path"] is None
    assert it["pr_title"] == "Update app/calc.py"
    f = it["files"][0]
    assert (f["path"], f["old_content"], f["new_content"]) == ("app/calc.py", OLD, NEW)
    assert apply_patch(f["old_content"], f["patch"]) == f["new_content"]
    assert f["patch"].startswith("--- a/app/calc.py\n+++ b/app/calc.py\n@@ ")
    out = capsys.readouterr().out
    for reason in ("security_keyword_message", "merge_or_root", "known_fix_commit",
                   "no_code_files", "security_keyword_linked_pr", "bot", "accepted"):
        assert f'"{reason}": 1' in out
    # The old side came from reverse-applying GitHub's patch: no parent fetch.
    assert not any(f"ref={PARENT}" in c for c in session.calls)
    n_calls = len(session.calls)
    assert n_calls == 7  # list, 3 details, 2 linked-PR lists, 1 content

    # Everything is cached: a re-run makes no call at all.
    assert run(_args(tmp_path, pr_eval), session=NoNetwork()) == 0
    assert (tmp_path / "out.jsonl").read_text().count("\n") == 1


def test_dry_run_makes_no_calls(tmp_path, capsys):
    pr_eval = _setup(tmp_path)
    assert run(_args(tmp_path, pr_eval, "--dry-run", "--max-commits", "300"),
               session=NoNetwork()) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and '"total_calls"' in out
    assert not (tmp_path / "out.jsonl").exists()


def test_rate_limit_stops_cleanly(tmp_path, capsys):
    pr_eval = _setup(tmp_path)
    routes = {f"{API}/repos/{REPO}/commits?per_page=100&page=1": FakeResponse(
        403, {}, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1"})}
    assert run(_args(tmp_path, pr_eval), session=FakeSession(routes)) == 0
    assert "stopped early" in capsys.readouterr().out
    assert (tmp_path / "out.jsonl").read_text() == ""


def test_secondary_rate_limit_waits_then_retries(tmp_path):
    url = f"{API}/repos/{REPO}/commits/{_sha('a')}"
    session = FakeSession({url: [FakeResponse(429, {}, {"Retry-After": "7"}),
                                 FakeResponse(200, {"sha": "x"})]})
    slept = []
    client = BenignClient(session, None, tmp_path, Stats(), wait_on_rate_limit=True,
                          sleep=slept.append)
    assert client.get_json(url) == {"sha": "x"} and slept == [7.0]
    no_wait = BenignClient(FakeSession({url: FakeResponse(429, {}, {"Retry-After": "7"})}),
                           None, tmp_path / "b", Stats())
    with pytest.raises(RateLimitExceeded):
        no_wait.get_json(url)


def test_call_budget_and_min_interval(tmp_path):
    routes = {f"{API}/a": FakeResponse(200, {}), f"{API}/b": FakeResponse(200, {})}
    slept = []
    clock = iter([0.0, 0.05, 0.3]).__next__
    client = BenignClient(FakeSession(routes), None, tmp_path, Stats(), max_calls=1,
                          min_interval=0.25, sleep=slept.append, clock=clock)
    client.get_json(f"{API}/a")
    with pytest.raises(BudgetExceeded):
        client.get_json(f"{API}/b")
    client.max_calls = 5
    client.get_json(f"{API}/b")
    assert slept and abs(slept[0] - 0.2) < 1e-9


def test_cache_path_matches_builder_client(tmp_path):
    client = GithubClient(NoNetwork(), None, tmp_path, Stats(), offline=True)
    url = f"{API}/repos/{REPO}/commits?per_page=100&page=1"
    assert cache_path(tmp_path, url) == client._cache_file(url)


def test_filters():
    known = known_prefixes({_sha("f"), "abcdef1"})
    assert list_reject_reason(_list_entry(_sha("f"), "x"), known) == "known_fix_commit"
    assert list_reject_reason(_list_entry("abcdef1" + "0" * 33, "x"), known) == \
        "known_fix_commit"
    assert list_reject_reason(_list_entry(_sha("1"), "Add feature"), known) is None
    assert list_reject_reason(_list_entry(_sha("1"), "x", login="renovate[bot]"), known) == "bot"
    assert list_reject_reason(_list_entry(_sha("1"), "x", login="renovate[bot]"), known,
                              include_bots=True) is None
    for msg in ("Fix CVE-2024-1234", "GHSA-xxxx", "prevent path traversal",
                "Sanitize input", "escape HTML", "ReDoS in parser", "auth bypass",
                "prototype pollution", "Upgrade to avoid vulnerability"):
        assert is_security_text(msg), msg
    for msg in ("Add pagination to list view", "Refactor tests", "Bump version to 2.1"):
        assert not is_security_text(msg), msg
    assert prs_reject_reason([{"title": "Harden config", "body": None}]) is not None
    assert prs_reject_reason([]) is None and prs_reject_reason({"__status__": 404}) is None

    def f(name, add=1, dele=1, patch="@@", status="modified"):
        return {"filename": name, "additions": add, "deletions": dele, "patch": patch,
                "status": status}
    assert detail_reject_reason({"files": [f("README.md")]}, 6, 2000) == "no_code_files"
    assert detail_reject_reason({"files": [f(f"a{i}.py") for i in range(7)]}, 6, 2000) == \
        "too_many_files"
    assert detail_reject_reason({"files": [f("a.py", 1500, 600)]}, 6, 2000) == \
        "too_many_changed_lines"
    assert detail_reject_reason({"files": [f("a.py", patch=None)]}, 6, 2000) == \
        "file_without_patch"
    assert detail_reject_reason({"files": [f("a.py"), f("test/x.py", 5000)]}, 6, 2000) is None


def test_repo_splits_order_and_known_commits(tmp_path):
    pr_eval = tmp_path / "p.jsonl"
    rows = [{"repo": "o/dev1", "split": "dev", "meta": {"commit": "c1" * 20}},
            {"repo": "o/res", "split": "reserve", "meta": {"repo_side": None}},
            {"repo": "O/Dev1", "split": "reserve", "meta": {"repo_side": "dev"}},
            {"repo": "o/test1", "split": "reserve", "meta": {"repo_side": "test"}}]
    pr_eval.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    splits, known = load_repo_splits(pr_eval)
    assert splits == {"o/dev1": "dev", "o/res": "reserve", "o/test1": "test"}
    assert known == {"c1" * 20}
    order = repo_order({"a/x": "reserve", "b/y": "dev", "c/z": "test", "d/w": "dev"}, 42)
    assert set(order[:2]) == {"b/y", "d/w"}
    assert order[2:] == ["c/z", "a/x"]
    rows.append({"repo": "o/test1", "split": "dev", "meta": {}})
    pr_eval.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_repo_splits(pr_eval)

    cache = tmp_path / "cache"
    cache.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("GHSA-1.json", json.dumps({"id": "GHSA-1", "references": [
            {"type": "FIX", "url": f"https://github.com/O/Dev1/commit/{'e' * 40}"},
            {"type": "FIX", "url": f"https://github.com/other/repo/commit/{'9' * 40}"}]}))
    (cache / "PyPI_all.zip").write_bytes(buf.getvalue())
    (cache / "processed_advisories.json").write_text(json.dumps(
        {"X": {"pairs": [{"commit": "7" * 40}]}}), encoding="utf-8")
    assert known_fix_commits(cache, {"o/dev1"}) == {"e" * 40, "7" * 40}


def test_estimate_calls():
    est = estimate_calls(n_repos=480, max_commits=300, per_repo=3, pages=1, check_prs=True)
    assert est["repos_needed"] == 100 and est["list_calls"] == 125
    assert est["total_calls"] == (est["list_calls"] + est["commit_detail_calls"]
                                  + est["linked_pr_calls"] + est["content_calls"])
    assert estimate_calls(480, 300, 3, 1, True, cached_lists=125)["list_calls"] == 0
    assert estimate_calls(10, 300, 3, 1, False)["repos_needed"] == 10
