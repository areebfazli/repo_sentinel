"""Unit tests for the pure logic in scripts/build_corpus_from_osv.py.

These only exercise module-level functions that do no network I/O (commit-URL
parsing, hunk-header parsing/overlap, CWE->category mapping, whitespace-only
change detection, the advisory split, and the dedupe key), matching the
module's own contract that this logic is importable without triggering
network calls.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.build_corpus_from_osv import (  # noqa: E402
    cwe_to_category,
    function_overlaps_hunks,
    is_whitespace_only_change,
    make_dedupe_key,
    parse_fix_commit_url,
    parse_hunk_ranges,
    split_by_advisory,
)

# --- commit-URL parsing -----------------------------------------------------


def test_parses_plain_commit_url():
    assert parse_fix_commit_url("https://github.com/django/django/commit/abc1234") == (
        "django",
        "django",
        "abc1234",
    )


def test_parses_plain_commit_url_with_full_sha_and_trailing_fragment():
    sha = "a" * 40
    url = f"https://github.com/owner/repo/commit/{sha}#diff-1234"
    assert parse_fix_commit_url(url) == ("owner", "repo", sha)


def test_parses_pull_commits_form():
    url = "https://github.com/owner/repo/pull/42/commits/deadbeef"
    assert parse_fix_commit_url(url) == ("owner", "repo", "deadbeef")


def test_parses_pull_commits_form_with_trailing_path():
    url = "https://github.com/owner/repo/pull/42/commits/deadbeef.patch"
    assert parse_fix_commit_url(url) == ("owner", "repo", "deadbeef")


def test_rejects_non_matching_urls():
    assert parse_fix_commit_url("https://github.com/owner/repo") is None
    assert parse_fix_commit_url("https://github.com/owner/repo/issues/1") is None
    assert parse_fix_commit_url("https://gitlab.com/owner/repo/commit/abc1234") is None
    assert parse_fix_commit_url("https://github.com/owner/repo/commits/abc1234") is None
    assert parse_fix_commit_url("") is None
    assert parse_fix_commit_url(None) is None


# --- hunk-header parsing + overlap ------------------------------------------


def test_parse_hunk_ranges_single_hunk():
    patch = "@@ -1,3 +1,4 @@\n a\n-b\n+b1\n+b2\n c\n"
    assert parse_hunk_ranges(patch) == [(1, 4)]


def test_parse_hunk_ranges_multiple_hunks():
    patch = "@@ -1,2 +1,2 @@\n a\n+b\n@@ -10,2 +11,3 @@\n j\n+k\n+l\n"
    assert parse_hunk_ranges(patch) == [(1, 2), (11, 13)]


def test_parse_hunk_ranges_default_count_is_one():
    patch = "@@ -5 +7 @@\n-old\n+new\n"
    assert parse_hunk_ranges(patch) == [(7, 7)]


def test_parse_hunk_ranges_zero_count_contributes_nothing():
    # A pure-deletion hunk (+c,0) adds no post-commit lines.
    patch = "@@ -5,2 +5,0 @@\n-a\n-b\n"
    assert parse_hunk_ranges(patch) == []


def test_function_overlaps_hunks_true_when_overlapping():
    hunks = [(10, 20)]
    assert function_overlaps_hunks(15, 25, hunks) is True  # partial overlap
    assert function_overlaps_hunks(5, 12, hunks) is True  # partial overlap other side
    assert function_overlaps_hunks(10, 20, hunks) is True  # exact match


def test_function_overlaps_hunks_false_when_disjoint():
    hunks = [(10, 20)]
    assert function_overlaps_hunks(1, 9, hunks) is False
    assert function_overlaps_hunks(21, 30, hunks) is False


def test_function_overlaps_hunks_false_for_empty_hunks():
    assert function_overlaps_hunks(1, 100, []) is False


# --- CWE -> category ---------------------------------------------------------


def test_cwe_to_category_sqli():
    assert cwe_to_category(["CWE-89"]) == "sqli"


def test_cwe_to_category_multi_cwe_arrow_cmd_injection():
    assert cwe_to_category(["CWE-78"]) == "cmd_injection"
    assert cwe_to_category(["CWE-94"]) == "cmd_injection"


def test_cwe_to_category_multi_cwe_arrow_secrets():
    assert cwe_to_category(["CWE-798"]) == "secrets"
    assert cwe_to_category(["CWE-522"]) == "secrets"


def test_cwe_to_category_first_match_wins_in_listed_order():
    # sqli (CWE-89) is listed before secrets (CWE-798); with both present, sqli wins.
    assert cwe_to_category(["CWE-798", "CWE-89"]) == "sqli"


def test_cwe_to_category_no_match_is_other():
    assert cwe_to_category(["CWE-999999"]) == "other"


def test_cwe_to_category_no_cwe_at_all_is_other():
    assert cwe_to_category([]) == "other"
    assert cwe_to_category(None) == "other"


# --- whitespace/comment-only change detection --------------------------------


def test_whitespace_only_change_python_reformatting():
    before = "def foo(x):\n    return x+1\n"
    after = "def foo(x):\n\n    return x + 1\n\n"
    assert is_whitespace_only_change(before, after, "python") is True


def test_whitespace_only_change_python_comment_added_only():
    before = "def foo(x):\n    return x + 1\n"
    after = "def foo(x):\n    # explain\n    return x + 1\n"
    assert is_whitespace_only_change(before, after, "python") is True


def test_whitespace_only_change_python_real_logic_change():
    before = "def foo(x):\n    return x + 1\n"
    after = "def foo(x):\n    return x + 2\n"
    assert is_whitespace_only_change(before, after, "python") is False


def test_whitespace_only_change_javascript_block_comment():
    before = "function foo(x) {\n  return x + 1;\n}\n"
    after = "function foo(x) {\n  /* explain */\n  return x + 1;\n}\n"
    assert is_whitespace_only_change(before, after, "javascript") is True


def test_whitespace_only_change_javascript_real_logic_change():
    before = "function foo(x) {\n  return x + 1;\n}\n"
    after = "function foo(x) {\n  return x - 1;\n}\n"
    assert is_whitespace_only_change(before, after, "javascript") is False


# --- advisory split invariant -------------------------------------------------


def _pair(advisory_id, fn):
    return {
        "advisory_id": advisory_id,
        "function_name": fn,
        "vulnerable_code": f"{advisory_id}:{fn}",
    }


def test_split_by_advisory_no_advisory_appears_on_both_sides():
    pairs = []
    for adv in ("GHSA-a", "GHSA-b", "GHSA-c", "GHSA-d", "GHSA-e", "GHSA-f", "GHSA-g", "GHSA-h"):
        for fn in ("f1", "f2"):
            pairs.append(_pair(adv, fn))

    corpus, held_out = split_by_advisory(pairs, eval_fraction=0.25, seed=42)

    corpus_advisories = {p["advisory_id"] for p in corpus}
    eval_advisories = {p["advisory_id"] for p in held_out}
    assert corpus_advisories.isdisjoint(eval_advisories)
    assert corpus_advisories | eval_advisories == {
        "GHSA-a",
        "GHSA-b",
        "GHSA-c",
        "GHSA-d",
        "GHSA-e",
        "GHSA-f",
        "GHSA-g",
        "GHSA-h",
    }
    assert len(corpus) + len(held_out) == len(pairs)


def test_split_by_advisory_is_deterministic_given_seed():
    pairs = [_pair(f"GHSA-{i}", "fn") for i in range(20)]
    corpus1, eval1 = split_by_advisory(pairs, eval_fraction=0.3, seed=7)
    corpus2, eval2 = split_by_advisory(pairs, eval_fraction=0.3, seed=7)
    assert [p["advisory_id"] for p in corpus1] == [p["advisory_id"] for p in corpus2]
    assert [p["advisory_id"] for p in eval1] == [p["advisory_id"] for p in eval2]


def test_split_by_advisory_holds_out_roughly_the_requested_fraction():
    pairs = [_pair(f"GHSA-{i}", "fn") for i in range(40)]
    corpus, held_out = split_by_advisory(pairs, eval_fraction=0.25, seed=1)
    eval_advisories = {p["advisory_id"] for p in held_out}
    # round(40 * 0.25) == 10 advisories held out, each contributing 1 pair here.
    assert len(eval_advisories) == 10
    assert len(held_out) == 10
    assert len(corpus) == 30


# --- dedupe key ---------------------------------------------------------------


def test_dedupe_key_identical_inputs_match():
    k1 = make_dedupe_key("owner/repo", "a/b.py", "foo", "def foo(): pass")
    k2 = make_dedupe_key("owner/repo", "a/b.py", "foo", "def foo(): pass")
    assert k1 == k2


def test_dedupe_key_differs_on_any_field():
    base = make_dedupe_key("owner/repo", "a/b.py", "foo", "def foo(): pass")
    assert base != make_dedupe_key("owner/other", "a/b.py", "foo", "def foo(): pass")
    assert base != make_dedupe_key("owner/repo", "a/c.py", "foo", "def foo(): pass")
    assert base != make_dedupe_key("owner/repo", "a/b.py", "bar", "def foo(): pass")
    assert base != make_dedupe_key("owner/repo", "a/b.py", "foo", "def foo(): return 1")
