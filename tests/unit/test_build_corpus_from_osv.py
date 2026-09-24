"""Unit tests for the pure logic in scripts/build_corpus_from_osv.py.

These only exercise module-level functions that do no network I/O (commit-URL
parsing, hunk-header parsing/overlap, CWE->category mapping, whitespace-only
change detection, the advisory split, the dedupe key, and the per-pair
quality score/filter), matching the module's own contract that this logic is
importable without triggering network calls. The one filesystem test writes
under pytest's ``tmp_path`` only.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.build_corpus_from_osv import (  # noqa: E402
    annotate_quality,
    apply_quality_filter,
    build_ecosystem_outputs,
    build_eval_lines,
    code_files_in_commit,
    compute_diff_quality,
    count_functions_per_commit,
    count_security_tokens,
    cwe_to_category,
    function_overlaps_hunks,
    is_whitespace_only_change,
    make_dedupe_key,
    parse_fix_commit_url,
    parse_hunk_ranges,
    rejected_path_for,
    split_by_advisory,
    tokenize_code,
    write_ecosystem_outputs,
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



# --- quality: rename-only detection -------------------------------------------
#
# Decision (documented in compute_diff_quality): a *consistent 1:1* identifier
# substitution across the whole function counts as rename-only even when it
# changes semantics (e.g. calling ``url_for`` instead of ``url``) — token shape
# can't tell a semantic rename from a cosmetic one. What is NOT a rename: an
# identifier mapped to two names, two identifiers merged into one (including
# into a name the function already used), keyword/literal/operator changes.

KERAS_OLD = """def save_weights(f, layers):
    \"\"\"Save weights to f.

    Args: f, an h5 file.
    \"\"\"
    for layer in layers:
        f.attrs[layer.name] = layer.get_weights()  # store
    return f
"""
KERAS_NEW = """def save_weights(group, layers):
    \"\"\"Save weights to an h5 group.

    Args:
        group: an h5 group.
    \"\"\"
    for layer in layers:
        group.attrs[layer.name] = layer.get_weights()
    return group
"""


def test_rename_only_param_rename_throughout_with_docstring_edits():
    q = compute_diff_quality(KERAS_OLD, KERAS_NEW, "python", "other")
    assert q["rename_only"] is True
    assert q["changed_stmt_lines"] == 0
    assert q["renamed_identifiers"] == {"f": "group"}


def test_rename_only_consistent_callee_substitution_counts_as_rename():
    # url(...) -> url_for(...) is a different function being called, but it is a
    # consistent 1:1 substitution, so by design it is rename-only.
    old = "def nxt():\n    n = request.args.get('next')\n    return url('index')\n"
    new = "def nxt():\n    n = request.args.get('next')\n    return url_for('index')\n"
    q = compute_diff_quality(old, new, "python", "open_redirect")
    assert q["rename_only"] is True
    assert q["renamed_identifiers"] == {"url": "url_for"}


def test_not_rename_when_target_name_already_used_in_function():
    # The pgadmin CVE-2024-4215 shape: url_for is already called elsewhere, so
    # url -> url_for merges two identifiers (many-to-one) — a real change.
    old = "def nxt():\n    r = url_for('reg')\n    return url('index')\n"
    new = "def nxt():\n    r = url_for('reg')\n    return url_for('index')\n"
    q = compute_diff_quality(old, new, "python", "open_redirect")
    assert q["rename_only"] is False
    assert q["changed_stmt_lines"] == 2


def test_not_rename_when_identifier_renamed_only_on_some_lines():
    old = "def f(x):\n    a = x + 1\n    b = x + 2\n    return a + b\n"
    new = "def f(x):\n    a = y + 1\n    b = x + 2\n    return a + b\n"
    q = compute_diff_quality(old, new, "python", "other")
    assert q["rename_only"] is False
    assert q["changed_stmt_lines"] == 2


def test_not_rename_when_keyword_or_literal_changes():
    kw = compute_diff_quality(
        "def f(x):\n    if x:\n        g()\n", "def f(x):\n    while x:\n        g()\n",
        "python", "other",
    )
    assert kw["rename_only"] is False and kw["changed_stmt_lines"] == 2
    lit = compute_diff_quality(
        "def f(x):\n    return x + 1\n", "def f(x):\n    return x + 2\n", "python", "other"
    )
    assert lit["rename_only"] is False and lit["changed_stmt_lines"] == 2


def test_rename_only_javascript():
    old = "function f(a) {\n  // add one\n  return a + 1;\n}\n"
    new = "function f(value) {\n  /* add one */\n  return value + 1;\n}\n"
    q = compute_diff_quality(old, new, "javascript", "xss")
    assert q["rename_only"] is True
    assert q["renamed_identifiers"] == {"a": "value"}


def test_rename_plus_real_change_counts_only_the_real_line():
    old = "def f(a):\n    x = a\n    return run(x)\n"
    new = "def f(b):\n    x = b\n    return run(x, safe=True)\n"
    q = compute_diff_quality(old, new, "python", "other")
    assert q["rename_only"] is False
    assert q["renamed_identifiers"] == {"a": "b"}
    assert q["changed_stmt_lines"] == 2  # the run(...) line, removed + added


def test_security_rename_flag():
    md5 = compute_diff_quality(
        "def h(b):\n    return hashlib.md5(b).hexdigest()\n",
        "def h(b):\n    return hashlib.sha256(b).hexdigest()\n",
        "python", "weak_crypto",
    )
    assert md5["rename_only"] is True and md5["security_rename"] is True
    yaml_fix = compute_diff_quality(
        "def l(s):\n    return yaml.load(s)\n",
        "def l(s):\n    return yaml.safe_load(s)\n",
        "python", "cmd_injection",
    )
    assert yaml_fix["rename_only"] is True and yaml_fix["security_rename"] is True
    cosmetic = compute_diff_quality(KERAS_OLD, KERAS_NEW, "python", "other")
    assert cosmetic["security_rename"] is False


# --- quality: comment/docstring-only diffs --------------------------------------


def test_docstring_and_comment_only_diff_has_no_stmt_changes():
    old = 'def f(x):\n    """Old doc."""\n    # old comment\n    return x\n'
    new = 'def f(x):\n    """New doc,\n    two lines."""\n    # new comment\n    return x  # tail\n'
    q = compute_diff_quality(old, new, "python", "other")
    assert q["changed_stmt_lines"] == 0
    assert q["rename_only"] is False


def test_docstring_quote_style_change_is_not_a_stmt_change():
    old = 'def f():\n    "Doc."\n    return 1\n'
    new = 'def f():\n    """Doc."""\n    return 1\n'
    assert compute_diff_quality(old, new, "python", "other")["changed_stmt_lines"] == 0


def test_javascript_comment_only_diff_has_no_stmt_changes():
    old = "function f(x) {\n  // a\n  return x;\n}\n"
    new = "function f(x) {\n  /* b\n     c */\n  return x; // d\n}\n"
    assert compute_diff_quality(old, new, "javascript", "other")["changed_stmt_lines"] == 0


def test_tokenizer_keeps_strings_that_are_not_statements():
    toks = tokenize_code('def f():\n    """doc"""\n    x = "kept"\n    g(\n        "arg"\n    )\n',
                         "python")
    strings = [t.text for t in toks if t.kind == "str"]
    assert strings == ["|kept", "|arg"]


# --- quality: changed_stmt_lines counting -----------------------------------------


def test_changed_stmt_lines_counts_removed_plus_added():
    old = "def f(q, cur):\n    cur.execute('SELECT %s' % q)\n    return cur\n"
    new = "def f(q, cur):\n    cur.execute('SELECT %s', (q,))\n    return cur\n"
    assert compute_diff_quality(old, new, "python", "sqli")["changed_stmt_lines"] == 2


def test_changed_stmt_lines_pure_insertion():
    old = "def f(p):\n    return open(p)\n"
    new = "def f(p):\n    check(p)\n    return open(p)\n"
    assert compute_diff_quality(old, new, "python", "path_traversal")["changed_stmt_lines"] == 1


def test_changed_stmt_lines_ignores_reflow_and_magic_trailing_comma():
    old = "def f(a):\n    return g(a, 1)\n"
    new = "def f(a):\n    return g(\n        a,\n        1,\n    )\n"
    q = compute_diff_quality(old, new, "python", "other")
    assert q["changed_stmt_lines"] == 0 and q["rename_only"] is False


def test_changed_stmt_lines_counts_multiline_string_change_once_per_side():
    old = 'def f(c):\n    c.execute("""\n        SELECT a\n        FROM t\n    """)\n'
    new = 'def f(c):\n    c.execute("""\n        SELECT a\n        FROM t WHERE x\n    """)\n'
    assert compute_diff_quality(old, new, "python", "sqli")["changed_stmt_lines"] == 2


# --- quality: security token counting ----------------------------------------------


def test_security_tokens_counts_category_keywords_in_changed_lines_only():
    old = "def f(q, cur):\n    cur.execute('x')\n    cur.execute('SELECT %s' % q)\n"
    new = "def f(q, cur):\n    cur.execute('x')\n    cur.execute('SELECT %s', (q,))\n"
    # Changed lines only (the unchanged cur.execute('x') line is not counted):
    # per side execute + SELECT + %s = 3, times two sides.
    assert compute_diff_quality(old, new, "python", "sqli")["security_tokens"] == 6


def test_count_security_tokens_is_case_insensitive_and_category_scoped():
    text = "subprocess.Popen(cmd, shell=True)"
    assert count_security_tokens(text, "cmd_injection") == 3  # subprocess, Popen, shell=
    assert count_security_tokens(text, "sqli") == 0


def test_count_security_tokens_other_uses_union_of_all_lists():
    text = "cursor.execute(q); pickle.loads(b); os.path.join(a, b)"
    assert count_security_tokens(text, "other") >= 5
    assert count_security_tokens(text, "other") == count_security_tokens(text, "no_such_category")
    assert count_security_tokens("x = 1", "other") == 0


# --- quality: per-commit counts + filter ---------------------------------------------


def _qpair(advisory_id, fn, *, commit="c1", category="sqli", old=None, new=None, file_path="a.py"):
    return {
        "cve_id": advisory_id,
        "category": category,
        "language": "python",
        "vulnerable_code": old or f"def {fn}(q, c):\n    c.execute('SELECT %s' % q)\n",
        "fixed_code": new or f"def {fn}(q, c):\n    c.execute('SELECT %s', (q,))\n",
        "repo": "o/r",
        "commit": commit,
        "file_path": file_path,
        "function_name": fn,
        "advisory_id": advisory_id,
    }


def test_count_functions_per_commit_counts_distinct_functions_across_files():
    pairs = [
        _qpair("A", "f1", file_path="a.py"),
        _qpair("A", "f1", file_path="b.py"),
        _qpair("B", "f1", file_path="a.py"),  # same commit via a 2nd advisory: no double count
        _qpair("C", "g", commit="c2"),
    ]
    assert count_functions_per_commit(pairs) == {("o/r", "c1"): 2, ("o/r", "c2"): 1}


def test_code_files_in_commit_counts_supported_non_test_files():
    commit = {"files": [{"filename": "pkg/a.py"}, {"filename": "pkg/b.js"},
                        {"filename": "tests/test_a.py"}, {"filename": "README.md"}]}
    assert code_files_in_commit(commit) == 2
    assert code_files_in_commit(None) is None
    assert code_files_in_commit({"__status__": 404}) is None


def test_functions_per_commit_cap_rejects_every_pair_of_a_broad_commit():
    broad = [_qpair("A", f"f{i}", commit="broad") for i in range(7)]
    narrow = [_qpair("B", f"g{i}", commit="narrow") for i in range(6)]
    annotated = annotate_quality(broad + narrow, {("o/r", "broad"): 7, ("o/r", "narrow"): 1})
    assert {e["quality"]["files_in_commit"] for e in annotated} == {7, 1}

    kept, rejected = apply_quality_filter(annotated, 1, 6, False)
    assert {e["function_name"] for e in kept} == {f"g{i}" for i in range(6)}  # exactly 6 is OK
    assert len(rejected) == 7
    assert {e["reject_reason"] for e in rejected} == {"broad_commit"}

    kept, rejected = apply_quality_filter(annotated, 1, 0, False)  # 0 disables the cap
    assert len(kept) == 13 and not rejected


def test_filter_reasons_and_drop_other():
    rename = _qpair("A", "r", commit="c1", old=KERAS_OLD, new=KERAS_NEW)
    doc = _qpair("B", "d", commit="c2", old='def d():\n    "a"\n    return 1\n',
                 new='def d():\n    "b"\n    return 1\n')
    other = _qpair("C", "o", commit="c3", category="other")
    real = _qpair("D", "s", commit="c4")
    annotated = annotate_quality([rename, doc, other, real], {})

    kept, rejected = apply_quality_filter(annotated, 1, 6, False)
    assert {e["function_name"] for e in kept} == {"o", "s"}
    assert {e["function_name"]: e["reject_reason"] for e in rejected} == {
        "r": "rename_only",
        "d": "comment_or_format_only",
    }

    kept, rejected = apply_quality_filter(annotated, 3, 6, True)
    reasons = {e["function_name"]: e["reject_reasons"] for e in rejected}
    assert reasons["o"] == ["few_changed_stmt_lines", "other_category"]
    assert reasons["s"] == ["few_changed_stmt_lines"]
    assert not kept


def test_keep_security_renames_exempts_only_flagged_renames():
    md5 = _qpair("A", "h", commit="c1", category="weak_crypto",
                 old="def h(b):\n    return hashlib.md5(b)\n",
                 new="def h(b):\n    return hashlib.sha256(b)\n")
    cosmetic = _qpair("B", "save_weights", commit="c2", category="other",
                      old=KERAS_OLD, new=KERAS_NEW)
    annotated = annotate_quality([md5, cosmetic], {})
    kept, rejected = apply_quality_filter(annotated, 1, 6, False)
    assert not kept and len(rejected) == 2
    kept, rejected = apply_quality_filter(annotated, 1, 6, False, keep_security_renames=True)
    assert [e["function_name"] for e in kept] == ["h"]
    assert [e["function_name"] for e in rejected] == ["save_weights"]


def test_annotate_quality_does_not_mutate_input_pairs():
    pairs = [_qpair("A", "f")]
    annotated = annotate_quality(pairs, {})
    assert "quality" not in pairs[0]
    assert set(annotated[0]["quality"]) == {
        "changed_stmt_lines", "rename_only", "renamed_identifiers", "security_rename",
        "security_tokens", "functions_in_commit", "files_in_commit",
    }


# --- outputs: rejected file + eval built from filtered pairs only ----------------------


def _mixed_pairs(n_advisories=40):
    """Per advisory: one real fix, one rename-only bystander; every 5th advisory
    is a broad commit (7 functions)."""
    pairs = []
    for i in range(n_advisories):
        adv = f"GHSA-{i:03d}"
        pairs.append(_qpair(adv, "fix", commit=f"c{i}"))
        pairs.append(_qpair(adv, "bystander", commit=f"c{i}", old=KERAS_OLD, new=KERAS_NEW))
        if i % 5 == 0:
            pairs.extend(_qpair(adv, f"wide{k}", commit=f"c{i}") for k in range(5))
    return pairs


def test_eval_items_never_come_from_rejected_pairs():
    pairs = _mixed_pairs()
    corpus, held_out, rejected = build_ecosystem_outputs(
        pairs, {}, eval_fraction=0.25, seed=3,
        min_changed_stmt_lines=1, max_functions_per_commit=6, drop_other=False,
    )
    assert held_out and rejected and corpus
    rejected_keys = {(e["advisory_id"], e["function_name"]) for e in rejected}
    assert not rejected_keys & {(e["advisory_id"], e["function_name"]) for e in held_out}
    assert not rejected_keys & {(e["advisory_id"], e["function_name"]) for e in corpus}
    assert len(corpus) + len(held_out) + len(rejected) == len(pairs)

    # Every eval line is a kept held-out pair (rename-only bystanders use KERAS
    # code, which must never show up in the eval set).
    eval_lines = build_eval_lines(held_out)
    assert len(eval_lines) == 2 * len(held_out)
    assert all(line["code"] not in (KERAS_OLD, KERAS_NEW) for line in eval_lines)
    assert all("_bystander_" not in line["id"] and "_wide" not in line["id"] for line in eval_lines)

    # Still split by advisory, and the held-out advisories are the ones the
    # unfiltered split picks (the filter only removes items, never reshuffles).
    assert {e["advisory_id"] for e in corpus}.isdisjoint({e["advisory_id"] for e in held_out})
    _, unfiltered_eval = split_by_advisory(pairs, 0.25, 3)
    held_out_rejected = {e["advisory_id"] for e in rejected if e["held_out"]}
    assert {e["advisory_id"] for e in held_out} | held_out_rejected == {
        e["advisory_id"] for e in unfiltered_eval
    }


def test_write_ecosystem_outputs_writes_rejected_file_outside_ingest_glob(tmp_path):
    corpus, _, rejected = build_ecosystem_outputs(
        _mixed_pairs(10), {}, eval_fraction=0.0, seed=1,
        min_changed_stmt_lines=1, max_functions_per_commit=6, drop_other=False,
    )
    out_path, rej_path = write_ecosystem_outputs(tmp_path, "PyPI", corpus, rejected)

    assert out_path == tmp_path / "osv_pypi.json"
    assert rej_path == rejected_path_for(tmp_path, "PyPI")
    assert rej_path == tmp_path / "rejected" / "osv_pypi_rejected.json"

    written_corpus = json.loads(out_path.read_text(encoding="utf-8"))
    written_rejected = json.loads(rej_path.read_text(encoding="utf-8"))
    assert len(written_corpus) == len(corpus) and len(written_rejected) == len(rejected)
    assert all("advisory_id" not in e and "quality" in e for e in written_corpus)
    assert all(e["reject_reason"] and e["advisory_id"] for e in written_rejected)
    assert {e["reject_reason"] for e in written_rejected} == {"rename_only", "broad_commit"}

    # The ingest script loads {out_dir}/*.json: it must see only the kept corpus.
    from scripts.ingest_cve_corpus import load_cves

    assert len(load_cves(tmp_path)) == len(corpus)


def test_eval_ids_are_unique_and_match_the_offline_migration():
    from scripts.migrate_eval_ids import migrate_lines

    base = {"cve_id": "CVE-1", "function_name": "parse", "language": "python",
            "category": "other"}
    pairs = [{**base, "vulnerable_code": "def parse(a): v1", "fixed_code": "def parse(a): f1"},
             {**base, "vulnerable_code": "def parse(b): v2", "fixed_code": "def parse(b): f2"},
             {**base, "vulnerable_code": "def parse(a): v1", "fixed_code": "def parse(a): f1"}]
    lines = build_eval_lines(pairs)
    ids = [ln["id"] for ln in lines]
    assert len(ids) == 4 == len(set(ids))  # exact duplicate pair emitted once
    assert ids[0].startswith("CVE-1_parse_") and ids[0].endswith("_vuln")
    assert ids[1] == ids[0][:-5] + "_safe"
    # Old-style ids (<advisory>_<function>), migrated offline, give the same ids.
    old = [{**ln, "id": "CVE-1_parse_" + ln["id"].rsplit("_", 1)[1]}
           for ln in build_eval_lines(pairs[:2]) + build_eval_lines(pairs[2:])]
    migrated, stats = migrate_lines(old)
    assert migrated == lines and stats["duplicates_dropped"] == 1
    assert migrate_lines(migrated)[0] == migrated  # idempotent
