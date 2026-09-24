"""Unit tests for the pure logic in scripts/build_ordinary_negatives.py.

No network, no models, no filesystem: hunk-overlap exclusion, dedupe against
eval/corpus bodies, the per-repo cap, stratification determinism, the
length-matched flag, the output schema and the ``kind`` convention.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.build_ordinary_negatives import (  # noqa: E402
    assign_length_matched,
    bin_index,
    body_hash,
    build_item,
    changed_line_ranges,
    held_out_advisories,
    infer_kind,
    language_quotas,
    quantile_edges,
    reject_reason,
    sample_candidates,
    source_commits,
)

PY_BODY = "def total(items):\n    s = 0\n    for i in items:\n        s += i\n    return s\n"


def _func(code=PY_BODY, start=10, name="total", repo="o/r", path="pkg/mod.py", lang="python"):
    return {
        "repo": repo, "commit": "c1", "advisory_id": "GHSA-1", "file_path": path,
        "name": name, "start_line": start, "end_line": start + len(code.splitlines()) - 1,
        "code": code, "language": lang,
    }


def _flt(**overrides):
    base = dict(
        hunk_ranges=[], min_lines=3, max_lines=120, drop_trivial=True,
        trivial_max_stmt_lines=3, mined_names=set(), known_hashes=set(),
    )
    base.update(overrides)
    return base


# --- hunk-overlap exclusion --------------------------------------------------


def test_changed_line_ranges_includes_pure_deletions():
    patch = "@@ -1,3 +1,4 @@\n ctx\n+new\n@@ -20,2 +21,0 @@\n-gone\n-gone\n@@ -40 +40 @@\n-a\n+b\n"
    assert changed_line_ranges(patch) == [(1, 4), (21, 22), (40, 40)]


def test_function_overlapping_a_hunk_is_excluded():
    f = _func(start=10)  # lines 10..14
    assert reject_reason(f, **_flt(hunk_ranges=[(14, 16)])) == "overlaps_fix_hunk"
    assert reject_reason(f, **_flt(hunk_ranges=[(1, 9), (15, 30)])) is None


def test_function_where_lines_were_only_deleted_is_excluded():
    f = _func(start=10)
    ranges = changed_line_ranges("@@ -12,2 +11,0 @@\n-x\n-y\n")
    assert reject_reason(f, **_flt(hunk_ranges=ranges)) == "overlaps_fix_hunk"


# --- other filters -----------------------------------------------------------


def test_length_bounds_and_trivial_flag():
    short = "def f():\n    return 1\n"
    assert reject_reason(_func(short), **_flt()) == "too_short"
    long_code = "def f():\n" + "    x = 1\n" * 130
    assert reject_reason(_func(long_code), **_flt()) == "too_long"
    accessor = 'def name(self):\n    """The name."""\n    # cached\n    return self._name\n'
    assert reject_reason(_func(accessor), **_flt()) == "trivial"
    assert reject_reason(_func(accessor), **_flt(drop_trivial=False)) is None


# --- dedupe against eval / corpus -------------------------------------------


def test_dedupe_against_eval_or_corpus_body_ignores_whitespace_and_comments():
    reformatted = (
        "def total(items):\n  # sum them\n  s=0\n  for i in items:\n      s+=i\n  return s"
    )
    known = {body_hash(reformatted, "python")}
    assert reject_reason(_func(), **_flt(known_hashes=known)) == (
        "duplicate_of_eval_or_corpus_body"
    )
    other = {body_hash(PY_BODY.replace("s += i", "s -= i"), "python")}
    assert reject_reason(_func(), **_flt(known_hashes=other)) is None


def test_same_function_as_a_mined_pair_is_excluded():
    mined = {("o/r", "pkg/mod.py", "total")}
    assert reject_reason(_func(), **_flt(mined_names=mined)) == "same_function_as_mined_pair"
    assert reject_reason(_func(repo="o/other"), **_flt(mined_names=mined)) is None


# --- split recomputation / source commits -----------------------------------


def _pair(aid, repo="o/r", commit="c1", fn="f"):
    return {"advisory_id": aid, "repo": repo, "commit": commit, "file_path": "a.py",
            "function_name": fn, "vulnerable_code": f"{aid}{fn}"}


def test_held_out_commits_drop_advisories_on_the_corpus_side_elsewhere():
    pypi = [_pair(f"A{i}", commit=f"c{i}") for i in range(10)]
    npm = [_pair(f"A{i}", commit=f"c{i}") for i in range(10)]  # same advisories
    eval_side, corpus_side = held_out_advisories({"PyPI": pypi, "npm": npm}, 0.3, 42)
    commits, dropped = source_commits(eval_side, corpus_side)
    corpus_ids = {p["advisory_id"] for ps in corpus_side.values() for p in ps}
    assert all(c["advisory_id"] not in corpus_ids for c in commits)
    assert set(dropped) <= corpus_ids
    # A commit shared with a corpus-side advisory is dropped too.
    ev = {"PyPI": [_pair("E1", commit="shared")]}
    co = {"PyPI": [_pair("C1", commit="shared")]}
    assert source_commits(ev, co) == ([], ["E1"])


# --- sampling ----------------------------------------------------------------


def _pool():
    out = []
    for r in range(12):
        lang = "javascript" if r % 4 == 0 else "python"
        for k in range(40):
            f = _func(repo=f"org/repo{r}", start=k * 10 + 1, name=f"f{k}", lang=lang)
            out.append(f)
    return out


def test_per_repo_cap_is_respected():
    picked = sample_candidates(_pool(), {"python": 200, "javascript": 60}, 7, seed=42)
    counts = Counter(c["repo"] for c in picked)
    assert max(counts.values()) <= 7
    assert len(picked) == 12 * 7  # every repo capped before the quotas are met


def test_stratified_sampling_is_deterministic_and_order_independent():
    pool = _pool()
    shuffled = pool[:]
    random.Random(1).shuffle(shuffled)
    quotas = {"python": 50, "javascript": 12}
    a = sample_candidates(pool, quotas, 25, seed=42)
    b = sample_candidates(shuffled, quotas, 25, seed=42)
    assert [build_item(c, set())["id"] for c in a] == [build_item(c, set())["id"] for c in b]
    assert Counter(c["language"] for c in a) == Counter(quotas)
    c = sample_candidates(pool, quotas, 25, seed=7)
    assert [build_item(x, set())["id"] for x in c] != [build_item(x, set())["id"] for x in a]


def test_language_quotas_follow_shares_without_backfill():
    assert language_quotas(1000, {"python": 0.798, "javascript": 0.202},
                           {"python": 5000, "javascript": 5000}) == {
        "python": 798, "javascript": 202}
    # A short pool is capped, not backfilled from the other language.
    assert language_quotas(1000, {"python": 0.8, "javascript": 0.2},
                           {"python": 5000, "javascript": 50}) == {
        "python": 800, "javascript": 50}


# --- length matching ---------------------------------------------------------


def test_quantile_bins():
    edges = quantile_edges(list(range(1, 101)), 4)
    assert edges[0] == 1 and edges[-1] == 100
    assert bin_index(1, edges) == 0 and bin_index(100, edges) == 3
    assert bin_index(0, edges) is None and bin_index(101, edges) is None


def test_length_matched_subset_follows_the_reference_distribution():
    ref = {"python": list(range(10, 60))}  # uniform 10..59
    # Heavily skewed short: many 10-19 line items, few long ones.
    lengths = [12] * 80 + [25] * 20 + [35] * 10 + [45] * 10 + [55] * 10 + [200] * 3
    items = [{"id": f"i{k}", "language": "python", "line_count": n}
             for k, n in enumerate(lengths)]
    matched = assign_length_matched(items, ref, 5, seed=42)
    chosen = [i["line_count"] for i in items if i["length_matched"]]
    assert matched == {"python": len(chosen)} and len(chosen) == 50
    edges = quantile_edges(ref["python"], 5)
    per_bin = Counter(bin_index(n, edges) for n in chosen)
    ref_bin = Counter(bin_index(n, edges) for n in ref["python"])
    # Same share per quintile bin as the reference (the long bins are the limit).
    assert per_bin == {b: round(50 * k / len(ref["python"])) for b, k in ref_bin.items()}
    assert 200 not in chosen
    again = [dict(i) for i in items]
    assign_length_matched(again, ref, 5, seed=42)
    assert [i["length_matched"] for i in again] == [i["length_matched"] for i in items]


def test_length_matched_is_false_without_a_reference_language():
    items = [{"id": "x", "language": "go", "line_count": 10}]
    assert assign_length_matched(items, {"python": [5, 10]}, 5, seed=1) == {"go": 0}
    assert items[0]["length_matched"] is False


# --- output schema / kind ----------------------------------------------------


def test_output_item_schema():
    item = build_item(_func(), corpus_repos={"o/r"})
    for key in ("id", "language", "label", "category", "expected_cve_id", "source", "code"):
        assert key in item  # the detection_eval*.jsonl fields
    assert item["label"] == "safe" and item["category"] == "none"
    assert item["expected_cve_id"] is None
    assert item["source"] == "osv_ordinary" and item["kind"] == "ordinary"
    assert item["repo"] == "o/r" and item["commit"] == "c1" and item["file_path"] == "pkg/mod.py"
    assert item["function_name"] == "total" and item["line_count"] == 5
    assert item["repo_in_corpus"] is True and item["length_matched"] is False
    assert not item["id"].endswith(("_vuln", "_safe"))  # a singleton for run_eval's pairing
    assert json.loads(json.dumps(item)) == item


def test_infer_kind_convention():
    assert infer_kind({"id": "x", "kind": "ordinary", "source": "osv_ordinary"}) == "ordinary"
    assert infer_kind({"id": "sqli_vuln_1", "source": "handwritten"}) == "handwritten"
    assert infer_kind({"id": "CVE-1_f_vuln", "source": "osv"}) == "vulnerable"
    assert infer_kind({"id": "CVE-1_f_safe", "source": "osv"}) == "fixed_twin"
