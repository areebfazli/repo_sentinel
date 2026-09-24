"""Tests for ml/evaluation/analyze_candidates.py on small synthetic inputs:
Semgrep row mapping across the id migration, candidate rules, union/overlap
counts, breakdowns and the static-first cost arithmetic. No Semgrep run, no
models, no network."""
import pytest

from ml.evaluation.analyze_candidates import (
    breakdown,
    calls_per_100,
    implications,
    map_semgrep_rows,
    ordinary_rates,
    pair_records,
    semgrep_flag,
    summarize_pairs,
    union_rate,
)

HIGH = {"rule_id": "python_eval_rule-eval", "severity": "high", "snippet": "eval(x)"}
LOW = {"rule_id": "python_tmpdir_rule-hardcodedtmp", "severity": "low", "snippet": "tmp"}
EXCLUDED_HIGH = {"rule_id": "python_assert_rule-assert-used", "severity": "high"}


def _line(i, suffix, code):
    return {"id": f"{i}_{suffix}", "code": code}


def test_map_semgrep_rows_skips_dropped_duplicates_and_checks_lines():
    old = [
        {"id": "A_f_vuln", "lines": 2, "hits": [HIGH]},
        {"id": "A_f_safe", "lines": 3, "hits": []},
        {"id": "A_f_vuln", "lines": 2, "hits": [HIGH]},  # exact duplicate, dropped
        {"id": "A_f_safe", "lines": 3, "hits": []},
        {"id": "A_f_vuln", "lines": 1, "hits": []},  # same prefix, different code
        {"id": "A_f_safe", "lines": 1, "hits": []},
    ]
    new = [
        _line("A_f_11111111", "vuln", "y = 1\neval(x)"),
        _line("A_f_11111111", "safe", "y = 1\nsafe(x)\nz"),
        _line("A_f_22222222", "vuln", "w"),
        _line("A_f_22222222", "safe", "v"),
    ]
    mapped, stats = map_semgrep_rows(old, new)
    assert mapped == {"A_f_11111111_vuln": [HIGH], "A_f_11111111_safe": [],
                      "A_f_22222222_vuln": [], "A_f_22222222_safe": []}
    assert stats["mapped_pairs"] == 2 and stats["skipped_old_pairs"] == ["A_f"]
    assert stats["unmapped_new_pairs"] == 0


def test_map_semgrep_rows_rejects_snippet_mismatch():
    old = [{"id": "A_f_vuln", "lines": 1, "hits": [HIGH]},
           {"id": "A_f_safe", "lines": 1, "hits": []}]
    new = [_line("A_f_11111111", "vuln", "nothing here"),
           _line("A_f_11111111", "safe", "x")]
    mapped, stats = map_semgrep_rows(old, new)
    assert mapped == {} and stats["unmapped_new_pairs"] == 1


def test_semgrep_flag_modes():
    assert semgrep_flag([HIGH], "evidence") and semgrep_flag([HIGH], "any")
    assert not semgrep_flag([LOW], "evidence") and semgrep_flag([LOW], "any")
    assert not semgrep_flag([EXCLUDED_HIGH], "evidence")
    assert not semgrep_flag([EXCLUDED_HIGH], "any")
    assert semgrep_flag([], "any") is False
    assert semgrep_flag(None, "any") is None


def _pairs():
    specs = [  # (base, category, vuln hits, fixed hits, rev guard, fix guard, date)
        ("P1_f_00000001", "cmd", [HIGH], [], "guard_removed", "none", "2020-01-01"),
        ("P2_f_00000002", "cmd", [], [], "guard_removed", "guard_added", "2020-01-01"),
        ("P3_f_00000003", "xss", [LOW], [HIGH], "none", "guard_removed", "2026-01-01"),
        ("P4_f_00000004", "xss", [], [], "none", "none", "2026-01-01"),
    ]
    pairs, hits, guard, dates = [], {}, {}, {}
    for base, cat, vh, fh, rev, fix, date in specs:
        adv = base.split("_")[0]
        v = {"id": f"{base}_vuln", "language": "python", "category": cat,
             "expected_cve_id": adv, "code": "x"}
        s = {**v, "id": f"{base}_safe"}
        pairs.append({"base": base, "vuln": v, "safe": s})
        hits[v["id"]], hits[s["id"]] = vh, fh
        guard[base] = {"rev_risk": rev, "fix_risk": fix, "rev_alert": False, "fix_alert": False}
        dates[adv] = {"public_since": date}
    split_of = {"P1_f_00000001_vuln": "dev", "P1_f_00000001_safe": "dev",
                "P3_f_00000003_vuln": "test", "P3_f_00000003_safe": "test"}
    return pair_records(pairs, hits, guard, split_of, dates, "2025-01-01")


def test_union_recall_fpr_and_overlap():
    recs = _pairs()
    s = summarize_pairs(recs, "evidence")
    rr, ff = s["recall_reverse_fix"], s["fpr_fix_direction"]
    assert s["pairs"] == 4
    assert rr["semgrep"]["k"] == 1 and rr["guard_diff"]["k"] == 2 and rr["union"]["k"] == 2
    assert rr["overlap"] == {"both": 1, "semgrep_only": 0, "guard_only": 1, "neither": 2}
    assert ff["semgrep"]["k"] == 1 and ff["guard_diff"]["k"] == 1 and ff["union"]["k"] == 1
    assert ff["overlap"]["both"] == 1
    s_any = summarize_pairs(recs, "any")
    assert s_any["recall_reverse_fix"]["union"]["k"] == 3  # the low-severity hit counts
    assert rr["union"]["ci95"][0] < 0.5 < rr["union"]["ci95"][1]


def test_breakdowns_by_split_period_category():
    recs = _pairs()
    by_split = breakdown(recs, "evidence", "split")
    assert by_split["unassigned"]["pairs"] == 2 and by_split["dev"]["pairs"] == 1
    by_period = breakdown(recs, "evidence", "period")
    assert by_period["before_cutoff"]["recall_reverse_fix"]["union"]["k"] == 2
    assert by_period["after_cutoff"]["recall_reverse_fix"]["union"]["k"] == 0
    by_cat = breakdown(recs, "evidence", "category")
    assert set(by_cat) == {"cmd", "xss"}


def test_pairs_without_semgrep_rows_are_excluded():
    recs = _pairs()
    recs[0]["sem_vuln_evidence"] = None
    assert summarize_pairs(recs, "evidence")["pairs"] == 3


def test_ordinary_rates():
    items = [{"id": f"o{i}", "length_matched": i < 2} for i in range(4)]
    hits = {"o0": [HIGH], "o1": [], "o2": [LOW], "o3": []}
    r = ordinary_rates(items, hits, {"o0": "dev", "o2": "test"}, "evidence")
    assert r["all"]["k"] == 1 and r["all"]["n"] == 4
    assert r["length_matched"]["k"] == 1 and r["not_length_matched"]["k"] == 0
    assert r["dev"]["k"] == 1 and r["test"]["k"] == 0
    assert ordinary_rates(items, hits, {}, "any")["all"]["k"] == 2


def test_cost_arithmetic():
    assert union_rate(0.0, 0.0) == 0.0
    assert union_rate(0.5, 0.5) == pytest.approx(0.75)
    # 1% base rate, recall 0.2, benign candidate rate 1% -> 0.2 + 0.99 calls per 100.
    assert calls_per_100(0.2, 0.01, 0.01) == pytest.approx(1.19)
    imp = implications(0.2, {"x": 0.01}, 0.25)
    assert imp["max_recall_perfect_verifier"] == 0.2
    assert imp["max_recall_with_verifier_tpr"] == pytest.approx(0.05)
    assert imp["calls_per_100_functions"]["x"]["1%"] == {"static_first": 1.19,
                                                         "llm_first": 100.0}
    assert implications(0.2, {}, None)["max_recall_with_verifier_tpr"] is None


def test_advisory_level_counts_any_flagged_pair():
    from ml.evaluation.analyze_candidates import advisory_level

    recs = _pairs()
    recs[1]["advisory"] = recs[0]["advisory"]  # P1 and P2 now share an advisory
    al = advisory_level(recs, "evidence")
    assert al["advisories"] == 3
    assert al["recall_reverse_fix"]["union"]["k"] == 1
    assert al["recall_reverse_fix"]["semgrep"]["k"] == 1
    assert al["fpr_fix_direction"]["union"]["k"] == 1  # only P3; guard_added is not a flag
