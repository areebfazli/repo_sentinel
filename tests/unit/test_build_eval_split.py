"""Tests for scripts/build_eval_split.py: the grouping invariant (no advisory,
repo or near-duplicate across dev/test; twins together), determinism,
disjointness, counts, date strata, and the committed v1 manifest's consistency
with the eval datasets. Synthetic inputs, no cache, no network, no models."""
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_eval_split import (
    DEFAULT_ORDINARY,
    DEFAULT_OUT,
    DEFAULT_PAIR_FILES,
    advisory_date,
    build_manifest,
    identifiers,
    jaccard,
    load_jsonl,
    near_duplicate_links,
    pair_lines,
    stratum_quotas,
    year_bin,
)

WORDS = [f"w{i}" for i in range(400)]


def _code(rng: random.Random, n: int = 12) -> str:
    names = rng.sample(WORDS, n)
    return "def f(" + ", ".join(names[:3]) + "):\n" + "\n".join(
        f"    {a} = {b}" for a, b in zip(names[3:], names[4:], strict=False)) + "\n"


def _dataset(n_adv=40, seed=0):
    """Pairs over ``n_adv`` advisories (1-4 pairs each, 2 advisories per repo),
    ordinary functions in the same repos (plus a few ordinary-only repos)."""
    rng = random.Random(seed)
    lines, provenance, dates, ordinary = [], {}, {}, []
    for a in range(n_adv):
        adv = f"CVE-{2019 + a % 8}-{1000 + a}"
        repo = f"org/repo{a // 2}"
        lang = "javascript" if a % 5 == 0 else "python"
        for k in range(1 + a % 4):
            base = f"{adv}_fn{k}_{a:04d}{k:04d}"
            vuln = _code(rng)
            for suffix, label, code in (("vuln", "vulnerable", vuln),
                                        ("safe", "safe", vuln + "    check()\n")):
                lines.append({"id": f"{base}_{suffix}", "language": lang, "label": label,
                              "category": "xss", "expected_cve_id": adv, "source": "osv",
                              "code": code})
            provenance[base] = {"aliases": [f"GHSA-{a}"], "repo": repo,
                                "commits": [(repo, f"c{a}")]}
        dates[adv] = advisory_date(adv, [f"{2019 + a % 8}-06-01T00:00:00Z"], [])
    for o in range(120):
        r = o % 25
        ordinary.append({"id": f"ordinary_{o}", "language": "python", "label": "safe",
                         "kind": "ordinary", "code": _code(rng), "repo": f"org/repo{r}",
                         "advisory_id": f"GHSA-{2 * r}" if r < 20 else f"GHSA-x{r}",
                         "length_matched": o % 3 == 0})
    rng.shuffle(lines)  # pairing must not depend on line order
    return pair_lines(lines), ordinary, provenance, dates


def _build(**kw):
    pairs, ordinary, prov, dates = _dataset()
    args = dict(seed=42, dev_pairs=20, test_pairs=12, dev_ordinary=50, test_ordinary=40,
                created="2026-01-01")
    args.update(kw)
    return build_manifest(pairs, ordinary, prov, dates, **args), pairs, ordinary, prov


def test_manifest_shape():
    m, *_ = _build()
    assert set(m) == {"version", "seed", "dev", "test", "meta"}
    assert m["version"] == 1 and m["seed"] == 42
    assert set(m["dev"]) == {"ids"} and set(m["test"]) == {"ids"}
    assert set(m["meta"]) == {"created", "counts", "grouping", "date_strata"}
    assert m["dev"]["ids"] == sorted(m["dev"]["ids"])


def test_dev_test_disjoint_and_twins_together():
    m, pairs, *_ = _build()
    dev, test = set(m["dev"]["ids"]), set(m["test"]["ids"])
    assert not dev & test
    for p in pairs:
        v, s = p["vuln"]["id"], p["safe"]["id"]
        assert (v in dev) == (s in dev) and (v in test) == (s in test)


def test_grouping_invariant_no_advisory_or_repo_straddles():
    m, pairs, ordinary, prov = _build()
    adv_of, repo_of = {}, {}
    for p in pairs:
        for x in (p["vuln"], p["safe"]):
            adv_of[x["id"]] = {x["expected_cve_id"], *prov[p["base"]]["aliases"]}
            repo_of[x["id"]] = prov[p["base"]]["repo"]
    for o in ordinary:
        adv_of[o["id"]], repo_of[o["id"]] = {o["advisory_id"]}, o["repo"]
    for key in (adv_of, repo_of):
        def vals(side, key=key):
            out = set()
            for i in m[side]["ids"]:
                v = key[i]
                out |= v if isinstance(v, set) else {v}
            return out
        assert not vals("dev") & vals("test")
    leak = m["meta"]["counts"]["leakage"]
    assert leak["advisories_in_both"] == 0 and leak["repos_in_both"] == 0


def test_advisory_grouping_mode_still_keeps_advisories_whole():
    m, pairs, *_ = _build(group_by="advisory")
    side = {i: "dev" for i in m["dev"]["ids"]} | {i: "test" for i in m["test"]["ids"]}
    by_adv = {}
    for p in pairs:
        s = side.get(p["vuln"]["id"])
        if s:
            assert by_adv.setdefault(p["vuln"]["expected_cve_id"], s) == s


def test_near_duplicates_are_grouped_together():
    pairs, ordinary, prov, dates = _dataset()
    # An ordinary function in an unrelated repo that copies a pair's vulnerable code.
    victim = pairs[0]
    ordinary.append({"id": "ordinary_copy", "language": "python", "label": "safe",
                     "kind": "ordinary", "code": victim["vuln"]["code"] + "    x9 = 1\n",
                     "repo": "other/fork", "advisory_id": "GHSA-fork", "length_matched": True})
    m = build_manifest(pairs, ordinary, prov, dates, dev_pairs=60, test_pairs=40,
                       dev_ordinary=200, test_ordinary=200, created="x")
    side = {i: "dev" for i in m["dev"]["ids"]} | {i: "test" for i in m["test"]["ids"]}
    if victim["vuln"]["id"] in side and "ordinary_copy" in side:
        assert side[victim["vuln"]["id"]] == side["ordinary_copy"]
    assert m["meta"]["counts"]["leakage"][
        "near_dup_code_pairs_across_splits_jaccard_ge_0.7"] == 0


def test_deterministic_for_seed_and_input_order():
    a, *_ = _build()
    b, *_ = _build()
    assert a == b
    pairs, ordinary, prov, dates = _dataset()
    c = build_manifest(pairs, list(reversed(ordinary)), prov, dates, seed=42, dev_pairs=20,
                       test_pairs=12, dev_ordinary=50, test_ordinary=40, created="2026-01-01")
    assert c["dev"] == a["dev"] and c["test"] == a["test"]


def test_counts_match_ids_and_targets():
    m, pairs, ordinary, _ = _build()
    counts = m["meta"]["counts"]
    ord_lm = {o["id"]: o["length_matched"] for o in ordinary}
    for side, n_p, n_o in (("dev", 20, 50), ("test", 12, 40)):
        ids = m[side]["ids"]
        c = counts[side]
        assert c["items"] == len(ids)
        assert c["vulnerable"] == sum(i.endswith("_vuln") for i in ids) == c["pairs"]
        assert c["ordinary"] == sum(i in ord_lm for i in ids) <= n_o
        assert c["ordinary_length_matched"] == sum(ord_lm.get(i, False) for i in ids)
        assert c["ordinary_length_matched"] + c["ordinary_not_length_matched"] == c["ordinary"]
        assert abs(c["pairs"] - n_p) <= 3
    assert (counts["dev"]["pairs"] + counts["test"]["pairs"] + counts["unassigned"]["pairs"]
            == len(pairs))
    assert (counts["dev"]["ordinary"] + counts["test"]["ordinary"]
            + counts["unassigned"]["ordinary"] == len(ordinary))
    strata = m["meta"]["date_strata"]["pairs_by_stratum"]
    assert sum(strata["dev"].values()) == counts["dev"]["pairs"]
    assert sum(strata["all"].values()) == len(pairs)


def test_length_matched_ordinary_kept_first_when_capped():
    m, _, ordinary, _ = _build(dev_ordinary=5, test_ordinary=5)
    lm = {o["id"] for o in ordinary if o["length_matched"]}
    for side in ("dev", "test"):
        picked = [i for i in m[side]["ids"] if i.startswith("ordinary_")]
        assert len(picked) <= 5
        assert all(i in lm for i in picked)


def test_pair_lines_rejects_unpaired():
    with pytest.raises(ValueError):
        pair_lines([{"id": "A_f_1234abcd_vuln"}])
    with pytest.raises(ValueError):
        pair_lines([{"id": "A_f_1234abcd"}])


def test_dates_and_bins():
    d = advisory_date("CVE-2009-1", ["2021-03-01T00:00:00Z", "2021-02-01T00:00:00Z"],
                      ["2022-05-01T00:00:00Z"])
    assert d["public_since"] == "2021-02-01" and d["date_source"] == "fix_commit"
    d = advisory_date("CVE-2009-1", [], ["2022-05-01T00:00:00Z"])
    assert d["public_since"] == "2022-05-01" and d["date_source"] == "osv_published"
    d = advisory_date("CVE-2009-1", [], [])
    assert d["public_since"] == "2009-01-01" and d["date_source"] == "cve_id_year"
    assert advisory_date("GHSA-x", [], [])["public_since"] is None
    assert year_bin("2021-12-31") == "<=2022"
    assert year_bin("2024-01-01") == "2024"
    assert year_bin("2027-01-01") == "2026+"
    assert year_bin(None) == "unknown"


def test_stratum_quotas_sum_and_proportion():
    q = stratum_quotas(150, {"a": 173, "b": 61, "c": 268})
    assert sum(q.values()) == 150
    assert q["c"] > q["a"] > q["b"]
    assert stratum_quotas(10, {}) == {}


def test_near_duplicate_links_and_jaccard():
    a = identifiers("def f(alpha, beta):\n    return alpha + beta + gamma + delta + eps")
    b = identifiers("def g(alpha, beta):\n    return alpha + beta + gamma + delta + eps")
    assert jaccard(a, b) >= 0.7
    assert "return" not in a and "def" not in a
    links = near_duplicate_links([a, b, frozenset({"x", "y", "z", "u", "v"})], 0.7, 5)
    assert [(i, j) for i, j, _ in links] == [(0, 1)]
    assert near_duplicate_links([a, b], 0.7, 50) == []  # too few identifiers


# --- the committed manifest --------------------------------------------------


@pytest.mark.skipif(not DEFAULT_OUT.exists(), reason="split manifest not built")
def test_committed_v1_manifest_consistent_with_datasets():
    m = json.loads(Path(DEFAULT_OUT).read_text())
    lines = [ln for p in DEFAULT_PAIR_FILES for ln in load_jsonl(p)]
    ordinary = load_jsonl(DEFAULT_ORDINARY)
    known = {ln["id"]: ln for ln in lines} | {o["id"]: o for o in ordinary}
    dev, test = set(m["dev"]["ids"]), set(m["test"]["ids"])
    assert not dev & test
    assert dev <= known.keys() and test <= known.keys()
    adv = {s: {known[i].get("expected_cve_id") or known[i].get("advisory_id") for i in ids}
           for s, ids in (("dev", dev), ("test", test))}
    assert not adv["dev"] & adv["test"]
    repo = {s: {known[i]["repo"].lower() for i in ids if known[i].get("repo")}
            for s, ids in (("dev", dev), ("test", test))}
    assert not repo["dev"] & repo["test"]
    for i in dev | test:
        if i.endswith("_vuln"):
            twin = i[:-5] + "_safe"
            assert (twin in dev) == (i in dev) and (twin in test) == (i in test)
    c = m["meta"]["counts"]
    assert c["dev"]["items"] == len(dev) and c["test"]["items"] == len(test)
    kinds = Counter(known[i].get("kind", "pair") for i in test)
    assert kinds["ordinary"] == c["test"]["ordinary"]
