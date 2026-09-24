"""Candidate recall of a static-first design, offline.

Question: if Semgrep hits and guard_diff ``guard_removed`` changes were the only
things that sent a function to the LLM ("static analysis first, LLM as triage"),
what fraction of vulnerabilities would even reach the LLM, and at what cost?

Framing each held-out vulnerable/fixed pair as a PR diff:

- **reverse fix** (fixed -> vulnerable): a simulated vulnerability-introducing
  PR. A candidate if Semgrep hits the post-PR (vulnerable) code OR guard_diff
  reports ``guard_removed``. Candidate recall = fraction of pairs flagged.
- **fix direction** (vulnerable -> fixed): a benign, security-adjacent PR.
  Candidate FPR = Semgrep hit on the fixed code OR ``guard_removed``.
- **ordinary functions**: Semgrep hit rate only; guard_diff needs a diff, so it
  is not measurable on them. Its FPR on ordinary changes is taken from
  ``guard_diff_eval.json``'s controls (synthetic benign edits; real bystander
  edits from broad fix commits) when that file exists.

Semgrep results come from ``results/semgrep_eval.json`` (per-item rows, run in
snippet mode before the eval id migration: rows are mapped to the new ids by
file order, prefix, line counts and hit snippets; the 4 exact-duplicate pairs
the migration dropped are skipped). guard_diff is recomputed here (pure
Python / tree-sitter, ~10 s). Two Semgrep candidate rules:

- ``evidence``: severity >= high, minus the rules production excludes
  (``SEMGREP_MIN_SEVERITY`` / ``SEMGREP_EXCLUDED_RULES`` defaults) — what the
  pipeline passes to the LLM today;
- ``any``: any severity, minus the same excluded rules — the widest net.

Breakdowns: overall, per split (``ml/evaluation/splits/v1.json``), language,
category and before / after the split's date cutoff; Wilson 95% CIs. No
network, no models, no Semgrep run.

    python -m ml.evaluation.analyze_candidates   # -> results/candidate_recall.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.eval_guard_diff import rate  # noqa: E402
from scripts.build_eval_split import load_jsonl, pair_lines  # noqa: E402

DATASETS = ROOT / "ml" / "evaluation" / "datasets"
RESULTS = ROOT / "ml" / "evaluation" / "results"
PAIR_FILES = [DATASETS / "detection_eval_osv_pypi.jsonl",
              DATASETS / "detection_eval_osv_npm.jsonl"]
ORDINARY = DATASETS / "detection_eval_ordinary.jsonl"
SEMGREP = RESULTS / "semgrep_eval.json"
GUARD_EVAL = RESULTS / "guard_diff_eval.json"
LLM_ARM = RESULTS / "llm_arm_current_groq_qwen_29.json"
SPLIT = ROOT / "ml" / "evaluation" / "splits" / "v1.json"
DEFAULT_OUT = RESULTS / "candidate_recall.json"

SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
# Mirrors the production defaults in backend/app/config.py (SEMGREP_EXCLUDED_RULES).
EXCLUDED_RULES = frozenset({
    "python_assert_rule-assert-used",
    "python_random_rule-random",
    "python_requests_rule-request-without-timeout",
})
MODES = {"evidence": "high", "any": "info"}
BASE_RATES = (0.01, 0.02, 0.05)
_HASH_SUFFIX = re.compile(r"_[0-9a-f]{8}$")


# ---------------------------------------------------------------------------
# Pure logic (unit-tested in tests/unit/test_analyze_candidates.py)
# ---------------------------------------------------------------------------


def _n_lines(code: str) -> int:
    return len((code or "").splitlines())


def _snippets_present(hits: list[dict], code: str) -> bool:
    flat = " ".join((code or "").split())
    for h in hits:
        first = (h.get("snippet") or "").strip().splitlines()[:1]
        if first and " ".join(first[0].split()) not in flat:
            return False
    return True


def map_semgrep_rows(old_rows: list[dict], new_lines: list[dict]) -> tuple[dict, dict]:
    """Map one eval file's pre-migration Semgrep rows (``<prefix>_vuln`` /
    ``_safe`` in file order, ids without the hash) onto the migrated lines
    (same order, ``<prefix>_<hash8>_vuln``; exact duplicate pairs dropped).
    Walks both in order; an old pair is matched to the next new pair when the
    prefix, both line counts and every hit snippet agree, else it is counted as
    a dropped duplicate. Returns ``({new_id: hits}, stats)``."""
    mapped: dict[str, list] = {}
    dropped = []
    j = 0
    for k in range(0, len(old_rows) - 1, 2):
        ov, os_ = old_rows[k], old_rows[k + 1]
        prefix = ov["id"][:-5]
        if j + 1 < len(new_lines):
            nv, ns = new_lines[j], new_lines[j + 1]
            if (_HASH_SUFFIX.sub("", nv["id"][:-5]) == prefix
                    and ov.get("lines") == _n_lines(nv["code"])
                    and os_.get("lines") == _n_lines(ns["code"])
                    and _snippets_present(ov["hits"], nv["code"])
                    and _snippets_present(os_["hits"], ns["code"])):
                mapped[nv["id"]], mapped[ns["id"]] = ov["hits"], os_["hits"]
                j += 2
                continue
        dropped.append(prefix)
    return mapped, {"old_pairs": len(old_rows) // 2, "mapped_pairs": len(mapped) // 2,
                    "skipped_old_pairs": dropped, "unmapped_new_pairs": (len(new_lines) - j) // 2}


def semgrep_flag(hits: list[dict] | None, mode: str) -> bool | None:
    """Would these hits make the unit a candidate under ``mode``? ``None`` if
    the unit has no Semgrep row."""
    if hits is None:
        return None
    floor = SEVERITY_RANK[MODES[mode]]
    return any(SEVERITY_RANK.get(str(h.get("severity", "")).lower(), 0) >= floor
               and h.get("rule_id") not in EXCLUDED_RULES for h in hits)


def pair_records(pairs: list[dict], semgrep_hits: dict[str, list], guard: dict[str, dict],
                 split_of: dict[str, str], dates: dict[str, dict], cutoff: str) -> list[dict]:
    """One record per pair with every candidate signal, both directions."""
    out = []
    for p in pairs:
        v, s = p["vuln"], p["safe"]
        g = guard[p["base"]]
        since = (dates.get(v["expected_cve_id"]) or {}).get("public_since")
        rec = {
            "base": p["base"], "advisory": v["expected_cve_id"], "language": v["language"],
            "category": v["category"], "split": split_of.get(v["id"], "unassigned"),
            "period": ("unknown" if not since else
                       "before_cutoff" if since < cutoff else "after_cutoff"),
            "guard_rev": g["rev_risk"] == "guard_removed",
            "guard_fix": g["fix_risk"] == "guard_removed",
            "alert_rev": bool(g.get("rev_alert")), "alert_fix": bool(g.get("fix_alert")),
        }
        for mode in MODES:
            rec[f"sem_vuln_{mode}"] = semgrep_flag(semgrep_hits.get(v["id"]), mode)
            rec[f"sem_fixed_{mode}"] = semgrep_flag(semgrep_hits.get(s["id"]), mode)
        out.append(rec)
    return out


def summarize_pairs(recs: list[dict], mode: str) -> dict:
    """Candidate recall (reverse) and FPR (fix direction) for Semgrep, guard_diff
    and their union, plus the overlap on each side. Pairs without a Semgrep row
    are excluded."""
    rs = [r for r in recs if r[f"sem_vuln_{mode}"] is not None
          and r[f"sem_fixed_{mode}"] is not None]
    n = len(rs)

    def side(sem_key: str, guard_key: str) -> dict:
        sem = [r[sem_key] for r in rs]
        grd = [r[guard_key] for r in rs]
        both = sum(a and b for a, b in zip(sem, grd, strict=True))
        sem_only = sum(a and not b for a, b in zip(sem, grd, strict=True))
        guard_only = sum(b and not a for a, b in zip(sem, grd, strict=True))
        return {
            "semgrep": rate(sum(sem), n), "guard_diff": rate(sum(grd), n),
            "union": rate(both + sem_only + guard_only, n),
            "overlap": {"both": both, "semgrep_only": sem_only, "guard_only": guard_only,
                        "neither": n - both - sem_only - guard_only},
        }

    return {"pairs": n,
            "recall_reverse_fix": side(f"sem_vuln_{mode}", "guard_rev"),
            "fpr_fix_direction": side(f"sem_fixed_{mode}", "guard_fix")}


def advisory_level(recs: list[dict], mode: str) -> dict:
    """Pairs cluster by advisory (one advisory contributes up to 6), so also
    report the share of advisories with at least one flagged pair."""
    by_adv: dict[str, list] = defaultdict(list)
    for r in recs:
        if r[f"sem_vuln_{mode}"] is not None and r[f"sem_fixed_{mode}"] is not None:
            by_adv[r["advisory"]].append(r)
    n = len(by_adv)

    def share(pred) -> dict:
        return rate(sum(any(pred(r) for r in rs) for rs in by_adv.values()), n)

    return {
        "advisories": n,
        "recall_reverse_fix": {
            "semgrep": share(lambda r: r[f"sem_vuln_{mode}"]),
            "guard_diff": share(lambda r: r["guard_rev"]),
            "union": share(lambda r: r[f"sem_vuln_{mode}"] or r["guard_rev"])},
        "fpr_fix_direction": {
            "semgrep": share(lambda r: r[f"sem_fixed_{mode}"]),
            "guard_diff": share(lambda r: r["guard_fix"]),
            "union": share(lambda r: r[f"sem_fixed_{mode}"] or r["guard_fix"])},
    }


def breakdown(recs: list[dict], mode: str, key: str) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for r in recs:
        groups[r[key]].append(r)
    return {g: summarize_pairs(rs, mode)
            for g, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))}


def ordinary_rates(items: list[dict], semgrep_hits: dict[str, list], split_of: dict[str, str],
                   mode: str) -> dict:
    def r(sub):
        flags = [semgrep_flag(semgrep_hits.get(o["id"]), mode) for o in sub]
        flags = [f for f in flags if f is not None]
        return rate(sum(flags), len(flags))

    out = {"all": r(items),
           "length_matched": r([o for o in items if o.get("length_matched")]),
           "not_length_matched": r([o for o in items if not o.get("length_matched")])}
    for split in ("dev", "test"):
        out[split] = r([o for o in items if split_of.get(o["id"]) == split])
    return out


def union_rate(*rates: float) -> float:
    """P(any flag) for independent flags."""
    p_none = 1.0
    for x in rates:
        p_none *= 1 - x
    return 1 - p_none


def calls_per_100(recall: float, benign_rate: float, base_rate: float) -> float:
    """LLM calls per 100 changed functions when only candidates are reviewed."""
    return round(100 * (base_rate * recall + (1 - base_rate) * benign_rate), 2)


def implications(candidate_recall: float, benign_rates: dict[str, float],
                 verifier_tpr: float | None) -> dict:
    """Recall ceiling and LLM-call volume of static-first vs reviewing every
    function (LLM-first: 100 calls per 100 functions, recall = verifier TPR)."""
    out = {
        "candidate_recall": round(candidate_recall, 4),
        "max_recall_perfect_verifier": round(candidate_recall, 4),
        "max_recall_with_verifier_tpr": (round(candidate_recall * verifier_tpr, 4)
                                         if verifier_tpr is not None else None),
        "llm_first_recall_at_verifier_tpr": verifier_tpr,
        "calls_per_100_functions": {},
    }
    for label, benign in benign_rates.items():
        out["calls_per_100_functions"][label] = {
            f"{b:.0%}": {"static_first": calls_per_100(candidate_recall, benign, b),
                         "llm_first": 100.0}
            for b in BASE_RATES
        }
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def guard_rows(pairs: list[dict]) -> dict[str, dict]:
    from backend.app.core.guard_diff import guard_diff

    out = {}
    for p in pairs:
        lang, vuln, fixed = p["vuln"]["language"], p["vuln"]["code"], p["safe"]["code"]
        fwd, rev = guard_diff(vuln, fixed, lang), guard_diff(fixed, vuln, lang)
        out[p["base"]] = {"fix_risk": fwd.risk, "rev_risk": rev.risk,
                          "fix_alert": fwd.alert, "rev_alert": rev.alert}
    return out


def load_semgrep(path: Path, lines_by_file: dict[str, list[dict]],
                 ordinary: list[dict]) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["items"]
    hits: dict[str, list] = {}
    stats = {}
    for name, lines in lines_by_file.items():
        old = [r for r in rows if r.get("set") == name]
        mapped, st = map_semgrep_rows(old, lines)
        hits.update(mapped)
        stats[name] = st
    ord_lines = {o["id"]: o for o in ordinary}
    ok = bad = 0
    for r in rows:
        if r.get("kind") == "ordinary" and r["id"] in ord_lines:
            if r.get("lines") == _n_lines(ord_lines[r["id"]]["code"]):
                hits[r["id"]] = r["hits"]
                ok += 1
            else:
                bad += 1
    stats["ordinary"] = {"mapped": ok, "line_count_mismatch": bad}
    return hits, stats


def llm_cross_check(path: Path, recs: list[dict], mode: str) -> dict | None:
    """Among the LLM arm's vulnerable items (pre-migration ids, matched by a
    unique prefix), how many were static candidates, and which did the LLM flag."""
    if not path.exists():
        return None
    items = json.loads(path.read_text(encoding="utf-8"))["llm"]["items"]
    by_prefix: dict[str, list] = defaultdict(list)
    for r in recs:
        by_prefix[_HASH_SUFFIX.sub("", r["base"])].append(r)
    rows = []
    for it in items:
        if it.get("kind") != "vulnerable":
            continue
        cands = by_prefix.get(it["id"][:-5], [])
        if len(cands) != 1:
            rows.append({"id": it["id"], "llm_flagged": it.get("prediction"), "candidate": None})
            continue
        r = cands[0]
        rows.append({"id": it["id"], "llm_flagged": bool(it.get("prediction")),
                     "candidate": bool(r[f"sem_vuln_{mode}"] or r["guard_rev"])})
    known = [r for r in rows if r["candidate"] is not None]
    return {
        "items": rows,
        "vulnerable_items": len(rows),
        "llm_flagged": sum(bool(r["llm_flagged"]) for r in rows),
        "candidates": sum(r["candidate"] for r in known),
        "llm_flagged_and_candidate": sum(r["candidate"] and r["llm_flagged"] for r in known),
        "unmatched": len(rows) - len(known),
    }


def _pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} [{lo:.3f}, {hi:.3f}]" if r["n"] else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--semgrep", type=Path, default=SEMGREP)
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    t0 = time.time()
    lines_by_file = {p.name: load_jsonl(p) for p in PAIR_FILES}
    pairs = pair_lines([ln for lines in lines_by_file.values() for ln in lines])
    ordinary = load_jsonl(ORDINARY)
    hits, map_stats = load_semgrep(args.semgrep, lines_by_file, ordinary)
    split_of: dict[str, str] = {}
    dates: dict[str, dict] = {}
    cutoff = "9999"
    if args.split.exists():
        manifest = json.loads(args.split.read_text(encoding="utf-8"))
        for side in ("dev", "test"):
            split_of.update({i: side for i in manifest[side]["ids"]})
        ds = manifest["meta"]["date_strata"]
        dates, cutoff = ds["by_advisory"], ds["cutoff"]
    guard = guard_rows(pairs)
    recs = pair_records(pairs, hits, guard, split_of, dates, cutoff)

    guard_ord = {}
    if GUARD_EVAL.exists():
        ge = json.loads(GUARD_EVAL.read_text(encoding="utf-8"))
        guard_ord = {
            "synthetic_benign_edits": ge["synthetic_ordinary_control"]["all_three"][
                "fpr_guard_removed"],
            "real_bystander_edits": ge["rejected_bystander_control"]["broad_commit"][
                "pre_to_post_guard_removed"],
        }
    verifier = None
    if LLM_ARM.exists():
        verifier = json.loads(LLM_ARM.read_text(encoding="utf-8"))["llm"]["realistic"][
            "tpr_vulnerable"]

    result: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "inputs": {"semgrep": str(args.semgrep.relative_to(ROOT)),
                               "split": str(args.split.relative_to(ROOT)),
                               "cutoff": cutoff, "semgrep_mapping": map_stats,
                               "semgrep_run_mode": "snippet (unit code only, no file context)",
                               "excluded_rules": sorted(EXCLUDED_RULES)},
                    "guard_diff_ordinary_proxies": guard_ord,
                    "llm_verifier_tpr": verifier, "modes": {}}
    for mode in MODES:
        overall = summarize_pairs(recs, mode)
        ordn = ordinary_rates(ordinary, hits, split_of, mode)
        r_c = overall["recall_reverse_fix"]["union"]["rate"]
        benign = {"fix_direction_pairs": overall["fpr_fix_direction"]["union"]["rate"]}
        for label, g in guard_ord.items():
            benign[f"ordinary_semgrep+guard_{label}"] = round(
                union_rate(ordn["all"]["rate"], g["rate"]), 4)
        benign["ordinary_semgrep_only"] = ordn["all"]["rate"]
        result["modes"][mode] = {
            "overall": overall,
            "advisory_level": advisory_level(recs, mode),
            "by_split": breakdown(recs, mode, "split"),
            "by_language": breakdown(recs, mode, "language"),
            "by_period": breakdown(recs, mode, "period"),
            "by_category": breakdown(recs, mode, "category"),
            "ordinary_semgrep": ordn,
            "implications": implications(r_c, benign, verifier["rate"] if verifier else None),
            "llm_arm_cross_check": llm_cross_check(LLM_ARM, recs, mode),
        }
    result["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    for mode, res in result["modes"].items():
        o = res["overall"]
        print(f"\n== Semgrep mode: {mode} ({o['pairs']} pairs)")
        print("| slice | n | recall Semgrep | recall guard_diff | recall union "
              "| FPR fix-dir union |")
        print("|---|---|---|---|---|---|")
        rows = [("all", o)] + [(f"split={k}", v) for k, v in res["by_split"].items()] + [
            (f"{k}", v) for k, v in res["by_period"].items()] + [
            (f"lang={k}", v) for k, v in res["by_language"].items()] + [
            (f"cat={k}", v) for k, v in res["by_category"].items()]
        for name, s in rows:
            rr, ff = s["recall_reverse_fix"], s["fpr_fix_direction"]
            print(f"| {name} | {s['pairs']} | {_pct(rr['semgrep'])} | {_pct(rr['guard_diff'])} "
                  f"| {_pct(rr['union'])} | {_pct(ff['union'])} |")
        al = res["advisory_level"]
        print(f"advisory level ({al['advisories']}): recall Semgrep "
              f"{_pct(al['recall_reverse_fix']['semgrep'])}, guard_diff "
              f"{_pct(al['recall_reverse_fix']['guard_diff'])}, union "
              f"{_pct(al['recall_reverse_fix']['union'])}; FPR union "
              f"{_pct(al['fpr_fix_direction']['union'])}")
        print(f"overlap (reverse): {o['recall_reverse_fix']['overlap']}; "
              f"(fix dir): {o['fpr_fix_direction']['overlap']}")
        on = res["ordinary_semgrep"]
        print(f"ordinary Semgrep: all {_pct(on['all'])}; length-matched "
              f"{_pct(on['length_matched'])}; dev {_pct(on['dev'])}; test {_pct(on['test'])}")
        print(f"implications: {json.dumps(res['implications'])}")
        if res["llm_arm_cross_check"]:
            x = res["llm_arm_cross_check"]
            print(f"LLM arm cross-check: {x['vulnerable_items']} vulnerable, LLM flagged "
                  f"{x['llm_flagged']}, candidates {x['candidates']}, both "
                  f"{x['llm_flagged_and_candidate']}, unmatched {x['unmatched']}")
    print(f"\nsemgrep mapping: {json.dumps(map_stats)}")
    print(f"wrote {args.out} ({result['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
