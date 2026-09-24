"""Measure the PR diff-direction check (``backend.app.core.guard_diff``).

Real fix commits give (vulnerable -> fixed) function pairs. Treat:

- **fix direction** (vulnerable -> fixed) as a benign change: correct output is
  ``guard_added`` or ``none``; ``guard_removed`` is a false positive (FPR);
- **reverse direction** (fixed -> vulnerable), a simulated vulnerability-
  introducing PR: correct output is ``guard_removed`` (TPR).

``guard_diff`` is symmetric by construction, so reverse ``guard_removed`` equals
fix-direction ``guard_added`` — the script checks that as a sanity test.

Controls for ordinary (non-security) changes, where the base rate lives:

- **synthetic**: each ordinary function (``detection_eval_ordinary.jsonl``)
  diffed against itself after a benign edit (rename a local, add a log line,
  swap two independent statements, and all three). Synthetic, so optimistic.
- **bystander**: real diffs of functions the builder rejected from fix commits
  (``data/cve_corpus/rejected``): ``broad_commit`` bystanders are real edits from
  commits touching many functions (some are part of the fix, so contaminated);
  ``rename_only``/``comment_or_format_only`` should all be ``none``.

Splits: held-out eval pairs (``detection_eval_osv_{pypi,npm}.jsonl``, primary)
and corpus pairs (``data/cve_corpus/osv_{pypi,npm}.json``, secondary; the
vocabularies were tuned on these). No model, no network, no Qdrant.

    python -m ml.evaluation.eval_guard_diff   # -> ml/evaluation/results/guard_diff_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.guard_diff import (  # noqa: E402
    GuardDiffResult,
    _language,
    _parse,
    guard_diff,
)

DATASETS = ROOT / "ml" / "evaluation" / "datasets"
CORPUS = ROOT / "data" / "cve_corpus"
DEFAULT_OUT = ROOT / "ml" / "evaluation" / "results" / "guard_diff_eval.json"
BASE_RATES = (0.01, 0.02, 0.05)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_heldout_pairs() -> list[dict]:
    """(vulnerable, fixed) pairs from the held-out eval sets. Lines come as
    consecutive ``<prefix>_vuln`` / ``<prefix>_safe`` (build_eval_lines order)."""
    pairs = []
    for eco in ("pypi", "npm"):
        rows = [json.loads(line) for line in (DATASETS / f"detection_eval_osv_{eco}.jsonl")
                .read_text().splitlines() if line.strip()]
        i = 0
        while i < len(rows) - 1:
            a, b = rows[i], rows[i + 1]
            if a["id"].endswith("_vuln") and b["id"] == a["id"][:-5] + "_safe":
                pairs.append({"id": a["id"][:-5], "language": a["language"],
                              "category": a["category"], "vulnerable": a["code"],
                              "fixed": b["code"], "split": "heldout"})
                i += 2
            else:
                i += 1
    return pairs


def load_corpus_pairs() -> list[dict]:
    pairs = []
    for eco in ("pypi", "npm"):
        for k, e in enumerate(json.loads((CORPUS / f"osv_{eco}.json").read_text())):
            if not e.get("fixed_code"):
                continue
            pairs.append({"id": f"{e['cve_id']}_{e['function_name']}_{k}",
                          "language": e["language"], "category": e["category"],
                          "vulnerable": e["vulnerable_code"], "fixed": e["fixed_code"],
                          "split": "corpus"})
    return pairs


def load_rejected_pairs() -> list[dict]:
    pairs = []
    for eco in ("pypi", "npm"):
        path = CORPUS / "rejected" / f"osv_{eco}_rejected.json"
        if not path.exists():
            continue
        for k, e in enumerate(json.loads(path.read_text())):
            reasons = set(e.get("reject_reasons") or [e.get("reject_reason")])
            group = ("cosmetic" if reasons & {"rename_only", "comment_or_format_only"}
                     else "broad_commit")
            pairs.append({"id": f"{e['cve_id']}_{e['function_name']}_{k}",
                          "language": e["language"], "category": e["category"],
                          "vulnerable": e["vulnerable_code"], "fixed": e["fixed_code"],
                          "group": group})
    return pairs


def load_ordinary() -> list[dict]:
    path = DATASETS / "detection_eval_ordinary.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Synthetic benign edits (ordinary-change control)
# ---------------------------------------------------------------------------

_PY_ASSIGN = re.compile(r"^[ \t]+([a-z_][a-z0-9_]*)\s*=(?!=)", re.M)
_JS_DECL = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=")


def _function_node(root):
    wanted = {"function_definition", "function_declaration", "method_definition",
              "arrow_function", "function_expression", "function"}
    stack = [root]
    while stack:
        n = stack.pop(0)
        if n.type in wanted:
            return n
        stack.extend(n.children)
    return None


def edit_rename_local(code: str, lang: str) -> str | None:
    rx = _PY_ASSIGN if lang == "python" else _JS_DECL
    skip = {"self", "cls", "this"}
    for m in rx.finditer(code):
        name = m.group(1)
        if name in skip or len(name) < 2:
            continue
        new = name + "_renamed"
        out = re.sub(rf"(?<![\w.$]){re.escape(name)}(?![\w$])", new, code)
        return out if out != code else None
    return None


def _body_statements(code: str, lang: str):
    root, src, off = _parse(code, lang)
    fn = _function_node(root)
    if fn is None:
        return None, None, off
    body = fn.child_by_field_name("body")
    if body is None or body.type not in ("block", "statement_block"):
        return None, None, off
    return [c for c in body.named_children if c.type != "comment"], src, off


def edit_add_log(code: str, lang: str) -> str | None:
    stmts, _src, off = _body_statements(code, lang)
    if not stmts:
        return None
    first = stmts[0]
    line_idx = first.start_point[0] - off
    col = first.start_point[1]
    lines = code.split("\n")
    if not 0 <= line_idx < len(lines):
        return None
    if line_idx == 0:  # one-line function body; not a safe insertion point
        return None
    log = 'logger.debug("processing request")' if lang == "python" else \
        'console.log("processing request");'
    lines.insert(line_idx, " " * col + log)
    return "\n".join(lines)


def _assigned_and_used(node, src) -> tuple[set[str], set[str]]:
    text = src[node.start_byte:node.end_byte].decode("utf-8", "replace")
    names = set(re.findall(r"[A-Za-z_$][\w$]*", text))
    m = re.match(r"\s*(?:const|let|var)?\s*([A-Za-z_$][\w$.]*)\s*=(?!=)", text)
    assigned = {m.group(1).split(".")[0]} if m else set()
    return assigned, names


def edit_reorder(code: str, lang: str) -> str | None:
    stmts, src, off = _body_statements(code, lang)
    if not stmts or len(stmts) < 2:
        return None
    simple = {"expression_statement", "lexical_declaration", "variable_declaration"}
    lines = code.split("\n")
    for a, b in zip(stmts, stmts[1:], strict=False):
        if a.type not in simple or b.type not in simple:
            continue
        if a.start_point[0] != a.end_point[0] or b.start_point[0] != b.end_point[0]:
            continue
        if b.start_point[0] != a.start_point[0] + 1:
            continue
        a_asg, a_use = _assigned_and_used(a, src)
        b_asg, b_use = _assigned_and_used(b, src)
        if not a_asg or not b_asg:  # only swap two independent assignments
            continue
        if a_asg & b_use or b_asg & a_use:
            continue
        i, j = a.start_point[0] - off, b.start_point[0] - off
        if not (0 < i < len(lines) and 0 < j < len(lines)):
            continue
        lines[i], lines[j] = lines[j], lines[i]
        return "\n".join(lines)
    return None


EDITS = {"rename_local": edit_rename_local, "add_log": edit_add_log,
         "reorder": edit_reorder}


def edit_all(code: str, lang: str) -> str | None:
    out, applied = code, 0
    for fn in (edit_reorder, edit_rename_local, edit_add_log):
        nxt = fn(out, lang)
        if nxt is not None:
            out, applied = nxt, applied + 1
    return out if applied else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def precision_at(tpr: float, fpr: float, base: float) -> float | None:
    den = tpr * base + fpr * (1 - base)
    return round(tpr * base / den, 4) if den > 0 else None


def _neg_kinds(r: GuardDiffResult) -> list[str]:
    return sorted({c.kind for c in r.changes if c.direction in ("removed", "weakened")})


def _pos_kinds(r: GuardDiffResult) -> list[str]:
    return sorted({c.kind for c in r.changes if c.direction in ("added", "strengthened")})


def _brief(r: GuardDiffResult) -> list[dict]:
    return [{"direction": c.direction, "kind": c.kind, "line": c.line,
             "old": c.old_text, "new": c.new_text, "conf": c.confidence,
             "why": c.rationale} for c in r.changes[:6]]


def evaluate_pairs(pairs: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for p in pairs:
        fwd = guard_diff(p["vulnerable"], p["fixed"], p["language"])
        rev = guard_diff(p["fixed"], p["vulnerable"], p["language"])
        rows.append({
            "id": p["id"], "language": p["language"], "category": p["category"],
            "fix_risk": fwd.risk, "rev_risk": rev.risk,
            "fix_alert": fwd.alert, "rev_alert": rev.alert,
            "fix_neg": fwd.removed_score, "fix_pos": fwd.added_score,
            "fix_neg_kinds": _neg_kinds(fwd), "fix_pos_kinds": _pos_kinds(fwd),
            "rev_neg_kinds": _neg_kinds(rev),
            "fix_changes": _brief(fwd), "rev_changes": _brief(rev),
            "note": fwd.note, "identical": p["vulnerable"].strip() == p["fixed"].strip(),
        })
    return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(r["rev_risk"] == "guard_removed" for r in rows)
    fp = sum(r["fix_risk"] == "guard_removed" for r in rows)
    tp_alert = sum(r["rev_alert"] for r in rows)
    fp_alert = sum(r["fix_alert"] for r in rows)
    fix_added = sum(r["fix_risk"] == "guard_added" for r in rows)
    rev_added = sum(r["rev_risk"] == "guard_added" for r in rows)
    none = sum(r["fix_risk"] == "none" for r in rows)
    out = {
        "pairs": n,
        "tpr_reverse_guard_removed": rate(tp, n),
        "fpr_fix_guard_removed": rate(fp, n),
        "alert_tier_tpr": rate(tp_alert, n),
        "alert_tier_fpr": rate(fp_alert, n),
        "fix_guard_added": rate(fix_added, n),
        "fix_none": rate(none, n),
        "symmetry_ok": tp == fix_added and fp == rev_added,
        "partial_parse": sum(bool(r["note"]) for r in rows),
    }
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out["per_category"] = {
        c: {"n": len(rs),
            "tpr": round(sum(r["rev_risk"] == "guard_removed" for r in rs) / len(rs), 3),
            "fpr": round(sum(r["fix_risk"] == "guard_removed" for r in rs) / len(rs), 3)}
        for c, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))
    }
    tp_kinds = Counter(k for r in rows if r["rev_risk"] == "guard_removed"
                       for k in r["rev_neg_kinds"])
    fp_kinds = Counter(k for r in rows if r["fix_risk"] == "guard_removed"
                       for k in r["fix_neg_kinds"])
    out["per_kind"] = {k: {"tp_pairs": tp_kinds.get(k, 0), "fp_pairs": fp_kinds.get(k, 0)}
                       for k in sorted(set(tp_kinds) | set(fp_kinds),
                                       key=lambda k: -(tp_kinds.get(k, 0) + fp_kinds.get(k, 0)))}
    return out


def evaluate_synthetic(ordinary: list[dict]) -> dict:
    out = {}
    for name, fn in [*EDITS.items(), ("all_three", edit_all)]:
        n = flagged_removed = flagged_added = alerts = 0
        examples = []
        for item in ordinary:
            lang = item["language"]
            try:
                edited = fn(item["code"], lang)
            except Exception:
                edited = None
            if edited is None or edited == item["code"]:
                continue
            n += 1
            r = guard_diff(item["code"], edited, lang)
            alerts += r.alert
            if r.risk == "guard_removed":
                flagged_removed += 1
                if len(examples) < 5:
                    examples.append({"id": item["id"], "changes": _brief(r)})
            elif r.risk == "guard_added":
                flagged_added += 1
        out[name] = {"edited": n, "fpr_guard_removed": rate(flagged_removed, n),
                     "fpr_alert": rate(alerts, n),
                     "guard_added": rate(flagged_added, n), "fp_examples": examples}
    out["ordinary_items"] = len(ordinary)
    return out


def evaluate_rejected(pairs: list[dict]) -> dict:
    out = {}
    for group in ("broad_commit", "cosmetic"):
        rows = [p for p in pairs if p["group"] == group]
        res = [guard_diff(p["vulnerable"], p["fixed"], p["language"]) for p in rows]
        n = len(rows)
        out[group] = {
            "pairs": n,
            "pre_to_post_guard_removed": rate(sum(r.risk == "guard_removed" for r in res), n),
            "pre_to_post_alert": rate(sum(r.alert for r in res), n),
            "pre_to_post_guard_added": rate(sum(r.risk == "guard_added" for r in res), n),
            "any_change_reported": rate(sum(bool(r.changes) for r in res), n),
        }
    return out


def base_rate_table(tpr: dict, fprs: dict[str, dict]) -> dict:
    table = {}
    for label, fpr in fprs.items():
        row = {}
        for b in BASE_RATES:
            row[f"{b:.0%}"] = {
                "point": precision_at(tpr["rate"] or 0.0, fpr["rate"] or 0.0, b),
                # Conservative: TPR lower bound, FPR upper bound.
                "conservative": precision_at(tpr["ci95"][0], fpr["ci95"][1], b),
            }
        table[label] = row
    return table


def pick_examples(rows: list[dict], k: int = 5) -> dict:
    correct = [r for r in rows if r["rev_risk"] == "guard_removed"
               and r["fix_risk"] != "guard_removed"]
    wrong_fp = [r for r in rows if r["fix_risk"] == "guard_removed"]
    missed = [r for r in rows if r["rev_risk"] != "guard_removed" and not r["identical"]]

    def spread(rs):  # one per category first, deterministic
        seen, out = set(), []
        for r in rs:
            if r["category"] not in seen:
                seen.add(r["category"])
                out.append(r)
        out += [r for r in rs if r not in out]
        return [{"id": r["id"], "category": r["category"], "language": r["language"],
                 "fix_risk": r["fix_risk"], "rev_risk": r["rev_risk"],
                 "rev_changes": r["rev_changes"][:3], "fix_changes": r["fix_changes"][:3]}
                for r in out[:k]]

    return {"correct_reverse_detections": spread(correct),
            "false_positives_fix_direction": spread(wrong_fp),
            "missed_reverse": spread(missed)}


def _fmt(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{r['rate']:.3f} ({r['k']}/{r['n']}, CI {lo:.3f}-{hi:.3f})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--split", choices=("heldout", "corpus", "both"), default="both")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args(argv)

    _language("python"), _language("javascript")
    t0 = time.time()
    result: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rows_by_split = {}
    splits = ["heldout", "corpus"] if args.split == "both" else [args.split]
    for split in splits:
        pairs = load_heldout_pairs() if split == "heldout" else load_corpus_pairs()
        summary, rows = evaluate_pairs(pairs)
        summary["examples"] = pick_examples(rows)
        result[split] = summary
        rows_by_split[split] = rows
        print(f"[{split}] pairs={summary['pairs']}")
        print(f"  TPR reverse guard_removed : {_fmt(summary['tpr_reverse_guard_removed'])}")
        print(f"  FPR fix-dir guard_removed : {_fmt(summary['fpr_fix_guard_removed'])}")
        print(f"  fix-dir guard_added       : {_fmt(summary['fix_guard_added'])}")
        print(f"  alert tier TPR / FPR      : {_fmt(summary['alert_tier_tpr'])} / "
              f"{_fmt(summary['alert_tier_fpr'])}")
        print(f"  symmetry ok               : {summary['symmetry_ok']}")

    if not args.skip_controls:
        synth = evaluate_synthetic(load_ordinary())
        result["synthetic_ordinary_control"] = synth
        for name, s in synth.items():
            if isinstance(s, dict):
                print(f"[synthetic:{name}] edited={s['edited']} "
                      f"FPR={_fmt(s['fpr_guard_removed'])}")
        rej = evaluate_rejected(load_rejected_pairs())
        result["rejected_bystander_control"] = rej
        for g, s in rej.items():
            print(f"[bystander:{g}] pairs={s['pairs']} "
                  f"guard_removed={_fmt(s['pre_to_post_guard_removed'])}")

    primary = result.get("heldout") or result.get("corpus")
    fprs = {"fix_direction": primary["fpr_fix_guard_removed"]}
    if "synthetic_ordinary_control" in result:
        fprs["synthetic_ordinary"] = result["synthetic_ordinary_control"]["all_three"][
            "fpr_guard_removed"]
        fprs["bystander_broad_commit"] = result["rejected_bystander_control"]["broad_commit"][
            "pre_to_post_guard_removed"]
    result["precision_at_base_rate"] = base_rate_table(
        primary["tpr_reverse_guard_removed"], fprs)
    alert_fprs = {"fix_direction": primary["alert_tier_fpr"]}
    if "synthetic_ordinary_control" in result:
        alert_fprs["synthetic_ordinary"] = result["synthetic_ordinary_control"]["all_three"][
            "fpr_alert"]
        alert_fprs["bystander_broad_commit"] = result["rejected_bystander_control"][
            "broad_commit"]["pre_to_post_alert"]
    result["alert_tier_precision_at_base_rate"] = base_rate_table(
        primary["alert_tier_tpr"], alert_fprs)
    for title, table in (("guard_removed", result["precision_at_base_rate"]),
                         ("alert tier", result["alert_tier_precision_at_base_rate"])):
        for label, row in table.items():
            cells = "  ".join(f"{b}: {v['point']} (cons {v['conservative']})"
                              for b, v in row.items())
            print(f"[precision {title} | FPR={label}] {cells}")
    result["rows"] = rows_by_split
    result["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out} ({result['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
