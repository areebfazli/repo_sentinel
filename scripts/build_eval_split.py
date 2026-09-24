"""Build a grouped, date-stratified dev/test split of the eval sets, offline.

Inputs: the OSV pair sets (``detection_eval_osv_{pypi,npm}.jsonl``: consecutive
``<advisory>_<fn>_<hash8>_vuln`` / ``_safe`` lines) and the ordinary negatives
(``detection_eval_ordinary.jsonl``: ``repo``, ``advisory_id``, ``length_matched``).
Output: ``ml/evaluation/splits/v1.json``::

    {"version": 1, "seed": 42, "dev": {"ids": [...]}, "test": {"ids": [...]},
     "meta": {"created": ..., "counts": {...}, "grouping": "...", "date_strata": {...}}}

**Grouping (no leakage between dev and test).** Every vulnerable/fixed pair and
every ordinary function is a unit; units are unioned (connected components) when
they share any of: advisory id (the display CVE/GHSA id plus every OSV alias of
the advisory that produced the pair, e.g. GHSA + PYSEC), repository (``--group-by
repo``, the default; ``advisory`` skips this), or near-duplicate code
(identifier Jaccard >= ``--near-dup``, both sides with >= ``--min-idents``
distinct identifiers). A pair's two twins are one unit, so they never straddle.
Whole components go to one side.

**Date strata.** Each advisory gets ``public_since`` = the earliest of the fix
commit's author/committer dates (GitHub API cache) and the earliest OSV
``published`` across its aliases (OSV zips) — a lower bound on when the fixed
code was public, so "after the cutoff" is conservative. Without the cache it
falls back to the CVE id's year (``date_source: cve_id_year``). Pairs are
stratified by ``public_since`` year bin x language, and ``date_strata.by_advisory`` keeps
the per-advisory date so results can be split at any training cutoff.

**Sizes.** Components are first split into a dev side and a test side, balancing
pairs per (year bin, language) cell (``dev_pairs : test_pairs``) and length-matched / other
ordinary functions (``dev_ordinary : test_ordinary``). Then, per side, whole
advisories are drawn (seeded) to per-cell quotas of ``--dev-pairs`` /
``--test-pairs``; pairs not drawn stay unassigned (re-run with larger sizes to
use them — each side's pool is fixed by the seed, so growing a split never
leaks). Ordinary functions are kept up to ``--dev-ordinary`` / ``--test-ordinary``
(length-matched first). Deterministic for a given seed and inputs.

    python scripts/build_eval_split.py                  # -> ml/evaluation/splits/v1.json
    python scripts/build_eval_split.py --check          # print counts, write nothing

No network, no models: reads ``data/osv_cache/`` (processed_advisories.json, the
GitHub API response cache, the OSV zips) if present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ROOT / "ml" / "evaluation" / "datasets"
DEFAULT_PAIR_FILES = [DATASETS / "detection_eval_osv_pypi.jsonl",
                      DATASETS / "detection_eval_osv_npm.jsonl"]
DEFAULT_ORDINARY = DATASETS / "detection_eval_ordinary.jsonl"
DEFAULT_OUT = ROOT / "ml" / "evaluation" / "splits" / "v1.json"
DEFAULT_CACHE_DIR = ROOT / "data" / "osv_cache"
OSV_ZIPS = ("PyPI_all.zip", "npm_all.zip")

# Year bins for public_since; the last bin is open-ended.
DEFAULT_BINS = ("<=2022", "2023", "2024", "2025", "2026+")
DEFAULT_CUTOFF = "2025-01-01"
UNKNOWN = "unknown"

_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_STOP = set(keyword.kwlist) | {
    "self", "cls", "this", "const", "let", "var", "function", "return", "new", "typeof",
    "undefined", "null", "true", "false", "async", "await", "if", "else", "for", "while",
    "of", "in", "do", "switch", "case", "break", "continue", "try", "catch", "finally",
    "throw", "export", "default", "require", "module", "exports",
}


# ---------------------------------------------------------------------------
# Pure logic (unit-tested in tests/unit/test_build_eval_split.py)
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pair_lines(lines: list[dict]) -> list[dict]:
    """``[{base, vuln, safe}]`` from eval lines: ``<base>_vuln`` + ``<base>_safe``
    matched by id (order-independent). Raises on an unmatched line."""
    by_base: dict[str, dict] = defaultdict(dict)
    for ln in lines:
        lid = ln["id"]
        if lid.endswith("_vuln"):
            by_base[lid[:-5]]["vuln"] = ln
        elif lid.endswith("_safe"):
            by_base[lid[:-5]]["safe"] = ln
        else:
            raise ValueError(f"not a pair line: {lid!r}")
    out = []
    for base in sorted(by_base):
        p = by_base[base]
        if set(p) != {"vuln", "safe"}:
            raise ValueError(f"unpaired eval line: {base!r}")
        out.append({"base": base, **p})
    return out


def identifiers(code: str) -> frozenset[str]:
    """Distinct identifiers of ``code`` minus language keywords."""
    return frozenset(t for t in _IDENT_RE.findall(code or "") if t not in _STOP)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def near_duplicate_links(
    sets: list[frozenset], threshold: float, min_idents: int
) -> list[tuple[int, int, float]]:
    """All index pairs ``(i, j, J)`` with Jaccard >= ``threshold``, skipping
    sets smaller than ``min_idents``. Uses the size bound J <= |small|/|large|
    to prune, so it is near-linear on real code."""
    order = sorted((i for i, s in enumerate(sets) if len(s) >= min_idents),
                   key=lambda i: (len(sets[i]), i))
    links = []
    for a, i in enumerate(order):
        si = sets[i]
        for j in order[a + 1:]:
            sj = sets[j]
            if len(si) < threshold * len(sj):
                break
            jac = jaccard(si, sj)
            if jac >= threshold:
                links.append((min(i, j), max(i, j), round(jac, 3)))
    return sorted(links)


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def year_bin(date: str | None, bins: tuple[str, ...] = DEFAULT_BINS) -> str:
    """Bin label for an ISO date (``<=YYYY`` first bin, ``YYYY+`` last bin)."""
    if not date or not date[:4].isdigit():
        return UNKNOWN
    year = int(date[:4])
    for b in bins:
        if b.startswith("<="):
            if year <= int(b[2:]):
                return b
        elif b.endswith("+"):
            if year >= int(b[:-1]):
                return b
        elif year == int(b):
            return b
    return UNKNOWN


def cve_year(advisory_id: str) -> int | None:
    m = re.match(r"CVE-(\d{4})-", advisory_id or "")
    return int(m.group(1)) if m else None


def advisory_date(
    advisory_id: str, commit_dates: list[str], osv_published: list[str]
) -> dict:
    """``{public_since, date_source, fix_commit_date, osv_published, cve_year}``:
    public_since is the earliest known public date (YYYY-MM-DD)."""
    commit = min((d for d in commit_dates if d), default=None)
    published = min((d for d in osv_published if d), default=None)
    known = [d for d in (commit, published) if d]
    year = cve_year(advisory_id)
    if known:
        since = min(known)[:10]
        source = "fix_commit" if since == (commit or "")[:10] else "osv_published"
    elif year:
        since, source = f"{year}-01-01", "cve_id_year"
    else:
        since, source = None, UNKNOWN
    return {"public_since": since, "date_source": source,
            "fix_commit_date": commit[:10] if commit else None,
            "osv_published": published[:10] if published else None, "cve_year": year}


def build_units(pairs: list[dict], ordinary: list[dict], provenance: dict[str, dict],
                strata: dict[str, str]) -> list[dict]:
    """One unit per pair (both twins) and per ordinary function, with the
    grouping keys it contributes: ``adv:<id>`` and ``repo:<owner/name>``."""
    units = []
    for p in pairs:
        adv = p["vuln"]["expected_cve_id"]
        prov = provenance.get(p["base"], {})
        keys = {f"adv:{adv}"} | {f"adv:{a}" for a in prov.get("aliases", ())}
        repo = prov.get("repo")
        units.append({
            "kind": "pair", "ids": [p["vuln"]["id"], p["safe"]["id"]], "advisory": adv,
            "adv_keys": keys, "repo": repo.lower() if repo else None,
            "stratum": strata.get(adv, UNKNOWN), "language": p["vuln"]["language"],
            "cell": f"{strata.get(adv, UNKNOWN)}|{p['vuln']['language']}",
            "category": p["vuln"]["category"], "code": p["vuln"]["code"],
            "codes": [p["vuln"]["code"], p["safe"]["code"]],
        })
    for o in ordinary:
        adv = o.get("advisory_id")
        units.append({
            "kind": "ordinary", "ids": [o["id"]], "advisory": adv,
            "adv_keys": {f"adv:{adv}"} if adv else set(),
            "repo": o["repo"].lower() if o.get("repo") else None,
            "stratum": None, "cell": None, "language": o["language"], "category": "none",
            "length_matched": bool(o.get("length_matched")), "code": o["code"],
            "codes": [o["code"]],
        })
    return units


def group_units(units: list[dict], group_by: str, near_dup: float,
                min_idents: int) -> tuple[list[int], dict]:
    """Component label per unit + stats. Links: shared advisory key, shared
    repo (``group_by == "repo"``), near-duplicate code."""
    dsu = _DSU(len(units))
    first: dict[str, int] = {}
    for i, u in enumerate(units):
        keys = set(u["adv_keys"])
        if group_by == "repo" and u["repo"]:
            keys.add(f"repo:{u['repo']}")
        for k in sorted(keys):
            if k in first:
                dsu.union(first[k], i)
            else:
                first[k] = i
    # Components before the near-duplicate merge, to count links it adds.
    pre = [dsu.find(i) for i in range(len(units))]
    links = near_duplicate_links([identifiers(u["code"]) for u in units], near_dup,
                                 min_idents) if near_dup <= 1 else []
    cross = [(i, j, jac) for i, j, jac in links if pre[i] != pre[j]]
    for i, j, _ in links:
        dsu.union(i, j)
    labels = [dsu.find(i) for i in range(len(units))]
    sizes = Counter(labels)
    stats = {
        "groups": len(sizes),
        "largest_group_units": max(sizes.values(), default=0),
        "near_dup_links_merged_across_groups": len(cross),
        "near_dup_examples": [[units[i]["ids"][0], units[j]["ids"][0], jac]
                              for i, j, jac in cross[:5]],
    }
    return labels, stats


def _vector(units: list[dict], members: list[int], strata: list[str]) -> dict[str, int]:
    v: Counter = Counter()
    for i in members:
        u = units[i]
        if u["kind"] == "pair":
            v[f"pairs:{u['cell']}"] += 1
        else:
            v["ord_lm" if u["length_matched"] else "ord_other"] += 1
    return {d: v.get(d, 0) for d in [f"pairs:{s}" for s in strata] + ["ord_lm", "ord_other"]}


def assign_sides(units: list[dict], labels: list[int], seed: int, dev_pair_share: float,
                 dev_ord_share: float) -> dict[int, str]:
    """``{component: "dev"|"test"}``: largest components first (seeded ties),
    each to the side whose targets it fills least, relative to that side's
    target, on the dimensions it has (pairs per (year bin, language) cell,
    ordinary length-matched / other)."""
    strata = sorted({u["cell"] for u in units if u["kind"] == "pair"})
    comps: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        comps[lab].append(i)
    vecs = {c: _vector(units, m, strata) for c, m in comps.items()}
    total: Counter = Counter()
    for v in vecs.values():
        total.update(v)
    share = {"dev": {}, "test": {}}
    for d in total:
        s = dev_ord_share if d.startswith("ord") else dev_pair_share
        share["dev"][d], share["test"][d] = s, 1 - s
    target = {side: {d: total[d] * share[side][d] for d in total} for side in share}
    rng = random.Random(f"{seed}:sides")
    keys = sorted(comps, key=lambda c: min(units[i]["ids"][0] for i in comps[c]))
    tiebreak = {c: rng.random() for c in keys}
    order = sorted(keys, key=lambda c: (-len(comps[c]), tiebreak[c]))
    filled = {"dev": Counter(), "test": Counter()}
    side_of: dict[int, str] = {}
    for c in order:
        v = vecs[c]
        costs = {}
        for side in ("dev", "test"):
            cost = 0.0
            for d, n in v.items():
                if n:
                    cost += n * (filled[side][d] + n) / max(target[side][d], 1e-9)
            costs[side] = cost
        if abs(costs["dev"] - costs["test"]) < 1e-12:
            side = "dev" if tiebreak[c] < dev_pair_share else "test"
        else:
            side = min(costs, key=costs.get)
        side_of[c] = side
        filled[side].update(v)
    return side_of


def stratum_quotas(target: int, counts: dict[str, int]) -> dict[str, int]:
    """``target`` split across strata proportionally to ``counts`` (largest
    remainder, deterministic)."""
    total = sum(counts.values())
    if not total:
        return {s: 0 for s in counts}
    raw = {s: target * n / total for s, n in counts.items()}
    quotas = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda k: (-(raw[k] - quotas[k]), k)):
        if sum(quotas.values()) >= target:
            break
        quotas[s] += 1
    return quotas


def select_pairs(units: list[dict], members: list[int], quotas: dict[str, int],
                 seed: int, tag: str) -> list[int]:
    """Whole advisories (all their pair units on this side) drawn in seeded
    order per (year bin, language) cell up to its quota; then, while short, the
    smallest remaining advisory if taking it lands closer to the quota. Last, a
    global top-up (cells with the largest remaining shortfall first) while the
    total is below ``sum(quotas)`` and an advisory brings it closer."""
    by_adv: dict[str, list[int]] = defaultdict(list)
    for i in members:
        if units[i]["kind"] == "pair":
            by_adv[units[i]["advisory"]].append(i)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for adv, idx in by_adv.items():
        by_stratum[units[idx[0]]["cell"]].append(adv)
    picked: list[int] = []
    taken: Counter = Counter()
    leftovers: list[tuple[str, int, str]] = []
    for s in sorted(by_stratum):
        quota = quotas.get(s, 0)
        advs = sorted(by_stratum[s])
        random.Random(f"{seed}:{tag}:{s}").shuffle(advs)
        rest = []
        for adv in advs:
            n = len(by_adv[adv])
            if taken[s] + n <= quota:
                picked.extend(by_adv[adv])
                taken[s] += n
            else:
                rest.append(adv)
        while taken[s] < quota and rest:
            adv = min(rest, key=lambda a: (len(by_adv[a]), advs.index(a)))
            n = len(by_adv[adv])
            if abs(taken[s] + n - quota) >= quota - taken[s]:
                break  # overshooting would miss the quota by more than stopping
            rest.remove(adv)
            picked.extend(by_adv[adv])
            taken[s] += n
        leftovers += [(s, rank, adv) for rank, adv in enumerate(rest)]
    target = sum(quotas.values())
    while len(picked) < target and leftovers:
        def key(item):
            s, rank, adv = item
            return (-(quotas.get(s, 0) - taken[s]), len(by_adv[adv]), s, rank)
        s, rank, adv = min(leftovers, key=key)
        n = len(by_adv[adv])
        if abs(len(picked) + n - target) >= target - len(picked):
            leftovers = [x for x in leftovers if len(by_adv[x[2]]) < n]
            continue
        leftovers.remove((s, rank, adv))
        picked.extend(by_adv[adv])
        taken[s] += n
    return sorted(picked)


def select_ordinary(units: list[dict], members: list[int], cap: int, seed: int,
                    tag: str) -> list[int]:
    """Up to ``cap`` ordinary units: length-matched ones first, then the rest,
    each in seeded order."""
    lm = sorted((i for i in members if units[i]["kind"] == "ordinary"
                 and units[i]["length_matched"]), key=lambda i: units[i]["ids"][0])
    other = sorted((i for i in members if units[i]["kind"] == "ordinary"
                    and not units[i]["length_matched"]), key=lambda i: units[i]["ids"][0])
    random.Random(f"{seed}:{tag}:lm").shuffle(lm)
    random.Random(f"{seed}:{tag}:other").shuffle(other)
    return sorted((lm + other)[:cap])


def _code_hash(code: str) -> str:
    return hashlib.sha1("".join((code or "").split()).encode()).hexdigest()


def leakage_checks(units: list[dict], chosen: dict[str, list[int]], near_dup: float,
                   min_idents: int) -> dict:
    """Cross-split overlap of advisories, repos, whitespace-insensitive exact
    code, and near-duplicate code (identifier Jaccard, every twin compared)."""
    def keys(side, fn):
        return {k for i in chosen[side] for k in fn(units[i])}
    adv = keys("dev", lambda u: u["adv_keys"]) & keys("test", lambda u: u["adv_keys"])
    repo = keys("dev", lambda u: [u["repo"]] if u["repo"] else []) & keys(
        "test", lambda u: [u["repo"]] if u["repo"] else [])
    code = {s: [(i, c) for i in chosen[s] for c in units[i]["codes"]] for s in chosen}
    exact = {_code_hash(c) for _, c in code["dev"]} & {_code_hash(c) for _, c in code["test"]}
    dev_n = len(code["dev"])
    sets = [identifiers(c) for _, c in code["dev"] + code["test"]]
    links = near_duplicate_links(sets, near_dup, min_idents)
    cross = [(a, b, j) for a, b, j in links if a < dev_n <= b]
    return {
        "advisories_in_both": len(adv),
        "repos_in_both": len(repo),
        "exact_code_in_both": len(exact),
        f"near_dup_code_pairs_across_splits_jaccard_ge_{near_dup}": len(cross),
    }


def build_manifest(
    pairs: list[dict], ordinary: list[dict], provenance: dict[str, dict],
    dates: dict[str, dict], *, seed: int = 42, dev_pairs: int = 150, test_pairs: int = 100,
    dev_ordinary: int = 500, test_ordinary: int = 500, group_by: str = "repo",
    near_dup: float = 0.7, min_idents: int = 5, bins: tuple[str, ...] = DEFAULT_BINS,
    cutoff: str = DEFAULT_CUTOFF, created: str | None = None,
) -> dict:
    strata = {adv: year_bin(d.get("public_since"), bins) for adv, d in dates.items()}
    units = build_units(pairs, ordinary, provenance, strata)
    labels, gstats = group_units(units, group_by, near_dup, min_idents)
    n_pairs = sum(u["kind"] == "pair" for u in units)
    n_ord = len(units) - n_pairs
    side_of = assign_sides(units, labels, seed,
                           dev_pairs / max(dev_pairs + test_pairs, 1),
                           dev_ordinary / max(dev_ordinary + test_ordinary, 1))
    members = {"dev": [], "test": []}
    for i, lab in enumerate(labels):
        members[side_of[lab]].append(i)
    all_strata = Counter(u["cell"] for u in units if u["kind"] == "pair")
    chosen: dict[str, list[int]] = {}
    for side, n_p, n_o in (("dev", dev_pairs, dev_ordinary), ("test", test_pairs, test_ordinary)):
        quotas = stratum_quotas(n_p, dict(all_strata))
        chosen[side] = sorted(select_pairs(units, members[side], quotas, seed, side)
                              + select_ordinary(units, members[side], n_o, seed, side))

    def side_counts(idx: list[int], side_members: list[int]) -> dict:
        us = [units[i] for i in idx]
        ps = [u for u in us if u["kind"] == "pair"]
        os_ = [u for u in us if u["kind"] == "ordinary"]
        return {
            "items": sum(len(u["ids"]) for u in us),
            "pairs": len(ps), "vulnerable": len(ps), "fixed_twin": len(ps),
            "ordinary": len(os_),
            "ordinary_length_matched": sum(u["length_matched"] for u in os_),
            "ordinary_not_length_matched": sum(not u["length_matched"] for u in os_),
            "advisories_with_pairs": len({u["advisory"] for u in ps}),
            "repos": len({u["repo"] for u in us if u["repo"]}),
            "groups": len({labels[i] for i in idx}),
            "pairs_by_language": dict(sorted(Counter(u["language"] for u in ps).items())),
            "ordinary_by_language": dict(sorted(Counter(u["language"] for u in os_).items())),
            "pairs_by_category": dict(sorted(Counter(u["category"] for u in ps).items())),
            "side_pool": {"pairs": sum(units[i]["kind"] == "pair" for i in side_members),
                          "ordinary": sum(units[i]["kind"] == "ordinary"
                                          for i in side_members)},
        }

    counts = {side: side_counts(chosen[side], members[side]) for side in ("dev", "test")}
    used = set(chosen["dev"]) | set(chosen["test"])
    counts["unassigned"] = {
        "pairs": sum(units[i]["kind"] == "pair" for i in range(len(units)) if i not in used),
        "ordinary": sum(units[i]["kind"] == "ordinary"
                        for i in range(len(units)) if i not in used),
    }
    counts["total"] = {"pairs": n_pairs, "ordinary": n_ord,
                       "ordinary_length_matched": sum(u.get("length_matched", False)
                                                      for u in units),
                       "advisories_with_pairs": len({u["advisory"] for u in units
                                                     if u["kind"] == "pair"}),
                       "repos": len({u["repo"] for u in units if u["repo"]})}
    counts["leakage"] = leakage_checks(units, chosen, near_dup, min_idents)

    def stratum_counts(idx) -> dict:
        c = Counter(units[i]["stratum"] for i in idx if units[i]["kind"] == "pair")
        return {b: c.get(b, 0) for b in (*bins, UNKNOWN) if c.get(b, 0) or b != UNKNOWN}

    def cutoff_counts(idx) -> dict:
        c = Counter()
        for i in idx:
            u = units[i]
            if u["kind"] != "pair":
                continue
            d = (dates.get(u["advisory"]) or {}).get("public_since")
            c[UNKNOWN if not d else ("before" if d < cutoff else "on_or_after")] += 1
        return dict(sorted(c.items()))

    all_idx = list(range(len(units)))
    date_strata = {
        "field": "public_since",
        "definition": "earliest of the fix commit's author/committer date (GitHub API cache) "
                      "and the earliest OSV `published` over the advisory's aliases; "
                      "CVE-id year as fallback. A lower bound on when the fixed code was "
                      "public; pairs are stratified by its year and by language.",
        "bins": list(bins),
        "cutoff": cutoff,
        "pairs_by_stratum": {"dev": stratum_counts(chosen["dev"]),
                             "test": stratum_counts(chosen["test"]),
                             "all": stratum_counts(all_idx)},
        "pairs_by_cutoff": {"dev": cutoff_counts(chosen["dev"]),
                            "test": cutoff_counts(chosen["test"]),
                            "all": cutoff_counts(all_idx)},
        "date_sources": dict(sorted(Counter(d["date_source"] for d in dates.values()).items())),
        "by_advisory": {adv: {"public_since": d["public_since"], "stratum": strata[adv],
                              "source": d["date_source"], "cve_year": d["cve_year"]}
                        for adv, d in sorted(dates.items())},
    }
    grouping = (
        f"connected components over shared advisory id (display id + OSV aliases), "
        f"{'shared repository, ' if group_by == 'repo' else ''}"
        f"and near-duplicate code (identifier Jaccard >= {near_dup}, >= {min_idents} "
        f"identifiers); a vulnerable/fixed pair is one unit; whole components go to one "
        f"side ({gstats['groups']} components, largest {gstats['largest_group_units']} "
        f"units, {gstats['near_dup_links_merged_across_groups']} near-duplicate links "
        f"merged across otherwise separate components)"
    )
    ids = {side: sorted(x for i in chosen[side] for x in units[i]["ids"]) for side in chosen}
    return {
        "version": 1,
        "seed": seed,
        "dev": {"ids": ids["dev"]},
        "test": {"ids": ids["test"]},
        "meta": {
            "created": created or time.strftime("%Y-%m-%d"),
            "counts": counts,
            "grouping": grouping,
            "date_strata": date_strata,
        },
    }


# ---------------------------------------------------------------------------
# Offline provenance / dates from data/osv_cache (not unit-tested; optional)
# ---------------------------------------------------------------------------


def load_provenance(cache_dir: Path, pairs: list[dict]) -> dict[str, dict]:
    """``{base: {aliases, repo, commits}}`` by matching each pair's exact code to
    the builder's state file. ``{}`` without the cache."""
    state_path = cache_dir / "processed_advisories.json"
    if not state_path.exists():
        return {}
    from scripts.build_corpus_from_osv import pair_code_hash

    wanted = {p["base"].rsplit("_", 1)[-1] for p in pairs}
    by_hash: dict[str, list[dict]] = defaultdict(list)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for entry in state.values():
        for sp in entry.get("pairs") or []:
            h = pair_code_hash(sp["vulnerable_code"], sp["fixed_code"])
            if h in wanted:
                by_hash[h].append(sp)
    del state
    out = {}
    for p in pairs:
        src = [sp for sp in by_hash.get(p["base"].rsplit("_", 1)[-1], [])
               if sp["vulnerable_code"] == p["vuln"]["code"]
               and sp["fixed_code"] == p["safe"]["code"]]
        if src:
            out[p["base"]] = {
                "aliases": sorted({sp["advisory_id"] for sp in src}),
                "repo": sorted({sp["repo"] for sp in src})[0],
                "commits": sorted({(sp["repo"], sp["commit"]) for sp in src}),
            }
    return out


def load_dates(cache_dir: Path, pairs: list[dict], provenance: dict[str, dict]) -> dict:
    """Per display advisory id: ``advisory_date`` from the cached fix-commit
    responses and OSV zip records (both optional)."""
    aliases: dict[str, set] = defaultdict(set)
    commits: dict[str, set] = defaultdict(set)
    for p in pairs:
        adv = p["vuln"]["expected_cve_id"]
        prov = provenance.get(p["base"], {})
        aliases[adv].update(prov.get("aliases", ()))
        aliases[adv].add(adv)
        commits[adv].update(tuple(c) for c in prov.get("commits", ()))
    published: dict[str, str] = {}
    wanted = set().union(*aliases.values()) if aliases else set()
    for name in OSV_ZIPS:
        path = cache_dir / name
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for aid in sorted(wanted):
                if f"{aid}.json" in names and aid not in published:
                    rec = json.loads(zf.read(f"{aid}.json"))
                    if rec.get("published"):
                        published[aid] = rec["published"]
    api = cache_dir / "api"
    out = {}
    for adv in sorted(aliases):
        cdates = []
        for repo, sha in sorted(commits[adv]):
            url = f"https://api.github.com/repos/{repo}/commits/{sha}"
            f = api / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
            if f.exists():
                c = (json.loads(f.read_text(encoding="utf-8")).get("commit") or {})
                cdates += [(c.get(k) or {}).get("date") for k in ("author", "committer")]
        out[adv] = advisory_date(adv, [d for d in cdates if d],
                                 [published[a] for a in aliases[adv] if a in published])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", nargs="+", type=Path, default=DEFAULT_PAIR_FILES)
    ap.add_argument("--ordinary", type=Path, default=DEFAULT_ORDINARY)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-pairs", type=int, default=150)
    ap.add_argument("--test-pairs", type=int, default=100)
    ap.add_argument("--dev-ordinary", type=int, default=500)
    ap.add_argument("--test-ordinary", type=int, default=500)
    ap.add_argument("--group-by", choices=("repo", "advisory"), default="repo")
    ap.add_argument("--near-dup", type=float, default=0.7)
    ap.add_argument("--min-idents", type=int, default=5)
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--created", default=None, help="override meta.created (YYYY-MM-DD)")
    ap.add_argument("--check", action="store_true", help="print counts, write nothing")
    args = ap.parse_args(argv)

    lines = [ln for path in args.pairs for ln in load_jsonl(path)]
    pairs = pair_lines(lines)
    ordinary = load_jsonl(args.ordinary) if args.ordinary else []
    provenance = load_provenance(args.cache_dir, pairs)
    dates = load_dates(args.cache_dir, pairs, provenance)
    print(f"pairs {len(pairs)} (provenance for {len(provenance)}), ordinary {len(ordinary)}, "
          f"advisories {len(dates)}")
    manifest = build_manifest(
        pairs, ordinary, provenance, dates, seed=args.seed, dev_pairs=args.dev_pairs,
        test_pairs=args.test_pairs, dev_ordinary=args.dev_ordinary,
        test_ordinary=args.test_ordinary, group_by=args.group_by, near_dup=args.near_dup,
        min_idents=args.min_idents, cutoff=args.cutoff, created=args.created,
    )
    meta = manifest["meta"]
    print(meta["grouping"])
    print(json.dumps({k: v for k, v in meta["counts"].items()}, indent=1))
    print(json.dumps({k: v for k, v in meta["date_strata"].items() if k != "by_advisory"},
                     indent=1))
    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
