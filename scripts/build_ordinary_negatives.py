"""Build an "ordinary functions" negative eval set from held-out OSV fix commits.

The OSV eval sets are 50% vulnerable (each vulnerable function plus its fixed
twin), so their precision says nothing about a realistic base rate, where only
1-5% of the functions in a PR are vulnerable. This script builds the missing
negatives: ordinary functions from the same real repos, *not* touched by any
security fix.

Source: the post-fix file contents of the **held-out** (eval-side) advisories of
``scripts/build_corpus_from_osv.py``, read offline from its GitHub response
cache (``data/osv_cache/``). The held-out split is recomputed with the builder's
own functions (``split_by_advisory`` over the same deduped pairs, same
``--seed``/``--eval-fraction``), so it is the exact advisory set behind
``detection_eval_osv_{pypi,npm}.jsonl``; drawing only from those advisories
keeps the negatives away from the corpus side. Every function in every cached
post-fix file (the files the fix commit modified) is extracted with
``CodeParser`` and dropped if it:

- sits on a test/doc/example path (the builder's ``_is_skipped_path``);
- overlaps any hunk of the fix commit's patch (context lines included, and a
  pure-deletion hunk counts at the deletion point), or its file has no patch;
- is shorter than ``--min-lines`` or longer than ``--max-lines`` (3 / 120, as
  for the corpus);
- is a trivial accessor (<= ``--trivial-max-stmt-lines`` statement lines,
  i.e. excluding blank/comment/docstring lines; ``--no-drop-trivial`` keeps them);
- is the same ``(repo, file_path, function_name)`` as any function any mined
  fix commit changed (corpus, eval or rejected pair - any version of it);
- has the same normalised body (sha256 of the code with comments and all
  whitespace removed) as any eval item, corpus entry or mined pair, or as an
  ordinary function already taken.

The survivors are sampled deterministically (``--seed``): up to ``--total``
functions, split across languages by the reference eval sets' language mix, at
most ``--per-repo-cap`` per repo. Each item also carries ``length_matched``:
a subset whose line-count distribution matches the OSV *vulnerable* eval items
of the same language by quantile bins (length confounds embedding similarity,
so compare FPR on that subset too).

These functions are not guaranteed bug-free - they are merely not changed by
a known security fix. Pure logic lives in module-level functions (unit-tested
in ``tests/unit/test_build_ordinary_negatives.py``); the script never makes a
network call and never loads an ML model.

``infer_kind`` is the convention for telling eval items apart:
``kind`` field if present (``"ordinary"`` here), else handwritten source ->
``"handwritten"``, id ending ``_vuln`` -> ``"vulnerable"``, ``_safe`` ->
``"fixed_twin"``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

# Add project root to path so this runs as a script (mirrors the other scripts/*.py).
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.core.code_parser import CodeParser  # noqa: E402
from scripts.build_corpus_from_osv import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_OUT_DIR,
    SUPPORTED_EXTENSIONS,
    GithubClient,
    Stats,
    _code_lines,
    _is_skipped_path,
    _normalize_code_for_diff,
    make_dedupe_key,
    ranges_overlap,
    split_by_advisory,
)

ECOSYSTEMS = ("PyPI", "npm")
DEFAULT_OUT = DEFAULT_EVAL_DIR / "detection_eval_ordinary.jsonl"
DEFAULT_REFERENCE = [
    DEFAULT_EVAL_DIR / "detection_eval.jsonl",
    DEFAULT_EVAL_DIR / "detection_eval_osv_pypi.jsonl",
    DEFAULT_EVAL_DIR / "detection_eval_osv_npm.jsonl",
]
DEFAULT_CORPUS = sorted(DEFAULT_OUT_DIR.glob("*.json"))

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


# ---------------------------------------------------------------------------
# Pure logic (no network, no filesystem) - unit-tested directly.
# ---------------------------------------------------------------------------


def changed_line_ranges(patch: str) -> list[tuple[int, int]]:
    """Post-commit line ranges covered by every hunk of a unified diff.

    Unlike the builder's ``parse_hunk_ranges`` (which drops pure-deletion
    hunks, since it only pairs functions that still exist), a ``+c,0`` hunk here
    yields ``(c, c + 1)``: the lines around the deletion point, so a function
    whose body lost lines counts as changed."""
    ranges: list[tuple[int, int]] = []
    for line in patch.splitlines():
        m = _HUNK_RE.match(line)
        if not m:
            continue
        start = int(m.group("start"))
        count = int(m.group("count")) if m.group("count") is not None else 1
        if count <= 0:
            ranges.append((max(start, 1), start + 1))
        else:
            ranges.append((start, start + count - 1))
    return ranges


def overlaps_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(ranges_overlap(start, end, a, b) for a, b in ranges)


def body_hash(code: str, language: str) -> str:
    """sha256 of the body with comments, blank lines and all whitespace removed
    (the builder's whitespace/comment-insensitive normalisation)."""
    return hashlib.sha256(_normalize_code_for_diff(code, language).encode("utf-8")).hexdigest()


def line_count(code: str) -> int:
    return len(code.splitlines())


def statement_line_count(code: str, language: str) -> int:
    """Lines carrying statement tokens (blank, comment and docstring lines vanish)."""
    return len(_code_lines(code, language))


def infer_kind(item: dict) -> str:
    """``kind`` of an eval item: explicit field, else handwritten / vulnerable /
    fixed_twin from ``source`` and the id suffix (``sqli_vuln_1`` is handwritten,
    not a pair member)."""
    if item.get("kind"):
        return item["kind"]
    if item.get("source") == "handwritten":
        return "handwritten"
    item_id = item.get("id", "")
    if item_id.endswith("_vuln"):
        return "vulnerable"
    if item_id.endswith("_safe"):
        return "fixed_twin"
    return "unknown"


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    """The builder's global per-ecosystem dedupe (first occurrence wins)."""
    seen: set[str] = set()
    out = []
    for p in pairs:
        key = make_dedupe_key(p["repo"], p["file_path"], p["function_name"], p["vulnerable_code"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def held_out_advisories(
    pairs_by_ecosystem: dict[str, list[dict]], eval_fraction: float, seed: int
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Recompute the builder's split: per ecosystem, dedupe then
    ``split_by_advisory``. Returns ``(eval_side, corpus_side)`` pairs by ecosystem."""
    eval_side, corpus_side = {}, {}
    for eco, pairs in pairs_by_ecosystem.items():
        corpus, held = split_by_advisory(dedupe_pairs(pairs), eval_fraction, seed)
        eval_side[eco], corpus_side[eco] = held, corpus
    return eval_side, corpus_side


def source_commits(
    eval_side: dict[str, list[dict]], corpus_side: dict[str, list[dict]]
) -> tuple[list[dict], list[str]]:
    """Distinct held-out fix commits to mine, as
    ``{"repo", "commit", "advisory_id", "ecosystem"}`` sorted by (repo, commit).

    An advisory held out in one ecosystem but on the corpus side of another
    (an advisory listed under both PyPI and npm), or a commit that any
    corpus-side advisory also uses, is dropped: its files fed the corpus.
    Returns ``(commits, dropped_advisory_ids)``."""
    corpus_ids = {p["advisory_id"] for ps in corpus_side.values() for p in ps}
    corpus_commits = {(p["repo"], p["commit"]) for ps in corpus_side.values() for p in ps}
    commits: dict[tuple[str, str], dict] = {}
    dropped: set[str] = set()
    for eco in sorted(eval_side):
        for p in eval_side[eco]:
            key = (p["repo"], p["commit"])
            if p["advisory_id"] in corpus_ids or key in corpus_commits:
                dropped.add(p["advisory_id"])
                continue
            commits.setdefault(
                key,
                {"repo": p["repo"], "commit": p["commit"], "advisory_id": p["advisory_id"],
                 "ecosystem": eco},
            )
    return [commits[k] for k in sorted(commits)], sorted(dropped)


def reject_reason(
    func: dict,
    *,
    hunk_ranges: list[tuple[int, int]],
    min_lines: int,
    max_lines: int,
    drop_trivial: bool,
    trivial_max_stmt_lines: int,
    mined_names: set[tuple[str, str, str]],
    known_hashes: set[str],
) -> str | None:
    """First filter an extracted function fails (``None`` = keep). ``func`` has
    ``repo``, ``file_path``, ``name``, ``start_line``, ``end_line``, ``code``,
    ``language``. Within-pool duplicates are handled by the caller."""
    if overlaps_any(func["start_line"], func["end_line"], hunk_ranges):
        return "overlaps_fix_hunk"
    n = line_count(func["code"])
    if n < min_lines:
        return "too_short"
    if n > max_lines:
        return "too_long"
    if drop_trivial and (
        statement_line_count(func["code"], func["language"]) <= trivial_max_stmt_lines
    ):
        return "trivial"
    if (func["repo"], func["file_path"], func["name"]) in mined_names:
        return "same_function_as_mined_pair"
    if body_hash(func["code"], func["language"]) in known_hashes:
        return "duplicate_of_eval_or_corpus_body"
    return None


def language_quotas(total: int, shares: dict[str, float], pools: dict[str, int]) -> dict[str, int]:
    """Per-language target counts: ``total`` split by ``shares`` (largest
    remainder), capped by each pool. A short pool is *not* backfilled from
    another language, so the mix never drifts far from the reference."""
    raw = {lang: total * s for lang, s in shares.items()}
    quotas = {lang: int(v) for lang, v in raw.items()}
    for lang in sorted(raw, key=lambda k: (-(raw[k] - quotas[k]), k)):
        if sum(quotas.values()) >= total:
            break
        quotas[lang] += 1
    return {lang: min(q, pools.get(lang, 0)) for lang, q in quotas.items()}


def _candidate_sort_key(c: dict) -> tuple:
    return (c["repo"], c["commit"], c["file_path"], c["start_line"], c["end_line"], c["name"])


def sample_candidates(
    candidates: list[dict],
    quotas: dict[str, int],
    per_repo_cap: int,
    seed: int,
) -> list[dict]:
    """Deterministic stratified sample: per language (sorted), candidates are
    put in a canonical order, shuffled with ``random.Random(f"{seed}:{lang}")``
    and taken while the repo is under ``per_repo_cap`` (<= 0 disables; the cap
    is across languages) until the language's quota is met. Independent of
    input order."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_lang[c["language"]].append(c)
    picked: list[dict] = []
    per_repo: Counter = Counter()
    for lang in sorted(by_lang):
        pool = sorted(by_lang[lang], key=_candidate_sort_key)
        random.Random(f"{seed}:{lang}").shuffle(pool)
        taken = 0
        for c in pool:
            if taken >= quotas.get(lang, 0):
                break
            if per_repo_cap > 0 and per_repo[c["repo"]] >= per_repo_cap:
                continue
            per_repo[c["repo"]] += 1
            picked.append(c)
            taken += 1
    return sorted(picked, key=lambda c: (c["language"], *_candidate_sort_key(c)))


def quantile_edges(values: list[int], n_bins: int) -> list[float]:
    """``n_bins + 1`` inclusive-range edges at evenly spaced quantiles
    (nearest-rank on the sorted values)."""
    s = sorted(values)
    return [s[min(round(q * (len(s) - 1) / n_bins), len(s) - 1)] for q in range(n_bins + 1)]


def bin_index(value: int, edges: list[float]) -> int | None:
    """Bin of ``value`` for ``edges`` (bins are ``(e[i], e[i+1]]``, the first
    also including ``e[0]``); ``None`` outside ``[e[0], e[-1]]``. Empty bins
    from tied edges simply never match."""
    if value < edges[0] or value > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if value <= edges[i + 1]:
            return i
    return None


def assign_length_matched(
    items: list[dict], reference: dict[str, list[int]], n_bins: int, seed: int
) -> dict[str, int]:
    """Set ``item["length_matched"]`` in place: per language, the largest
    subset whose share in each quantile bin (bins from ``reference[lang]``, the
    vulnerable items' line counts) equals the reference's share. Picks within a
    bin are seeded, so it is deterministic. Returns matched counts by language."""
    matched: dict[str, int] = {}
    for item in items:
        item["length_matched"] = False
    for lang in sorted({i["language"] for i in items}):
        ref = reference.get(lang) or []
        mine = [i for i in items if i["language"] == lang]
        if not ref:
            matched[lang] = 0
            continue
        edges = quantile_edges(ref, n_bins)
        ref_bins = Counter(bin_index(v, edges) for v in ref)
        target = {b: ref_bins[b] / len(ref) for b in ref_bins if b is not None}
        in_bin: dict[int, list[dict]] = defaultdict(list)
        for it in mine:
            b = bin_index(it["line_count"], edges)
            if b is not None:
                in_bin[b].append(it)
        size = min(int(len(in_bin[b]) / share) for b, share in target.items() if share > 0)
        count = 0
        for b in sorted(target):
            take = min(round(size * target[b]), len(in_bin[b]))
            pool = sorted(in_bin[b], key=lambda i: i["id"])
            random.Random(f"{seed}:{lang}:{b}").shuffle(pool)
            for it in pool[:take]:
                it["length_matched"] = True
            count += take
        matched[lang] = count
    return matched


def make_item_id(c: dict) -> str:
    fn_token = re.sub(r"[^A-Za-z0-9_-]+", "_", c["name"])
    digest = hashlib.sha1(
        f"{c['repo']}|{c['commit']}|{c['file_path']}|{c['start_line']}".encode()
    ).hexdigest()[:10]
    return f"ordinary_{fn_token}_{digest}"


def build_item(c: dict, corpus_repos: set[str]) -> dict:
    """One eval line in the ``detection_eval*.jsonl`` shape (+ provenance)."""
    return {
        "id": make_item_id(c),
        "language": c["language"],
        "label": "safe",
        "category": "none",
        "expected_cve_id": None,
        "source": "osv_ordinary",
        "kind": "ordinary",
        "code": c["code"],
        "repo": c["repo"],
        "commit": c["commit"],
        "advisory_id": c["advisory_id"],
        "file_path": c["file_path"],
        "function_name": c["name"],
        "start_line": c["start_line"],
        "line_count": line_count(c["code"]),
        "repo_in_corpus": c["repo"] in corpus_repos,
        "length_matched": False,
    }


def describe(values: list[int]) -> str:
    if not values:
        return "n=0"
    s = sorted(values)

    def q(p: float) -> int:
        return s[min(int(p * len(s)), len(s) - 1)]

    return (
        f"n={len(s):4d} mean={sum(s) / len(s):5.1f} p10={q(0.1):3d} p25={q(0.25):3d} "
        f"med={q(0.5):3d} p75={q(0.75):3d} p90={q(0.9):3d} max={s[-1]:3d}"
    )


# ---------------------------------------------------------------------------
# Filesystem side (cache reads only; no network).
# ---------------------------------------------------------------------------


def ecosystem_advisory_ids(cache_dir: Path, ecosystem: str) -> set[str]:
    """Advisory ids in the cached OSV zip (one ``{id}.json`` per advisory)."""
    zip_path = Path(cache_dir) / f"{ecosystem}_all.zip"
    if not zip_path.exists():
        sys.exit(f"{zip_path} is not cached; run build_corpus_from_osv.py first.")
    with zipfile.ZipFile(zip_path) as zf:
        return {Path(n).stem for n in zf.namelist() if n.endswith(".json")}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--reference", nargs="+", default=[str(p) for p in DEFAULT_REFERENCE],
                   help="Eval sets: language mix, length reference (OSV vulnerable items) "
                   "and body dedupe.")
    p.add_argument("--corpus", nargs="+", default=[str(p) for p in DEFAULT_CORPUS],
                   help="Corpus files whose vulnerable/fixed bodies are excluded.")
    p.add_argument("--eval-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total", type=int, default=1000)
    p.add_argument("--per-repo-cap", type=int, default=25)
    p.add_argument("--min-lines", type=int, default=3)
    p.add_argument("--max-lines", type=int, default=120)
    p.add_argument("--drop-trivial", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop trivial accessors (<= --trivial-max-stmt-lines statement lines).")
    p.add_argument("--trivial-max-stmt-lines", type=int, default=3)
    p.add_argument("--length-bins", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    state_path = cache_dir / "processed_advisories.json"
    if not state_path.exists():
        sys.exit(f"{state_path} missing; run build_corpus_from_osv.py first.")
    state: dict[str, dict] = json.loads(state_path.read_text(encoding="utf-8"))

    # 1. Recompute the held-out split exactly as the builder does.
    pairs_by_eco: dict[str, list[dict]] = {}
    for eco in ECOSYSTEMS:
        ids = ecosystem_advisory_ids(cache_dir, eco)
        pairs_by_eco[eco] = [p for aid in sorted(ids & state.keys()) for p in state[aid]["pairs"]]
    eval_side, corpus_side = held_out_advisories(pairs_by_eco, args.eval_fraction, args.seed)
    commits, dropped = source_commits(eval_side, corpus_side)
    held_ids = {p["advisory_id"] for ps in eval_side.values() for p in ps}

    # 2. Exclusion sets: every mined function (any side, any filter outcome) by
    #    name and body, plus every eval item and corpus entry body.
    mined_names: set[tuple[str, str, str]] = set()
    known_hashes: set[str] = set()
    for entry in state.values():
        for p in entry.get("pairs", []):
            mined_names.add((p["repo"], p["file_path"], p["function_name"]))
            known_hashes.add(body_hash(p["vulnerable_code"], p["language"]))
            known_hashes.add(body_hash(p["fixed_code"], p["language"]))
    reference_items: list[dict] = []
    for path in args.reference:
        reference_items.extend(load_jsonl(Path(path)))
    for it in reference_items:
        known_hashes.add(body_hash(it["code"], it["language"]))
    corpus_repos: set[str] = set()
    for path in args.corpus:
        for e in json.loads(Path(path).read_text(encoding="utf-8")):
            lang = e.get("language", "python")
            for key in ("vulnerable_code", "fixed_code"):
                if e.get(key):
                    known_hashes.add(body_hash(e[key], lang))
            if e.get("repo"):
                corpus_repos.add(e["repo"])
    del state

    # 3. Extract every function of every cached post-fix file.
    stats = Stats()
    client = GithubClient(requests.Session(), None, cache_dir, stats, offline=True)
    parser = CodeParser()
    reasons: Counter = Counter()
    candidates: list[dict] = []
    seen_hashes: set[str] = set()
    extracted = 0
    commits_used = 0
    files_used = 0
    missing_commit = 0
    missing_files = 0
    advisories_used: set[str] = set()
    flt = dict(
        min_lines=args.min_lines, max_lines=args.max_lines, drop_trivial=args.drop_trivial,
        trivial_max_stmt_lines=args.trivial_max_stmt_lines, mined_names=mined_names,
        known_hashes=known_hashes,
    )
    for src in commits:
        owner, repo = src["repo"].split("/", 1)
        commit_data = client.get_cached_json(
            f"https://api.github.com/repos/{owner}/{repo}/commits/{src['commit']}"
        )
        if not commit_data or commit_data.get("__status__") == 404:
            missing_commit += 1
            continue
        used_file = False
        for f in commit_data.get("files") or []:
            path = f.get("filename", "")
            ext = Path(path).suffix.lower()
            if f.get("status") != "modified" or ext not in SUPPORTED_EXTENSIONS:
                continue
            if _is_skipped_path(path):
                reasons["skipped_path_file"] += 1
                continue
            patch = f.get("patch")
            if not patch:
                reasons["no_patch_file"] += 1
                continue
            content = client.get_file(owner, repo, path, src["commit"])
            if content is None:
                missing_files += 1
                continue
            used_file = True
            files_used += 1
            hunks = changed_line_ranges(patch)
            language = SUPPORTED_EXTENSIONS[ext]
            for fn in parser.extract_functions(content, ext):
                extracted += 1
                func = {**fn, **src, "file_path": path, "language": language}
                reason = reject_reason(func, hunk_ranges=hunks, **flt)
                if reason is None:
                    h = body_hash(fn["code"], language)
                    if h in seen_hashes:
                        reason = "duplicate_within_pool"
                    else:
                        seen_hashes.add(h)
                if reason:
                    reasons[reason] += 1
                    continue
                candidates.append(func)
        if used_file:
            commits_used += 1
            advisories_used.add(src["advisory_id"])

    # 4. Stratified, per-repo-capped sample.
    lang_counts = Counter(it["language"] for it in reference_items)
    shares = {lang: n / sum(lang_counts.values()) for lang, n in lang_counts.items()}
    pools = Counter(c["language"] for c in candidates)
    quotas = language_quotas(args.total, shares, pools)
    picked = sample_candidates(candidates, quotas, args.per_repo_cap, args.seed)
    items = [build_item(c, corpus_repos) for c in picked]

    # 5. Length comparison + length-matched flag against OSV vulnerable items.
    ref_lengths: dict[str, list[int]] = defaultdict(list)
    for it in reference_items:
        if it.get("source") == "osv" and it["label"] == "vulnerable":
            ref_lengths[it["language"]].append(line_count(it["code"]))
    matched = assign_length_matched(items, ref_lengths, args.length_bins, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    # 6. Summary.
    print("=" * 78)
    print("Ordinary-function negatives")
    print("=" * 78)
    print(f"Held-out advisories (recomputed split): {len(held_ids)}")
    print(f"  dropped (also on the corpus side):    {len(dropped)} {dropped}")
    print(f"Distinct held-out fix commits:          {len(commits)}")
    print(f"  commit response not cached:           {missing_commit}")
    print(f"  post-fix files not cached (skipped):  {missing_files}")
    print(f"Commits / advisories / files used:      {commits_used} / {len(advisories_used)} "
          f"/ {files_used}")
    print(f"Functions extracted:                    {extracted}")
    print(f"Candidates after filters:               {len(candidates)} "
          f"{dict(sorted(pools.items()))}")
    print("Filtered by reason (file-level reasons count files):")
    for reason, n in reasons.most_common():
        print(f"  {reason:34s} {n}")
    print(f"Reference language shares: "
          f"{ {k: round(v, 3) for k, v in sorted(shares.items())} }; quotas {quotas}")
    by_lang = Counter(it["language"] for it in items)
    print(f"Sampled: {len(items)} {dict(sorted(by_lang.items()))} "
          f"(per-repo cap {args.per_repo_cap})")
    per_repo = Counter(it["repo"] for it in items)
    print(f"Repos covered: {len(per_repo)}; max per repo {max(per_repo.values(), default=0)}; "
          f"items from repos also in the corpus: {sum(it['repo_in_corpus'] for it in items)}")
    print("\nLine counts (vulnerable = OSV eval vulnerable items):")
    for lang in sorted(set(ref_lengths) | set(by_lang)):
        print(f"  {lang}")
        print(f"    vulnerable      {describe(ref_lengths.get(lang, []))}")
        pool_n = [line_count(c["code"]) for c in candidates if c["language"] == lang]
        mine = [it for it in items if it["language"] == lang]
        print(f"    candidate pool  {describe(pool_n)}")
        print(f"    sampled         {describe([it['line_count'] for it in mine])}")
        print(f"    length_matched  "
              f"{describe([it['line_count'] for it in mine if it['length_matched']])}")
    print(f"length_matched: {sum(matched.values())} {matched}")
    print(f"\nWrote {len(items)} items to {out_path}")
    print("GitHub API calls made: 0 (offline cache only)")


if __name__ == "__main__":
    main()
