# Eval splits

`v1.json` is a fixed dev/test split of the OSV pair sets and the ordinary negatives. Tune
prompts, thresholds and gates on **dev**. Report numbers on **test**, and look at it only once
per decision. Built offline and deterministically by `scripts/build_eval_split.py` (seed 42):

```bash
python scripts/build_eval_split.py            # -> ml/evaluation/splits/v1.json
python scripts/build_eval_split.py --check    # print counts, write nothing
```

Format: `{"version": 1, "seed": 42, "dev": {"ids": [...]}, "test": {"ids": [...]},
"meta": {"created", "counts", "grouping", "date_strata"}}`. The ids are eval item ids: both
twins of a pair (`<advisory>_<fn>_<hash8>_vuln` / `_safe`) plus the ordinary ids.

## Grouping

Items are grouped into connected components. Two items share a component if they have:

- the same advisory: the display CVE/GHSA id, or any OSV alias (GHSA/PYSEC) of the source
  advisory, taken from `data/osv_cache/processed_advisories.json`;
- the same repository. Ordinary functions come from the held-out advisories' own fix commits,
  so a repo's ordinary functions stay with its pairs;
- near-duplicate code (identifier Jaccard ≥ 0.7, at least 5 identifiers).

Each component goes to one side as a whole. v1 has 203 components (one per repo). The largest
has 44 units, and the near-duplicate rule merged nothing extra: the only cross-advisory
near-duplicate is in one repo.

## Dates

Each advisory has `public_since`, the earliest of:

- the fix commit's author/committer date (from the GitHub API cache);
- the earliest OSV `published` across its aliases.

This is a lower bound on when the fixed code was public, so a split taken at "after the cutoff"
is conservative. OSV `published` alone is misleading: GHSA backfilled many old CVEs in 2022.

Pairs are stratified by `public_since` year (≤2022 / 2023 / 2024 / 2025 / 2026+) × language.
`meta.date_strata.by_advisory` keeps the date for every advisory, so results can be split at
any model's training cutoff. `pairs_by_cutoff` uses 2025-01-01 (change it with `--cutoff`).

## v1 counts

| | pairs (vuln + fixed) | ordinary | … length-matched | … not length-matched | advisories | repos |
|---|---|---|---|---|---|---|
| dev | 151 | 498 | 161 | 337 | 82 | 96 |
| test | 100 | 500 | 163 | 337 | 60 | 86 |
| unassigned | 251 | 2 | | | | |
| total | 502 | 1,000 | 324 | 676 | 276 | 203 |

- Pairs by year: dev 52 / 18 / 19 / 17 / 45 and test 34 / 13 / 11 / 12 / 30 (≤2022 … 2026+).
  Split at 2025-01-01, that is dev 89 before / 62 after and test 58 / 42.
- JavaScript pairs: dev 31/151, test 20/100.
- Leakage checks: 0 advisories, 0 repos and 0 exact code bodies in both splits. There are 0
  cross-split code pairs with identifier Jaccard ≥ 0.7; every twin and ordinary function was
  compared.
- The 251 unassigned pairs are reserved. Each belongs to a component whose side is fixed by the
  seed. To grow a split without leakage, re-run the builder with larger `--dev-pairs` /
  `--test-pairs`. The 2 unassigned ordinary functions are over the 500 cap.
- Per-category counts are in `meta.counts.<split>.pairs_by_category`. Categories are not
  stratified: most have fewer than 50 pairs.

## Ordinary pool is too small to certify a low FPR

A split with 500 ordinary functions and **zero** false positives still has a Wilson 95% upper
bound of 0.76%. The full 1,000 with zero false positives gives 0.38%, and one false positive
already gives 0.56%. To show FPR ≤ 0.5% with a true FPR of 0.1% / 0.25% / 0.35%, one split
needs ≈ 1,120 / 2,590 / 7,540 ordinary functions. Functions from the same repo are correlated,
so the real requirement is higher still.

`ml/evaluation/datasets/detection_eval_ordinary_ext.jsonl` is a larger pool. It has 3,377
functions (1,062 length-matched) from 198 repos, still capped at 25 per repo. It was built
offline in 14–18 s at 265 MB peak RSS:

```bash
python scripts/build_ordinary_negatives.py --total 20000 \
    --out ml/evaluation/datasets/detection_eval_ordinary_ext.jsonl
# then drop the 4 items whose ids collide (see below):
python - <<'EOF'
import json, collections
p = "ml/evaluation/datasets/detection_eval_ordinary_ext.jsonl"
rows = [json.loads(l) for l in open(p)]
c = collections.Counter(r["id"] for r in rows)
open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows if c[r["id"]] == 1))
EOF
```

Notes on the larger pool:

- The current 1,000 functions are almost all in it (951 of 1,000).
- `--per-repo-cap 100` gives 7,824 functions, but more of them come from the same few repos.
- The candidate pool has 11,638 functions.
- v1 does not use this file. A v2 split can reuse v1's repo-to-side assignment.
- **Known builder issue:** `make_item_id` hashes repo|commit|file|start_line only. Nested
  anonymous functions that start and end on the same line get the same id, and their order in
  the output isn't stable between runs. At `--total 1000` this doesn't happen; at 20,000 it
  affects 2 ids (4 items), which the step above drops.
- Cost: an LLM review of 3,377 functions is about 4–5M tokens, weeks of Groq's free tier
  (200K tokens/day).
