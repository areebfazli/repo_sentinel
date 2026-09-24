# RepoSentinel

**An AI security reviewer for pull requests.** RepoSentinel embeds a developer's code and
runs it concurrently against two vector collections: **Ghost Hunter** (known CVE
vulnerable-code snippets) and **Team Memory** (the team's own past PR review discussions),
ranks the matches (by similarity by default; an optional cross-encoder rerank is available),
then asks an LLM to write a structured, actionable PR comment.

It ships as three surfaces:

- a **FastAPI** backend with an async job API,
- a static **vanilla-JS dashboard**, and
- a **GitHub Action** that posts inline review comments on pull requests.

---

## Why two collections?

Most AI reviewers only know about public vulnerabilities. RepoSentinel adds a second memory:
the review comments your team has *already written*. When a new PR repeats a mistake a senior
reviewer flagged six months ago, RepoSentinel surfaces that past discussion, weighted by how
recent it was and how senior the reviewer is.

| Collection | Qdrant name | Source | What it catches |
|------------|-------------|--------|-----------------|
| **Ghost Hunter** | `cve_corpus` | CVE vulnerable-code snippets plus, where known, the patched version (`data/cve_corpus/*.json`) | Known vulnerability patterns (SQLi, command injection, XSS, path traversal, …) |
| **Team Memory** | `team_history` | Closed-PR review comments crawled from GitHub | Team-specific conventions and past mistakes |

---

## Architecture

```
POST /api/v1/analyze/  ──▶  Scan row (queued)  ──▶  202 + job_id
                                    │
                                    ▼  (FastAPI BackgroundTask)
                            ┌─────────────────┐
                            │   RagMerger     │  loads models once, shared
                            └───────┬─────────┘
                     ┌──────────────┴──────────────┐
                     ▼                              ▼
             CVE retriever                  Team retriever
        (Ghost Hunter, gather)          (Team Memory, gather)
                     │                              │
   Embedder (jina-embeddings-v2-base-code, mean-pooled, 768-d, cached)
   → VectorStore ANN top-N (optional language filter)
   → similarity gate → similarity order (optional cross-encoder rerank, code-vs-code)
   → feedback suppression / downweight → top-k
                     └──────────────┬──────────────┘
                                    ▼
            LLM report (Groq gpt-oss-120b → Groq qwen3.8-27b → Gemini)
                   structured JSON, allowlist-validated
                                    │
                                    ▼
                     deterministic server-side Markdown

GET /api/v1/analyze/{job_id}  ◀── clients poll for the result
```

**Key design points**

- **Async jobs.** `POST /api/v1/analyze/` returns `202 + job_id` immediately and runs the work
  in a background task; clients poll `GET /api/v1/analyze/{job_id}`. Findings are persisted so
  the feedback loop works regardless of who's polling.
- **Retrieval is tuned for recall, the LLM is the precision filter.** Embedding similarity
  alone can't reliably separate a safe snippet from its vulnerable twin (they embed alike), so
  the similarity gate is deliberately low (`SIM_THRESHOLD_CVE=0.25`) and the LLM report does the
  final precision filtering.
- **Anti-hallucination.** The LLM prompt embeds an allowlist of the retrieved CVE / PR IDs;
  any finding that references an ID outside that list is dropped server-side.
- **Feedback loop.** Every finding ties back to a specific vector (`point_id`). Thumbs-up/down
  votes are aggregated across scans and used to suppress or downweight noisy matches.
- **Files mode.** Changed files are split into functions with tree-sitter; only functions that
  overlap changed lines are analyzed (`# reposentinel-ignore` skips one). Findings are anchored
  to file + line for inline PR comments.

---

## Quick start

All Python is run **from the project root** (the code uses absolute `backend.app.*` imports;
there is no installed package).

```bash
pip install -r requirements.txt
cp .env.example .env                 # fill in keys; defaults to ENVIRONMENT=development
```

### 1. Seed both collections

The API returns useful results only once **both** collections are seeded. Local Qdrant
(`./qdrant_data`) is single-process, so **stop the API before running any ingest script.**

```bash
# Ghost Hunter: CVE corpus (reads data/cve_corpus/*.json). Entries with `fixed_code`
# also store the patched twin, and the LLM prompt shows the fix diff.
python scripts/ingest_cve_corpus.py --recreate

# Team Memory: demo data (offline)
python scripts/ingest_team_history.py --mock --recreate

# Team Memory: real closed-PR review comments (needs GITHUB_TOKEN)
python scripts/ingest_team_history.py --repo owner/name
```

> Re-run with `--recreate` on **both** collections whenever the embedding model changes;
> mixed-model vectors silently destroy relevance.

### 2. Run the API

```bash
python -m backend.app.main            # serves 0.0.0.0:8000 (loads models into RAM on boot)

# No LLM key handy? Run with the explicit mock provider:
LLM_PROVIDER=mock python -m backend.app.main
```

### 3. Serve the dashboard

```bash
cd frontend && python -m http.server 8080
# then open http://localhost:8080
```

CORS is scoped to `CORS_ORIGINS` (default `localhost:8080`). Opening `index.html` directly via
`file://` won't work; `Origin: null` isn't allowed. The dashboard POSTs to
`http://127.0.0.1:8000/api/v1/analyze/` and polls for the result.

---

## Configuration

Settings come from `.env` via Pydantic (`backend/app/config.py`). Highlights:

| Setting | Default | Notes |
|---------|---------|-------|
| `ENVIRONMENT` | `development` | `development` = file Qdrant + SQLite (Docker-free); `production` = networked Qdrant + Postgres |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Mean-pooled, 768-dim; needs `EMBEDDING_TRUST_REMOTE_CODE=True` |
| `SIM_THRESHOLD_CVE` | `0.25` | Similarity gate, tuned for high recall |
| `RERANKER_ENABLED` | `False` | Cross-encoder rerank stage. Off: no measurable gain on the held-out OSV eval (category hit 0.373 off vs 0.36–0.41 on, at 7–49 s/item) and it costs ~3 GB RAM; when off it is never loaded and matches keep similarity order |
| `RERANKER_MODEL` / `RERANKER_MAX_TOKENS` | `BAAI/bge-reranker-v2-m3` / `512` | Used only when enabled; 512 was the best-measured length and half the cost of 1024 |
| `RERANK_THRESHOLD` | `0.0` | Gate on `sigmoid(logit)`; only applies with `RERANKER_ENABLED`. Keep-all: rerank probability didn't separate vulnerable from fixed at any threshold |
| `TWIN_MARGIN_MIN` | (off) | Drop CVE matches that look at least as much like the stored fix as like the bug; calibrate with `run_eval --margin-sweep` |
| `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` | `groq` / `gemini` | OpenAI-compatible endpoints; primary → fallback |
| `GROQ_MODEL` / `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-120b` / `qwen/qwen3.8-27b` | With a Groq primary the chain is Groq primary model → Groq fallback model → `LLM_FALLBACK_PROVIDER` (Groq rate-limits per model). The job result's `llm_provider_used` names the model that answered, e.g. `groq:qwen/qwen3.8-27b` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | (none) | A configured provider with a missing key **hard-fails at startup** |
| `REPOSENTINEL_API_KEY` | (none) | `X-RepoSentinel-Key` header, optional in dev, required in production |
| `HYBRID_ENABLED` | `False` | Dense + sparse (BM25) RRF fusion; off by default (measured F1 gain below the adoption bar) |

See `.env.example` for the full list.

---

## GitHub Action

`.github/workflows/repo_sentinel.yml` runs `github_action/scan_pr.py` on pull requests. It
collects the PR's changed files, POSTs them in files mode, and posts **inline review comments**
anchored to changed lines (deduped across pushes via hidden
`<!-- reposentinel:f:<sha1> -->` markers), plus a summary comment and a configurable severity
gate.

Configure two repository secrets:

- `REPOSENTINEL_URL`: where your API is reachable
- `REPOSENTINEL_API_KEY`: matches the backend's `REPOSENTINEL_API_KEY`

`INPUT_FAIL_ON_SEVERITY` (default `high`) controls when the check fails the build.

---

## Tests & lint

```bash
python -m pytest -m "not slow"        # fast suite (no model loads)
python -m pytest -m slow              # eval regression (needs a seeded cve_corpus + API stopped)
ruff check backend scripts tests ml github_action
```

The fast suite never downloads models (`conftest.py` sets `PRELOAD_MODELS=false` and mocks the
LLM). Each run uses a fresh temp SQLite database.

## Detection eval harness

Calibrate thresholds or compare embedding models:

```bash
python -m ml.evaluation.run_eval --write-baseline \
    --dataset ml/evaluation/datasets/detection_eval.jsonl \
    ml/evaluation/datasets/detection_eval_osv_pypi.jsonl \
    ml/evaluation/datasets/detection_eval_osv_npm.jsonl
```

The eval quantifies the precision ceiling (~0.5 across thresholds, since safe and vulnerable
near-twins embed alike), which is *why* the gate favors recall and the LLM report is the real
precision filter. **Category hit rate** (did we retrieve the right CVE?) is the meaningful
retrieval metric; the calibrated baseline lives in `ml/evaluation/baseline.json`.
The reranker follows `RERANKER_ENABLED` (off, so runs are fast retrieval-only metrics);
`--rerank` forces it on and `--no-rerank` off, and the baseline records which was used. The
default `--sim-sweep` (0.20–0.95) always includes the 0.25 operating point.
`--sample N --seed S` runs a label-balanced subsample for quick config comparisons (e.g. of
`--reranker-model`/`--reranker-max-tokens`); a sample can't be written as the baseline.

**Realistic metrics.** Every run also reports metrics by item `kind` (an explicit `kind`
field, else from the id: `<p>_vuln` = vulnerable, `<p>_safe` = its fixed twin, anything else
= handwritten; `detection_eval_ordinary.jsonl` holds ordinary, non-security functions with
`kind: "ordinary"`). These are TPR on vulnerable items and FPR on fixed twins, ordinary and
handwritten-safe items (each with a Wilson 95% interval). They also include precision at
realistic base rates, `TPR·π / (TPR·π + FPR_ordinary·(1−π))` for π = 0.01/0.02/0.05, the
balanced 50/50 precision, and pairwise discrimination: the share of vuln/twin pairs where
only the vulnerable one is flagged, and the reverse. `--sample-kinds
vulnerable=40,fixed_twin=40,ordinary=80 --seed S` draws a stratified sample that keeps
pairs together. With the same kinds and seed, raising the quotas gives a superset.

**LLM report stage (`--llm`).** This measures the real precision filter end to end. For
every item with retrieved CVEs, the eval builds the production prompt (the top
`RETRIEVAL_TOP_K` matches, including the fix diff) and calls the configured `LLMRouter`. It
then validates the findings against the retrieved-ID allowlist exactly as the API does. An
item counts as vulnerable when a validated finding references a retrieved CVE.
`realistic_any_finding` also scores the API's `is_vulnerable`, which is true for any
validated finding. The run makes real provider calls, so it has several limits:
- `--llm-max-calls N` is required and capped at 200.
- Calls are paced by `--llm-sleep` (2.5 s) and `--llm-tpm` (8000 tokens/min).
- `--llm-token-budget T` sets a hard token stop.
- The run stops cleanly on a daily-limit error or on repeated rate limits.
- Results are cached in `ml/evaluation/results/llm_cache.jsonl` by (item id, prompt
  sha256, model), so re-running the same command resumes without re-calling.
- `--llm-primary-only` keeps every answer on one model.

Results go to `--out` only, never to the baseline:

```bash
python -m ml.evaluation.run_eval --no-rerank --sample-kinds vulnerable=14,fixed_twin=14,ordinary=22 \
    --seed 42 --llm --llm-primary-only --llm-max-calls 60 --llm-token-budget 150000 \
    --out ml/evaluation/results/llm_run.json \
    --dataset ml/evaluation/datasets/detection_eval_osv_pypi.jsonl \
    ml/evaluation/datasets/detection_eval_osv_npm.jsonl ml/evaluation/datasets/detection_eval_ordinary.jsonl
```

---

## Project layout

```
backend/app/
  api/            FastAPI routes + auth dependency
  core/           retrieval pipeline, LLM client, renderer, parsers, scoring
  db/             SQLAlchemy models + session (scans, findings, feedback, …)
  services/       scan_runner (the background job)
  main.py         app entrypoint (module-only: python -m backend.app.main)
data/cve_corpus/  CVE snippet corpus (JSON)
frontend/         static vanilla-JS dashboard
github_action/    PR scanner script
ml/evaluation/    detection eval harness + baseline
scripts/          ingestion scripts for both collections
tests/            pytest suite (unit + integration; slow eval regression)
```

---

## Notes & gotchas

- **The LLM is Groq/Gemini, never a silent mock.** Mock output only happens with an explicit
  `LLM_PROVIDER=mock`. Defaults are `openai/gpt-oss-120b` with `qwen/qwen3.8-27b` as a
  same-provider Groq fallback before Gemini; a retired model id (HTTP 404) is logged as a
  "not found or decommissioned" warning naming the model and falls through to the next one.
- **Local Qdrant is single-process.** The API, ingest scripts, and eval can never run at the
  same time; scripts print "stop the API first" on a lock error.
- **Any embedding-model change requires `--recreate` on both collections.** Payloads carry
  `embedding_model` so drift is detectable.

---

## Growing the corpus from real CVE fixes

`scripts/build_corpus_from_osv.py` mines the [OSV](https://osv.dev) bulk export for an
ecosystem (`PyPI`, `npm`) for advisories that reference a GitHub fix commit, fetches that
commit via the GitHub API, and pairs up the pre-commit ("vulnerable") and post-commit
("fixed") version of every function whose body actually changed inside the commit's diff. It
produces a corpus grounded in real fixes rather than handwritten snippets, plus a held-out eval
set split **by advisory** (never by individual function) so nothing in eval shares an advisory
with the training corpus.

```bash
python scripts/build_corpus_from_osv.py --ecosystem PyPI --ecosystem npm --max-advisories 200
```

- Writes `data/cve_corpus/osv_{pypi,npm}.json` (same shape as `sample_cves.json`, plus
  `fixed_code`/`repo`/`commit`/`file_path`/`function_name`) — feed it to
  `scripts/ingest_cve_corpus.py` like any other corpus file.
- Writes `ml/evaluation/datasets/detection_eval_osv_{pypi,npm}.jsonl` (two lines per held-out pair: one
  `vulnerable`, one `safe`), in the same format as `detection_eval.jsonl`.
- Caches the OSV zip and every raw GitHub API response under `data/osv_cache/` (gitignored),
  keyed by request URL, so re-runs — especially `--resume` — make zero redundant API calls.
- Without `GITHUB_TOKEN` set, GitHub's unauthenticated rate limit (60 requests/hour) is the
  practical ceiling; the script always respects `--max-advisories` and, on hitting the rate
  limit, stops cleanly with a clear message rather than crashing or hammering the API (pass
  `--wait-on-rate-limit` to sleep until it resets instead).
- **Quality filter.** Fix commits also touch bystander functions, so every pair gets a
  `quality` object (an added field; ingest ignores it): `changed_stmt_lines` (removed + added
  lines after dropping blank, comment and docstring lines, formatter re-wraps, and consistent
  identifier renames), `rename_only`, `renamed_identifiers`, `security_rename`,
  `security_tokens` (hits from the per-category `SECURITY_TOKENS` keyword table in the changed
  lines), `functions_in_commit` and `files_in_commit`. The filter flags:
  - `--min-changed-stmt-lines 1` (default) drops rename-only and comment/format-only pairs.
  - `--max-functions-per-commit 6` (default; `0` disables) drops every pair from a broad
    commit.
  - `--drop-other` (off by default) drops pairs with no or an unmapped CWE.
  - `--keep-security-renames` (on by default; `--no-keep-security-renames` to disable) keeps rename-only pairs whose renamed
    identifier looks security-relevant (`md5` → `sha256`, `load` → `safe_load`).

  A rename counts only if it is a consistent 1:1 mapping across the whole function. A semantic
  swap such as `url(...)` → `url_for(...)` still counts as a rename, because token shape can't
  tell it apart from a cosmetic one. Rejected pairs, each with a `reject_reason`, go to
  `data/cve_corpus/rejected/osv_{eco}_rejected.json`. They sit in a subdirectory because ingest
  loads every `data/cve_corpus/*.json`. The advisory split runs before the filter, so the
  held-out advisories don't depend on the filter flags, and a rejected pair never reaches the
  eval set. The summary prints kept and rejected counts by reason, the kept category mix and
  the `other` share.
- `--offline` reads only the cache and makes no network calls. To re-filter a finished run
  (for example after changing the flags), run
  `python scripts/build_corpus_from_osv.py --ecosystem PyPI --resume --offline --max-advisories 100000`.
- All of the pure logic lives in importable, network-free module-level functions. That covers
  commit-URL parsing, diff-hunk overlap, the CWE→category table, whitespace-only-change
  detection, the advisory split, the dedupe key, and the quality score and filter. See
  `tests/unit/test_build_corpus_from_osv.py`.

### Ordinary-function negatives

The OSV eval sets are half vulnerable, so their precision doesn't reflect a real PR, where
maybe 1–5% of functions are vulnerable. `scripts/build_ordinary_negatives.py` builds
`ml/evaluation/datasets/detection_eval_ordinary.jsonl`: 1,000 ordinary functions from the same
real repos that no security fix touched. It works offline from `data/osv_cache/`, makes no API
calls and loads no models.

```bash
python scripts/build_ordinary_negatives.py        # --seed 42 --eval-fraction 0.15, as the builder
```

- **Source.** The functions come from the files each **held-out** advisory's fix commit modified,
  taken at the fix commit. The script recomputes the split with the builder's own functions. An
  advisory or commit that also appears on the corpus side is dropped.
- **Exclusions.** A function is dropped if it:
  - overlaps any patch hunk (context lines and pure deletions included);
  - sits on a test, doc or example path;
  - is outside 3–120 lines;
  - is a trivial accessor with ≤ 3 statement lines (`--no-drop-trivial` keeps these);
  - has the same repo, file and name as any function a mined fix commit changed;
  - has the same normalised body (sha256, with comments and whitespace removed) as any eval item,
    corpus entry or earlier pick.
- **Sampling.** Selection is seeded. The language mix follows the eval sets (80% Python, 20%
  JavaScript), with at most `--per-repo-cap 25` functions per repo.
- **Item fields.** Items use the eval format with `label: "safe"`, `category: "none"`,
  `expected_cve_id: null`, `source: "osv_ordinary"` and `kind: "ordinary"`. They also carry
  `repo`, `commit`, `advisory_id`, `file_path`, `function_name`, `line_count` and
  `repo_in_corpus`.
- **`length_matched`.** Ordinary functions are much shorter than the vulnerable eval items
  (median 14 vs 29 lines). `length_matched: true` marks the largest subset that matches the OSV
  vulnerable items' line-count quintiles per language. Report false-positive rate on that
  subset too.
- **`kind` for older eval items.** Older items have no `kind` field; it comes from the id:
  - `source: "handwritten"` → `handwritten`
  - an id ending `_vuln` → `vulnerable`
  - an id ending `_safe` → `fixed_twin`

  `infer_kind` in the script implements this.
- **Caveat.** These functions are not known to be bug-free. They were only never changed by a
  known security fix.
