# RepoSentinel

**An AI security reviewer for pull requests.** RepoSentinel embeds a developer's code and
runs it concurrently against two vector collections: **Ghost Hunter** (known CVE
vulnerable-code snippets) and **Team Memory** (the team's own past PR review discussions). It
reranks the matches with a cross-encoder, then asks an LLM to write a structured, actionable
PR comment.

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
   → similarity gate → cross-encoder rerank (code-vs-code)
   → feedback suppression / downweight → top-k
                     └──────────────┬──────────────┘
                                    ▼
                        LLM report (Groq → Gemini)
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
| `RERANK_THRESHOLD` | `0.0` | bge gives near-0.5 sigmoid scores on code pairs; it only provides ordering |
| `TWIN_MARGIN_MIN` | (off) | Drop CVE matches that look at least as much like the stored fix as like the bug; calibrate with `run_eval --margin-sweep` |
| `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` | `groq` / `gemini` | OpenAI-compatible endpoints; primary → fallback |
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
python -m ml.evaluation.run_eval --sim-sweep 0.20:0.70:0.05 --write-baseline
```

The eval quantifies the precision ceiling (~0.5 across thresholds, since safe and vulnerable
near-twins embed alike), which is *why* the gate favors recall and the LLM report is the real
precision filter. **Category hit rate** (did we retrieve the right CVE?) is the meaningful
retrieval metric; the calibrated baseline lives in `ml/evaluation/baseline.json`.

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
  `LLM_PROVIDER=mock`.
- **Local Qdrant is single-process.** The API, ingest scripts, and eval can never run at the
  same time; scripts print "stop the API first" on a lock error.
- **Any embedding-model change requires `--recreate` on both collections.** Payloads carry
  `embedding_model` so drift is detectable.
