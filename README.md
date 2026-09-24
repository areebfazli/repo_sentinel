# RepoSentinel

**An AI security reviewer for pull requests.** An LLM reviews each changed function for
vulnerabilities on its own merits, with evidence attached: matches from two vector
collections, **Ghost Hunter** (known CVE vulnerable-code snippets, with how each was fixed) and
**Team Memory** (the team's own past PR review discussions), ranked by similarity (an optional
cross-encoder rerank is available). Every finding must quote the offending code, and the
structured result is rendered into an actionable PR comment.

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
     LLM review of every unit (OpenRouter → Groq), CVE/team matches as reference
       untrusted text in nonce-tagged blocks; findings must quote the code
                                    │
                                    ▼
          deterministic server-side Markdown (all LLM text escaped)

GET /api/v1/analyze/{job_id}  ◀── clients poll for the result
```

**Key design points**

- **Async jobs.** `POST /api/v1/analyze/` returns `202 + job_id` immediately and runs the work
  in a background task; clients poll `GET /api/v1/analyze/{job_id}`. Findings are persisted so
  the feedback loop works regardless of who's polling.
- **The LLM reviews the code; retrieval is reference context.** Embedding similarity can't
  separate a safe snippet from its vulnerable twin (they embed alike) and picks the right bug
  class only ~28% of the time on named categories, so retrieved CVEs are shown to the LLM as
  "similar known vulnerabilities, may or may not apply" (with the fix diff where stored), not
  as the only findings it may report. The old retrieval-only prompt found 1 of 14 vulnerable
  functions end to end (ROADMAP, Findings 2026-09-24).
- **Anti-hallucination.** Each finding must quote the offending line(s); a finding whose
  quote isn't in the reviewed code is dropped server-side, and its line anchors the PR
  comment. The check is whitespace-insensitive (a multi-line statement quoted on one line, a
  list of lines, or text as it appeared after sanitising all match) but every quoted
  character sequence must be in the code. A cited CVE / team-PR id outside the retrieved allowlist is removed from the finding
  (the finding stays).
- **Prompt-injection hardening.** PR code, file names, corpus code, advisory text and team
  comments are untrusted: each goes into a `<untrusted_<nonce>>` block with a per-prompt random
  nonce, after HTML-comment openers are defused in place (never stripped: stripping hid code
  between `# <!--` and `# -->`), zero-width / bidi characters are made visible, and backtick
  fences and tag look-alikes are defused. Nothing in the code under review is deleted and
  line numbers stay 1:1 with the file; the system prompt says
  instructions inside those blocks are data. On the way out, every LLM-written field is
  escaped when rendered (server report and the Action's inline comments), so a finding can't
  inject links, images, HTML, @-mentions or `<!-- reposentinel:... -->` markers
  (`backend/app/core/untrusted.py`).
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

Settings come from `.env` via Pydantic (`backend/app/config.py`). Defaults, including LLM
routing and model choice, live in code, so `.env` only needs secrets (API keys); any setting
can still be overridden there. Highlights:

| Setting | Default | Notes |
|---------|---------|-------|
| `ENVIRONMENT` | `development` | `development` = file Qdrant + SQLite (Docker-free); `production` = networked Qdrant + Postgres |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Mean-pooled, 768-dim; needs `EMBEDDING_TRUST_REMOTE_CODE=True` |
| `SIM_THRESHOLD_CVE` | `0.25` | Similarity gate, tuned for high recall |
| `RERANKER_ENABLED` | `False` | Cross-encoder rerank stage. Off: no measurable gain on the held-out OSV eval (category hit 0.373 off vs 0.36–0.41 on, at 7–49 s/item) and it costs ~3 GB RAM; when off it is never loaded and matches keep similarity order |
| `RERANKER_MODEL` / `RERANKER_MAX_TOKENS` | `BAAI/bge-reranker-v2-m3` / `512` | Used only when enabled; 512 was the best-measured length and half the cost of 1024 |
| `RERANK_THRESHOLD` | `0.0` | Gate on `sigmoid(logit)`; only applies with `RERANKER_ENABLED`. Keep-all: rerank probability didn't separate vulnerable from fixed at any threshold |
| `TWIN_MARGIN_MIN` | (off) | Drop CVE matches that look at least as much like the stored fix as like the bug; calibrate with `run_eval --margin-sweep` |
| `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` | `openrouter` / `groq` | `groq` \| `gemini` \| `openrouter` \| `mock`; OpenAI-compatible endpoints. Default chain: `openrouter:qwen/qwen3.8-27b:free` → `openrouter:google/gemma-4-31b-it:free` → `groq:openai/gpt-oss-120b` → `groq:qwen/qwen3.8-27b` (each provider, primary or fallback, tries its own fallback model after its default one). OpenRouter's free models share an upstream pool and often 429 (`upstream_provider_shared_pool`), so the Groq fallback is what keeps reports flowing |
| `GROQ_MODEL` / `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-120b` / `qwen/qwen3.8-27b` | Groq default model → Groq fallback model (Groq rate-limits per model); set `GROQ_FALLBACK_MODEL=` to disable the second. The job result's `llm_provider_used` names the model that answered, e.g. `groq:qwen/qwen3.8-27b` |
| `OPENROUTER_MODEL` / `OPENROUTER_FALLBACK_MODEL` | `qwen/qwen3.8-27b:free` / `google/gemma-4-31b-it:free` | OpenRouter free models (20 req/min, 1,000 req/day with ≥ $10 credits, fewer without; they need "allow free endpoints that may train on inputs" in OpenRouter's privacy settings or return 404; see [openrouter.ai/docs](https://openrouter.ai/docs)). The qwen primary is always sent without `response_format` (it rejects it); set `OPENROUTER_FALLBACK_MODEL=` to disable the gemma fallback. OpenRouter default model → OpenRouter fallback model → `LLM_FALLBACK_PROVIDER`. 429 is retried; 402 (insufficient credits) is not, and skips OpenRouter's other models; a model that rejects `response_format` is retried once without it |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | `/chat/completions` is appended |
| `OPENROUTER_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` | (none) | The only LLM settings `.env` needs. A missing key for `LLM_PROVIDER` **hard-fails at startup**; a missing key for `LLM_FALLBACK_PROVIDER` just logs a warning and that provider is skipped |
| `LLM_MAX_PROMPT_TOKENS` / `LLM_MAX_UNITS_PER_PROMPT` / `LLM_MAX_CALLS_PER_SCAN` | `6000` / `6` / `6` | Per-scan LLM budget. Units are ordered by evidence (guard_diff alert, Semgrep hit, `guard_removed`, then retrieval similarity) and packed into as few prompts as fit (one call when everything fits). Tokens are estimated as chars / 4 × 1.25; 6000 keeps a prompt plus a reasoning model's answer under Groq's free-tier 8K tokens/min. An oversized unit loses its references, then its code is elided around its changed lines (head and tail without a diff) with explicit `[... N line(s) omitted ...]` markers and real line numbers; units beyond the call cap are listed in `units_not_reviewed` and in the report, never dropped silently |
| `LLM_TPM_LIMITS` / `LLM_OUTPUT_TOKENS_ESTIMATE` | `{"groq": 8000, "openrouter": null}` / `1500` | Per-model tokens per rolling minute (key: provider or `provider:model`). Each call reserves its estimated prompt tokens plus the output allowance and waits for room, so back-to-back calls stay under Groq's free-tier 8K tokens/min |
| `LLM_MAX_WAIT_S` / `LLM_SCAN_MAX_WALL_S` | `60` / `480` | Longest single wait (rate budget or `Retry-After`, which is honoured in full) before trying the next client; wall-time budget of a scan's LLM stage (units that can't be sent in time: `units_not_reviewed`, reason `time_budget`) |
| `GUARD_ALERT_SEVERITY` | `medium` | Severity of deterministic guard_diff alert findings (evidence, not a verdict) |
| `LLM_MAX_CVES_PER_UNIT` | `2` | Retrieved CVE matches shown per unit (team matches likewise) |
| `SEMGREP_ENABLED` / `SEMGREP_MIN_SEVERITY` | `True` / `high` | Static-analysis evidence for the review (see below); a missing engine only logs a warning |
| `REPOSENTINEL_API_KEY` | (none) | `X-RepoSentinel-Key` header, optional in dev, required in production |
| `HYBRID_ENABLED` | `False` | Dense + sparse (BM25) RRF fusion; off by default (measured F1 gain below the adoption bar) |

See `.env.example` for the full list.

---

## GitHub Action

`.github/workflows/repo_sentinel.yml` runs `github_action/scan_pr.py` on pull requests. It
collects the PR's changed files, POSTs them in files mode, and posts **inline review comments**
anchored to the finding's quoted line when it is in the diff (added or context line), else the
nearest changed line. Comments are deduped across pushes via hidden
`<!-- reposentinel:f:<sha1> -->` markers (hash of file, function and the finding's
`dedupe_key`; only a marker at the very end of a comment counts). An LLM finding's
`dedupe_key` is its source plus the code line it is anchored to (whitespace-insensitive), never
LLM wording, so CWE / title drift between runs updates the same comment. Findings also carry
`legacy_dedupe_keys` (the previous key formula): on the first run after upgrading, a comment
posted under an old marker is adopted and updated in place instead of deleted and re-posted
(one whose CWE / title had already drifted can't be matched and is replaced once). It also
posts a summary comment and applies a configurable severity gate. Finding text is escaped
before it is posted.

The job result also carries the review's inputs and coverage: `static_analysis` (Semgrep
evidence), `guard_diff` (units whose diff removed or added a guard), `llm_calls`,
`review_status` (`complete` | `partial` | `failed`) with `units_total` / `units_reviewed` /
`units_partially_reviewed`, and `units_not_reviewed` (units left out by the token budget, too
large for a prompt, the LLM time budget, or a failed LLM call; the report lists them too). Only
a complete review is reported as clean; a partial one leads with "Partial review: N of M
unit(s) not reviewed". If every LLM call fails the scan still completes (`review_status:
failed`) with its Semgrep and guard_diff results.

Configure two repository secrets:

- `REPOSENTINEL_URL`: where your API is reachable
- `REPOSENTINEL_API_KEY`: matches the backend's `REPOSENTINEL_API_KEY`

Gates (workflow `env`):

- `INPUT_FAIL_ON_SEVERITY` (default `high`): fail on a finding at/above this severity.
- `INPUT_FAIL_ON_PARTIAL` (default `true`): fail when the review was `partial` or `failed`
  (older servers without `review_status`: partial when `units_not_reviewed` is non-empty).
- `INPUT_GATE_ON_DETERMINISTIC` (default `false`): let a deterministic-only finding (guard_diff
  alert) fail the severity gate on its own; by default it needs the LLM review or Semgrep to
  flag the same spot (`corroborated_by`).

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

**LLM stage (`--llm`).** This measures the reviewer end to end. `--llm-prompt` picks the arm:

- `current` (default): the production review prompt, built exactly as a snippet scan builds
  it: the item's top retrieved CVEs (capped at `LLM_MAX_CVES_PER_UNIT`, with fix diffs), the
  Semgrep evidence from one engine run over all items as snippets, the prompt budget, and a
  deterministic per-item nonce (so prompts and cache keys are reproducible). guard_diff is
  not applicable (a snippet has no previous version). Every item gets a call. An item counts
  as vulnerable when a finding survives validation (its quote is in the code), which is what
  the API's `is_vulnerable` means; `realistic_cve_finding` also counts findings that cite a
  shown CVE.
- `no_retrieval`: the same prompt and Semgrep evidence with zero CVEs.
- `legacy`: the old retrieval-only prompt (`ml/evaluation/legacy_prompt.py`). Items without
  retrieved CVEs get no call, and an item counts as vulnerable when a validated finding
  references a retrieved CVE (`realistic_any_finding`: any validated finding).

Realistic metrics also report the false-positive rate on length-matched ordinary functions
(`fpr_ordinary_length_matched`). The run makes real provider calls, so it has several limits:
- `--llm-max-calls N` is required and capped at 200.
- Calls are paced by `--llm-sleep` (2.5 s) and `--llm-tpm` (8000 tokens/min).
- `--llm-token-budget T` sets a hard token stop.
- The run stops cleanly on a daily-limit error or on repeated rate limits.
- Results are cached in `ml/evaluation/results/llm_cache.jsonl` by (item id, prompt
  sha256, model, temperature, repeat index), so re-running the same command resumes without
  re-calling. Entries written before temperature and repeat were recorded count as 0.2 and
  repeat 0.
- `--llm-primary-only` keeps every answer on one model. Compare arms on the same model.
  `--llm-model PROVIDER:MODEL` (with `--llm-primary-only`) pins exactly that model, whatever
  the provider settings say, and needs only that provider's key. On OpenRouter a primary-only
  run sends `provider: {"allow_fallbacks": false}`, and `--llm-upstream SLUG` also pins the
  upstream (`order: [SLUG]`). Each item records the upstream `provider` that OpenRouter
  reports, so a silent upstream change shows up.
- `--llm-temperature` defaults to 0.0 in the eval. Production uses `LLM_TEMPERATURE` (0.2).
- `--llm-repeat K` runs every item K times, each repeat as its own cached call. It reports
  the flip rate (the share of items whose prediction changes between repeats, with a Wilson
  CI) and per-repeat rates. Metrics come from repeat 0.
- `--split PATH --split-name dev|test` evaluates only the ids in a split manifest
  (`{"version": 1, "seed", "dev": {"ids"}, "test": {"ids"}, "meta"}`), before any sampling.
  The test split is refused unless `--i-know-this-is-the-test-set` is passed. The output JSON
  records the split, the manifest's sha256 and any ids that aren't in the datasets.

**Localised scoring (the primary LLM metric).** "Any validated finding" counts a vulnerable
function as detected even when the finding is about another line or another bug. For a
`_vuln` item whose `_safe` twin is loaded, the eval computes the **fix lines**: the lines the
fix deleted or modified (difflib over stripped lines, so re-indenting doesn't count). A hunk
that only inserts lines contributes the insertion point ±2 lines, and blank-only hunks are
ignored. A vulnerable item is a *localised TP* when a validated finding's line range (or its
quoted code, located in the item) is within `--localise-tolerance` (2) lines of a fix line.
It also counts when the finding's CWE is one of the item's expected CWEs (a `cwe`/`cwe_ids`
field; no current eval set has one, so today only line overlap counts). The false-positive
rate on fixed twins and ordinary code stays "any validated finding". The eval also reports
`fpr_fixed_twin_localised`: twins flagged on the lines the fix added or changed. The
`headline` block puts localised TPR first, next to the any-finding TPR. The legacy prompt's
findings have no quote or line, so that arm gets no localised numbers.

Offline tools (they read saved results; no model load and no LLM call):
- `--rescore RESULT.json [--out NEW.json]` recomputes every metric. Per-item records now
  store their findings (quote, line, CWE), fix lines and localised outcome. Older results are
  backfilled from the raw responses in `--llm-cache` and the item code in `--dataset`
  (pre-migration ids are mapped when unambiguous). The output lists what couldn't be
  recomputed and why.
- `--compare A.json B.json` pairs two runs by item id. It prints exact McNemar tests on
  vulnerable localised TP and fixed-twin FP, with the discordant counts, plus exact
  (Clopper-Pearson) CIs on ordinary FPR.

Measured 2026-09-24 on one model (`groq:qwen/qwen3.8-27b`, 8 vuln/fixed pairs + 13 ordinary
functions): the new prompt found 2/8 vulnerable functions vs 1/8 for the legacy prompt, flagged
1/8 fixed twins (legacy 2/8) and 1/13 ordinary functions (legacy 0/13); the `no_retrieval` arm
flagged exactly the same items as `current` with 44% fewer tokens. Rescored offline with
localised scoring, the new prompt found **1/8**: the other "detection" flagged the SSRF host
check on lines 11–12, while the fix changed an XSS on line 19, and it flagged the fixed twin
the same way. No fixed twin was flagged on its fix lines (0/8). All differences are within the
confidence intervals; see ROADMAP "Results 2026-09-24".

Results go to `--out` only, never to the baseline:

```bash
python -m ml.evaluation.run_eval --no-rerank --sample-kinds vulnerable=14,fixed_twin=14,ordinary=22 \
    --seed 42 --llm --llm-primary-only --llm-max-calls 60 --llm-token-budget 150000 \
    --out ml/evaluation/results/llm_run.json \
    --dataset ml/evaluation/datasets/detection_eval_osv_pypi.jsonl \
    ml/evaluation/datasets/detection_eval_osv_npm.jsonl ml/evaluation/datasets/detection_eval_ordinary.jsonl
```

---

## Static-analysis evidence (Semgrep)

`backend/app/core/semgrep_scanner.py` runs Semgrep CE (`semgrep` in `requirements.txt`,
LGPL-2.1 engine; Opengrep's binary works too) with a vendored, permissively licensed rule set
(`backend/app/rules/semgrep/`, GitLab `sast-rules`: MIT / Apache-2.0 / LGPL-3.0 — see its
README for why the Semgrep Registry rules are not used). `SemgrepScanner().scan_units(units,
sources=None)` scans all units in one engine run and returns `{(file_path, function_name,
start_line): [hit, ...]}`, each hit `rule_id, message, severity, cwe, line, end_line, snippet`.
Pass `sources={file_path: full_text}` when you have the files: they are scanned whole (imports
in context) and hits are assigned to units by line. Engine missing / timeout / crash logs a
warning and returns `{}`. It is blocking; call it with `asyncio.to_thread`.

**In scans** (`backend/app/core/evidence.py`): every scan runs it in a worker thread alongside
embedding and retrieval — files mode over the full file text, snippet mode over the snippet
(language guessed when the client sends none). Hits at or above `SEMGREP_MIN_SEVERITY`
(default `high`) not in `SEMGREP_EXCLUDED_RULES` (Bandit's `assert`, Python `random`,
requests-without-timeout) go into the prompt as *static-analysis evidence* (rule id, CWE,
line; regex / ReDoS rules marked low-confidence) and into the result's `static_analysis`. They
are evidence for the LLM, never findings on their own. `SEMGREP_ENABLED=false` turns it off;
`SEMGREP_TIMEOUT_S` (60) bounds the run.

Measured on the eval sets (snippet mode, no imports), a hit is weak evidence, not a gate: with
Bandit's `assert` rule excluded, 4.3% of vulnerable functions vs 2.2% of their fixed twins and
1.7% of ordinary functions have a hit (high/critical severity only: 3.2% / 1.4% / 0.5%). The
engine costs ~7 s fixed per run (Python rules; ~13 s with JS as well), then ~0.05 s per unit.

## Diff-direction evidence (did the PR remove a guard?)

`backend/app/core/guard_diff.py` compares a unit's old and new code (tree-sitter, Python + JS,
no model, ~15 ms per unit). It reports `GuardChange`s: removed/added sanitisers, auth checks,
path-containment and bounds checks, `raise`/`return`/`throw` guard blocks, unsafe-API swaps
(`yaml.safe_load` to `yaml.load`), flag flips (`shell=True`, `verify=False`) and parameterised
SQL turned into interpolated SQL. It nets them into `risk` (`guard_removed` / `guard_added` /
`none`) plus a high-precision `alert` tier (swap/flag/SQL evidence only). Features are counted
per kind with identifier-anonymised keys, so renames, reformatting and moved statements don't
count. The vocabularies are module-level tables. Files mode already has what it needs:
`old_code_for_units` reverse-applies the Action's `patch` to `content` to get the old file.

**In scans** (files mode only; a snippet has no previous version): units are planned with
`patch_touched_lines(patch)` (deletion points count as changes, so a function whose only
change is a deleted guard is analysed), each unit's old code is rebuilt, and units with
`risk: guard_removed` get their removed / weakened changes in the prompt as *change-direction
evidence*. The `alert` tier is also reported as its own deterministic finding (`source:
"guard_diff"`, severity `GUARD_ALERT_SEVERITY` (default medium), rendered under "Removed
security guards (deterministic check, no LLM)"), whatever the LLM says, with `corroborated_by`
listing `llm` / `semgrep` when either flags the same spot (within 3 lines); only a corroborated
one fails the Action's severity gate by default. Every unit with a signal is listed in the result's `guard_diff`.

`python -m ml.evaluation.eval_guard_diff` treats real fix commits as benign changes and their
reversal as vulnerability-introducing PRs. On 506 held-out pairs: TPR 0.227 and fix-direction
FPR 0.030. The `alert` tier has TPR 0.032 and FPR 0/506. Synthetic benign edits of 994 ordinary
functions produce 0 flags. Results are written to `ml/evaluation/results/guard_diff_eval.json`.

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

- **The LLM is OpenRouter/Groq/Gemini, never a silent mock.** Mock output only happens with an
  explicit `LLM_PROVIDER=mock`. The default chain is OpenRouter (`qwen/qwen3.8-27b:free`, then
  `google/gemma-4-31b-it:free`) → Groq (`openai/gpt-oss-120b`, then `qwen/qwen3.8-27b`). OpenRouter's
  free models share an upstream pool and often 429 (`upstream_provider_shared_pool`), so set
  `GROQ_API_KEY` too; a retired model id (HTTP 404) is logged as a "not found or decommissioned"
  warning naming the model and falls through to the next one. For OpenRouter an error object
  inside an HTTP 200 is treated as a failed call (not a crash), and replies wrapped in
  ```` ```json ```` fences or preceded by reasoning text are still parsed.
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
  `vulnerable`, one `safe`), in the same format as `detection_eval.jsonl`. Ids are
  `<advisory>_<function>_<hash8>_{vuln,safe}` (the hash is of the pair's code, so ids are unique;
  an exact duplicate pair is written once). `scripts/migrate_eval_ids.py` applies that scheme
  to files built before it, offline.
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

### Dev/test split and static-first candidate recall

`ml/evaluation/splits/v1.json` (built by `scripts/build_eval_split.py`, offline, seed 42) is
the fixed dev/test split. Dev has 151 pairs + 498 ordinary functions; test has 100 pairs + 500
ordinary functions. Items are grouped by advisory (including OSV aliases), repository and
near-duplicate code, so no advisory, repo or near-duplicate function straddles the two splits.
Pairs are stratified by the year the fix became public and by language. See
`ml/evaluation/splits/README.md` for strata, leakage checks and why 500 ordinary functions
can't certify an FPR ≤ 0.5%.

`python -m ml.evaluation.analyze_candidates` (about 10 s, offline) asks: if only Semgrep hits
and guard_diff `guard_removed` changes reached the LLM, what share of vulnerabilities would it
ever see? It reuses the per-item rows in `results/semgrep_eval.json` and recomputes guard_diff,
then writes `results/candidate_recall.json`.
