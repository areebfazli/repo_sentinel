# RepoSentinel: technical details

The [README](../README.md) explains what RepoSentinel does, its key numbers and how to run it.
This file holds the rest: architecture, configuration, the evidence stages, the PR-level
review, the eval harnesses and how the datasets are built. Paths are relative to the project
root. Plans and dated results live in [ROADMAP.md](../ROADMAP.md).

## Contents

- [Why two collections?](#why-two-collections)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [GitHub Action](#github-action)
- [Tests & lint](#tests--lint)
- [Detection eval harness](#detection-eval-harness)
- [Static-analysis evidence (Semgrep)](#static-analysis-evidence-semgrep)
- [Diff-direction evidence (did the PR remove a guard?)](#diff-direction-evidence-did-the-pr-remove-a-guard)
- [PR-level review (files mode)](#pr-level-review-files-mode)
  - [PR-level eval](#pr-level-eval)
- [Project layout](#project-layout)
- [Notes & gotchas](#notes--gotchas)
- [Growing the corpus from real CVE fixes](#growing-the-corpus-from-real-cve-fixes)
  - [Ordinary-function negatives](#ordinary-function-negatives)
  - [Dev/test split and static-first candidate recall](#devtest-split-and-static-first-candidate-recall)
  - [PR-shaped eval (`pr_eval_v1`)](#pr-shaped-eval-pr_eval_v1)
  - [PR-shaped eval v2 (real benign commits)](#pr-shaped-eval-v2-real-benign-commits)

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
  files mode (REVIEW_MODE=pr): PR-level audit of the whole change (diff + changed
    functions before/after + leads), context rounds, then one verifier call per candidate
  snippet mode / REVIEW_MODE=units: LLM review of every unit, CVE/team matches as reference
       (OpenRouter → Groq; untrusted text in nonce-tagged blocks; findings must quote the code)
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
- **PR-level review in files mode.** The model is asked what the *change* newly introduces,
  over the unified diff and each changed function before and after, with leads (guard_diff,
  permissive Semgrep, sensitive sinks on added lines) to focus attention; it may ask for code
  defined elsewhere in the PR (the option exists, but the model made no such request in the
  160 scored PRs of the dev run), and every candidate is re-judged by a separate verifier call
  before it is reported. See [PR-level review](#pr-level-review-files-mode).

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
| `LLM_MAX_OUTPUT_TOKENS` / `LLM_SAMPLING` / `LLM_TEMPERATURE` | `16000` / qwen/qwen3.8-27b: `1.0` / `0.95` / top_k `20` / `0.2` | `max_tokens` sent with every call (reasoning counts against it; a client with a tokens/min limit gets at most what the prompt leaves of it, under 512 skips the client; output cut at the cap is a failed call, never an empty review). Per-model temperature / top_p / top_k (code default, `config.py`; top_k only to OpenRouter); `LLM_TEMPERATURE` for models without an entry |
| `LLM_TPM_LIMITS` / `LLM_OUTPUT_TOKENS_ESTIMATE` | `{"groq": 8000, "openrouter": null}` / `1500` | Per-model tokens per rolling minute (key: provider or `provider:model`). Each call reserves its estimated prompt tokens plus the output allowance and waits for room, so back-to-back calls stay under Groq's free-tier 8K tokens/min |
| `LLM_MAX_WAIT_S` / `LLM_SCAN_MAX_WALL_S` | `60` / `480` | Longest single wait (rate budget or `Retry-After`, which is honoured in full) before trying the next client; wall-time budget of a scan's LLM stage (units that can't be sent in time: `units_not_reviewed`, reason `time_budget`) |
| `GUARD_ALERT_SEVERITY` | `medium` | Severity of deterministic guard_diff alert findings (evidence, not a verdict) |
| `LLM_MAX_CVES_PER_UNIT` | `2` | Retrieved CVE matches shown per unit (team matches likewise); `0` also skips retrieval (and its embedding cost) entirely |
| `SEMGREP_ENABLED` / `SEMGREP_MIN_SEVERITY` | `True` / `high` | Static-analysis evidence for the review (see below); a missing engine only logs a warning |
| `REVIEW_MODE` | `pr` | Files mode: `pr` = PR-level audit + verification (below); `units` = the per-function review (kept for comparison). Snippet mode always uses the per-unit review |
| `PR_REVIEW_MAX_PROMPT_TOKENS` / `PR_REVIEW_MAX_AUDIT_CALLS` / `PR_REVIEW_CONTEXT_ROUNDS` / `PR_REVIEW_CONTEXT_MAX_TOKENS` | `12000` / `4` / `2` / `1500` | Audit budget: prompt size (estimated tokens, incl. the context reserve; also the verifier prompt's size; a prompt too big for Groq's 8K tokens/min skips the Groq clients, so with Groq as the only provider set `6000`, see [the Groq limitation](#notes--gotchas)), audit calls per scan (context rounds included; a PR too big for one prompt is split by file), extra rounds answering the model's context requests, tokens of requested context per prompt |
| `PR_REVIEW_MAX_VERIFIER_CALLS` / `PR_REVIEW_MIN_CONFIDENCE` / `PR_REVIEW_MIN_AUDIT_CONFIDENCE` | `8` / `7` / `5` | One verifier call per candidate, at most this many; a finding is reported only when confirmed with confidence ≥ 7/10; candidates the audit itself rates below 5 are not verified. An unverified candidate is not reported and makes the review `partial` |
| `PR_REVIEW_MAX_SUGGESTIONS` / `PR_REVIEW_SUGGEST_MIN_CONFIDENCE` | `5` / `4` | The non-blocking "worth a look" tier (below): at most this many per scan (`0` = off), verifier confidence needed. Never findings, never gating |
| `PR_REVIEW_SEMGREP_LEAD_MIN_SEVERITY` / `PR_REVIEW_SINK_LEADS` / `PR_REVIEW_HARD_EXCLUSIONS` | `low` / `True` / `True` | Semgrep hits shown as leads (marked "lead only" below `SEMGREP_MIN_SEVERITY`), sensitive-sink leads on added lines, regex hard exclusions before verification |
| `PR_REVIEW_FIX_EXAMPLES` / `VERIFIER_MODEL` | `0` / (none) | Retrieved "how a similar bug was fixed" examples in the audit prompt (off); `<provider>:<model>` tried first for verifier calls (the normal chain stays the fallback; each finding records `verifier`) |
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
before it is posted. The PR review's non-blocking "worth a look" items
(`review_suggestions`) appear only in the summary comment (the server's report section plus
a count line): no inline comments, never part of either gate.

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
(`yaml.safe_load` to `yaml.load`), flag flips (`shell=True`, `verify=False`), dropped safe-load
keywords (`Loader=SafeLoader`, `resolve_entities=False`, `weights_only=True` on `torch.load`:
dropping the explicit `True` or writing `False` is weakened, at 0.6, i.e. `guard_removed` but not
the alert tier, as dropping it is harmless from torch 2.6, where it is the default; numpy's
`allow_pickle` only counts as an explicit `True`, its default being safe) and parameterised
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
Adding `weights_only` (re-run on the current 502 held-out / 2865 corpus pairs): `guard_removed`
TPR 114 → 115 / 502 held-out and 686 → 692 / 2865 corpus (all torch.load deserialisation
CVEs), fix-direction FPR unchanged (14 / 502, 72 / 2865), the alert tier unchanged (16 / 502 at
0 / 502; 69 / 2865 at 6 / 2865), broad-commit bystanders 49 → 50 / 2364 (`_load_pyfunc`
gaining a `weights_only=False` parameter default); on the 476 PR-eval items (dev200 +
misleading) only InspireMusic's CVE-2025-5148 changes (intro → `guard_removed`, fix →
`guard_added`), no alert changes.

## PR-level review (files mode)

`backend/app/core/pr_review.py` (default `REVIEW_MODE=pr`) reviews a PR the way a security
reviewer reads one, not function by function:

1. **Bundle** (`core/pr_context.py`): per changed file the unified diff (new-file line
   numbers), and per changed function its version after and before the change (old content is
   rebuilt by reverse-applying the patch; a patch is synthesised when only old + new are
   given). A tree-sitter symbol index over the PR's files (definitions, methods, module-level
   variables, call sites; Python / JS / Go / Java) adds direct callers / callees of the changed
   functions when they fit.
2. **Leads**, which focus attention and are never findings: guard_diff changes in every
   direction, Semgrep hits down to `PR_REVIEW_SEMGREP_LEAD_MIN_SEVERITY` (medium / low marked
   "lead only"), and sensitive sinks on the diff's added lines (exec / eval / subprocess /
   `os.system` / pickle / `yaml.load` / formatted SQL / file paths / redirects / outbound
   requests / `innerHTML` / `dangerouslySetInnerHTML` / `child_process` / `fs` / `new Function`).
3. **Audit**: "identify vulnerabilities NEWLY INTRODUCED by this change", data flow from
   attacker-controlled sources to sinks; each finding gives file, new-file line, CWE, severity,
   `source`, `sink`, `missing_control`, `exploit_scenario`, a verbatim quote and a 1–10
   confidence. The PR title / body are never in the prompt (descriptions framing a change as
   safe can collapse detection) and commit messages are called untrustworthy. Files are packed
   by lead strength into prompts of `PR_REVIEW_MAX_PROMPT_TOKENS`; a file too big alone loses
   neighbouring context, then before-versions, then after-code away from the changes, then
   (partial) diff lines.
4. **Context loop**: instead of answering, the model may send `{"need_context": [{"symbol",
   "file", "want": "definition|callers"}]}`; symbols are resolved from the PR's own files,
   appended and the audit re-asked (≤ `PR_REVIEW_CONTEXT_ROUNDS`, ≤
   `PR_REVIEW_CONTEXT_MAX_TOKENS`, stops early when nothing new resolves). In the 160 scored
   PRs of the dev run the model never used it (0 context requests).
5. **Validation + filters**: the quote must be in the NEW file (searched near the claimed line
   first; diff markers tolerated), then regex hard exclusions (DoS, rate limiting, resource
   leaks, memory safety outside C/C++, docs, tests) and the audit-confidence floor.
6. **Verification**: one fresh-context call per candidate with the file after the change
   (whole, or a window around the finding plus its function), its diff and the claim; it must
   establish source, control, sink, reachable path and counterevidence, with precedents for
   Python / JS web code (React XSS only via `dangerouslySetInnerHTML`, no SSRF / path traversal
   in client-side JS, env vars trusted, ReDoS only with an attacker-controlled pattern, ...).
   Kept only if `confirmed` with confidence ≥ `PR_REVIEW_MIN_CONFIDENCE`.
7. **Worth a look (not blocking)**: the verifier prompt answers `uncertain` when the change
   removes an existing security control but the attack path isn't visible, so such
   regressions were never reported. A candidate becomes a `review_suggestions` item (status
   `review_suggested`) only when the verdict is `uncertain` or `confirmed` below the cutoff,
   the verifier's confidence is ≥ `PR_REVIEW_SUGGEST_MIN_CONFIDENCE` (4), and deterministic
   checks show the change deleted code that looks like a security control at that spot
   (deleted lines, recognised as a control, not found elsewhere in the new code). Either:
   - **guard_diff**: a removed / weakened change of a `guard_removed` unit containing the
     candidate, located inside the candidate's function (the *outermost* function of the new
     file containing the candidate's line, `PRBundle.outermost_function_at`, decorators
     included: a removed admin check in an Express handler counts for the sink in its
     `ids.map(async id => ...)` callback, a view's check for its local `_do()` helper; a
     method stays the method, as its class is not a function; a swap by its new-side line,
     a deleted guard by its old line within the old version of that function, or by its
     deletion point when the function is new), whose old code is in no file's NEW content
     (moved, not removed). The innermost function missed those callbacks and closures.
     A candidate in no function (module-level code) falls back to
     `GUARD_EVIDENCE_FALLBACK_WINDOW` (30) new-file lines around it. An earlier fixed
     10-line window around the candidate removed no false alarm at any width on the dev200
     rescore and lost genuine catches (a removed cross-channel check 29 lines above the
     candidate, a removed `validate_git_ref` 12 lines away, both in the same function); or
   - **the verifier's quote**: its optional `removed_control_quote` is found
     (whitespace-insensitive, diff markers tolerated) in the OLD file within 40 lines of the
     candidate, every non-blank line it matches is a line the patch deletes (not kept or
     context), none of them is comment-only (`#`, `//`, `/* */`, `*`, docstring lines), the
     lines are a control by guard_diff's own classifier (guard calls such as sanitisers,
     auth / permission checks or `compare_digest`, guard blocks, bounds checks, safe API /
     flag / SQL forms; Python / JavaScript only), and the quote is in no file's NEW content.

   Neither kind counts where a guard_diff alert finding already reports the removal: the
   change is itself an alert change, an alert finding is on the same file and line (for the
   quote: a matched line's deletion point), or a quoted line holds an alert change's old code.
   The alert is in the report already; the candidate keeps its verifier status (`uncertain` /
   below the cutoff). Both routes rely on guard_diff's grammar (Python / JavaScript /
   TypeScript): for **Go, Java** and other languages the verifier's quote can never qualify
   (no control classifier) and guard_diff gives no evidence, so the tier never fires there.

   A claim that doesn't check out never qualifies; `rejected` / unverified never do. At most
   `PR_REVIEW_MAX_SUGGESTIONS` (5); candidates past the cap keep their verifier status.
   They get their own "👀 Worth a look (not blocking)" report section (escaped like
   findings; an otherwise clean report keeps its ✅ heading) and are never in
   `report_findings`, `is_vulnerable` or any gate.

Deterministic guard_diff alerts, Semgrep evidence, `review_status` and `units_not_reviewed`
work as in the per-unit review. The result adds `review_mode`, a `pr_review` block (audit /
verifier calls, context rounds and requests, candidates → excluded / rejected / uncertain /
below the cutoff / `review_suggested` / confirmed, which partition the candidates; estimated
prompt tokens), `review_suggestions` (file / line, title, the removed control's old code,
`evidence`: `verifier_quote` and / or `guard_diff`, verdict, confidence, the verifier's reason)
and per-finding `taint_source`, `sink`, `missing_control`,
`exploit_scenario`, `confidence`, `audit_confidence`, `verifier` (all optional; the Action and
dashboard ignore fields they don't know). Report findings are not rows of the `findings`
table (that holds retrieval matches for feedback); they and the review suggestions live in
the scan's `result_json`.

**Answer format checks.** Every audit / verifier call passes its schema check
(`audit_schema_problem`, `verifier_schema_problem`; the per-unit review
`units_schema_problem`) to `LLMRouter.generate(..., validate=...)`. The answer is the LAST
JSON object in the reply that passes it (`extract_json`: a trailing example / note object
or an unclosed brace / quote in reasoning prose no longer hides the answer; nothing nested in
a cut-off object is ever returned); a reply with none is a bad-output failure of that client
and the next model / provider in the chain answers instead, as are output cut at
`max_tokens`, null content and a malformed HTTP 200. If every client that was called ENDS
that way (none ends on a 429 / 5xx / timeout / HTTP error; a 429 retried on the same client
before it answered unusably doesn't count; clients skipped because the prompt doesn't fit
their tokens/min limit don't count) the `LLMError` has `bad_output` set and the
units are `not_reviewed` (reason `bad_output`) or the candidate `unverified`, never
"reviewed, no findings"; any other failure mix is `llm_error`. Each client's final outcome is
in `LLMError.client_outcomes` (`bad_output` / `rate_limit` / `failed`), and the eval's
`classify_llm_error` uses the same rule: `bad_output` as above, `rate_limit` only if some
client's last attempt was a 429 (or its retry after one was skipped for the rate budget),
else `other`; a daily-limit 429 anywhere in the call is always `daily_limit` and stops the
run. An `{"error": {...}}` object inside an HTTP 200 is handled like the HTTP status of its
integer or numeric-string `code` (429 / 5xx a same-client retry, 402 skips the provider, any
other 4xx a non-retriable error, i.e. `llm_error`); one without a usable code is bad output.
The checks also run after the call
(defence in depth). `review_pr(files, router, ...)` runs the whole stage
on plain `{path, old_content, new_content, patch}` dicts without the API or DB (the eval's
entry point).

**Cost per PR** (estimated tokens = chars / 4 × 1.25, plus `LLM_OUTPUT_TOKENS_ESTIMATE` = 1500
per call): a typical PR (1–5 files, one audit prompt) makes 1 audit call, +0–2 context rounds,
+1 verifier call per candidate (usually 0–3): **~1–6 calls, ~5K–40K tokens**. The ceiling is
`PR_REVIEW_MAX_AUDIT_CALLS` + `PR_REVIEW_MAX_VERIFIER_CALLS` = 12 calls (~90K tokens). On
Groq's free tier (8K tokens/min per model) each call takes about a minute of one model's
budget, so a 6-call review needs a few minutes; `LLM_SCAN_MAX_WALL_S` (480 s) bounds it and
anything cut is reported (`time_budget`). Groq only serves prompts up to ~7.5K estimated
tokens (see the Groq limitation under [Notes & gotchas](#notes--gotchas)): with the 12K default
the larger audit / verifier prompts run on OpenRouter only. The per-unit review makes at most
`LLM_MAX_CALLS_PER_SCAN` = 6 calls.

Adapted text: the audit prompt, the verifier's exclusions / precedents and the hard-exclusion
regexes come from [anthropics/claude-code-security-review](https://github.com/anthropics/claude-code-security-review)
(MIT); the data-flow method and verification steps from
[openai/codex-security](https://github.com/openai/codex-security) (Apache-2.0). Both are
modified; notices and licence texts: `backend/app/core/prompts/THIRD_PARTY_NOTICES.md`.

### PR-level eval

`ml/evaluation/run_pr_eval.py` measures the PR review against the per-unit review on PR-shaped items
(`ml/evaluation/datasets/pr_eval/`, built by `scripts/build_pr_eval.py`; the default dataset
is the 200-item dev sample: 60 vulnerability-introducing PRs (reversed fixes), their 60 real
fixes and 80 bystander benign PRs). Retrieval is off in every arm; no embedder or Qdrant.

- `--arm pr`: `review_pr` on the item's files. One run gives three headlines: `verified`
  (the report: confirmed findings + deterministic guard alerts), `verified_plus_review` (the
  report plus the non-blocking worth-a-look items) and `audit_only` (every candidate that
  reached the verifier, i.e. before verification). The output also says what the tier adds
  (catches gained, false alarms added) and has policy-curve rows "confirmed ≥ 8 / ≥ 7 + worth
  a look". `--rescore` of a run recorded before the tier replays `verified_plus_review` from
  the cached verdicts with **guard_diff evidence only** (recomputed from the dataset; the old
  verifier prompt had no `removed_control_quote`) at the run's own cutoff, and labels it so;
  without the dataset the view is reported n/a. Rescore of `pr_eval_pr_dev200.json` (old
  prompt, cutoff 8, guard_diff evidence only, limited to the candidate's function, with
  `weights_only` in guard_diff's vocabulary): introducing strict 16/55 (verified 12/55),
  function-level 21/55 (16/55); fix PRs flagged 3/57 (2/57); benign 0/48 (0/48; only 48 of 80
  benign PRs scored). The function scope alone gave strict 15/55, function-level 20/55
  (open-webui back; the 10-line window had 19/55); `weights_only` adds InspireMusic
  CVE-2025-5148 (a dropped `weights_only=True`) at both levels.
- `--arm units`: the per-unit review (`REVIEW_MODE=units`), run in-process exactly as
  `scan_runner` runs files mode, on the same PRs.
- `--arm pr_misleading`: the `pr` arm on the `_misleading` variants of the selected
  introducing PRs (a "harmless refactor" title / body, passed to `review_pr`, which must ignore
  it). Nonces come from the base id, so the prompts are byte-identical to the `pr` arm's when
  the text is ignored (the dry run checks this offline).

Scoring (tolerance `--localise-tolerance`, 2): an introducing PR is a **localised TP** when a
kept finding is in `target.path` within 2 lines of a `vuln_lines_new` line (also reported:
any finding, right file, any vulnerable path); a fix PR is an FP when anything is reported
(also: on the fix's own changed lines); a benign PR is an FP (alert) when anything is
reported. The output adds pair outcomes (introducing vs its fix), precision at 1 / 2 / 5 %
base rates from the benign FPR, Wilson CIs, calls / tokens / latency per PR (mean, p90), the
candidate funnel, context rounds (symbols requested vs resolved), verifier outcomes (a
heuristic bucketing of the verifier's reasons), `review_status` and per category / language.

Calls go through a gate: a JSONL cache (`ml/evaluation/results/pr_llm_cache.jsonl`) keyed by
(item id, call prompt sha256, model, temperature, repeat) so a re-run resumes (the
pipeline's per-call format check is not part of the key, so older entries replay; a cached
answer it rejects is a miss, re-asked within the call budget, and replaced); pacing
(`--llm-sleep`, `--llm-tpm`); `--llm-max-calls` / `--llm-token-budget`; a clean stop on a
daily limit or repeated rate limits. Items with an unfinished or failed call are `not_run` /
`error` and excluded (re-run to resume). That includes a call whose answer fails the format
check on every model, or is cut off / empty (`error_kind` `bad_output`): the item stays
`error` and is retried on the next run (a failed call is never cached), rather than being
recorded as a scored item with a `partial` review; any partial result it has only enters the
sensitivity view and the bounds. The wall-clock budget is disabled in the eval.
`--llm-model` / `--llm-upstream` / `--verifier-model` pin models; `--llm-temperature T`
forces a temperature (default: each model's `LLM_SAMPLING`, as production sends; the cache
key carries the sampling sent, so `--llm-temperature 0` replays temperature-0 entries); `--split dev|test` (test needs `--i-know-this-is-the-test-set`);
`--sample-kinds vulnerable=N,fix=N,benign=N --seed S` keeps pairs together. A selection
holding any test-split item (e.g. `pr_eval_v2_test.jsonl`) is refused without `--split test
--i-know-this-is-the-test-set`. Benign FPR is also reported per provenance
(`benign_by_source`: `real_commit` vs `bystander`, see the v2 dataset below).

Semgrep leads are precomputed once (one engine run over every file of the selection, all
severities) and served from the cache by a stub scanner; without `--semgrep-cache` there are
no Semgrep leads.

The dry run's wall-time estimate has two pacing profiles. `groq_free_8k_tpm` paces only the
calls Groq can serve (estimated prompt ≤ 8000 − 512 tokens) and reports the rest as
`not_servable_calls` / `items_with_unservable_calls`: with the 12K prompt default many PR
prompts can't run on Groq at all (see the Groq limitation).

```bash
# offline: every prompt with a stub model, calls / tokens / time estimates (~1 min, 4 workers)
python -m ml.evaluation.run_pr_eval --dry-run --arm pr      # also: --arm units / pr_misleading
# Semgrep leads for the selection (one engine run; see ROADMAP for time / RAM)
python -m ml.evaluation.run_pr_eval --semgrep-precompute \
    --semgrep-cache ml/evaluation/results/pr_semgrep_dev200.json
# live runs (real provider calls), then offline comparison
python -m ml.evaluation.run_pr_eval --arm pr --llm-model openrouter:qwen/qwen3.8-27b:free \
    --llm-tpm 0 --llm-sleep 3 --llm-max-calls 950 \
    --semgrep-cache ml/evaluation/results/pr_semgrep_dev200.json \
    --out ml/evaluation/results/pr_eval_pr_dev200.json
python -m ml.evaluation.run_pr_eval --compare ml/evaluation/results/pr_eval_units_dev200.json \
    ml/evaluation/results/pr_eval_pr_dev200.json                # exact McNemar, paired by id
python -m ml.evaluation.run_pr_eval --compare A.json A.json --view-a audit_only --view-b verified
python -m ml.evaluation.run_pr_eval --rescore A.json --localise-tolerance 5 --out A5.json
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
ml/evaluation/    detection eval harness + baseline; PR-level eval (run_pr_eval.py)
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
  inside an HTTP 200 is treated like the HTTP status of its code (not a crash), and replies wrapped in
  ```` ```json ```` fences or preceded by reasoning text are still parsed. An answer that
  fails the call's format check (e.g. a bare finding instead of `{"findings": [...]}`) falls
  through to the next model too.
- **Groq's free tier only serves small prompts.** Groq allows ~8K tokens/min per model and
  counts `max_tokens` against it, so the router caps a Groq call's output at what the prompt
  leaves of the 8K and skips a Groq client whose prompt leaves under 512 tokens: prompts over
  ~7.5K estimated tokens never run on Groq, and those that do may get little room to answer
  (`openai/gpt-oss-120b` as little as ~1.7K output tokens for a 6K units prompt). With the
  12K `PR_REVIEW_MAX_PROMPT_TOKENS` default, OpenRouter serves the big PR-review prompts and
  Groq is the fallback for small ones. A **Groq-only** setup (`LLM_PROVIDER=groq`, no
  OpenRouter key) fails every PR-review prompt over ~7.5K tokens, cleanly (`LLMError`: the
  units are `not_reviewed`, the review `partial` / `failed`, never clean); set
  `PR_REVIEW_MAX_PROMPT_TOKENS=6000` there. Recommended: OpenRouter primary, Groq fallback.
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

### PR-shaped eval (`pr_eval_v1`)

The function-level sets ask "is this function vulnerable?". The product sees a PR (changed
files with full content and a unified diff) and must answer "does this change introduce a
vulnerability?". `scripts/build_pr_eval.py` rebuilds the 276 held-out fix commits behind the
OSV eval sets as PRs, offline from `data/osv_cache/` (about 30 s, 330 MB peak, no API calls,
no models):

```bash
python scripts/build_pr_eval.py            # -> ml/evaluation/datasets/pr_eval/
python scripts/build_pr_eval.py --check    # summary only
```

- **Items** (one JSON object per line): `id`, `kind`, `language`, `repo`, `advisory_id`,
  `cve_id`, `category`, `cwe`, `split`, `pr_title`, `pr_body`, `files` (`path`,
  `old_content`, `new_content`, `patch`), `target` (`path`, `vuln_lines_new`,
  `changed_lines_new`), `source`, `notes`, plus a `meta` provenance object.
- **Kinds.**
  - `vuln_introducing`: the fix commit reversed (`source: reversed_fix`).
    `vuln_lines_new` holds the vulnerable-file lines, inside the eval set's vulnerable
    functions, that the fix deleted or changed. For an insertion-only fix it holds the
    insertion point ±2 (`meta.vuln_lines_mode`).
  - `vuln_fix`: the real fix (`real_fix`). The correct verdict is "nothing introduced".
  - `benign`: bystander edits (`bystander`). These are the files of a fix commit that hold no
    function the corpus builder paired as changed by the fix. They come from the eval commits
    first, then from other mined fix commits in the same repos (at most 5 per repo).
- **Files.** Each PR has every `.py`/`.js`/`.ts` file the commit changed, minus
  test/doc/example paths. It is capped at 6 files and 2,000 changed lines, vulnerable files
  first, and truncation is noted in `notes`.
  - A side missing from the cache is rebuilt from GitHub's patch when the patch applies
    exactly. Renamed files are omitted, because their contents were never fetched.
  - Patches are regenerated with difflib: standard `@@` headers, 3 context lines, `a/`/`b/`
    paths. `apply_patch(old, patch) == new` holds for all 2,537 files. `changed_lines_new`
    matches `diff_utils.parse_patch_changed_lines`.
- **PR text.** Titles and bodies are synthetic and neutral ("Update <path> and N other
  files"). They never use the commit message or advisory text. In
  `pr_eval_v1_misleading.jsonl`, the 276 `vuln_introducing` items (id suffix `_misleading`)
  instead say "Refactor: simplify input handling, no behaviour change".
- **Split.** Each item takes the v1 side of its eval pair ids, and bystanders take their
  repo's side. No advisory or repo is in both dev and test. `meta.repo_side` records the
  repo's v1 side even for `reserve` items.
- **Counts.** 1,212 items in total:

  | | vuln_introducing | vuln_fix | benign |
  |---|---|---|---|
  | dev | 82 | 82 | 134 |
  | test | 60 | 60 | 117 |
  | reserve | 134 | 134 | 409 |

  - 276 of 276 `vuln_introducing` items have `vuln_lines_new`: 219 from deleted/changed lines,
    57 from an insertion point.
  - The median PR has 1 file and 21–22 changed lines.
  - `pr_eval_v1_sample_dev.jsonl` is the first-run sample: 60 `vuln_introducing` items
    stratified by language × category, their 60 `vuln_fix` twins and 80 benign items (at most
    3 per repo).
- **Storage.** The JSONL files hold full file contents (130 MB), so they are gitignored.
  `pr_eval_v1_manifest.json` is committed. It pins each file's sha256, the counts and every
  id by split, and the build is deterministic.
- **Caveat.** Bystander benigns come from security-adjacent commits, so they don't give a
  realistic benign base rate. `scripts/fetch_benign_commits.py` fetches real benign commits
  from the same repos through the GitHub API. It skips known fix commits, merges, bots, and
  any message or linked PR that matches a wide security-keyword net. It caches every
  response, honours rate limits and `--max-api-calls`, and writes
  `pr_eval_benign_commits_v1.jsonl` in the same format (`source: benign_commit`).
  `--dry-run` prints the plan and a call estimate without touching the network:

  ```bash
  python scripts/fetch_benign_commits.py --dry-run --max-commits 300 --per-repo 3
  python scripts/fetch_benign_commits.py --max-commits 300 --per-repo 3 --wait-on-rate-limit
  ```

### PR-shaped eval v2 (real benign commits)

The fetch kept 300 real commits (dev 203 from 80 repos, test 97 from 40; Python 231,
JavaScript 69). `python scripts/build_pr_eval.py --v2` (offline, a few seconds) turns them
into eval items and the next run samples. It reads the built v1 files after checking their
sha256 against the v1 manifest, and never rebuilds or rewrites v1.

- `pr_eval_v2_real_commits.jsonl`: all 300 commits, validated like the builder's own items
  (each patch is the difflib diff of the stored old/new contents, ≤ 6 files and ≤ 2,000
  changed lines, neutral PR text, split equal to the repo's v1 side, not a mined fix or eval
  commit). `source: real_commit`; the fetcher's label stays in `meta.fetched_source`, the
  commit sha in `meta.commit`. Real commits are kept apart from the v1 `bystander` benigns.
- `pr_eval_v2_sample_dev.jsonl` (240): the v1 dev sample's 60 `vuln_introducing` + 60
  `vuln_fix` items (same ids and content as v1, so runs compare; label overrides applied)
  plus 120 dev real commits. Their language mix follows the sampled vulnerable items (96
  Python / 24 JavaScript). Repos are taken round-robin, at most 2 per repo, which covers all
  80 dev repos. There are no bystanders.
- `pr_eval_v2_test.jsonl` (217): all 60 + 60 test vuln items and all 97 test real commits.
- `pr_eval_v2_manifest.json` (committed): input and output sha256s, counts, ids, the leak
  check and size stats. The build fails if any item, repo, advisory, commit or eval-pair id
  would cross dev/test. It checks every item against its file's split, every repo against
  its v1 side and every vuln item's eval-pair ids against `splits/v1.json`. Result: 0 shared
  repos, advisories, commits or ids.
- **Size.** Real commits are smaller changes than bystanders: 1.5 files per PR vs 2.2, median
  8 changed lines vs 22 (p90 72 vs 157). The files they touch are as large (median 546 vs 645
  new-file lines, with a longer tail up to 34.5K), so prompt sizes are similar.
- **Running.** `run_pr_eval` reports benign FPR per provenance (`benign_by_source`:
  `real_commit` vs `bystander`, each with its exact 95 % upper bound and precision at the
  base rates; the real-commit row is the realistic one). Any selection holding a test-split
  item, such as `pr_eval_v2_test.jsonl`, needs `--split test --i-know-this-is-the-test-set`,
  in every mode. The v2 dev sample runs without it.
- **Dry run** (`--arm pr`, current defaults incl. 12K prompts, no Semgrep cache yet): dev
  sample 296 / 680 / 2,677 requests (floor / scenario / ceiling), 2.3M / 5.7M / 32M tokens;
  test file 257 / 588 / 2,417 requests, 2.0M / 5.2M / 29M tokens. No dev PR and one test PR
  (a real commit touching a 23.6K-line Ghost file) come out pipeline-partial.
