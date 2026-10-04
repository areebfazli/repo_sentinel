# RepoSentinel Roadmap

Decided 2026-09-22. Ordered by expected impact.

## Order

1. Record findings (this entry).
2. Realistic eval: add ordinary (non-security) functions from the same repos as negatives; report FPR and precision at 1/2/5% base rates.
3. ~~Measure the existing end-to-end LLM prompt in `run_eval` on that set.~~ **Done**: TPR 1/14, FPR twin 1/14, FPR ordinary 1/22 (gpt-oss-120b, 50 items).
4. ~~Three-arm test~~ **Done 2026-09-24** as legacy prompt vs new LLM-first prompt vs the same with no retrieval (see "Results 2026-09-24" below). Retrieval added nothing measurable -> LLM-first review with Semgrep evidence shipped; CVEs stay as optional reference / citations.
5. ~~Semgrep/Opengrep with full language rule packs~~ **Done** (b946269), wired into scans as prompt evidence (41c855f).
6. ~~PR diff-direction check~~ **Done** (4567539), wired into files-mode scans (41c855f): `guard_removed` as prompt evidence, `alert` tier as deterministic findings.
7. ~~Prompt-injection hardening + per-scan token budget / CVE cap.~~ **Done** (663138c, 0bf6274).
8. Later: embedding speed (findings cache per function hash), neural judge experiments.
9. **PR-level review** (2026-09-30, design below): files mode now audits the change as a whole
   and verifies each candidate separately (`REVIEW_MODE=pr`, default; `units` kept). Next:
   measure it against the per-unit review on the PR eval set (`ml/evaluation/datasets/pr_eval/`)
   before tuning any threshold. Harness built (`ml/evaluation/run_pr_eval.py`, "PR-level eval
   harness" below; dry run done); the live runs are pending.

## Findings 2026-09-24 (precision research + adversarial review)

**Precision experiments** (506 held-out vulnerable/fixed pairs, per-item metrics; 0.50 = chance):
- Whole-function retrieval similarity: AUC 0.495. Existing twin_margin: AUC 0.534, P@R0.8 0.512 — only cheap signal above chance after length control, too weak to gate on.
- Hunk-level twin (embed only the fix's removed vs added lines): AUC 0.515 — identical to a random-fix control, so nothing transfers across advisories. Lexical fix-line matching: 0.501.
- Guard-token vocabulary (tokens fixes add/remove): AUC 0.547 but a length artefact — 0.47 on length-matched pairs.
- Paired accuracy is length-confounded: "shorter function = vulnerable" alone scores 0.806 paired accuracy. Report per-item AUC, precision@recall 0.8 and length-balanced paired accuracy instead.
- LLM judge (gpt-oss-120b, top-1 neighbour's fix diff, "does the query still contain the flaw this fix removes?"), 49 pairs: 10 pairs fully correct vs 1 reversed (p≈0.01), precision 0.655 at recall 0.39, 59% both "no". Confound: it scored precision 0.73 when the retrieved fix was the WRONG bug class vs 0.61 when right — suggests general judgement, not diff matching. Small n.
- Literature: no published method separates vulnerable from fixed for non-clone code; LLM+knowledge methods plateau ~0.30 pairwise accuracy (Vul-RAG replication, https://arxiv.org/pdf/2606.04739); high-precision tools (MVP, MOVERY) need clones.

**Adversarial review findings (verified):**
- Category hit 0.43 is inflated by the catch-all "other" class: 0.279 on named categories (280 items) vs 0.602 on "other"; always guessing "other" scores 0.447 — better than retrieval.
- Precision on a 50/50 eval misleads. At the judge's measured FPR 0.204 and recall 0.388, precision would be 0.019 / 0.037 / 0.091 at 1% / 2% / 5% vulnerable base rate. False-positive rate on ordinary code is unmeasured.
- The fix-diff prompt already exists (`markdown_renderer.build_user_prompt`); `run_eval` never measured the LLM stage end-to-end.
- A per-CVE LLM call would be up to 150 calls / ~134K tokens per PR; Groq free tier for these models is ~30 req/min, 8K tokens/min, 200K tokens/day (https://console.groq.com/docs/rate-limits).
- No prompt-injection handling exists; third-party fix code and PR code go straight into the prompt.
- LLM defaults: llama-3.3-70b-versatile retired on Groq (404); now openai/gpt-oss-120b with qwen/qwen3.8-27b same-provider fallback.

## Results 2026-09-24 (LLM-first review)

Changes: the LLM reviews every unit on its own merits (retrieved CVEs are "similar known
vulnerabilities, may or may not apply"); every finding must quote the offending code (dropped
if the quote isn't there); Semgrep high/critical hits and guard_diff changes go in as
evidence; untrusted text is nonce-tagged and sanitised, output escaped; per-scan token budget.

Three arms on ONE model (`groq:qwen/qwen3.8-27b`, `--llm-primary-only`; OpenRouter's
`qwen/qwen3.8-27b:free` returned 4/4 `upstream_provider_shared_pool` 429s at the start). To stay
under a 190K-token Groq daily budget across arms the sample was halved, keeping pairs:
`--sample-kinds vulnerable=8,fixed_twin=8,ordinary=13 --seed 42` (a subset of the earlier 14/14/22
sample; pypi + npm + ordinary sets). Snippet mode, so guard_diff is N/A. Wilson 95% CIs.

| arm | TPR vulnerable | FPR fixed twin | FPR ordinary | FPR ordinary (length-matched) | precision @1/2/5% (point; with the ordinary-FPR upper 95% bound) | pairs vuln-only / twin-only / both / neither | calls / tokens |
|---|---|---|---|---|---|---|---|
| legacy prompt | 1/8 = 0.13 [0.02, 0.47] | 2/8 = 0.25 [0.07, 0.59] | 0/13 [0, 0.23] | 0/7 [0, 0.35] | 1.0 (FPR 0/13); 0.006 / 0.011 / 0.028 | 1 / 2 / 0 / 5 | 29 / 57.2K |
| current (new prompt + CVEs + Semgrep) | 2/8 = 0.25 [0.07, 0.59] | 1/8 = 0.13 [0.02, 0.47] | 1/13 = 0.08 [0.01, 0.33] | 1/7 = 0.14 [0.03, 0.51] | 0.032 / 0.062 / 0.146; 0.008 / 0.015 / 0.038 | 1 / 0 / 1 / 6 | 29 / 69.0K |
| no_retrieval (new prompt + Semgrep, 0 CVEs) | 2/8 = 0.25 [0.07, 0.59] | 1/8 = 0.13 [0.02, 0.47] | 1/13 = 0.08 [0.01, 0.33] | 1/7 = 0.14 [0.03, 0.51] | 0.032 / 0.062 / 0.146; 0.008 / 0.015 / 0.038 | 1 / 0 / 1 / 6 | 29 / 38.9K |

- `current` and `no_retrieval` flagged exactly the same four items; retrieval changed no verdict
  and cost +77% tokens. Only one `current` finding cited a CVE (on an SSRF pair, both twins).
- The legacy prompt on the same model found 1/8 and flagged 2 fixed twins (twin-only pairs 2 vs
  0). The new prompt's ordinary FP is a plausible path-join finding in `ckan/lib/uploader.py`.
- The quote check dropped nothing on this model (no hallucinated quotes observed).
- Semgrep (snippet mode, high/critical) fired on 1/29 items (pickle, a true vulnerable one, which
  both new arms flagged). Standalone numbers (506 pairs + 1,000 ordinary): high/critical hit on
  3.2% vulnerable / 1.4% fixed / 0.5% ordinary. guard_diff standalone (held-out 506 pairs):
  `guard_removed` TPR 0.227 / FPR 0.030; `alert` TPR 0.032 / FPR 0/506.
- Caveats: n = 8 pairs + 13 ordinary; every difference above is within the CIs. One model;
  gpt-oss-120b (the earlier legacy run, TPR 1/14) is not comparable. OSV functions may be in
  the model's training data. Results: `ml/evaluation/results/llm_arm_{legacy,current,no_retrieval}_groq_qwen_29.json`.
- Next: a larger run on a request-limited model (OpenRouter when its pool is free) to tighten
  the CIs; consider dropping CVE context from the default prompt (`LLM_MAX_CVES_PER_UNIT=0`) if
  the larger run confirms no gain.

## Fixes 2026-09-24 (adversarial review of the LLM-first pipeline)

All done; fast suite + ruff green; no API calls or model loads were needed to verify them.

1. ~~HIGH: code hidden from the LLM.~~ **Done** (32db980): `sanitize_untrusted` stripped HTML
   comments from the code under review (`# <!--` ... `# -->` hid `os.system(cmd)`; a `"<!--"`
   literal blanked the rest of the function). Openers are now defused in place, nothing is
   deleted, U+2028/U+2029 and lone CRs become placeholders so line numbers stay 1:1. Output:
   prose fields still drop complete comments, code fields keep everything.
2. ~~Oversized units truncated from the end.~~ **Done** (5c7845d): elided around the changed
   lines (head + tail without a diff), explicit omission markers, real line numbers, `partial`
   flag when changed code was cut.
3. ~~Partial reviews reported as clean.~~ **Done** (f8bcdd3): `review_status`
   complete / partial / failed + counts; "Partial review: N of M unit(s) not reviewed" instead
   of the clean message; Action summary shows coverage, `INPUT_FAIL_ON_PARTIAL` (default true).
4. ~~Quote check too strict.~~ **Done** (1635e0a): whitespace-insensitive, sanitiser-aware,
   list quotes, trailing comments / punctuation; hallucinated quotes still rejected.
5. ~~Deterministic guard alerts fail the build unvetoed.~~ **Done** (e9365a9):
   `GUARD_ALERT_SEVERITY` (medium), `corroborated_by` llm / semgrep, Action gates on them only
   when corroborated or `INPUT_GATE_ON_DETERMINISTIC`.
6. ~~Groq TPM overrun.~~ **Done** (fa15521): per-model token pacing (`LLM_TPM_LIMITS`),
   Retry-After honoured up to `LLM_MAX_WAIT_S`, `LLM_SCAN_MAX_WALL_S` (unsent units:
   `time_budget`). Not yet measured against the live Groq limits.
7. ~~Dedupe key drift.~~ **Done** (cfadf88): key = source + anchored code line; old comments
   adopted via `legacy_dedupe_keys`.
8. ~~All LLM calls failing discards deterministic evidence.~~ **Done** (f8bcdd3): the scan
   completes with `review_status: failed` and its Semgrep / guard_diff results.
9. ~~Duplicate eval ids.~~ **Done** (01905e8): ids carry a code hash, migrated offline
   (`scripts/migrate_eval_ids.py`: 1,012 -> 1,004 OSV items, 4 exact duplicate pairs dropped),
   `run_eval` pairs robustly. **Open:** `baseline.json` still records the old dataset sha256
   (the slow regression test fails until `run_eval --write-baseline` is re-run on the three
   sets), and the results above (`ml/evaluation/results/`, now versioned) used the old ids, so
   a `--seed 42` sample drawn now is not the same sample.

## Eval split + static-first candidate recall (2026-09-24)

- **Split** `ml/evaluation/splits/v1.json` (`scripts/build_eval_split.py`, seed 42):
  - dev: 151 pairs + 498 ordinary functions (161 length-matched);
  - test: 100 pairs + 500 ordinary functions (163 length-matched);
  - 251 pairs held in reserve.
  - Grouped by advisory, OSV aliases, repo and near-duplicate code: 0 advisories, repos, exact
    or near-duplicate bodies in both splits.
  - Stratified by the year the fix became public × language. `by_advisory` dates allow a split
    at any training cutoff.
  - 500 ordinary functions can't certify FPR ≤ 0.5%: 0/500 still has a 0.76% upper bound. A
    larger pool (3,377) is in `detection_eval_ordinary_ext.jsonl`. See
    `ml/evaluation/splits/README.md`.
- **Static-first candidate recall** (`python -m ml.evaluation.analyze_candidates`, offline, all
  502 held-out pairs):
  - Candidate = Semgrep evidence hit (high+, production exclusions) on the new code, or
    guard_diff `guard_removed`.
  - Reverse fix (simulated vulnerability-introducing PR): Semgrep 16/502 = 0.032, guard_diff
    0.227, union **117/502 = 0.233 [0.198, 0.272]**. At the advisory level the union is 86/276
    = 0.312.
  - Overlap: guard_diff alone 101, Semgrep alone 3, both 13. Semgrep adds 3 pairs to
    guard_diff. Semgrep at any severity: union 0.235.
  - Fix direction (benign): union FPR 20/502 = 0.040.
  - Ordinary functions: Semgrep 5/1000 = 0.005. guard_diff can't run without a diff; its proxy
    is 0 on synthetic edits and 0.021 on real bystander edits.
  - By category, union recall is 0.38 cmd_injection, 0.38 xss, 0.32 path_traversal, 0.23
    redos, 0.19 ssrf, 0.21 authz, 0.16 "other" and 0/5 sqli.
  - Before vs after 2025-01-01: 0.231 vs 0.236.
- **Implication.** Static-first can never recall more than ~23% of these vulnerabilities, even
  with a perfect LLM verifier. In exchange it sends ~1–4 functions per 100 to the LLM instead
  of 100.
  - If the verifier's TPR (0.25, n = 8) were independent of candidacy, recall would be ~0.06.
  - It may not be independent: in the 8-item LLM arm, both LLM hits were static candidates and
    the other 6 were not. n is tiny.
  - Next measurement: LLM-first on the test split's 100 vulnerable functions (22 are candidates,
    78 not). If the LLM's hits fall almost entirely inside the candidates, static-first loses
    little recall for ~25x fewer calls.

## PR-level review (2026-09-30)

**Why.** Tools that produce useful PR security reviews (claude-code-security-review, Codex
Security, vulnhuntr, Semgrep Assistant, ZeroPath) share: PR-level framing ("what does this
change newly introduce?", diff + changed files), context pulled on demand, a separate
per-finding verification pass with false-positive precedents, evidence-carrying findings
(source -> control -> sink, exploit scenario, counterevidence), and static tools as leads.
Diff-only prompting reaches ~6-9% recall in the literature; structured context + verification
~48% recall at 70% precision (VIC-RAGENT). A PR description framing a change as safe can
collapse detection (97% -> 4%). Ours so far: per-function prompt ~1/8 on the right lines, CVE
retrieval added nothing, guard_diff ~23% of simulated vulnerability-introducing diffs at ~3-4%
FP, Semgrep >= high fires rarely.

**Design** (`core/pr_context.py`, `core/pr_review.py`, `core/prompts/`):
- Bundle: per file the diff with new-file line numbers; per changed function after + before
  (old content from the reverse-applied patch); tree-sitter symbol index of the PR's files
  (definitions, methods, module variables, call sites) for context requests and auto-included
  direct callers / callees. Token-budgeted: drop neighbours, then before-versions, then narrow
  the after-code to the changes, then diff-only, then clip the diff (partial).
- Leads (not findings): guard_diff in every direction, Semgrep down to `low` (marked "lead
  only" below `SEMGREP_MIN_SEVERITY`), sensitive sinks on added lines.
- Audit prompt adapted from claude-code-security-review (MIT) + codex-security (Apache-2.0)
  guidance; PR title/body never included; JSON findings with source / sink / missing_control /
  exploit_scenario / quote / confidence 1-10.
- Context loop: `need_context` requests resolved on PR files only, <= 2 rounds, <= 1500 tokens,
  early stop when nothing new resolves (one final call). Our own code; nothing from vulnhuntr
  (AGPL).
- Validation: quote must be in the NEW file; regex hard exclusions (DoS, rate limiting, leaks,
  memory safety outside C/C++, docs, tests; ReDoS left to the verifier's "attacker-controlled
  pattern" precedent; open redirects kept); audit confidence >= 5.
- Verifier: one fresh call per candidate (<= 8), whole new file or a window + the diff + the
  claim as untrusted text; keep only `confirmed` with confidence >= 8; `VERIFIER_MODEL` can put
  another model first. Unverified candidates are not reported and make the review partial.
- Cost: typical PR 1-6 calls (~5K-40K estimated tokens); ceiling 12 calls.

**Open questions / to measure (eval integration is a separate step):**
- Recall / FPR vs the per-unit review on the PR eval set, per stage (audit-only candidates vs
  verified), so the verifier's cost in recall is known. `review_pr` returns every candidate
  with its status for this.
- `PR_REVIEW_MIN_CONFIDENCE` 8 and the audit floor 5 are uncalibrated defaults.
- Whether the sink leads help or just add noise (off switch: `PR_REVIEW_SINK_LEADS`).
- Whether a stronger verifier model (`VERIFIER_MODEL`) is worth its rate budget.
- Free-tier throughput: a 6-call review is a few minutes on Groq's 8K TPM.

## PR-level eval harness (2026-09-30)

`ml/evaluation/run_pr_eval.py` ([docs/DETAILS.md, "PR-level eval"](docs/DETAILS.md#pr-level-eval)): arms `pr` (scored as `verified` and
`audit_only` from one run), `units` (the per-unit review in-process, same PRs) and
`pr_misleading` (the introducing PRs with a "harmless refactor" title / body). Retrieval off,
deterministic nonces, per-call JSONL cache (resume), pacing / call / token budgets,
localised scoring, pair stats, precision at 1/2/5 %, exact McNemar `--compare`, offline
`--rescore` and `--dry-run`. Shared LLM-eval plumbing moved to `ml/evaluation/llm_eval_common.py`
(`run_eval` re-exports it; importing it doesn't load torch). `CodeParser.extract_functions` now
breaks same-line ties by byte offset: the order of functions on one line (minified JS) differed
between processes, which changed prompts and would have broken the cache.

**Dry run on `pr_eval_v1_sample_dev.jsonl`** (200 PRs: 60 introducing / 60 fix / 80 benign;
offline, 4 worker processes, ~1 min and < 0.5 GB per arm). Prompt tokens are estimates
(chars / 4 x 1.25); totals add `LLM_OUTPUT_TOKENS_ESTIMATE` = 1500 completion tokens per call.
`floor` = audit calls only (no candidate, no context round); `scenario` = the stub reports one
candidate per file, each verified; `ceiling` = every context round the audit budget allows
(+1500 tokens each) and all 8 verifier calls at 6000 tokens.

| arm | requests floor / scenario / ceiling | calls per PR (mean, p90) floor / scenario | tokens incl. completion floor / scenario / ceiling |
|---|---|---|---|
| `pr` (200) | 320 / 689 / 2,274 | 1.6, 3 / 3.4, 7 | 1.75M / 4.13M / 16.2M |
| `units` (200) | 392 (exact: no follow-up calls) | 1.96, 4 | 2.29M |
| `pr_misleading` (60) | 98 / 200 / 683 | 1.6, 3 / 3.3, 6 | 0.54M / 1.21M / 4.89M |

- Prompt size per call: `pr` mean 4.5K, p90 6.0K; `units` mean 4.3K, p90 6.0K.
- Wall time on OpenRouter's free tier (20 requests/min, 1,000/day with credits), 3 s spacing
  plus an assumed 20 s per call: `pr` ~2.0 h (floor) / 4.4 h (scenario) / 14.5 h (ceiling, 3
  days of requests); `units` ~2.5 h; `pr_misleading` ~0.6-1.3 h. On Groq's free tier (8K
  tokens/min, ~200K tokens/day per model) even the `pr` floor is ~5.4 h and ~9 days of tokens:
  use OpenRouter.
- Partial by the pipeline's own budget (independent of the model's answers): `pr` 29/200 PRs
  (25 with a file's diff clipped to fit one 6,000-token audit prompt, 4 with files left
  unaudited by the 4-call audit budget); `units` 1/200.
- `pr_misleading`: prompts byte-identical to the `pr` arm's for 60/60 PRs; the PR title / body
  reached no prompt. A live run of that arm therefore measures only provider nondeterminism.

**Semgrep leads**: `--semgrep-precompute` runs the engine once over every file of the selection
(`pr_eval_v1_sample_dev`: 388 files, 282 Python / 106 JS/TS, 7.6 MB; the `pr_misleading` items
reuse their base items' entries). 2-item smoke test: 12 s engine time (mostly fixed start-up),
~0.9 GB peak for the engine processes; the cached hits served per PR matched a direct engine
run on the same units. Not yet run on the dataset.

## 1. Detection quality

### 1a. Grow the corpus (25 entries -> thousands of fix-commit pairs)
- Sources with vulnerable *and* fixed versions per function:
  - CVEfixes — 12K commits / 139K functions, multi-language, MIT / CC-BY-4.0.
  - ReposVul — 6.1K CVEs, multi-language, MIT.
  - OSV / GHSA advisories — CC-BY-4.0; mine fix commits for Python/JS.
- Avoid BigVul as a primary source: labels are only 25–60% accurate (https://arxiv.org/abs/2403.18624).

### 1b. Store the patched twin and retrieve both
- Directly targets safe-vs-vulnerable confusion (precision ~0.5 at every threshold).
- Vul-RAG: +7–11pp precision, +9–14pp recall (https://arxiv.org/abs/2406.11147). FVF: removed 65% of similar-but-patched false alarms with zero new FPs (https://arxiv.org/abs/2412.20740).
- Store `vulnerable_code` + `fixed_code` per entry; at query time compute `sim_vuln - sim_fixed`; feed the LLM the fix diff.

### 1c. Embedder — DECISION: jinaai/jina-embeddings-v2-base-code
- 161M, 768-dim, 8192-token context, Apache-2.0, contrastively trained for code retrieval. Same size class as UniXcoder (dev box: 8 cores, 11 GB RAM, no GPU).
- Why over UniXcoder: a real embedding model (no self-pooled hidden states, so comments stop diluting the vector); no 512-token truncation; better code-to-code ranking (UniXcoder CoIR 37.3, weakest of the field).
- Will NOT fix safe-vs-vulnerable twins; that is 1b.
- Code changes: `vector_store.py:33` hardcodes `vector_size = 768` (make it a setting); `embedder.py:84` hardcodes `max_length=512` (make it `EMBEDDING_MAX_TOKENS`, ~2048); `trust_remote_code=True` in `AutoModel.from_pretrained`; pooling stays `mean`.
- **Done 2026-09-22** (see 1d for the numbers). Peak RSS with both models 3.6 GB. (The ~70 ms/function first quoted here was wrong for real functions: ~1.5 s, see section 2.)

### 1d. Reranker — DECISION: BAAI/bge-reranker-v2-m3, off by default
- 568M cross-encoder, Apache-2.0. Current ms-marco MiniLM (web-search trained) scores ~0 on every code pair, so the stage is a no-op.
- Why over jina-reranker-v3: Jina rerankers are CC-BY-NC-4.0; v3 is listwise, so a pair's score changes with batch composition (https://huggingface.co/jinaai/jina-reranker-v3/discussions/2), which breaks a fixed `RERANK_THRESHOLD`. bge is pairwise, loads via `sentence_transformers.CrossEncoder` with no custom code, and has ONNX/int8 builds.
- Use `max_length=1024`, batch 8–16; expect ~2.5–3 GB RAM and a few hundred ms per pair on CPU. Fallback if too heavy: bge-reranker-base (278M).
- **Result 2026-09-24: reranker OFF by default (`RERANKER_ENABLED=false`), kept available.** Overnight eval on the same 150 held-out items from the OSV eval set (sim 0.25):

  | config | category hit | s/item |
  |---|---|---|
  | no reranker | 0.373 (28/75) | 0.05 |
  | bge-reranker-v2-m3 @1024 | 0.387 (29/75) | 48.6 |
  | bge-reranker-v2-m3 @512 | 0.413 (31/75) | 25.0 |
  | bge-reranker-base @512 | 0.360 (27/75) | 7.2 |

  All within noise (27–31 hits of 75). Rerank probability also doesn't separate vulnerable from fixed at any threshold (precision ≤ 0.5 at 0.1–0.9), so `RERANK_THRESHOLD` stays 0.0 and only applies when enabled. Full 1,062-item eval without the reranker (the new `baseline.json`, `"reranker": null`): P 0.5, R 1.0, F1 0.667, category hit 0.431. When off, the cross-encoder is never loaded (~3 GB RAM and its load time saved) and candidates keep similarity order. If enabled, use v2-m3 @512 (`RERANKER_MAX_TOKENS` default; best-measured and half the cost of 1024).
- Revisit when a code-trained cross-encoder exists, or after fine-tuning one on our vulnerable/fixed pairs (1a/1b data). Untested speed options if it comes back: batch size 1 (no padding to the longest pair) and reranking only the top-N=5 ANN candidates instead of 10.
- History: the first 50-item eval (2026-09-22) showed recall 0.96 -> 1.0 and the right category on top 23/25 vs 19/25 from ANN; the "0.50–0.73 on every code pair" probabilities were a double-sigmoid bug in `Reranker`, since fixed. `transformers` pinned `<5` (jina's remote code uses the 4.x API).

### 1e. Strip comments before embedding
- Tree-sitter is already loaded. Commented SQLi embeds at ~0.30 vs ~0.70 clean.

### 1f. Judge stage between retrieval and the LLM — DECISION: layered, Semgrep first
- No model solves safe-vs-vulnerable twins off the shelf: PrimeVul paired eval has fine-tuned code models at 1–3% and GPT-4 at 5–13% (https://arxiv.org/abs/2403.18624). Combine non-hallucinating signals instead. Laya and TypeSafe Jev rejected: no code training, no vuln evidence.
1. **Semgrep / Opengrep (+ Bandit for Python).** LGPL-2.1 / Apache-2.0, milliseconds on CPU, deterministic. ~~Map each corpus `category` to a rule set~~ — **superseded 2026-09-24**: run full language rule packs, not category-selected; retrieval gets the category wrong 72% of the time on named categories (see Findings 2026-09-24), so category-scoping the rules would inherit that error. Pass hits (rule id, line) into the LLM prompt as evidence to cite. Recall on novel patterns is 14–22% (https://arxiv.org/abs/2606.21071), so it is a precision booster, not a gate. Skip CodeQL (not free for private repos).
2. **`sim_vuln - sim_fixed`** from 1b.
3. **Optional neural judge**, validated on `run_eval` first: R2Vul (1.5B, MIT, https://github.com/martin-wey/R2Vul; verify its benchmark is PrimeVul-style) or Qwen2.5-Coder-3B-Instruct as a logprob yes/no judge (Apache-2.0, GGUF int4 on llama.cpp).
4. **Confidence score** from Semgrep hit, `sim_vuln - sim_fixed`, rerank score, judge probability, vote history. LLM sees only candidates above a threshold.

### 1g. Grow the eval set
- 50 items today; build from 1a's paired data. Assert category-hit-rate as well as F1; stamp `baseline.json` with date and git SHA.

## 2. Speed

Measured on the dev box (8 cores, 11 GB RAM, CPU only), 2026-09-23/24:
- jina embeds ~1.5 s per real function, not the ~70 ms quoted in 1c. With the reranker off (1d) the embedder is the whole model cost, so caching (embeddings, and findings per body hash below) matters most.
- `EMBEDDING_MAX_TOKENS` 1024 instead of 2048 saves only 7%; keep 2048.
- torch with 8 threads is 18% *slower* than 4.
- ONNX int8 export ran out of RAM on 11 GB; needs a bigger box (or a pre-exported model) before it can be measured.

Ideas:
- ONNX int8 for the embedder (and the reranker, if re-enabled): ~2.7–3.4x faster at 94–98% quality (export OOMed here, see above).
- Qdrant scalar quantization: 75% less memory, up to 60% faster, ~0.3% precision loss (https://qdrant.tech/articles/scalar-quantization/).
- Cache findings per function-body hash, not only embeddings.
- Stream inline comments per unit instead of after the whole scan.

## 3. Features (ranked)

1. **SARIF upload to GitHub code scanning** — Security tab, native dismissal, fingerprint dedupe replaces hidden markers. Needs a distinct `category` per run (https://github.blog/changelog/2025-07-21-code-scanning-will-stop-combining-multiple-sarif-runs).
2. **Prompt-injection hardening** — PR titles, comments and diffs go straight into the prompt; "Comment and Control" hijacked Claude Code Security Review, Gemini CLI and Copilot this way. Delimit untrusted content, treat as data, strip HTML comments / hidden markdown, never let LLM output trigger actions.
3. **` ```suggestion ` blocks** — one-click apply of `fix_snippet`.
4. **"Previously dismissed" context** — surface vote history in the comment.
5. **Per-repo category/severity scoping** (`.reposentinel.yml`).
6. **Check run instead of comment-only gate** — can be a required status.
7. **Resolve review threads via GraphQL** when a finding disappears, instead of deleting.
8. **Reachability filter** — only flag functions reachable from an entrypoint (tree-sitter call graph).
