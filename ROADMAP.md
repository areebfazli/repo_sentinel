# RepoSentinel Roadmap

Decided 2026-09-22. Ordered by expected impact.

## Order

1. Record findings (this entry).
2. Realistic eval: add ordinary (non-security) functions from the same repos as negatives; report FPR and precision at 1/2/5% base rates.
3. Measure the existing end-to-end LLM prompt in `run_eval` on that set.
4. Three-arm test: LLM with no retrieval vs top-1 fix diff vs random wrong fix diff (+ ordinary negatives). If equal, CVE retrieval is not useful as evidence -> LLM-first review with Semgrep evidence; keep CVEs for citations/category hints.
5. Semgrep/Opengrep with full language rule packs (not category-selected: retrieval gets the class wrong 72% of the time on named categories).
6. PR diff-direction check ("did this PR remove a guard?") — the judge got 0/7 guard-added pairs.
7. Prompt-injection hardening + per-scan token budget / CVE cap.
8. Later: embedding speed (findings cache per function hash), neural judge experiments.

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
