# RepoSentinel Roadmap

Decided 2026-09-22. Ordered by expected impact.

## Order

1. Corpus import (1a) + patched twin (1b) + bigger eval set (1g)
2. Comment stripping (1e) + ONNX / Qdrant quantization (2)
3. Semgrep confirmatory signal (1f step 1)
4. Prompt-injection hardening (3.2)
5. SARIF upload (3.1) + suggestion blocks (3.3)
6. Embedder swap to jina (1c) + reranker swap to bge (1d), evaluated on the bigger eval set
7. Neural judge experiments (1f step 3) + confidence score (1f step 4)

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
- **Done 2026-09-22** (see 1d for the numbers). Embeds a function in ~70 ms on CPU; peak RSS with both models 3.6 GB.

### 1d. Reranker — DECISION: BAAI/bge-reranker-v2-m3
- 568M cross-encoder, Apache-2.0. Current ms-marco MiniLM (web-search trained) scores ~0 on every code pair, so the stage is a no-op.
- Why over jina-reranker-v3: Jina rerankers are CC-BY-NC-4.0; v3 is listwise, so a pair's score changes with batch composition (https://huggingface.co/jinaai/jina-reranker-v3/discussions/2), which breaks a fixed `RERANK_THRESHOLD`. bge is pairwise, loads via `sentence_transformers.CrossEncoder` with no custom code, and has ONNX/int8 builds.
- Use `max_length=1024`, batch 8–16; expect ~2.5–3 GB RAM and a few hundred ms per pair on CPU. Fallback if too heavy: bge-reranker-base (278M).
- **Done 2026-09-22.** Eval (jina + bge, 50 items): recall 0.96 -> 1.0, F1 0.6575 -> 0.667, category hit rate 0.96; precision flat at 0.5 as expected. Thresholds unchanged (0.25 / 0.0); bge puts the right category on top 23/25 vs 19/25 from ANN. (Correction: the "0.50–0.73 on every code pair" probabilities were a double-sigmoid bug in `Reranker`, since fixed; true probabilities and any `RERANK_THRESHOLD` await a re-run of this eval.) Cost: 436 ms/pair on CPU, ~4.4 s per unit, 6x the embedder; first target for ONNX int8 (section 2). `transformers` pinned `<5` (jina's remote code uses the 4.x API).

### 1e. Strip comments before embedding
- Tree-sitter is already loaded. Commented SQLi embeds at ~0.30 vs ~0.70 clean.

### 1f. Judge stage between retrieval and the LLM — DECISION: layered, Semgrep first
- No model solves safe-vs-vulnerable twins off the shelf: PrimeVul paired eval has fine-tuned code models at 1–3% and GPT-4 at 5–13% (https://arxiv.org/abs/2403.18624). Combine non-hallucinating signals instead. Laya and TypeSafe Jev rejected: no code training, no vuln evidence.
1. **Semgrep / Opengrep (+ Bandit for Python).** LGPL-2.1 / Apache-2.0, milliseconds on CPU, deterministic. Map each corpus `category` to a rule set, run only those rules on the flagged function, pass hits (rule id, line) into the LLM prompt as evidence to cite. Recall on novel patterns is 14–22% (https://arxiv.org/abs/2606.21071), so it is a precision booster, not a gate. Skip CodeQL (not free for private repos).
2. **`sim_vuln - sim_fixed`** from 1b.
3. **Optional neural judge**, validated on `run_eval` first: R2Vul (1.5B, MIT, https://github.com/martin-wey/R2Vul; verify its benchmark is PrimeVul-style) or Qwen2.5-Coder-3B-Instruct as a logprob yes/no judge (Apache-2.0, GGUF int4 on llama.cpp).
4. **Confidence score** from Semgrep hit, `sim_vuln - sim_fixed`, rerank score, judge probability, vote history. LLM sees only candidates above a threshold.

### 1g. Grow the eval set
- 50 items today; build from 1a's paired data. Assert category-hit-rate as well as F1; stamp `baseline.json` with date and git SHA.

## 2. Speed

- ONNX int8 for embedder and reranker: ~2.7–3.4x faster at 94–98% quality.
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
