# PR eval v2 — held-out test run: pre-registration

Written before the run; the test set has never been evaluated. One run, no tuning afterwards.

## Configuration (frozen, = shipped default as of commit f7e4072)
- Model: `openrouter:nvidia/nemotron-3-super-120b-a12b:free`, no fallback, temperature 0.2 (no LLM_SAMPLING entry),
  LLM_MAX_OUTPUT_TOKENS 16000, LLM_CALL_DEADLINE_S 300.
- PR review defaults: 12K prompts, verifier cutoff PR_REVIEW_MIN_CONFIDENCE=7, worth-a-look tier on.
- Dataset: `ml/evaluation/datasets/pr_eval/pr_eval_v2_test.jsonl` (60 vuln_introducing, 60 vuln_fix,
  97 real everyday commits; manifest `pr_eval_v2_manifest.json`). Semgrep cache `pr_semgrep_v2_test217.json`.

## Primary metrics (verified report)
1. Bug-introducing PRs caught at function level (`localised_function`), with 95% CI.
2. Fix PRs flagged (any finding).
3. Everyday commits flagged (any finding), with exact 95% upper bound.

## Secondary metrics
- Strict localisation (exact lines ±2); verified + worth-a-look; audit-only (before verifier).
- In-scope recall (excluding DoS and timing/race CWEs).
- False-alarm rates split by PR size: < 50 changed lines vs >= 50 (benign and fix).
- Precision at 1% / 2% / 5% base rates (point and FPR-upper-bound).
- Errors / partial reviews, calls and latency per PR.

## Reporting rules
- Errored items: primary metrics on completed items, plus worst-case bounds over all items.
- Results are reported as-is in README/DETAILS, including if worse than dev.
