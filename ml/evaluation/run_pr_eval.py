"""PR-level eval: the PR review pipeline vs the per-unit review on PR-shaped items.

Items (``ml/evaluation/datasets/pr_eval/``, built by ``scripts/build_pr_eval.py``)
are PRs: ``files[{path, old_content, new_content, patch}]`` plus a ``kind``
(``vuln_introducing`` | ``vuln_fix`` | ``benign``) and a ``target``
(``path``, ``vuln_lines_new``, ``changed_lines_new``). Retrieval is OFF in every
arm (no embedder, no Qdrant): this compares review designs, not retrieval.

Arms (``--arm``):

- ``pr``: ``core.pr_review.review_pr`` on the item's files (audit with context
  rounds, then per-candidate verification). One run is scored twice:
  ``verified`` (the report: confirmed findings + deterministic guard_diff
  alerts, what production posts) and ``audit_only`` (every audit candidate
  that would be sent to the verifier - past the quote check, the hard
  exclusions and the audit's own confidence floor - plus the guard alerts).
  ``audit_raw`` (every validated candidate, exclusions included) is recorded
  too.
- ``units``: the old per-function review (``REVIEW_MODE=units``), run
  in-process exactly as ``scan_runner`` does for files mode (plan_units with
  deletion points, Semgrep at SEMGREP_MIN_SEVERITY, guard_diff, the review
  prompt budget, deterministic guard alerts) minus the API, DB and retrieval.
- ``pr_misleading``: the ``pr`` arm on the ``_misleading`` variants of the
  selected ``vuln_introducing`` items (``--misleading-dataset``), whose
  pr_title / pr_body frame the change as a harmless refactor. They are passed
  to ``review_pr``, which must ignore them. Nonces are derived from the BASE
  item id, so when the pipeline ignores the PR text its prompts are byte-for-byte
  the ``pr`` arm's (``--dry-run`` checks this offline); the cache key still
  carries the variant's own id, so a live run makes its own calls (it then
  measures end-to-end behaviour incl. provider nondeterminism).

Every prompt uses deterministic nonces (``eval_nonce`` of the base id and the
call's position), so prompts, and the cache, are reproducible. The LLM calls
go through ``EvalGate``: a JSONL cache keyed by (item id, call prompt sha256,
model, sampling, ``--llm-repeat-index``) so a re-run replays finished calls and
resumes. The pipeline's per-call format check (``validate``: the audit /
verifier / units schema checks) is not part of the key, so entries cached
before it existed still replay; a cached answer it rejects is a miss, re-asked
within the run's call budget, and the new answer replaces the entry. Sampling
defaults to each model's recommended parameters (settings.LLM_SAMPLING, what
production sends; e.g. temperature 1.0, top_p 0.95, top_k 20 for
qwen/qwen3.8-27b) and is keyed as
``llm_eval_common.sampling_key``: a plain number for a temperature-only run
(``--llm-temperature 0`` replays the existing temperature-0 entries), a
``"sampling:{...}"`` string otherwise, so the two never replay each other.
Sampled runs are stochastic: repeat a selection with ``--llm-repeat-index 1,
2, ...`` (separate ``--out`` files, ``--compare`` to pair them). Also:
pacing (``--llm-sleep``, ``--llm-tpm``); a hard cap on real calls
(``--llm-max-calls``) and on tokens (``--llm-token-budget``); a clean stop on
a daily limit or repeated rate limits. An item whose calls could not all be
made (stop) is ``not_run``; one whose call failed is ``error``; both are
excluded from the primary metrics (re-run to resume / retry) and reported in
the coverage, sensitivity and bounds sections. The wall-clock budget
(``LLM_SCAN_MAX_WALL_S``) is disabled in the eval (a frozen clock): pacing
must not change results.

Semgrep leads are precomputed once per selection (``--semgrep-precompute
--semgrep-cache PATH``: ONE engine run over every file of every selected item,
whole files, all severities, no exclusions) and served per item by
``CachedSemgrepScanner``, which assigns cached hits to the planner's units like
the real scanner. Without ``--semgrep-cache`` there are no Semgrep leads (the
config records it).

Scoring (primary = localised; tolerance ``--localise-tolerance``, default 2).
The metrics are over fully completed items (status ok / cached); errored and
not-run items are reported separately (below), never silently dropped.

- ``vuln_introducing``: localised TP iff a kept finding is in ``target.path``
  and its line range overlaps a ``vuln_lines_new`` line within the tolerance.
  Also: any-finding TP, right-file TP, localised on any vulnerable path
  (``meta.vuln_lines_by_path``; 15 of the 60 dev-sample items have several),
  and two looser, pre-registered secondary levels from the item's change facts
  (``pr_eval_facts``; ``change_anchored``): change-anchored (within the
  tolerance of any line the fix changed or of a deletion point, in any file)
  and function-level (also anywhere in a function the fix touched, or in a
  file the PR adds whole). The strict level misses catches on a deleted
  control, at the sink, or in a wholly new file, because ``vuln_lines_new``
  only covers the corpus-paired functions.
- ``vuln_fix``: FP iff any kept finding; also a finding on the fix's own
  added lines in ``target.path`` (``target.changed_lines_new``), and the same
  two looser levels (on a fix anchor / in a fixed function).
- ``benign``: FP (alert) iff any kept finding; findings per PR; FPR on all
  benign PRs and on those sharing no commit with a vuln item of the selection
  (provenance: every benign PR is a bystander of a security-fix commit).
- Coverage per kind (scored / errored with a partial result / errored with no
  result / not run); a sensitivity view adding the errored items' partial
  results; bounds over all items with every unknown outcome either way.
- In-scope recall (``pr_eval_facts.scope_of``: without the DoS family the
  audit prompt excludes, and without timing / race too) and the leaky-diff
  split (security vocabulary in the deleted lines; exact Fisher p).
- Pairs (introducing vs its fix), precision at 1/2/5% base rates from the
  localised TPR and the benign FPR (no point estimate when 0 benign FPs were
  observed; always at the FPR's exact 95% upper bound), Wilson CIs, calls /
  tokens / latency per PR, the candidate funnel, context rounds, verifier
  outcomes, review_status and per category / language.
- What the verifier drops (audit_only vs verified, per kind) and the verifier
  policy curve: confirmed >= k (k = 9..5), confirmed at any confidence,
  confirmed or uncertain, no verifier, replayed from the cached verdicts.

Offline modes: ``--rescore RESULT.json`` (recompute every metric from the
stored findings, e.g. at another tolerance; the facts and labels are
recomputed from the run's dataset), ``--compare A.json B.json``
(exact McNemar on introducing localised TP and on fix FP, exact binomial CIs
+ paired table on benign FPR; items matched by base id; ``--view-a`` /
``--view-b`` pick verified / audit_only; two views of ONE run get the
verifier-loss counts instead of a degenerate McNemar), and ``--dry-run`` (every prompt
built with a stub router, no network: calls and estimated tokens per PR, run
totals, pacing time and budget cut-offs).

    python -m ml.evaluation.run_pr_eval --dry-run --arm pr
    python -m ml.evaluation.run_pr_eval --semgrep-precompute \\
        --semgrep-cache ml/evaluation/results/pr_semgrep_dev200.json
    python -m ml.evaluation.run_pr_eval --arm pr --llm-model openrouter:qwen/qwen3.8-27b:free \\
        --llm-max-calls 900 --semgrep-cache ml/evaluation/results/pr_semgrep_dev200.json \\
        --out ml/evaluation/results/pr_eval_pr_dev200.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from backend.app.config import BASE_DIR, settings  # noqa: E402
from backend.app.core.llm_client import LLMError  # noqa: E402
from backend.app.core.markdown_renderer import SYSTEM_PROMPT as UNITS_SYSTEM_PROMPT  # noqa: E402
from backend.app.core.pr_context import diff_lines  # noqa: E402
from backend.app.core.pr_review import PRReviewConfig, review_pr  # noqa: E402
from backend.app.core.prompts.pr_audit import (  # noqa: E402
    AUDIT_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)
from backend.app.core.review_plan import estimate_tokens  # noqa: E402
from backend.app.core.semgrep_scanner import _owner, _unit_span  # noqa: E402
from backend.app.core.semgrep_scanner import unit_key as semgrep_unit_key  # noqa: E402
from ml.evaluation.llm_eval_common import (  # noqa: E402
    BASE_RATES,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS,
    DEFAULT_LLM_SLEEP,
    DEFAULT_LLM_TPM,
    TEST_SPLIT_FLAG,
    HttpUsageTap,
    LLMCache,
    _exact_rate,
    _fmt_exact,
    _fmt_rate,
    _rate,
    _usage_totals,
    build_llm_router,
    build_pinned_router,
    call_sha256,
    classify_llm_error,
    client_sampling,
    eval_nonce,
    fisher_exact_2x2,
    fpr_upper95_exact,
    llm_model_key,
    mcnemar_exact,
    openrouter_routing,
    parse_llm_model,
    precision_at_base_rate,
    precision_at_observed_fpr,
    router_temperature,
    set_router_temperature,
)
from ml.evaluation.pr_eval_facts import (  # noqa: E402
    SCOPE_DOS,
    SCOPE_IN,
    SCOPE_TIMING_RACE,
    scope_of,
    selection_facts,
)

DATASET_DIR = BASE_DIR / "ml" / "evaluation" / "datasets" / "pr_eval"
DEFAULT_DATASET = DATASET_DIR / "pr_eval_v1_sample_dev.jsonl"
DEFAULT_MISLEADING = DATASET_DIR / "pr_eval_v1_misleading.jsonl"
RESULTS_DIR = BASE_DIR / "ml" / "evaluation" / "results"
DEFAULT_PR_LLM_CACHE = RESULTS_DIR / "pr_llm_cache.jsonl"

ARMS = ("pr", "units", "pr_misleading")
PR_ARMS = ("pr", "pr_misleading")
VIEWS_BY_ARM = {"pr": ("verified", "audit_only", "audit_raw"),
                "pr_misleading": ("verified", "audit_only", "audit_raw"),
                "units": ("verified",)}
HEADLINE_VIEWS = {"pr": ("verified", "audit_only"), "pr_misleading": ("verified", "audit_only"),
                  "units": ("verified",)}
KIND_INTRO, KIND_FIX, KIND_BENIGN = "vuln_introducing", "vuln_fix", "benign"
PR_KINDS = (KIND_INTRO, KIND_FIX, KIND_BENIGN)
KIND_ALIASES = {
    "vulnerable": KIND_INTRO, "vuln_introducing": KIND_INTRO, "intro": KIND_INTRO,
    "fix": KIND_FIX, "vuln_fix": KIND_FIX,
    "benign": KIND_BENIGN,
}
SPLITS = ("dev", "test")
MISLEADING_SUFFIX = "_misleading"
DEFAULT_LOCALISE_TOLERANCE = 2
# Real calls per run: the dev sample needs ~500-900 (see --dry-run); OpenRouter's
# free tier allows 1,000 requests/day with credits on the account.
LLM_MAX_CALLS_CAP = 5000
# Candidate statuses of a candidate that reached (or was due for) the verifier.
VERIFIER_BOUND = ("confirmed", "rejected", "uncertain", "below_min_confidence", "unverified")
SEMGREP_CACHE_VERSION = 1
FINDING_KEYS = ("file_path", "line", "end_line", "cwe", "severity", "title", "source")
# Pacing profiles for the dry-run's wall-time estimate: OpenRouter free models
# allow 20 requests/min (1,000/day with credits); Groq's free tier ~30 req/min and
# 8K tokens/min per model.
PACING_PROFILES = {
    "openrouter_free": {"sleep_s": 3.0, "tpm": 0, "requests_per_day": 1000},
    "groq_free_8k_tpm": {"sleep_s": 2.0, "tpm": 8000, "requests_per_day": 1000},
}
DEFAULT_DRY_RUN_LATENCY_S = 20.0


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def base_id(item_id: str) -> str:
    """The id without the ``_misleading`` variant suffix."""
    return item_id[: -len(MISLEADING_SUFFIX)] if item_id.endswith(MISLEADING_SUFFIX) else item_id


def pair_key(item: dict) -> str:
    """Introducing / fix pair key: ``meta.pair_base``, else the base id minus
    ``_intro`` / ``_fix``; a benign item is its own group."""
    pb = (item.get("meta") or {}).get("pair_base")
    if pb:
        return pb
    bid = base_id(item["id"])
    for suffix in ("_intro", "_fix"):
        if bid.endswith(suffix):
            return bid[: -len(suffix)]
    return bid


def _as_paths(value) -> list[Path]:
    return [Path(p) for p in (value if isinstance(value, (list, tuple)) else [value])]


def load_items(paths, split: str | None = None, ids: set[str] | None = None) -> list[dict]:
    """Items of the JSONL file(s), streamed; only ``split`` / ``ids`` kept
    (the full pr_eval_v1.jsonl is ~130 MB)."""
    out: list[dict] = []
    for path in _as_paths(paths):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if split is not None and item.get("split") != split:
                    continue
                if ids is not None and item["id"] not in ids:
                    continue
                out.append(item)
    return out


def parse_sample_kinds(spec: str) -> dict[str, int]:
    """'vulnerable=30,fix=30,benign=40' -> {kind: quota} (aliases: vulnerable /
    intro / vuln_introducing, fix / vuln_fix, benign). ValueError otherwise."""
    quotas: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        kind = KIND_ALIASES.get(name.strip())
        if not sep or kind is None:
            raise ValueError(f"expected <kind>=<n> with kind in {', '.join(KIND_ALIASES)}; "
                             f"got {part!r}")
        if kind in quotas:
            raise ValueError(f"kind {kind!r} given twice")
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"quota for {name!r} is not an integer: {value!r}") from None
        if n < 0:
            raise ValueError(f"quota for {name!r} must be >= 0")
        quotas[kind] = n
    if not quotas:
        raise ValueError("no <kind>=<n> quotas given")
    return quotas


def sample_by_kind(items: list[dict], quotas: dict[str, int], seed: int = 42) -> list[dict]:
    """Deterministic stratified sample keeping introducing / fix pairs together:
    pair groups are shuffled once and walked in that order, taking the
    introducing item of the first ``quotas[intro]`` groups that have one and
    the fix of the first ``quotas[fix]`` that have one (so every fix whose
    introducing twin is in the pool comes with it while both quotas allow);
    benign items are shuffled separately. Unlisted kinds are left out. The
    result keeps the input order."""
    rng = random.Random(seed)
    groups: dict[str, dict[str, dict]] = {}
    for item in items:
        if item["kind"] in (KIND_INTRO, KIND_FIX):
            groups.setdefault(pair_key(item), {})[item["kind"]] = item
    keys = sorted(groups)
    rng.shuffle(keys)
    chosen: set[str] = set()
    for kind in (KIND_INTRO, KIND_FIX):
        quota = quotas.get(kind, 0)
        for key in keys:
            if quota <= 0:
                break
            if kind in groups[key]:
                chosen.add(groups[key][kind]["id"])
                quota -= 1
    benign = sorted(i["id"] for i in items if i["kind"] == KIND_BENIGN)
    rng.shuffle(benign)
    chosen.update(benign[: quotas.get(KIND_BENIGN, 0)])
    return [i for i in items if i["id"] in chosen]


def misleading_variants(items: list[dict], misleading_path) -> tuple[list[dict], list[str]]:
    """The ``_misleading`` variant of every selected vuln_introducing item
    (other kinds have none). Returns (variants in selection order, base ids
    without a variant)."""
    wanted = {i["id"] + MISLEADING_SUFFIX for i in items if i["kind"] == KIND_INTRO}
    found = {i["id"]: i for i in load_items(misleading_path, ids=wanted)}
    out, missing = [], []
    for i in items:
        if i["kind"] != KIND_INTRO:
            continue
        v = found.get(i["id"] + MISLEADING_SUFFIX)
        if v is None:
            missing.append(i["id"])
        else:
            out.append(v)
    return out, missing


def select_items(args) -> tuple[list[dict], dict]:
    """Load, split-filter, sample and (pr_misleading) swap in the variants.
    Returns (items, selection meta)."""
    items = load_items(args.dataset, split=args.split)
    meta: dict = {"datasets": [_rel(p) for p in _as_paths(args.dataset)], "split": args.split,
                  "n_loaded": len(items)}
    if args.sample_kinds is not None:
        items = sample_by_kind(items, args.sample_kinds, args.seed)
        meta.update(sample_kinds=args.sample_kinds, seed=args.seed)
    if args.arm == "pr_misleading":
        items, missing = misleading_variants(items, args.misleading_dataset)
        meta.update(misleading_dataset=_rel(args.misleading_dataset),
                    n_without_misleading_variant=len(missing))
    meta["n_items"] = len(items)
    meta["by_kind"] = dict(Counter(i["kind"] for i in items))
    return items, meta


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Semgrep: one precomputed engine run, served per item
# ---------------------------------------------------------------------------

HIT_FIELDS = ("rule_id", "message", "severity", "cwe", "line", "end_line")


def precompute_semgrep(items: list[dict], scanner) -> dict:
    """ONE ``scanner.scan_units`` call over every live file of every item
    (each file a whole-file unit with its text as ``sources``, so it is
    scanned whole exactly like files mode). Returns the cache payload
    ``{"version", "meta", "items": {base id: {path: [hit]}}}``; hits keep all
    severities (the scanner should be built without exclusions: the stub
    applies SEMGREP_EXCLUDED_RULES and each arm its own severity floor)."""
    units: list[dict] = []
    sources: dict[str, str] = {}
    where: dict[str, tuple[str, str]] = {}
    out: dict[str, dict[str, list]] = {}
    for n, item in enumerate(items):
        bid = base_id(item["id"])
        out.setdefault(bid, {})
        for f in item["files"]:
            if f.get("new_content") is None:
                continue
            key = f"i{n:05d}/{f['path']}"
            units.append({"file_path": key, "function_name": None, "start_line": 1,
                          "code": f["new_content"], "language": item.get("language")})
            sources[key] = f["new_content"]
            where[key] = (bid, f["path"])
            out[bid][f["path"]] = []
    t0 = time.perf_counter()
    hits = scanner.scan_units(units, sources=sources) if units else {}
    seconds = round(time.perf_counter() - t0, 2)
    for (key, _fn, _start), file_hits in hits.items():
        if key not in where:
            continue
        bid, path = where[key]
        out[bid][path] = [{k: h.get(k) for k in HIT_FIELDS} for h in file_hits]
    return {
        "version": SEMGREP_CACHE_VERSION,
        "meta": {"n_items": len(out), "n_files": len(units), "engine_seconds": seconds,
                 "n_hits": sum(len(v) for f in out.values() for v in f.values())},
        "items": out,
    }


def load_semgrep_cache(path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("version") != SEMGREP_CACHE_VERSION or not isinstance(data.get("items"), dict):
        raise ValueError(f"{path}: not a version-{SEMGREP_CACHE_VERSION} Semgrep cache")
    return data


class CachedSemgrepScanner:
    """A ``SemgrepScanner`` stand-in serving one item's precomputed whole-file
    hits: each hit goes to the innermost requested unit of its file containing
    its line (``semgrep_scanner._owner``), clipped / de-duplicated / sorted and
    with the snippet taken from the unit's code, as ``parse_results`` does;
    rules in ``exclude_rules`` are dropped like the scanner's own exclusions."""

    def __init__(self, file_hits: dict[str, list[dict]], exclude_rules=None):
        self.file_hits = file_hits or {}
        self.exclude_rules = frozenset(settings.SEMGREP_EXCLUDED_RULES
                                       if exclude_rules is None else exclude_rules)
        self.calls = 0

    def available(self) -> bool:
        return True

    def scan_units(self, units: list[dict], sources: dict[str, str] | None = None) -> dict:
        self.calls += 1
        by_file: dict[str, list[dict]] = {}
        for u in units:
            by_file.setdefault(u.get("file_path"), []).append(u)
        out: dict[tuple, list[dict]] = {}
        seen: set[tuple] = set()
        for path, hits in self.file_hits.items():
            file_units = by_file.get(path)
            if not file_units:
                continue
            for h in hits:
                if h.get("rule_id") in self.exclude_rules:
                    continue
                line = int(h["line"])
                unit = _owner(file_units, line)
                if unit is None:
                    continue
                start, last = _unit_span(unit)
                end_line = min(max(int(h.get("end_line") or line), line), last)
                key = semgrep_unit_key(unit)
                if (key, h["rule_id"], line) in seen:
                    continue
                seen.add((key, h["rule_id"], line))
                code_lines = (unit.get("code") or "").splitlines() or [""]
                snippet = code_lines[line - start: min(end_line, line + 2) - start + 1]
                out.setdefault(key, []).append({
                    "rule_id": h["rule_id"], "message": h.get("message") or "",
                    "severity": h.get("severity"), "cwe": list(h.get("cwe") or []),
                    "line": line, "end_line": end_line,
                    "snippet": "\n".join(s.rstrip()[:200] for s in snippet),
                })
        for hits in out.values():
            hits.sort(key=lambda h: (h["line"], h["rule_id"]))
        return out


def build_precompute_scanner(timeout_s: float):
    """The real engine, all severities, no rule exclusions (the cache is
    filtered per arm), a timeout sized for one run over the whole selection."""
    from backend.app.core.semgrep_scanner import SemgrepScanner

    return SemgrepScanner(timeout_s=timeout_s, exclude_rules=frozenset())


def engine_version(scanner) -> str | None:
    try:
        proc = subprocess.run([scanner.engine, "--version"], capture_output=True, text=True,
                              timeout=60)
        return (proc.stdout or proc.stderr).strip()[:100] or None
    except (OSError, subprocess.SubprocessError, TypeError):
        return None


# ---------------------------------------------------------------------------
# The LLM gate: cache, budgets, pacing, stop conditions
# ---------------------------------------------------------------------------


class EvalStopped(LLMError):
    """A call the eval refused to make (a stop condition): the item is not_run."""


def model_key(router) -> str:
    """``llm_model_key`` for real routers; a stub's ``label`` otherwise."""
    if getattr(router, "mock", False) or getattr(router, "clients", None):
        return llm_model_key(router)
    return str(getattr(router, "label", type(router).__name__))


def call_role(system: str) -> str:
    if system == AUDIT_SYSTEM_PROMPT:
        return "audit"
    if system == VERIFIER_SYSTEM_PROMPT:
        return "verifier"
    if system == UNITS_SYSTEM_PROMPT:
        return "units"
    return "other"


class EvalGate:
    """Every LLM call of a run goes through ``call``: the cache first (a hit
    costs no call, no wait and no budget), then the stop conditions, pacing,
    the real call (usage from the HTTP tap), the cache write. Failures are
    re-raised as ``LLMError`` so the pipeline degrades exactly as in
    production, and the item is marked (``error`` / ``not_run``)."""

    def __init__(self, *, cache: LLMCache | None, max_calls: float, token_budget: int | None,
                 sleep_s: float, tpm: int | None, max_rate_limit_errors: int,
                 tap: HttpUsageTap | None = None, sleep=asyncio.sleep,
                 temperature: float | None = None, keep_prompts: bool = False,
                 repeat: int = 0):
        self.cache = cache
        self.max_calls = max_calls
        self.token_budget = token_budget
        self.sleep_s, self.tpm = sleep_s, tpm
        self.max_rate_limit_errors = max_rate_limit_errors
        self.tap = tap
        self._sleep = sleep
        self.temperature = temperature
        self.keep_prompts = keep_prompts
        self.repeat = int(repeat)
        self.calls = self.tokens_used = self.last_tokens = self.consecutive_rl = 0
        self.stopped: str | None = None
        self.item_id: str | None = None
        self.log: list[dict] = []
        self.item_error: str | None = None
        self.item_error_kind: str | None = None
        self.item_not_run = False

    def begin_item(self, item_id: str) -> None:
        self.item_id = item_id
        self.log = []
        self.item_error = self.item_error_kind = None
        self.item_not_run = False

    def _temperature(self, inner) -> float | str:
        """Cache-key form of the sampling: ``--llm-temperature`` when forced,
        else what the router's primary client sends (``router_temperature``:
        the model's recommended sampling under the default)."""
        return router_temperature(inner) if self.temperature is None else float(self.temperature)

    async def call(self, inner, system: str, user: str, validate=None) -> tuple:
        """``validate``: the pipeline's format check (``LLMRouter.generate``).
        It never enters the cache key (prompt sha, model, sampling, repeat), so
        entries cached before it existed still replay; a cached answer it
        rejects is a miss (logged as ``cached_invalid``): a real call is made,
        within the run's call / token budget like any other, and its answer
        replaces the entry."""
        role = call_role(system)
        sha = call_sha256(system, user)
        model = model_key(inner)
        temp = self._temperature(inner)
        est = estimate_tokens(system) + estimate_tokens(user)
        entry = {"role": role, "sha": sha, "model": model, "prompt_tokens_est": est}
        if self.keep_prompts:
            entry["user"] = user
        self.log.append(entry)
        cached = (self.cache.get(self.item_id, sha, model, temp, self.repeat)
                  if self.cache is not None else None)
        if cached is not None and "llm_json" in cached:
            problem = validate(cached["llm_json"]) if validate is not None else None
            if problem is None:
                entry.update(cached=True, provider=cached.get("provider_used"),
                             latency_s=cached.get("latency_s"), usage=cached.get("usage"))
                return cached["llm_json"], cached.get("provider_used")
            entry["cached_invalid"] = problem
        if self.stopped is None:
            if self.calls >= self.max_calls:
                self.stopped = "max_calls"
            elif self.token_budget is not None and self.tokens_used + est > self.token_budget:
                self.stopped = "token_budget"
        if self.stopped is not None:
            entry.update(cached=False, not_run=True)
            self.item_not_run = True
            raise EvalStopped(f"eval stopped: {self.stopped}")
        if self.calls:
            wait = max(self.sleep_s, 60.0 * self.last_tokens / self.tpm if self.tpm else 0.0)
            if wait > 0:
                await self._sleep(wait)
        self.calls += 1
        if self.tap is not None:
            self.tap.take()
        t0 = time.perf_counter()
        try:
            kwargs = {"validate": validate} if validate is not None else {}
            llm_json, provider = await inner.generate(system, user, **kwargs)
        except asyncio.CancelledError:
            self.stopped = "interrupted"
            self.item_not_run = True
            entry.update(cached=False, not_run=True)
            raise EvalStopped("eval interrupted") from None
        except Exception as exc:  # noqa: BLE001 - any failure: this call errored
            latency = round(time.perf_counter() - t0, 3)
            events = self.tap.take() if self.tap is not None else []
            kind = classify_llm_error(exc, events)
            entry.update(cached=False, error=f"{type(exc).__name__}: {str(exc)[:300]}",
                         error_kind=kind, latency_s=latency, usage=_usage_totals(events))
            self.item_error = entry["error"]
            self.item_error_kind = kind
            if kind == "daily_limit":
                self.stopped = "daily_limit"
            elif kind == "rate_limit":
                self.consecutive_rl += 1
                if self.consecutive_rl >= self.max_rate_limit_errors:
                    self.stopped = "rate_limit"
            else:
                self.consecutive_rl = 0
            self.last_tokens = (entry["usage"] or {}).get("total_tokens") or 0
            self.tokens_used += self.last_tokens
            raise LLMError(f"eval call failed: {entry['error']}",
                           bad_output=bool(getattr(exc, "bad_output", False))) from exc
        latency = round(time.perf_counter() - t0, 3)
        events = self.tap.take() if self.tap is not None else []
        usage = _usage_totals(events)
        upstreams = [ev["upstream_provider"] for ev in events if ev.get("upstream_provider")]
        self.consecutive_rl = 0
        self.last_tokens = usage["total_tokens"] if usage else est
        self.tokens_used += self.last_tokens
        entry.update(cached=False, provider=provider, latency_s=latency, usage=usage)
        if upstreams:
            entry["upstream_provider"] = upstreams[-1]
        if self.cache is not None:
            self.cache.put({
                "id": self.item_id, "prompt_sha256": sha, "model": model, "temperature": temp,
                "repeat": self.repeat, "role": role, "provider_used": provider,
                "llm_json": llm_json,
                "upstream_provider": entry.get("upstream_provider"), "latency_s": latency,
                "usage": usage, "prompt_tokens_est": est,
            })
        return llm_json, provider


class GatedRouter:
    """What the pipeline sees as its router: every call through ``gate``, a
    frozen clock (the wall-clock budget never fires in the eval)."""

    mock = False

    def __init__(self, inner, gate: EvalGate):
        self.inner, self.gate = inner, gate

    @property
    def clock(self):
        return lambda: 0.0

    async def generate(self, system, user, *, deadline=None, validate=None):
        return await self.gate.call(self.inner, system, user, validate)


class NonceSeq:
    """Deterministic ``nonce_factory``: the n-th call of an item gets
    ``eval_nonce("<base id>#<n>")``."""

    def __init__(self, key: str):
        self.key, self.n = key, 0

    def __call__(self) -> str:
        self.n += 1
        return eval_nonce(f"{self.key}#{self.n}")


# ---------------------------------------------------------------------------
# Running one item
# ---------------------------------------------------------------------------


def _parser():
    from backend.app.core.code_parser import CodeParser

    return CodeParser()


def pr_review_config() -> PRReviewConfig:
    """Production settings (the eval's frozen clock makes ``wall_s`` inert)."""
    return PRReviewConfig.from_settings()


async def review_units_pr(files: list[dict], router, *, parser, semgrep_scanner=None,
                          nonce_factory=None) -> dict:
    """The per-unit review (``REVIEW_MODE=units``) of one PR, in-process:
    ``scan_runner._analyze_request``'s files branch (deletion points count as
    changes) without retrieval, then ``scan_runner._units_review_result``."""
    from backend.app.core.analysis_planner import plan_units
    from backend.app.core.evidence import (
        guard_evidence,
        plan_with_touched_lines,
        semgrep_evidence,
    )
    from backend.app.models.schemas import FileInput
    from backend.app.services.scan_runner import NO_RETRIEVAL, _units_review_result

    inputs = [FileInput(path=f["path"], content=f["new_content"], patch=f.get("patch"))
              for f in files if f.get("new_content") is not None]
    units, dropped = await asyncio.to_thread(
        plan_units, plan_with_touched_lines(inputs), parser, settings.MAX_UNITS_PER_SCAN)
    hits = await asyncio.to_thread(
        semgrep_evidence, semgrep_scanner, units, {f.path: f.content for f in inputs},
        settings.SEMGREP_MIN_SEVERITY, frozenset(settings.SEMGREP_EXCLUDED_RULES))
    guard = await asyncio.to_thread(guard_evidence, inputs, units, parser)
    notes = ([f"_Analysis capped at {settings.MAX_UNITS_PER_SCAN} functions; {dropped} not "
              "scanned._"] if dropped else [])
    analysis = {"raw": dict(NO_RETRIEVAL), "units": units, "semgrep": hits, "guard": guard,
                "notes": notes, "bundle": None}
    kwargs = {"nonce_factory": nonce_factory} if nonce_factory is not None else {}
    result = await _units_review_result(analysis, dict(analysis["raw"]), router, [], **kwargs)
    result["bundle_units"] = [
        {"file_path": u["file_path"], "function_name": u.get("function_name"),
         "start_line": u.get("start_line"), "end_line": u.get("end_line")} for u in units]
    return result


def _finding(f: dict) -> dict:
    return {k: f.get(k) for k in FINDING_KEYS}


def _candidate(c: dict) -> dict:
    verdict = c.get("verdict") or {}
    return {
        **{k: c.get(k) for k in ("file_path", "line", "end_line", "cwe", "severity", "title",
                                 "audit_confidence", "status", "status_reason")},
        "source": "llm",
        "verdict": verdict.get("verdict"),
        "verdict_confidence": verdict.get("confidence"),
        "verdict_reason": (verdict.get("reason") or "")[:300] or None,
        "verifier": c.get("verifier"),
    }


def view_findings(result: dict, arm: str) -> dict[str, list[dict]]:
    """The findings each view keeps (see the module docstring)."""
    report = [_finding(f) for f in result.get("report_findings") or []]
    if arm not in PR_ARMS:
        return {"verified": report}
    guard = [f for f in report if f.get("source") == "guard_diff"]
    cands = [_candidate(c) for c in result.get("candidates") or []]
    return {
        "verified": report,
        "audit_only": guard + [c for c in cands if c["status"] in VERIFIER_BOUND],
        "audit_raw": guard + cands,
    }


STAT_KEYS = ("audit_calls", "audit_prompts_planned", "verifier_calls", "context_rounds_used",
             "context_requested", "context_resolved", "candidates", "quote_not_found",
             "hard_excluded", "below_audit_confidence", "verified", "confirmed", "rejected",
             "uncertain", "below_min_confidence", "unverified", "bad_output", "files_total",
             "files_reviewed", "leads")


def item_record(item: dict, arm: str) -> dict:
    target = dict(item.get("target") or {})
    target["vuln_lines_by_path"] = (item.get("meta") or {}).get("vuln_lines_by_path") or {}
    return {
        "id": item["id"], "base_id": base_id(item["id"]), "arm": arm, "kind": item["kind"],
        **{k: item.get(k) for k in ("language", "category", "cwe", "repo", "source", "split")},
        "pair_key": pair_key(item) if item["kind"] in (KIND_INTRO, KIND_FIX) else None,
        "n_files": len(item.get("files") or []),
        "target": target,
    }


def _call_summary(log: list[dict]) -> dict:
    real = [e for e in log if e.get("cached") is False and not e.get("not_run")]
    done = [e for e in log if not e.get("not_run") and not e.get("error")]
    usage = [(e.get("usage") or {}).get("total_tokens") for e in done]
    return {
        "total": len(done),
        "real": sum(1 for e in real if not e.get("error")),
        "cached": sum(1 for e in log if e.get("cached") is True),
        "failed": sum(1 for e in log if e.get("error")),
        "by_role": dict(Counter(e["role"] for e in done)),
        "prompt_tokens_est": sum(e["prompt_tokens_est"] for e in done),
        "usage_total_tokens": sum(u for u in usage if u) if any(usage) else None,
        "usage_known_calls": sum(1 for u in usage if u),
        "latency_s": round(sum(e.get("latency_s") or 0.0 for e in done), 3),
    }


async def run_item(item: dict, arm: str, *, router, verifier_router, gate: EvalGate,
                   parser, semgrep_scanner=None, config: PRReviewConfig | None = None) -> dict:
    """One PR through the arm's pipeline; returns its per-item record."""
    rec = item_record(item, arm)
    gate.begin_item(item["id"])
    nonces = NonceSeq(base_id(item["id"]))
    t0 = time.perf_counter()
    if arm in PR_ARMS:
        result = await review_pr(
            item["files"], router, language=item.get("language"), parser=parser,
            semgrep_scanner=semgrep_scanner, verifier_router=verifier_router,
            config=config or pr_review_config(), retrieval=None, nonce_factory=nonces,
            pr_title=item.get("pr_title"), pr_body=item.get("pr_body"))
    else:
        result = await review_units_pr(item["files"], router, parser=parser,
                                       semgrep_scanner=semgrep_scanner, nonce_factory=nonces)
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    log = gate.log
    if gate.item_not_run:
        status = "not_run"
    elif gate.item_error:
        status = "error"
    elif log and all(e.get("cached") for e in log):
        status = "cached"
    else:
        status = "ok"
    rec["status"] = status
    if gate.item_error:
        rec.update(error=gate.item_error, error_kind=gate.item_error_kind)
    rec["calls"] = _call_summary(log)
    rec["call_log"] = [{k: e.get(k) for k in ("role", "sha", "cached", "provider", "latency_s",
                                               "prompt_tokens_est", "usage", "error_kind",
                                               "not_run", "upstream_provider", "cached_invalid")
                        if k in e}
                       for e in log]
    rec["prompt_shas"] = [e["sha"] for e in log]
    if gate.keep_prompts:
        rec["prompts"] = [e["user"] for e in log]
    rec["providers"] = sorted({e["provider"] for e in log if e.get("provider")})
    rec.update({k: result.get(k) for k in ("review_status", "units_total", "units_reviewed",
                                            "units_partially_reviewed")})
    rec["not_reviewed_reasons"] = dict(Counter(
        u.get("reason") for u in result.get("units_not_reviewed") or []))
    rec["static_hits"] = len(result.get("static_analysis") or [])
    rec["guard_alerts"] = sum(1 for f in result.get("report_findings") or []
                              if f.get("source") == "guard_diff")
    rec["n_units"] = len(result.get("bundle_units") or [])
    if arm in PR_ARMS:
        stats = result.get("pr_review") or {}
        rec["pr_review"] = {k: stats.get(k) for k in STAT_KEYS}
        rec["candidates"] = [_candidate(c) for c in result.get("candidates") or []]
    rec["findings"] = view_findings(result, arm)
    return rec


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def overlaps(f: dict, lines, tol: int) -> bool:
    """The finding's [line, end_line] is within ``tol`` lines of one of ``lines``
    (a finding without a line - a pure-deletion guard alert - never is)."""
    if f.get("line") is None:
        return False
    lo = int(f["line"])
    hi = int(f.get("end_line") or lo)
    lo, hi = min(lo, hi), max(lo, hi)
    return any(lo - tol <= int(v) <= hi + tol for v in lines or ())


def _in_spans(f: dict, spans) -> bool:
    """The finding's [line, end_line] intersects one of ``spans``
    (``[start, end, name]``); a finding without a line never does."""
    if f.get("line") is None:
        return False
    lo = int(f["line"])
    hi = int(f.get("end_line") or lo)
    lo, hi = min(lo, hi), max(lo, hi)
    return any(lo <= int(s[1]) and int(s[0]) <= hi for s in spans or ())


def change_anchored(rec: dict, findings: list[dict], tol: int) -> tuple:
    """``(anchor, function)`` localisation of ``findings`` against the item's
    change facts (``pr_eval_facts``), or ``(None, None)`` without facts.

    Pre-registered secondary definition (the strict ``localised`` stays the
    primary), applied to every file of the PR, not only ``target.path``:

    - anchor: a finding within ``tol`` lines of a change anchor of its file:
      a line the diff added, or a deletion point (the new-file lines just
      before / after a run of removed lines; ``patch_touched_lines``). In a
      reversed fix that is every line the real fix modified, and where the fix
      inserted a control (now deleted: e.g. the removed ``if hmac_key is not
      None:`` guard).
    - function: an anchor hit, OR a finding intersecting a function of the new
      file that contains a change anchor (innermost named function: the sink
      of a function whose guard the PR removed), OR any finding in a file the
      PR adds whole (a module the real fix deleted).

    Each level includes the stricter ones (the caller ORs in ``localised``)."""
    facts = rec.get("facts") or {}
    if "anchors_by_path" not in facts:
        return None, None
    anchors = facts.get("anchors_by_path") or {}
    spans = facts.get("touched_functions_by_path") or {}
    added = set(facts.get("added_files") or ())
    anchor = any(overlaps(f, anchors.get(f.get("file_path")), tol) for f in findings)
    function = anchor or any(
        f.get("file_path") in added or _in_spans(f, spans.get(f.get("file_path")))
        for f in findings)
    return anchor, function


def score_findings(rec: dict, findings: list[dict], tol: int) -> dict:
    """Per-item outcome of one view (see the module docstring). ``None`` for a
    change-anchored level means the record has no facts (not computable)."""
    t = rec.get("target") or {}
    out = {"any": bool(findings), "n_findings": len(findings)}
    anchor, function = change_anchored(rec, findings, tol)
    if rec["kind"] == KIND_INTRO:
        path = t.get("path")
        out["right_file"] = any(f.get("file_path") == path for f in findings)
        out["localised"] = any(f.get("file_path") == path
                               and overlaps(f, t.get("vuln_lines_new"), tol) for f in findings)
        by_path = t.get("vuln_lines_by_path") or {path: t.get("vuln_lines_new") or []}
        out["localised_any_vuln_path"] = any(
            f.get("file_path") in by_path and overlaps(f, by_path[f["file_path"]], tol)
            for f in findings)
        out["localised_fix_anchor"] = None if anchor is None else bool(
            out["localised"] or anchor)
        out["localised_function"] = None if function is None else bool(
            out["localised_fix_anchor"] or function)
    elif rec["kind"] == KIND_FIX:
        out["fp"] = bool(findings)
        out["on_fixed_lines"] = any(
            f.get("file_path") == t.get("path") and overlaps(f, t.get("changed_lines_new"), tol)
            for f in findings)
        out["on_fix_anchor"] = None if anchor is None else bool(out["on_fixed_lines"] or anchor)
        out["in_fixed_function"] = None if function is None else bool(
            out["on_fix_anchor"] or function)
    else:
        out["fp"] = bool(findings)
    return out


def scored(rec: dict) -> bool:
    """Fully completed: every call made (the strict, primary population)."""
    return rec.get("status") in ("ok", "cached")


def partial_result(rec: dict) -> bool:
    """An errored item with a partial result: some calls failed but at least
    one succeeded, so its findings are real but possibly incomplete."""
    return rec.get("status") == "error" and bool((rec.get("calls") or {}).get("total"))


def attach_outcomes(records: list[dict], tol: int) -> None:
    """Outcomes for scored items and for errored items with a partial result
    (the latter only enter the sensitivity view and the bounds)."""
    for r in records:
        r["outcomes"] = ({v: score_findings(r, fs, tol) for v, fs in r["findings"].items()}
                         if scored(r) or partial_result(r) else None)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _p90(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)]


def _rate_of(recs: list[dict], view: str, kind: str, key: str) -> dict:
    """``_rate`` of ``outcomes[view][key]`` over ``recs`` of ``kind``, records
    whose value is None (not computable) left out."""
    return _rate([bool(v) for r in recs if r["kind"] == kind
                  for v in [r["outcomes"][view].get(key)] if v is not None])


def item_scope(rec: dict) -> str:
    """``pr_eval_facts.scope_of`` the record's (current) labels."""
    return scope_of(rec.get("category"), rec.get("cwe"))


INTRO_KEYS = (("localised", "strict"), ("localised_function", "function-level"),
              ("any", "any finding"))


def scope_metrics(recs: list[dict], view: str) -> dict:
    """Introducing recall over all items and over the in-scope ones: without
    the DoS family the audit prompt excludes (``excl_dos``), and also without
    the timing / race group (``in_scope_only``)."""
    intro = [r for r in recs if r["kind"] == KIND_INTRO]
    groups = {"all": intro,
              "excl_dos": [r for r in intro if item_scope(r) != SCOPE_DOS],
              "in_scope_only": [r for r in intro if item_scope(r) == SCOPE_IN]}
    out = {name: {key: _rate_of(g, view, KIND_INTRO, key) for key, _ in INTRO_KEYS}
           for name, g in groups.items()}
    out["by_scope"] = {s: sum(1 for r in intro if item_scope(r) == s)
                       for s in (SCOPE_IN, SCOPE_TIMING_RACE, SCOPE_DOS)}
    out["out_of_scope_ids"] = sorted(r["base_id"] for r in intro if item_scope(r) != SCOPE_IN)
    return out


def leaky_split(recs: list[dict], view: str) -> dict | None:
    """Introducing recall split by whether the PR's deleted lines contain
    ``SECURITY_VOCAB`` terms, with a two-sided exact Fisher p per key
    (descriptive only). None without facts."""
    intro = [r for r in recs if r["kind"] == KIND_INTRO
             and "deleted_security_terms" in (r.get("facts") or {})]
    if not intro:
        return None
    leaky = [r for r in intro if r["facts"]["deleted_security_terms"]]
    clean = [r for r in intro if not r["facts"]["deleted_security_terms"]]
    out: dict = {"n_leaky": len(leaky), "n_not_leaky": len(clean),
                 "terms": dict(Counter(t for r in leaky
                                       for t in r["facts"]["deleted_security_terms"])
                               .most_common())}
    for key, _ in INTRO_KEYS:
        a, b = _rate_of(leaky, view, KIND_INTRO, key), _rate_of(clean, view, KIND_INTRO, key)
        out[key] = {"leaky": a, "not_leaky": b, "p_fisher_exact": round(fisher_exact_2x2(
            a["k"], a["n"] - a["k"], b["k"], b["n"] - b["k"]), 6)}
    return out


def _clean_benign(r: dict) -> bool:
    """A benign item sharing no commit with a vuln item of the selection."""
    prov = (r.get("facts") or {}).get("provenance")
    return prov is not None and not prov["shares_commit_with"]


def benign_provenance_metrics(recs: list[dict], view: str) -> dict | None:
    """Benign FPR on all benign items and on the ``_clean_benign`` subset, with
    the provenance counts. None without facts."""
    benign = [r for r in recs if r["kind"] == KIND_BENIGN]
    provs = [(r.get("facts") or {}).get("provenance") for r in benign]
    if not benign or any(p is None for p in provs):
        return None
    clean = [r for r in benign if _clean_benign(r)]
    fpr_clean = _rate_of(clean, view, KIND_BENIGN, "fp")
    return {
        "fpr_all": _rate_of(benign, view, KIND_BENIGN, "fp"),
        "fpr_no_shared_commit": fpr_clean,
        "fpr_no_shared_commit_upper95_exact": fpr_upper95_exact(fpr_clean["k"],
                                                                fpr_clean["n"]),
        "n": len(benign),
        "by_bystander_of": dict(Counter(str(p["bystander_of"]) for p in provs)),
        "shares_commit_with_intro": sum(p["shares_commit_with_intro"] for p in provs),
        "shares_commit_with_fix": sum(p["shares_commit_with_fix"] for p in provs),
        "shares_commit_with_any_vuln_item": sum(bool(p["shares_commit_with"]) for p in provs),
        "identical_subset_of_fix_files": sum(p["identical_subset_of_fix_files"] for p in provs),
        "no_shared_commit": len(clean),
    }


def view_metrics(records: list[dict], view: str, base_rates=BASE_RATES) -> dict:
    """Detection metrics of one view over the scored (fully completed) records."""
    recs = [r for r in records if scored(r)]

    def rate(kind, key):
        return _rate_of(recs, view, kind, key)

    tpr = rate(KIND_INTRO, "localised")
    fpr_benign = rate(KIND_BENIGN, "fp")
    upper = fpr_upper95_exact(fpr_benign["k"], fpr_benign["n"])
    pairs = Counter()
    by_pair: dict[str, dict] = {}
    for r in recs:
        if r["kind"] in (KIND_INTRO, KIND_FIX) and r.get("pair_key"):
            by_pair.setdefault(r["pair_key"], {})[r["kind"]] = r
    for p in by_pair.values():
        if KIND_INTRO in p and KIND_FIX in p:
            hit = p[KIND_INTRO]["outcomes"][view]["localised"]
            fp = p[KIND_FIX]["outcomes"][view]["fp"]
            pairs[{(True, False): "intro_only", (True, True): "both",
                   (False, True): "fix_only", (False, False): "neither"}[(hit, fp)]] += 1
    benign_counts = [r["outcomes"][view]["n_findings"] for r in recs if r["kind"] == KIND_BENIGN]

    def breakdown(field):
        out: dict = {}
        for value in sorted({str(r.get(field)) for r in recs}):
            sub = [r for r in recs if str(r.get(field)) == value]
            row = {}
            for kind, key, name in ((KIND_INTRO, "localised", "tpr_localised"),
                                    (KIND_INTRO, "localised_function", "tpr_localised_function"),
                                    (KIND_INTRO, "any", "tpr_any"),
                                    (KIND_FIX, "fp", "fpr_fix"),
                                    (KIND_BENIGN, "fp", "fpr_benign")):
                if any(r["kind"] == kind for r in sub):
                    row[name] = _rate_of(sub, view, kind, key)
            out[value] = row
        return out

    no_fp = bool(fpr_benign["n"]) and not fpr_benign["k"]
    return {
        "n_scored": len(recs),
        "tpr_localised": tpr,
        "tpr_localised_any_vuln_path": rate(KIND_INTRO, "localised_any_vuln_path"),
        "tpr_localised_fix_anchor": rate(KIND_INTRO, "localised_fix_anchor"),
        "tpr_localised_function": rate(KIND_INTRO, "localised_function"),
        "tpr_right_file": rate(KIND_INTRO, "right_file"),
        "tpr_any_finding": rate(KIND_INTRO, "any"),
        "fpr_fix": rate(KIND_FIX, "fp"),
        "fpr_fix_on_fixed_lines": rate(KIND_FIX, "on_fixed_lines"),
        "fpr_fix_on_fix_anchor": rate(KIND_FIX, "on_fix_anchor"),
        "fpr_fix_in_fixed_function": rate(KIND_FIX, "in_fixed_function"),
        "fpr_benign": fpr_benign,
        "fpr_benign_upper95_exact": upper,
        "benign_findings_per_pr": {"mean": _mean(benign_counts),
                                   "histogram": dict(sorted(Counter(benign_counts).items()))},
        # Point estimate at the observed FPR; None when no benign FP was seen
        # (an FPR of exactly 0 would print a meaningless precision of 1.0).
        "precision_at_base_rate": {
            str(pi): precision_at_observed_fpr(tpr["rate"], fpr_benign["k"], fpr_benign["n"],
                                               pi)
            for pi in base_rates},
        "precision_at_base_rate_note": (f"n/a: 0 of {fpr_benign['n']} benign PRs flagged; "
                                        "only the FPR-upper-bound precision is meaningful"
                                        if no_fp else None),
        # Conservative: at the exact (Clopper-Pearson) 95% upper bound of the FPR.
        "precision_at_base_rate_fpr_upper95": {
            str(pi): precision_at_base_rate(tpr["rate"], upper, pi) for pi in base_rates},
        "pairs_intro_localised_vs_fix_fp": {k: pairs[k] for k in
                                            ("intro_only", "both", "fix_only", "neither")},
        "in_scope": scope_metrics(recs, view),
        "leaky_split": leaky_split(recs, view),
        "benign_provenance": benign_provenance_metrics(recs, view),
        "by_category": breakdown("category"),
        "by_language": breakdown("language"),
    }


# ---------------------------------------------------------------------------
# Coverage, the errored-item sensitivity view, bounds
# ---------------------------------------------------------------------------


def coverage(records: list[dict]) -> dict:
    """Per kind: items, fully scored, errored with a partial result, errored
    with no result, not run (so "160 of 200" shows WHICH kinds are missing)."""
    out = {}
    for kind in PR_KINDS:
        rs = [r for r in records if r["kind"] == kind]
        if not rs:
            continue
        out[kind] = {
            "n": len(rs),
            "scored": sum(scored(r) for r in rs),
            "errored_partial": sum(partial_result(r) for r in rs),
            "errored_no_result": sum(r.get("status") == "error" and not partial_result(r)
                                     for r in rs),
            "not_run": sum(r.get("status") == "not_run" for r in rs),
        }
    return out


SENSITIVITY_KEYS = ((KIND_INTRO, "localised", "tpr_localised"),
                    (KIND_INTRO, "localised_function", "tpr_localised_function"),
                    (KIND_INTRO, "any", "tpr_any_finding"),
                    (KIND_FIX, "fp", "fpr_fix"),
                    (KIND_BENIGN, "fp", "fpr_benign"))


def sensitivity(records: list[dict], view: str) -> dict:
    """Two robustness checks around the strict (fully completed) metrics.

    ``with_partial``: the rates over scored items PLUS errored items with a
    partial result, their findings taken as they are (a partial item without
    a finding counts as negative, though a complete run might have found one).

    ``bounds``: over ALL items of the kind, each rate's range when every item
    without a known outcome went either way. A known positive is a positive
    outcome of a scored or partial item (its finding exists whatever the
    failed calls would have said); a known negative is a negative outcome of a
    scored item; everything else (partial without a finding, errored without a
    result, not run) is unknown. ``worst_case`` is the lower end for recall
    (every unknown introducing PR a miss) and the upper end for FP rates
    (every unknown fix / benign PR flagged)."""
    recs = [r for r in records if scored(r) or partial_result(r)]
    out: dict = {"n_items": len(recs), "with_partial": {}, "bounds": {}}
    for kind, key, name in SENSITIVITY_KEYS:
        out["with_partial"][name] = _rate_of(recs, view, kind, key)
        rs = [r for r in records if r["kind"] == kind]
        known_pos = known_neg = 0
        for r in rs:
            value = (r.get("outcomes") or {}).get(view, {}).get(key) if r.get(
                "outcomes") else None
            if value:
                known_pos += 1
            elif value is not None and scored(r):
                known_neg += 1
        n = len(rs)
        if not n or not any(((r.get("outcomes") or {}).get(view) or {}).get(key) is not None
                            for r in rs):
            continue
        lo, hi = known_pos / n, (n - known_neg) / n
        out["bounds"][name] = {
            "n": n, "known_positive": known_pos, "known_negative": known_neg,
            "unknown": n - known_pos - known_neg,
            "lower": round(lo, 4), "upper": round(hi, 4),
            "worst_case": round(lo if kind == KIND_INTRO else hi, 4)}
    return out


# ---------------------------------------------------------------------------
# What the verifier removes; offline verifier policies
# ---------------------------------------------------------------------------

LOSS_KEYS = ((KIND_INTRO, "localised", "intro_catches_lost_strict"),
             (KIND_INTRO, "localised_function", "intro_catches_lost_function"),
             (KIND_INTRO, "any", "intro_any_finding_lost"),
             (KIND_FIX, "fp", "fix_fps_removed"),
             (KIND_BENIGN, "fp", "benign_fps_removed"))


def view_losses(records: list[dict], strict_view: str = "verified",
                loose_view: str = "audit_only") -> dict:
    """Per kind, items positive in ``loose_view`` but not in ``strict_view``
    (and the reverse, which should be 0 when the strict view is a subset), over
    the scored records. For verified vs audit_only this is what the verifier
    drops: introducing catches lost, fix / benign false alarms removed. (A
    McNemar test between two nested views of one run is degenerate: the strict
    view is a subset by construction, so every discordant pair goes one way.)"""
    recs = [r for r in records if scored(r)]
    out: dict = {"strict_view": strict_view, "loose_view": loose_view}
    for kind, key, name in LOSS_KEYS:
        lost = gained = n = 0
        ids = []
        for r in recs:
            if r["kind"] != kind:
                continue
            a = r["outcomes"][strict_view].get(key)
            b = r["outcomes"][loose_view].get(key)
            if a is None or b is None:
                continue
            n += 1
            if b and not a:
                lost += 1
                ids.append(r["base_id"])
            elif a and not b:
                gained += 1
        out[name] = {"n": n, "count": lost, "strict_only": gained, "ids": ids}
    return out


def _policy_confirmed_at_least(k: int):
    return lambda c: c.get("verdict") == "confirmed" and (c.get("verdict_confidence") or 0) >= k


VERIFIER_POLICIES = (
    *((f"confirmed>={k}", _policy_confirmed_at_least(k)) for k in (9, 8, 7, 6, 5)),
    ("confirmed (any confidence)", lambda c: c.get("verdict") == "confirmed"),
    ("confirmed or uncertain", lambda c: c.get("verdict") in ("confirmed", "uncertain")),
    ("no verifier (every verifier-bound candidate)", lambda c: True),
)


def policy_findings(rec: dict, keep) -> list[dict]:
    """The findings a verifier decision policy ``keep(candidate)`` would report:
    the deterministic guard_diff findings plus the verifier-bound candidates it
    keeps (replayed from the cached verdicts: no LLM call)."""
    guard = [f for f in rec["findings"].get("verified") or [] if f.get("source") == "guard_diff"]
    return guard + [c for c in rec.get("candidates") or []
                    if c.get("status") in VERIFIER_BOUND and keep(c)]


def policy_curve(records: list[dict], tol: int) -> list[dict]:
    """Every ``VERIFIER_POLICIES`` entry over the scored records: introducing
    catches (strict / function-level), fix PRs flagged, benign PRs flagged."""
    recs = [r for r in records if scored(r) and r.get("candidates") is not None]
    rows = []
    for name, keep in VERIFIER_POLICIES:
        outs = [(r, score_findings(r, policy_findings(r, keep), tol)) for r in recs]

        def rate(kind, key, outs=outs):
            return _rate([bool(o[key]) for r, o in outs if r["kind"] == kind
                          and o.get(key) is not None])

        rows.append({"policy": name,
                     "intro_localised": rate(KIND_INTRO, "localised"),
                     "intro_localised_function": rate(KIND_INTRO, "localised_function"),
                     "fix_flagged": rate(KIND_FIX, "fp"),
                     "benign_flagged": rate(KIND_BENIGN, "fp")})
    return rows


# Heuristic buckets for the verifier's free-text reason on rejected / uncertain
# candidates (first match wins).
REASON_BUCKETS = (
    ("control_present", r"sanitiz|saniti[sz]|escap|validat|allowlist|whitelist|check(ed|s)? "
                        r"(before|for)|guard|normali[sz]"),
    ("not_attacker_controlled", r"not (user|attacker)[- ]controlled|trusted|internal|constant|"
                                r"hard[- ]?coded|not reachable from|no (user|external) input"),
    ("pre_existing", r"pre-?existing|already (present|there|existed)|not introduced|"
                     r"before (this|the) change"),
    ("unreachable", r"unreachable|dead code|never called|not reachable"),
    ("test_or_example", r"\btests?\b|example|fixture|mock"),
    ("insufficient_evidence", r"speculat|insufficient|cannot (confirm|determine)|unclear|"
                              r"no evidence|not enough"),
)


def reason_bucket(text: str | None) -> str:
    t = (text or "").lower()
    for name, pattern in REASON_BUCKETS:
        if re.search(pattern, t):
            return name
    return "other" if t else "no_reason"


def funnel_stats(rec: dict) -> dict:
    """The record's ``pr_review`` stats with ``confirmed`` = reported findings
    (candidates with status "confirmed"), the meaning ``run_pr_review`` gives it
    now. Runs recorded before the stats partitioned the candidates counted
    every confirmed verdict there, below_min_confidence ones included; the
    candidate statuses were partitioned all along, so counting them gives one
    meaning for old and new results alike."""
    stats = dict(rec.get("pr_review") or {})
    cands = rec.get("candidates")
    if cands is not None:
        stats["confirmed"] = sum(1 for c in cands if c.get("status") == "confirmed")
    return stats


def operational_metrics(records: list[dict], arm: str) -> dict:
    """Cost, coverage and funnel over the scored records."""
    recs = [r for r in records if scored(r)]
    calls = [r["calls"]["total"] for r in recs]
    tokens = [r["calls"]["usage_total_tokens"] or r["calls"]["prompt_tokens_est"] for r in recs]
    latency = [r["calls"]["latency_s"] for r in recs]
    out = {
        "calls_per_pr": {"mean": _mean(calls), "p90": _p90(calls), "total": sum(calls)},
        "tokens_per_pr": {"mean": _mean(tokens), "p90": _p90(tokens), "total": sum(tokens),
                          "note": "provider usage (prompt + completion) where recorded, else "
                                  "the prompt estimate"},
        "prompt_tokens_est_per_pr": {
            "mean": _mean([r["calls"]["prompt_tokens_est"] for r in recs]),
            "p90": _p90([r["calls"]["prompt_tokens_est"] for r in recs])},
        "latency_s_per_pr": {"mean": _mean(latency), "p90": _p90(latency)},
        "review_status": dict(Counter(r.get("review_status") for r in recs)),
        "not_reviewed_reasons": dict(sum((Counter(r.get("not_reviewed_reasons") or {})
                                          for r in recs), Counter())),
    }
    if arm in PR_ARMS:
        roles = Counter()
        for r in recs:
            roles.update(r["calls"]["by_role"])
        out["calls_by_role"] = dict(roles)
        funnel = Counter()
        for r in recs:
            stats = funnel_stats(r)
            for k in ("candidates", "quote_not_found", "hard_excluded", "below_audit_confidence",
                      "confirmed", "rejected", "uncertain", "below_min_confidence",
                      "unverified", "bad_output", "context_rounds_used", "context_requested",
                      "context_resolved", "audit_calls", "verifier_calls"):
                funnel[k] += stats.get(k) or 0
        out["funnel"] = dict(funnel)
        out["context"] = {
            "rounds_used": funnel["context_rounds_used"],
            "prs_with_context_rounds": sum(
                1 for r in recs if (r.get("pr_review") or {}).get("context_rounds_used")),
            "symbols_requested": funnel["context_requested"],
            "symbols_resolved": funnel["context_resolved"],
        }
        statuses = Counter()
        reasons = Counter()
        for r in recs:
            for c in r.get("candidates") or []:
                statuses[c["status"] + (f": {c['status_reason']}" if c.get("status_reason")
                                        else "")] += 1
                if c["status"] in ("rejected", "uncertain", "below_min_confidence"):
                    reasons[reason_bucket(c.get("verdict_reason"))] += 1
        out["candidate_statuses"] = dict(statuses.most_common())
        out["verifier_rejection_reasons_heuristic"] = dict(reasons.most_common())
    return out


def summarize(records: list[dict], arm: str, tol: int) -> dict:
    attach_outcomes(records, tol)
    views = VIEWS_BY_ARM[arm]
    out = {
        "status_counts": dict(Counter(r["status"] for r in records)),
        "coverage": coverage(records),
        "views": {v: view_metrics(records, v) for v in views},
        "sensitivity": {v: sensitivity(records, v) for v in views},
        "operational": operational_metrics(records, arm),
    }
    if arm in PR_ARMS:
        out["verifier_losses"] = view_losses(records, "verified", "audit_only")
        out["verifier_policy_curve"] = policy_curve(records, tol)
    return out


def _fmt_opt(r: dict) -> str:
    return "n/a (no facts: rescore with the dataset)" if not r["n"] else _fmt_rate(r)


def _frac(r: dict) -> str:
    return f"{r['k']}/{r['n']}" if r["n"] else "n/a"


def print_view(title: str, m: dict) -> None:
    print(f"\n{title} (n scored={m['n_scored']}):")
    print(f"  introducing, localised TP (primary)  {_fmt_rate(m['tpr_localised'])}")
    print(f"  introducing, localised any vuln path {_fmt_rate(m['tpr_localised_any_vuln_path'])}")
    print(f"  introducing, change-anchored         {_fmt_opt(m['tpr_localised_fix_anchor'])}")
    print(f"  introducing, function-level          {_fmt_opt(m['tpr_localised_function'])}")
    print(f"  introducing, right file              {_fmt_rate(m['tpr_right_file'])}")
    print(f"  introducing, any finding             {_fmt_rate(m['tpr_any_finding'])}")
    print(f"  fix PRs, any finding (FP)            {_fmt_rate(m['fpr_fix'])}")
    print(f"  fix PRs, finding on the fixed lines  {_fmt_rate(m['fpr_fix_on_fixed_lines'])}")
    print(f"  fix PRs, finding on a fix anchor     {_fmt_opt(m['fpr_fix_on_fix_anchor'])}")
    print(f"  fix PRs, finding in a fixed function {_fmt_opt(m['fpr_fix_in_fixed_function'])}")
    print(f"  benign PRs, alert rate (FP)          {_fmt_rate(m['fpr_benign'])}"
          f"  (findings/PR {m['benign_findings_per_pr']['mean']}; exact 95% upper "
          f"{m['fpr_benign_upper95_exact']})")
    point = m["precision_at_base_rate_note"] or m["precision_at_base_rate"]
    print(f"  precision @ base rate (point)        {point}")
    print(f"  precision @ base rate, FPR at upper  {m['precision_at_base_rate_fpr_upper95']}")
    print(f"  pairs (intro localised vs fix FP)    {m['pairs_intro_localised_vs_fix_fp']}")
    sc = m["in_scope"]
    print(f"  in-scope recall (by scope {sc['by_scope']}):")
    for name, label in (("all", "all items"), ("excl_dos", "excluding DoS"),
                        ("in_scope_only", "excluding DoS + timing/race")):
        g = sc[name]
        print(f"    {label:<29} strict {_frac(g['localised'])}, function-level "
              f"{_frac(g['localised_function'])}, any {_frac(g['any'])}")
    lk = m["leaky_split"]
    if lk:
        print(f"  leaky diff split (deleted lines with security vocabulary: {lk['n_leaky']} "
              f"leaky / {lk['n_not_leaky']} not):")
        for key, label in INTRO_KEYS:
            row = lk[key]
            print(f"    {label:<16} leaky {_frac(row['leaky'])}, not leaky "
                  f"{_frac(row['not_leaky'])}; exact Fisher p = {row['p_fisher_exact']:.3g}")
    bp = m["benign_provenance"]
    if bp:
        print(f"  benign provenance: {bp['n']} bystanders {bp['by_bystander_of']}; share a "
              f"commit with an intro item {bp['shares_commit_with_intro']}, with a fix item "
              f"{bp['shares_commit_with_fix']}; identical subset of the fix item's files "
              f"{bp['identical_subset_of_fix_files']}; no shared commit {bp['no_shared_commit']}")
        print(f"    benign FP, all {_fmt_rate(bp['fpr_all'])}; no shared commit "
              f"{_fmt_rate(bp['fpr_no_shared_commit'])} (exact 95% upper "
              f"{bp['fpr_no_shared_commit_upper95_exact']})")


def print_summary(summary: dict, arm: str) -> None:
    print(f"\nStatuses: {summary['status_counts']}")
    print("Coverage by kind (n / scored / errored with partial result / errored, no result / "
          "not run):")
    for kind, c in (summary.get("coverage") or {}).items():
        print(f"  {kind:<17} {c['n']:>3} / {c['scored']:>3} / {c['errored_partial']:>3} / "
              f"{c['errored_no_result']:>3} / {c['not_run']:>3}")
    labels = {"verified": f"{arm}: verified (the report)" if arm in PR_ARMS else "units",
              "audit_only": f"{arm}: audit only (candidates before verification)"}
    for view in HEADLINE_VIEWS[arm]:
        print_view(labels[view], summary["views"][view])
        sens = (summary.get("sensitivity") or {}).get(view)
        if sens:
            wp = sens["with_partial"]
            print(f"  sensitivity, + {sens['n_items'] - summary['views'][view]['n_scored']} "
                  f"errored items' partial results: "
                  + ", ".join(f"{name} {_frac(wp[name])}" for _, _, name in SENSITIVITY_KEYS))
            print("  bounds over ALL items (unknown outcomes either way; worst case first):")
            for name, b in sens["bounds"].items():
                print(f"    {name:<24} worst {b['worst_case']:.3f}, range [{b['lower']:.3f}, "
                      f"{b['upper']:.3f}] (known +{b['known_positive']} / "
                      f"-{b['known_negative']}, unknown {b['unknown']} of {b['n']})")
    if arm in PR_ARMS and summary.get("verifier_losses"):
        vl = summary["verifier_losses"]
        print("\nWhat the verifier drops (audit_only positive, verified negative; scored items):")
        for _, _, name in LOSS_KEYS:
            row = vl[name]
            print(f"  {name:<28} {row['count']} of {row['n']}"
                  + (f" (verified-only {row['strict_only']})" if row["strict_only"] else ""))
        print("\nVerifier policy curve (cached verdicts replayed, no LLM call; scored items):")
        print(f"  {'policy':<46} {'intro strict':>12} {'intro fn':>9} {'fix':>7} {'benign':>7}")
        for row in summary.get("verifier_policy_curve") or []:
            print(f"  {row['policy']:<46} {_frac(row['intro_localised']):>12} "
                  f"{_frac(row['intro_localised_function']):>9} {_frac(row['fix_flagged']):>7} "
                  f"{_frac(row['benign_flagged']):>7}")
    op = summary["operational"]
    print(f"\nCost per PR: calls {op['calls_per_pr']}, tokens {op['tokens_per_pr']['mean']} "
          f"(p90 {op['tokens_per_pr']['p90']}), latency {op['latency_s_per_pr']}")
    print(f"review_status {op['review_status']}; not reviewed {op['not_reviewed_reasons']}")
    if arm in PR_ARMS:
        print(f"Funnel {op['funnel']}")
        print(f"Context {op['context']}")
        print(f"Verifier reasons (heuristic buckets) {op['verifier_rejection_reasons_heuristic']}")


# ---------------------------------------------------------------------------
# Offline: rescore, compare
# ---------------------------------------------------------------------------


def refresh_facts(records: list[dict], items: list[dict], parser=None) -> dict:
    """Recompute every record's ``facts`` from the dataset ``items`` (matched by
    base id; ``pr_eval_facts.selection_facts`` over the items) and refresh its
    ``category`` / ``cwe`` from the dataset (labels may have been corrected
    since the run: ``build_pr_eval.LABEL_OVERRIDES``). Returns what changed."""
    facts = selection_facts(items, parser)
    changes, missing = [], []
    for r in records:
        f = facts.get(r["base_id"])
        if f is None:
            missing.append(r["base_id"])
            continue
        r["facts"] = f
        new = f["labels"]
        if (r.get("category"), r.get("cwe")) != (new["category"], new["cwe"]):
            changes.append({"id": r["base_id"],
                            "from": {"category": r.get("category"), "cwe": r.get("cwe")},
                            "to": {"category": new["category"], "cwe": new["cwe"]},
                            "reason": (new.get("label_override") or {}).get("reason")})
            r["category"], r["cwe"] = new["category"], new["cwe"]
    return {"facts": "recomputed from the dataset", "label_changes": changes,
            "records_without_dataset_item": missing}


def rescore(data: dict, tol: int, items: list[dict] | None = None, parser=None) -> dict:
    """Recompute every metric from the stored findings. With the run's dataset
    ``items``, the facts (change anchors, scope labels, leaky-diff terms,
    benign provenance) are recomputed first (``refresh_facts``); without, the
    records' stored facts are used (none in runs older than the facts)."""
    records = [dict(r) for r in data["items"]]
    meta = (refresh_facts(records, items, parser) if items is not None
            else {"facts": "as stored in the records"})
    summary = summarize(records, data["config"]["arm"], tol)
    return {**data, "config": {**data["config"], "localise_tolerance": tol},
            "rescore": meta, "summary": summary, "items": records}


def run_datasets(data: dict, fallback) -> list[Path]:
    """The dataset file(s) a saved run selected from (``config.selection``),
    else ``fallback``; only those that exist."""
    sel = (data.get("config") or {}).get("selection") or {}
    paths = [BASE_DIR / p if not Path(p).is_absolute() else Path(p)
             for p in sel.get("datasets") or []] or _as_paths(fallback)
    return [p for p in paths if p.exists()]


def load_run_items(data: dict, fallback) -> tuple[list[dict] | None, dict]:
    """The dataset items of a saved run's records (by base id), with the files'
    sha256; (None, meta) when no dataset file is available."""
    paths = run_datasets(data, fallback)
    if not paths:
        return None, {"datasets": [], "note": "dataset not found: facts not recomputed"}
    ids = {r["base_id"] for r in data["items"]}
    items = load_items(paths, ids=ids)
    return items, {"datasets": [_rel(p) for p in paths],
                   "sha256": {_rel(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}}


def _flat(records: list[dict], view: str, tol: int) -> dict[str, dict]:
    out = {}
    for r in records:
        if not scored(r):
            continue
        fs = r["findings"].get(view)
        if fs is None:
            raise ValueError(f"view {view!r} not in the run's findings "
                             f"({', '.join(r['findings'])})")
        out[r["base_id"]] = {"kind": r["kind"], **score_findings(r, fs, tol)}
    return out


def paired(a: dict[str, dict], b: dict[str, dict], kind: str, key: str) -> dict:
    ids = sorted(i for i in a if i in b and a[i]["kind"] == kind)
    pairs = [(bool(a[i][key]), bool(b[i][key])) for i in ids]
    c = Counter(pairs)
    a_only, b_only = c[(True, False)], c[(False, True)]
    return {"kind": kind, "key": key, "n": len(pairs),
            "a": _exact_rate([x for x, _ in pairs]), "b": _exact_rate([y for _, y in pairs]),
            "both": c[(True, True)], "a_only": a_only, "b_only": b_only,
            "neither": c[(False, False)],
            "p_mcnemar_exact": round(mcnemar_exact(a_only, b_only), 6)}


# The PR arm's views, narrowest first: each keeps a subset of the next one's
# findings by construction (verified = confirmed candidates + guard alerts).
NESTED_VIEWS = ("verified", "audit_only", "audit_raw")


def same_run(a: dict, b: dict) -> bool:
    """Two saved results of the same run: same config and the same records."""
    return a.get("config") == b.get("config") and [r["id"] for r in a["items"]] == [
        r["id"] for r in b["items"]]


def compare_runs(a_recs: list[dict], b_recs: list[dict], view_a: str = "verified",
                 view_b: str = "verified", tol: int = DEFAULT_LOCALISE_TOLERANCE,
                 same: bool = False) -> dict:
    """Paired comparison by base id: exact McNemar on introducing localised TP
    (and any-finding TP) and on fix FP; exact binomial CIs (all scored) plus
    the paired table on benign FPR.

    ``same`` (one run, two of its nested views, e.g. verified vs audit_only):
    no McNemar, which would be degenerate (the narrower view is a subset by
    construction, so every discordant pair points one way and the "test" only
    measures how much the verifier removed); the result is ``view_losses``
    instead: catches lost and false alarms removed per kind."""
    if same and view_a != view_b and {view_a, view_b} <= set(NESTED_VIEWS):
        strict, loose = sorted((view_a, view_b), key=NESTED_VIEWS.index)
        recs = [dict(r) for r in a_recs]
        attach_outcomes(recs, tol)
        return {"view_a": view_a, "view_b": view_b, "localise_tolerance": tol,
                "nested_views": True,
                "note": f"{strict} is a subset of {loose} in one run: losses, not McNemar",
                "losses": view_losses(recs, strict, loose)}
    a, b = _flat(a_recs, view_a, tol), _flat(b_recs, view_b, tol)
    return {
        "view_a": view_a, "view_b": view_b, "localise_tolerance": tol,
        "n_common": len(set(a) & set(b)), "n_only_in_a": len(set(a) - set(b)),
        "n_only_in_b": len(set(b) - set(a)),
        "intro_localised_tp": paired(a, b, KIND_INTRO, "localised"),
        "intro_any_finding_tp": paired(a, b, KIND_INTRO, "any"),
        "fix_fp": paired(a, b, KIND_FIX, "fp"),
        "benign_fpr": {
            "a": _exact_rate([bool(v["fp"]) for v in a.values() if v["kind"] == KIND_BENIGN]),
            "b": _exact_rate([bool(v["fp"]) for v in b.values() if v["kind"] == KIND_BENIGN]),
            "paired": paired(a, b, KIND_BENIGN, "fp"),
        },
    }


def print_comparison(label_a: str, label_b: str, cmp: dict) -> None:
    print(f"\nPaired comparison (by base id)  A = {label_a} [{cmp['view_a']}]\n"
          f"                                B = {label_b} [{cmp['view_b']}]")
    if cmp.get("nested_views"):
        print(f"  Nested views of one run ({cmp['note']}):")
        for _, _, name in LOSS_KEYS:
            row = cmp["losses"][name]
            print(f"    {name:<28} {row['count']} of {row['n']}")
        return
    print(f"  common {cmp['n_common']} (only in A {cmp['n_only_in_a']}, only in B "
          f"{cmp['n_only_in_b']})")
    for title, t in (("Introducing, localised TP (primary)", cmp["intro_localised_tp"]),
                     ("Introducing, any-finding TP", cmp["intro_any_finding_tp"]),
                     ("Fix PRs, FP", cmp["fix_fp"]),
                     ("Benign, FP (paired)", cmp["benign_fpr"]["paired"])):
        print(f"  {title} (n={t['n']}): A {_fmt_exact(t['a'])} | B {_fmt_exact(t['b'])}")
        print(f"    discordant: A only {t['a_only']}, B only {t['b_only']} (both {t['both']}, "
              f"neither {t['neither']}); exact McNemar p = {t['p_mcnemar_exact']:.4g}")
    o = cmp["benign_fpr"]
    print(f"  Benign FPR, exact 95% CI (all scored): A {_fmt_exact(o['a'])} | "
          f"B {_fmt_exact(o['b'])}")


# ---------------------------------------------------------------------------
# Dry run: every prompt, no network
# ---------------------------------------------------------------------------


def _first_changed_line(f: dict) -> tuple[int, str] | None:
    """(new-file line, text) of the file's first added line; for a
    deletion-only diff, the first new-file line after a deletion."""
    after_deletion = None
    for n, text in diff_lines(f.get("patch")):
        body = text[1:].strip()
        if n is not None and text.startswith("+") and len(body) >= 3:
            return n, body
        if text.startswith("-"):
            after_deletion = after_deletion or "pending"
        elif after_deletion == "pending" and n is not None and len(body) >= 3:
            after_deletion = (n, body)
    return after_deletion if isinstance(after_deletion, tuple) else None


class DryRunRouter:
    """Stub LLM (no network). Audit: one synthetic candidate per file of the
    prompt (its first changed line quoted, confidence 9) so the verifier
    prompts are built for real; verifier: "rejected"; units: no findings."""

    mock = False
    label = "dry-run"

    def __init__(self, item: dict):
        self.files = [f for f in item["files"] if f.get("new_content") is not None]

    async def generate(self, system, user, *, deadline=None, validate=None):
        if system == AUDIT_SYSTEM_PROMPT:
            listed = next((ln for ln in user.splitlines()
                           if ln.startswith("Files in this request:")), "")
            out = []
            for f in self.files:
                first = _first_changed_line(f)
                if first and f"`{f['path']}`" in listed:
                    out.append({"file": f["path"], "line": first[0], "quoted_code": first[1],
                                "severity": "high", "cwe": "CWE-20",
                                "title": "Dry-run candidate", "confidence": 9})
            return {"findings": out}, "dry-run"
        if system == VERIFIER_SYSTEM_PROMPT:
            return {"verdict": "rejected", "confidence": 2, "reason": "dry run"}, "dry-run"
        return {"findings": []}, "dry-run"


def context_call_bound(n_chunks: int, max_audit_calls: int, rounds: int) -> list[int]:
    """Extra (context-round) audit calls each chunk could make at most, by
    ``run_pr_review``'s rule "keep one call for every later chunk"."""
    calls_left = max(max_audit_calls, 0)
    out = []
    for i in range(n_chunks):
        if calls_left <= 0:
            out.append(0)
            continue
        later = n_chunks - i - 1
        extra = min(rounds, max(calls_left - 1 - later, 0))
        out.append(extra)
        calls_left -= 1 + extra
    return out


def dry_run_item_estimate(rec: dict, arm: str, config: PRReviewConfig) -> dict:
    """Calls / prompt tokens of one dry-run item: ``floor`` (audit only, no
    candidate, no context), ``scenario`` (the stub's one candidate per file,
    all verified) and ``ceiling`` (every context round the budget allows at
    +CONTEXT_MAX_TOKENS each, and the full verifier budget at
    PR_REVIEW_MAX_PROMPT_TOKENS per call)."""
    log = rec["call_log"]
    audit = [e["prompt_tokens_est"] for e in log if e["role"] in ("audit", "units")]
    verifier = [e["prompt_tokens_est"] for e in log if e["role"] == "verifier"]
    out = {"audit_calls": len(audit), "audit_prompt_tokens": audit,
           "verifier_calls_scenario": len(verifier), "verifier_prompt_tokens": verifier}
    floor_tokens = sum(audit)
    if arm not in PR_ARMS:
        out.update(floor={"calls": len(audit), "tokens": floor_tokens},
                   scenario={"calls": len(audit), "tokens": floor_tokens},
                   ceiling={"calls": len(audit), "tokens": floor_tokens})
        return out
    extras = context_call_bound(len(audit), config.max_audit_calls, config.context_rounds)
    ctx_tokens = sum(x * (t + config.context_max_tokens) for x, t in zip(extras, audit,
                                                                        strict=False))
    vmax = config.max_verifier_calls
    out.update(
        context_calls_upper=sum(extras),
        verifier_calls_upper=vmax,
        floor={"calls": len(audit), "tokens": floor_tokens},
        scenario={"calls": len(audit) + len(verifier), "tokens": floor_tokens + sum(verifier)},
        ceiling={"calls": len(audit) + sum(extras) + vmax,
                 "tokens": floor_tokens + ctx_tokens + vmax * config.max_prompt_tokens},
    )
    return out


def pacing_seconds(call_tokens: list[int], sleep_s: float, tpm: int | None,
                   latency_s: float) -> float:
    """Wall time of calls made one after another: each call's latency, plus
    the gate's wait before every call but the first (max(sleep, 60 x the
    previous call's tokens / tpm))."""
    total = latency_s * len(call_tokens)
    for prev in call_tokens[:-1]:
        total += max(sleep_s, 60.0 * prev / tpm if tpm else 0.0)
    return total


def summarize_dry_run(records: list[dict], arm: str, config: PRReviewConfig, *,
                      completion_tokens: int, latency_s: float, max_calls: float | None,
                      token_budget: int | None) -> dict:
    ests = [r["dry_run"] for r in records]
    per_call = [t for r in records for t in (r["dry_run"]["audit_prompt_tokens"]
                                             + r["dry_run"]["verifier_prompt_tokens"])]
    out: dict = {"n_items": len(records),
                 "assumed_completion_tokens_per_call": completion_tokens,
                 "assumed_latency_s_per_call": latency_s,
                 "prompt_tokens_per_call": {"mean": _mean(per_call), "p90": _p90(per_call),
                                            "max": max(per_call) if per_call else None}}
    for scen in ("floor", "scenario", "ceiling"):
        calls = [e[scen]["calls"] for e in ests]
        prompt = [e[scen]["tokens"] for e in ests]
        total_calls = sum(calls)
        total_tokens = sum(prompt) + completion_tokens * total_calls
        flat: list[float] = []
        for e, c in zip(ests, calls, strict=True):
            avg = (e[scen]["tokens"] / c if c else 0) + completion_tokens
            flat.extend([avg] * c)
        cut = None
        if max_calls is not None or token_budget is not None:
            run_calls = run_tokens = 0
            for i, e in enumerate(ests):
                run_calls += e[scen]["calls"]
                run_tokens += e[scen]["tokens"] + completion_tokens * e[scen]["calls"]
                if (max_calls is not None and run_calls > max_calls) or (
                        token_budget is not None and run_tokens > token_budget):
                    cut = {"first_item_not_covered": i, "items_covered": i}
                    break
        out[scen] = {
            "calls_per_pr": {"mean": _mean(calls), "p90": _p90(calls), "max": max(calls)
                             if calls else None},
            "prompt_tokens_per_pr": {"mean": _mean(prompt), "p90": _p90(prompt)},
            "total_requests": total_calls,
            "total_prompt_tokens": sum(prompt),
            "total_tokens_incl_completion": total_tokens,
            "budget_cutoff": cut,
            "wall_time_h": {
                name: {
                    "pacing_only": round(pacing_seconds(flat, p["sleep_s"], p["tpm"], 0) / 3600,
                                         2),
                    "with_latency": round(pacing_seconds(flat, p["sleep_s"], p["tpm"],
                                                         latency_s) / 3600, 2),
                    "days_at_requests_per_day": math.ceil(total_calls / p["requests_per_day"])
                    if total_calls else 0,
                } for name, p in PACING_PROFILES.items()},
        }
    partial = [r["id"] for r in records if r.get("review_status") != "complete"
               and (r.get("not_reviewed_reasons") or r.get("units_partially_reviewed"))]
    out["pipeline_partial_items"] = {
        "n": len(partial), "ids": partial[:50],
        "reasons": dict(sum((Counter(r.get("not_reviewed_reasons") or {}) for r in records),
                            Counter())),
        "units_partially_reviewed": sum(r.get("units_partially_reviewed") or 0
                                        for r in records),
    }
    out["review_status_stub"] = dict(Counter(r.get("review_status") for r in records))
    if arm in PR_ARMS:
        out["context_calls_upper_total"] = sum(e.get("context_calls_upper", 0) for e in ests)
        out["verifier_unverified_in_scenario"] = sum(
            (r.get("pr_review") or {}).get("unverified") or 0 for r in records)
    return out


async def dry_run_records(items: list[dict], arm: str, *, parser, semgrep_cache=None,
                          config: PRReviewConfig | None = None, progress: bool = False,
                          keep_prompts: bool = False) -> list[dict]:
    config = config or pr_review_config()
    gate = EvalGate(cache=None, max_calls=math.inf, token_budget=None, sleep_s=0.0, tpm=0,
                    max_rate_limit_errors=10**9, temperature=0.0, keep_prompts=keep_prompts)
    out = []
    for n, item in enumerate(items, start=1):
        stub = GatedRouter(DryRunRouter(item), gate)
        scanner = _item_scanner(semgrep_cache, item)
        rec = await run_item(item, arm, router=stub, verifier_router=stub, gate=gate,
                             parser=parser, semgrep_scanner=scanner, config=config)
        rec["dry_run"] = dry_run_item_estimate(rec, arm, config)
        out.append(rec)
        if progress and n % 25 == 0:
            print(f"  dry-run {n}/{len(items)}", flush=True)
    return out


def _dry_worker(items: list[dict], arm: str, semgrep_items: dict | None,
                keep_prompts: bool) -> list[dict]:
    cache = {"items": semgrep_items} if semgrep_items is not None else None
    return asyncio.run(dry_run_records(items, arm, parser=_parser(), semgrep_cache=cache,
                                       keep_prompts=keep_prompts))


def dry_run_all(items: list[dict], arm: str, *, semgrep_cache: dict | None = None,
                keep_prompts: bool = False, workers: int = 1) -> list[dict]:
    """``dry_run_records`` over ``items``, split across ``workers`` processes
    (the dry run is CPU-bound: tree-sitter + guard_diff, ~1.4 s per PR);
    records come back in item order."""
    if workers <= 1 or len(items) < 2 * workers:
        return asyncio.run(dry_run_records(items, arm, parser=_parser(),
                                           semgrep_cache=semgrep_cache, progress=True,
                                           keep_prompts=keep_prompts))
    from concurrent.futures import ProcessPoolExecutor

    chunks = [items[i::workers] for i in range(workers)]
    subsets = [None if semgrep_cache is None else
               {base_id(it["id"]): semgrep_cache["items"].get(base_id(it["id"])) or {}
                for it in chunk} for chunk in chunks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_dry_worker, chunks, [arm] * workers, subsets,
                              [keep_prompts] * workers))
    by_id = {r["id"]: r for part in parts for r in part}
    return [by_id[it["id"]] for it in items]


def _item_scanner(semgrep_cache: dict | None, item: dict):
    if semgrep_cache is None:
        return None
    return CachedSemgrepScanner(semgrep_cache["items"].get(base_id(item["id"])) or {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PR-level eval: pr vs units review on PR items.")
    p.add_argument("--dataset", nargs="+", default=[str(DEFAULT_DATASET)],
                   help="PR eval JSONL file(s) (default: the 200-item dev sample).")
    p.add_argument("--split", choices=SPLITS, default=None,
                   help=f"Keep only items of this split ('test' also needs {TEST_SPLIT_FLAG}).")
    p.add_argument(TEST_SPLIT_FLAG, dest="allow_test_split", action="store_true",
                   help="Allow --split test (the held-out test set; tune on dev).")
    p.add_argument("--sample-kinds", default=None, metavar="KIND=N,...",
                   help="Stratified sample, e.g. vulnerable=30,fix=30,benign=40 (pairs kept "
                        "together; deterministic with --seed).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arm", choices=ARMS, default="pr")
    p.add_argument("--misleading-dataset", default=str(DEFAULT_MISLEADING),
                   help="Where --arm pr_misleading finds the _misleading variants.")
    p.add_argument("--out", default=None, help="Result JSON (required for a real run).")
    p.add_argument("--localise-tolerance", type=int, default=DEFAULT_LOCALISE_TOLERANCE,
                   metavar="N")
    p.add_argument("--semgrep-cache", default=None, metavar="PATH",
                   help="Precomputed Semgrep hits (see --semgrep-precompute); without it "
                        "there are no Semgrep leads / evidence.")
    p.add_argument("--semgrep-precompute", action="store_true",
                   help="Run the Semgrep engine ONCE over every file of the selected items "
                        "and write --semgrep-cache (no LLM).")
    p.add_argument("--semgrep-timeout", type=float, default=3600.0, metavar="S",
                   help="Engine timeout for --semgrep-precompute (default 3600).")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true",
                       help="Build every prompt with a stub router (no network) and report "
                            "calls / tokens / time estimates.")
    modes.add_argument("--rescore", default=None, metavar="RESULT.json",
                       help="Offline: recompute a saved run's metrics (e.g. another "
                            "--localise-tolerance).")
    modes.add_argument("--compare", nargs=2, default=None, metavar=("A.json", "B.json"),
                       help="Offline: paired exact tests between two saved runs.")
    p.add_argument("--view-a", default="verified", choices=("verified", "audit_only",
                                                            "audit_raw"))
    p.add_argument("--view-b", default="verified", choices=("verified", "audit_only",
                                                            "audit_raw"))
    p.add_argument("--dry-run-completion-tokens", type=int,
                   default=settings.LLM_OUTPUT_TOKENS_ESTIMATE, metavar="T",
                   help="Assumed completion tokens per call in the dry-run totals (default "
                        "LLM_OUTPUT_TOKENS_ESTIMATE).")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="--dry-run only: worker processes (CPU-bound, ~80 MB each; default 4).")
    p.add_argument("--dry-run-latency-s", type=float, default=DEFAULT_DRY_RUN_LATENCY_S,
                   metavar="S", help="Assumed provider latency per call for the wall-time "
                                     "estimate.")
    llm = p.add_argument_group("LLM calls (real provider calls)")
    llm.add_argument("--llm-model", default=None, metavar="PROVIDER:MODEL",
                     help="Pin exactly this model (no fallback), e.g. "
                          "openrouter:qwen/qwen3.8-27b:free; else the production chain.")
    llm.add_argument("--llm-primary-only", action="store_true",
                     help="Without --llm-model: only the chain's primary model.")
    llm.add_argument("--llm-upstream", default=None, metavar="SLUG",
                     help="OpenRouter only: pin the upstream (provider.order=[SLUG]); applies "
                          "to every OpenRouter request of the run, verifier included.")
    llm.add_argument("--verifier-model", default=None, metavar="PROVIDER:MODEL",
                     help="Pin the verifier to this model (default: the audit's router).")
    llm.add_argument("--llm-temperature", type=float, default=DEFAULT_EVAL_TEMPERATURE,
                     metavar="T",
                     help="Force this sampling temperature on every call. Default: no "
                          "override, each model's recommended sampling (settings.LLM_SAMPLING, "
                          "e.g. qwen/qwen3.8-27b: temperature 1.0, top_p 0.95, top_k 20), as "
                          "production sends. Part of the cache key.")
    llm.add_argument("--llm-repeat-index", type=int, default=0, metavar="R",
                     help="Cache repeat index of this run's calls (default 0). With stochastic "
                          "sampling, run the same selection with R = 0, 1, ... (one --out "
                          "each) for independent samples of every call; --compare pairs them.")
    llm.add_argument("--llm-max-calls", type=int, default=None, metavar="N",
                     help=f"Hard cap on real calls this run (1..{LLM_MAX_CALLS_CAP}; cached "
                          "calls don't count). Required for a real run.")
    llm.add_argument("--llm-token-budget", type=int, default=None, metavar="T")
    llm.add_argument("--llm-sleep", type=float, default=DEFAULT_LLM_SLEEP, metavar="S")
    llm.add_argument("--llm-tpm", type=int, default=DEFAULT_LLM_TPM, metavar="T",
                     help=f"Pace to T tokens/min (default {DEFAULT_LLM_TPM}; 0 = off, e.g. "
                          "OpenRouter).")
    llm.add_argument("--llm-max-rate-limit-errors", type=int,
                     default=DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS, metavar="K")
    llm.add_argument("--llm-cache", default=str(DEFAULT_PR_LLM_CACHE), metavar="PATH")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.localise_tolerance < 0:
        parser.error("--localise-tolerance must be >= 0.")
    if args.split == "test" and not args.allow_test_split:
        parser.error(f"--split test is the held-out test set: pass {TEST_SPLIT_FLAG} only for "
                     "the final, pre-registered run (tune on dev).")
    if args.allow_test_split and args.split != "test":
        parser.error(f"{TEST_SPLIT_FLAG} only goes with --split test.")
    if args.sample_kinds is not None:
        try:
            args.sample_kinds = parse_sample_kinds(args.sample_kinds)
        except ValueError as exc:
            parser.error(f"--sample-kinds: {exc}")
    offline = args.rescore is not None or args.compare is not None
    if args.semgrep_precompute:
        if offline or args.dry_run:
            parser.error("--semgrep-precompute is its own mode.")
        if not args.semgrep_cache:
            parser.error("--semgrep-precompute requires --semgrep-cache PATH (the output).")
    real = not (offline or args.dry_run or args.semgrep_precompute)
    if real:
        if args.llm_max_calls is None:
            parser.error("a real run requires --llm-max-calls N (a hard cap on real calls).")
        if not 1 <= args.llm_max_calls <= LLM_MAX_CALLS_CAP:
            parser.error(f"--llm-max-calls must be between 1 and {LLM_MAX_CALLS_CAP}.")
        if args.out is None:
            parser.error("a real run requires --out.")
    if args.llm_token_budget is not None and args.llm_token_budget <= 0:
        parser.error("--llm-token-budget must be positive.")
    if args.llm_sleep < 0 or args.llm_tpm < 0:
        parser.error("--llm-sleep and --llm-tpm must be >= 0.")
    if args.llm_max_rate_limit_errors < 1:
        parser.error("--llm-max-rate-limit-errors must be >= 1.")
    if args.llm_temperature is not None and not 0.0 <= args.llm_temperature <= 2.0:
        parser.error("--llm-temperature must be between 0 and 2.")
    if args.llm_repeat_index < 0:
        parser.error("--llm-repeat-index must be >= 0.")
    for flag, spec in (("--llm-model", args.llm_model), ("--verifier-model",
                                                          args.verifier_model)):
        if spec:
            try:
                parse_llm_model(spec)
            except ValueError as exc:
                parser.error(f"{flag}: {exc}")
    if args.llm_model and args.llm_primary_only:
        parser.error("--llm-model already pins one model; drop --llm-primary-only.")
    if args.llm_upstream:
        provider = (parse_llm_model(args.llm_model)[0] if args.llm_model
                    else settings.LLM_PROVIDER)
        if provider != "openrouter" or not (args.llm_model or args.llm_primary_only):
            parser.error("--llm-upstream needs a pinned / primary-only OpenRouter model.")
    return args


def build_routers(args):
    """(audit router, verifier router or None), every client at --llm-temperature
    (None: no override, each model's recommended sampling)."""
    router = (build_pinned_router(args.llm_model) if args.llm_model
              else build_llm_router(args.llm_primary_only))
    set_router_temperature(router, args.llm_temperature)
    verifier = None
    if args.verifier_model:
        verifier = build_pinned_router(args.verifier_model)
        set_router_temperature(verifier, args.llm_temperature)
    return router, verifier


def _routing(args, router) -> dict | None:
    ns = SimpleNamespace(llm_primary_only=bool(args.llm_model or args.llm_primary_only),
                         llm_upstream=args.llm_upstream)
    return openrouter_routing(ns, router)


def run_config(args, router, verifier, semgrep_meta: dict | None) -> dict:
    cfg = pr_review_config()
    chain = ["mock"] if getattr(router, "mock", False) else [c.label for c in router.clients]
    return {
        "arm": args.arm,
        "model": llm_model_key(router),
        "chain": chain,
        "verifier_model": llm_model_key(verifier) if verifier is not None else None,
        "pinned_model": args.llm_model,
        "primary_only": bool(args.llm_model or args.llm_primary_only),
        "openrouter_provider_routing": _routing(args, router),
        "temperature": args.llm_temperature,
        "sampling": None if getattr(router, "mock", False) else client_sampling(
            router.clients[0]),
        "sampling_key": router_temperature(router),
        "repeat_index": args.llm_repeat_index,
        "max_calls": args.llm_max_calls,
        "token_budget": args.llm_token_budget,
        "sleep_s": args.llm_sleep,
        "tpm": args.llm_tpm,
        "cache": _rel(args.llm_cache),
        "localise_tolerance": args.localise_tolerance,
        "retrieval": "off (retrieval=None / no CVE references in any arm)",
        "wall_clock_budget": "disabled in the eval (frozen clock; pacing must not change "
                             "results)",
        "pr_review": {k: (sorted(v) if isinstance(v, frozenset) else v)
                      for k, v in vars(cfg).items()} if args.arm in PR_ARMS else None,
        "units_review": {
            "max_prompt_tokens": settings.LLM_MAX_PROMPT_TOKENS,
            "max_units_per_prompt": settings.LLM_MAX_UNITS_PER_PROMPT,
            "max_calls": settings.LLM_MAX_CALLS_PER_SCAN,
            "semgrep_min_severity": settings.SEMGREP_MIN_SEVERITY,
        } if args.arm == "units" else None,
        "semgrep": semgrep_meta,
    }


def _load_semgrep(args, items) -> tuple[dict | None, dict]:
    if not args.semgrep_cache:
        return None, {"leads": "off (no --semgrep-cache)"}
    cache = load_semgrep_cache(args.semgrep_cache)
    missing = [i["id"] for i in items if base_id(i["id"]) not in cache["items"]]
    if missing:
        raise SystemExit(f"--semgrep-cache {args.semgrep_cache} lacks {len(missing)} selected "
                         f"item(s), e.g. {missing[0]}: re-run --semgrep-precompute with the "
                         "same selection.")
    return cache, {"cache": _rel(args.semgrep_cache), **cache.get("meta", {}),
                   "excluded_rules": sorted(settings.SEMGREP_EXCLUDED_RULES)}


def run_semgrep_precompute(args) -> dict:
    items, meta = select_items(args)
    scanner = build_precompute_scanner(args.semgrep_timeout)
    if not scanner.available():
        raise SystemExit("Semgrep engine unavailable (pip install semgrep, or opengrep).")
    print(f"Semgrep precompute: {len(items)} item(s), one engine run "
          f"(timeout {args.semgrep_timeout:g}s)...", flush=True)
    data = precompute_semgrep(items, scanner)
    data["meta"].update(engine=scanner.engine, engine_version=engine_version(scanner),
                        rules_dir=_rel(scanner.rules_dir), selection=meta)
    out = Path(args.semgrep_cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"Wrote {out}: {data['meta']['n_files']} files, {data['meta']['n_hits']} hits, "
          f"{data['meta']['engine_seconds']}s.")
    return data


def run_dry(args) -> dict:
    items, meta = select_items(args)
    semgrep_cache, semgrep_meta = _load_semgrep(args, items)
    config = pr_review_config()
    print(f"Dry run ({args.arm}): {len(items)} items {meta['by_kind']}; no network.",
          flush=True)
    records = dry_run_all(items, args.arm, semgrep_cache=semgrep_cache,
                          keep_prompts=args.arm == "pr_misleading", workers=args.workers)
    summary = summarize_dry_run(records, args.arm, config,
                                completion_tokens=args.dry_run_completion_tokens,
                                latency_s=args.dry_run_latency_s,
                                max_calls=args.llm_max_calls, token_budget=args.llm_token_budget)
    if args.arm == "pr_misleading":
        base = load_items(args.dataset, split=args.split,
                          ids={base_id(r["id"]) for r in records})
        base_recs = dry_run_all(base, "pr", semgrep_cache=semgrep_cache,
                                workers=args.workers)
        shas = {r["id"]: r["prompt_shas"] for r in base_recs}
        same = sum(1 for r in records if shas.get(r["base_id"]) == r["prompt_shas"])
        leaked = sum(1 for r, it in zip(records, items, strict=True)
                     if pr_text_in_prompts(r, it))
        summary["misleading_vs_pr_prompts_identical"] = {"identical": same,
                                                          "n": len(records)}
        summary["pr_text_in_prompts"] = leaked
    print(json.dumps({k: v for k, v in summary.items() if k != "pipeline_partial_items"},
                     indent=1))
    print(f"pipeline-partial items: {summary['pipeline_partial_items']['n']} "
          f"{summary['pipeline_partial_items']['reasons']}")
    out = {"mode": "dry_run", "config": {"arm": args.arm, "selection": meta,
                                         "semgrep": semgrep_meta},
           "summary": summary,
           "items": [{k: r.get(k) for k in ("id", "kind", "language", "n_files", "n_units",
                                            "review_status", "not_reviewed_reasons",
                                            "units_partially_reviewed", "dry_run")}
                     for r in records]}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"Wrote {args.out}")
    return out


def pr_text_in_prompts(rec: dict, item: dict) -> bool:
    """Did the item's PR title or body reach any prompt (``rec["prompts"]``,
    kept by a ``keep_prompts`` gate)?"""
    texts = [t for t in (item.get("pr_title"), item.get("pr_body")) if t]
    return any(t in p for t in texts for p in rec.get("prompts") or [])


async def run_live(args, items, router, verifier, semgrep_cache, tap
                   ) -> tuple[list[dict], EvalGate]:
    cache = LLMCache(args.llm_cache)
    gate = EvalGate(cache=cache, max_calls=args.llm_max_calls,
                    token_budget=args.llm_token_budget, sleep_s=args.llm_sleep,
                    tpm=args.llm_tpm or None,
                    max_rate_limit_errors=args.llm_max_rate_limit_errors, tap=tap,
                    temperature=args.llm_temperature, repeat=args.llm_repeat_index)
    audit = GatedRouter(router, gate)
    verify = GatedRouter(verifier, gate) if verifier is not None else audit
    parser = _parser()
    config = pr_review_config()
    facts = selection_facts(items, parser)
    records = []
    announced = False
    for n, item in enumerate(items, start=1):
        rec = await run_item(item, args.arm, router=audit, verifier_router=verify, gate=gate,
                             parser=parser, semgrep_scanner=_item_scanner(semgrep_cache, item),
                             config=config)
        rec["facts"] = facts[item["id"]]
        records.append(rec)
        views = rec["findings"]
        print(f"  [{n}/{len(items)}] {item['id']} ({item['kind']}): {rec['status']}, "
              f"{rec['calls']['total']} call(s) ({rec['calls']['real']} real), "
              f"findings {', '.join(f'{v}={len(fs)}' for v, fs in views.items())}, "
              f"{rec.get('review_status')} | run calls {gate.calls}/{args.llm_max_calls}, "
              f"tokens {gate.tokens_used}", flush=True)
        if gate.stopped and not announced and n < len(items):
            announced = True
            print(f"  Stopped: {gate.stopped}; the remaining items only replay cached calls "
                  "(the rest are not_run).", flush=True)
    return records, gate


def run_real(args) -> dict:
    router, verifier = build_routers(args)  # fails fast on a missing key
    items, meta = select_items(args)
    semgrep_cache, semgrep_meta = _load_semgrep(args, items)
    config = run_config(args, router, verifier, semgrep_meta)
    config["selection"] = meta
    routing = config["openrouter_provider_routing"]
    temp = ("model default" if args.llm_temperature is None
            else f"forced {args.llm_temperature:g}")
    print(f"PR eval ({args.arm}): {len(items)} items {meta['by_kind']}; model "
          f"{config['model']}, verifier {config['verifier_model'] or 'same'}, sampling "
          f"{temp} {config['sampling']}, repeat {args.llm_repeat_index}; max "
          f"{args.llm_max_calls} real calls; cache "
          f"{args.llm_cache}." + (f" OpenRouter routing: {routing}." if routing else ""),
          flush=True)

    async def _go():
        with HttpUsageTap(inject={"provider": routing} if routing else None,
                          inject_url_prefix=settings.OPENROUTER_BASE_URL.rstrip("/")) as tap:
            return await run_live(args, items, router, verifier, semgrep_cache, tap)

    records, gate = asyncio.run(_go())
    summary = summarize(records, args.arm, args.localise_tolerance)
    out = {"config": config,
           "run": {"calls": gate.calls, "tokens_used": gate.tokens_used,
                   "stopped": gate.stopped,
                   "providers": dict(Counter(p for r in records for p in r["providers"]))},
           "summary": summary, "items": records}
    print_summary(summary, args.arm)
    if gate.stopped:
        print(f"Stopped early: {gate.stopped}. Re-run the same command to resume from the cache.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"Wrote {args.out}")
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.rescore is not None:
        data = json.loads(Path(args.rescore).read_text())
        items, source = load_run_items(data, args.dataset)
        data = rescore(data, args.localise_tolerance, items,
                       parser=_parser() if items is not None else None)
        data["rescore"].update(source)
        print(f"Rescore: facts {data['rescore']['facts']} {source.get('datasets')}; "
              f"label changes {len(data['rescore'].get('label_changes') or [])}")
        for ch in data["rescore"].get("label_changes") or []:
            print(f"  {ch['id']}: {ch['from']} -> {ch['to']} ({ch['reason']})")
        print_summary(data["summary"], data["config"]["arm"])
        if args.out:
            Path(args.out).write_text(json.dumps(data, indent=1))
            print(f"Wrote {args.out}")
        return data
    if args.compare is not None:
        a_path, b_path = args.compare
        a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
        same = Path(a_path).resolve() == Path(b_path).resolve() or same_run(a, b)
        parser = None
        for run in (a, b):  # current facts / labels for the change-anchored levels
            items, _ = load_run_items(run, args.dataset)
            if items is not None:
                parser = parser or _parser()
                refresh_facts(run["items"], items, parser)
        cmp = compare_runs(a["items"], b["items"], args.view_a, args.view_b,
                           args.localise_tolerance, same=same)
        print_comparison(f"{_rel(a_path)} ({a['config']['arm']}, {a['config'].get('model')})",
                         f"{_rel(b_path)} ({b['config']['arm']}, {b['config'].get('model')})",
                         cmp)
        if args.out:
            Path(args.out).write_text(json.dumps(cmp, indent=1))
        return cmp
    if args.semgrep_precompute:
        return run_semgrep_precompute(args)
    if args.dry_run:
        return run_dry(args)
    return run_real(args)


if __name__ == "__main__":
    main()
