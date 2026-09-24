"""Detection eval harness.

Measures the Ghost Hunter (CVE) retrieval pipeline against a labeled dataset of
vulnerable/safe snippets. Embeds + ANN-searches + reranks each snippet ONCE, then
sweeps (similarity_threshold x rerank_threshold) in-memory over the cached
candidates — so a full sweep costs one model pass, not one per threshold combo.

Run from the project root (stop the API first — local Qdrant is single-process):

    python -m ml.evaluation.run_eval --write-baseline \
        --dataset ml/evaluation/datasets/detection_eval.jsonl <other.jsonl ...>
    python -m ml.evaluation.run_eval --margin-sweep -0.10:0.20:0.02 \
        --dataset ml/evaluation/datasets/detection_eval.jsonl <other.jsonl ...>
    python -m ml.evaluation.run_eval --rerank --rerank-sweep 0.0:0.9:0.1 --dataset <a.jsonl>
    python -m ml.evaluation.run_eval --sample 150 --seed 42 \
        --reranker-model BAAI/bge-reranker-base --out ml/evaluation/results/base.json

The prediction rule at a given (sim_t, rerank_t, margin_t): keep candidates with
similarity_score >= sim_t and (margin_t off, or no patched twin, or twin_margin
>= margin_t — TWIN_MARGIN_MIN semantics); if the best-reranked survivor has
rerank_prob >= rerank_t, predict "vulnerable" with that candidate's category.

``twin_coverage`` is the fraction of eval items whose top candidate (best
reranked after the similarity gate, before the margin gate) has a patched twin —
the margin sweep only means something when it is well above zero.

The reranker follows ``settings.RERANKER_ENABLED`` (off by default, like the
API): without it the cross-encoder is never loaded, candidates are ranked by
similarity (rerank_prob := similarity_score) and the rerank sweep collapses to
a no-op — fast retrieval-only metrics on the full set. ``--rerank`` (or a
``--reranker-model``/``--reranker-max-tokens`` override) forces it on;
``--no-rerank`` forces it off.
``--sample N --seed S`` evaluates a deterministic, label-balanced subsample that
keeps OSV ``<prefix>_vuln``/``<prefix>_safe`` pairs together — for comparing
configs quickly; it can never be written as the baseline.
``--reranker-model`` / ``--reranker-max-tokens`` override the settings for this
run. Output JSON (and baseline.json) records the reranker actually used (null
when off), the sample/seed, and gather() timing (embed / ANN / rerank seconds).
The sim sweep always includes the operating point (SIM_THRESHOLD_CVE).

Realistic metrics (``realistic`` in --out) split the items by ``kind``: an
item's explicit ``kind`` field, else derived from its id — ``<p>_vuln`` ->
``vulnerable``, ``<p>_safe`` -> ``fixed_twin`` (OSV pairs), anything else ->
``handwritten``; ordinary (non-security) negatives carry ``kind: "ordinary"``.
They report TPR on vulnerable, FPR on fixed twins / ordinary / handwritten-safe
items (Wilson 95% intervals), precision at realistic base rates
pi in {0.01, 0.02, 0.05} — TPR*pi / (TPR*pi + FPR_ordinary*(1-pi)) — plus the
balanced 50/50 precision, and pairwise vuln-vs-twin discrimination. The
retrieval-only prediction is "at least one CVE match survives the operating
point" (exactly what score() counts at SIM_THRESHOLD_CVE).
``--sample-kinds vulnerable=40,fixed_twin=40,ordinary=80 --seed S`` draws a
deterministic stratified sample (vuln/twin pairs kept together).

``--llm`` adds the LLM stage; ``--llm-prompt`` picks the arm:

- ``current`` (default): the production review prompt, built exactly as
  scan_runner builds a snippet scan (one unit; its top-K retrieved CVEs capped
  to LLM_MAX_CVES_PER_UNIT; Semgrep evidence from one engine run over all items
  as snippets; the LLM_MAX_PROMPT_TOKENS budget; a deterministic per-item nonce
  so prompts and cache keys are reproducible). guard_diff is not applicable:
  a snippet has no previous version. Every item gets a call; prediction =
  production's ``is_vulnerable`` = at least one reviewed finding whose quote is
  in the code (``realistic_cve_finding``: one that also cites a shown CVE).
- ``no_retrieval``: the same prompt and Semgrep evidence with zero CVEs.
- ``legacy``: the pre-2026-09-24 prompt (``ml/evaluation/legacy_prompt.py``):
  items with retrieved CVEs only (others "safe" without a call, as the old
  production did), prediction = a validated finding references a retrieved CVE
  (``realistic_any_finding``: any validated finding).
Calls are capped (``--llm-max-calls``, <= 200 per run), paced
(``--llm-sleep``, ``--llm-tpm``), stop cleanly on repeated rate limits or a
daily limit, and are cached in a JSONL file keyed by (item id, prompt sha256,
model) so a re-run resumes without re-calling. LLM results are recorded in
``--out`` only, never in the baseline.
"""
import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_ROOT))

from backend.app.config import BASE_DIR, settings  # noqa: E402
from backend.app.core import markdown_renderer as review_prompt  # noqa: E402
from backend.app.core.cve_retriever import passes_twin_margin  # noqa: E402
from backend.app.core.embedder import Embedder  # noqa: E402
from backend.app.core.embedding_cache import EmbeddingCache  # noqa: E402
from backend.app.core.evidence import semgrep_evidence  # noqa: E402
from backend.app.core.llm_client import LLMRouter, TokenPacer  # noqa: E402
from backend.app.core.reranker import Reranker  # noqa: E402
from backend.app.core.review_plan import (  # noqa: E402
    assign_uids,
    build_review_units,
    plan_review_prompts,
    snippet_unit,
    unit_key,
)
from backend.app.core.vector_store import VectorStore  # noqa: E402
from ml.evaluation.legacy_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    validate_findings,
)

DEFAULT_DATASET = BASE_DIR / "ml" / "evaluation" / "datasets" / "detection_eval.jsonl"
BASELINE_PATH = BASE_DIR / "ml" / "evaluation" / "baseline.json"
RESULTS_DIR = BASE_DIR / "ml" / "evaluation" / "results"
DEFAULT_RERANK_SWEEP = "0.0:0.9:0.1"
# Starts below SIM_THRESHOLD_CVE (0.25) so the operating point is in the table.
DEFAULT_SIM_SWEEP = "0.20:0.95:0.05"
# rerank_t used without a reranker: rerank_prob is the similarity score there, and
# every candidate that survives the similarity gate (sim_t >= 0) already has
# similarity >= 0, so rerank_t = 0.0 is a no-op rather than a second sim gate.
NO_RERANK_THRESHOLD = 0.0
PAIR_SUFFIXES = ("_vuln", "_safe")

KIND_VULNERABLE = "vulnerable"
KIND_FIXED_TWIN = "fixed_twin"
KIND_ORDINARY = "ordinary"
KIND_HANDWRITTEN = "handwritten"
KINDS = (KIND_VULNERABLE, KIND_FIXED_TWIN, KIND_ORDINARY, KIND_HANDWRITTEN)
PAIR_KINDS = (KIND_VULNERABLE, KIND_FIXED_TWIN)
BASE_RATES = (0.01, 0.02, 0.05)

# Candidate payload fields kept by gather() so the LLM stage can rebuild the
# production prompt (build_user_prompt reads cve_id/severity/category/
# description/vulnerable_code/fixed_code).
PROMPT_FIELDS = (
    "cve_id", "severity", "category", "description", "vulnerable_code", "fixed_code",
    "point_id", "language",
)

LLM_MAX_CALLS_CAP = 200
DEFAULT_LLM_SLEEP = 2.5
DEFAULT_LLM_TPM = 8000  # Groq free tier tokens/minute per model
DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS = 3
DEFAULT_LLM_CACHE = RESULTS_DIR / "llm_cache.jsonl"
# Groq names the exhausted window in its 429 body ("... tokens per day (TPD)");
# OpenRouter's free tier says "Rate limit exceeded: free-models-per-day" (its
# per-minute throttle is "free-models-per-min", which matches none of these).
DAILY_LIMIT_MARKERS = ("per day", "(TPD)", "(RPD)", "free-models-per-day")
# A Retry-After this long only happens once a daily (not per-minute) window is spent.
DAILY_LIMIT_RETRY_AFTER_S = 300.0


def load_dataset(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _as_paths(dataset_path) -> list[Path]:
    """One path or a list of paths (--dataset takes several) -> list[Path]."""
    if isinstance(dataset_path, (str, Path)):
        return [Path(dataset_path)]
    return [Path(p) for p in dataset_path]


def load_datasets(dataset_path) -> list[dict]:
    """Concatenate one or more JSONL eval sets, in the order given. Warns on
    duplicate ids (per-id dicts would merge them; pairing copes, see
    ``pair_group_keys``)."""
    items: list[dict] = []
    for path in _as_paths(dataset_path):
        items.extend(load_dataset(path))
    dupes = [k for k, n in Counter(i["id"] for i in items).items() if n > 1]
    if dupes:
        print(f"WARNING: {len(dupes)} eval item id(s) occur more than once (e.g. {dupes[0]!r}); "
              "run scripts/migrate_eval_ids.py", file=sys.stderr)
    return items


def dataset_sha256(dataset_path) -> str:
    """sha256 of the eval set. A single file hashes exactly as before (so older
    baselines stay comparable); several hash their per-file digests in order."""
    paths = _as_paths(dataset_path)
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if len(digests) == 1:
        return digests[0]
    return hashlib.sha256("\n".join(digests).encode()).hexdigest()


def frange(spec: str) -> list[float]:
    """Parse 'start:stop:step' into an inclusive list of floats."""
    start, stop, step = (float(x) for x in spec.split(":"))
    n = int(round((stop - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def use_reranker(
    no_rerank: bool | None = None, model: str | None = None, max_tokens: int | None = None
) -> bool:
    """Whether a run uses the cross-encoder. ``no_rerank`` True/False forces
    off/on; None (the default) follows settings.RERANKER_ENABLED, except that a
    model/max-tokens override implies on (it only means something with one)."""
    if no_rerank is not None:
        return not no_rerank
    return settings.RERANKER_ENABLED or model is not None or max_tokens is not None


def reranker_config(
    no_rerank: bool | None = None, model: str | None = None, max_tokens: int | None = None
) -> dict | None:
    """The reranker a run uses (overrides resolved against settings), or None.

    This is both what build_components() constructs and what output JSON records.
    ``no_rerank`` semantics as in ``use_reranker``.
    """
    if not use_reranker(no_rerank, model, max_tokens):
        return None
    return {
        "model": settings.RERANKER_MODEL if model is None else model,
        "max_tokens": settings.RERANKER_MAX_TOKENS if max_tokens is None else max_tokens,
    }


def build_components(
    no_rerank: bool | None = None, reranker_model: str | None = None,
    reranker_max_tokens: int | None = None,
):
    """(embedder, store, reranker). Without a reranker (the default unless
    RERANKER_ENABLED, see ``use_reranker``) it is None and no cross-encoder is
    ever constructed (no model load)."""
    cfg = reranker_config(no_rerank, reranker_model, reranker_max_tokens)
    reranker = (
        None if cfg is None else Reranker(model_name=cfg["model"], max_tokens=cfg["max_tokens"])
    )
    return Embedder(cache=EmbeddingCache()), VectorStore(), reranker


def _pair_key(item_id: str) -> str:
    """Shared prefix of an OSV ``<prefix>_vuln``/``<prefix>_safe`` pair; any
    other id (e.g. handwritten ``sqli_vuln_1``) is its own group."""
    for suffix in PAIR_SUFFIXES:
        if item_id.endswith(suffix):
            return item_id[: -len(suffix)]
    return item_id


def item_kind(item: dict) -> str:
    """The item's ``kind``: its explicit field when valid, else from its id —
    ``_vuln`` -> vulnerable, ``_safe`` -> fixed_twin, others -> handwritten."""
    kind = item.get("kind")
    if kind in KINDS:
        return kind
    item_id = item["id"]
    if item_id.endswith("_vuln"):
        return KIND_VULNERABLE
    if item_id.endswith("_safe"):
        return KIND_FIXED_TWIN
    return KIND_HANDWRITTEN


def pair_group_keys(items: list[dict]) -> list[str]:
    """Sampling / pairing group per item, robust to reused ids.

    A vulnerable / fixed-twin item's group is its pair prefix (``_pair_key``),
    anything else (an ordinary item is never paired, whatever its id looks
    like) its own id. If a (prefix, kind) repeats - ids that aren't unique, as
    in eval files built before ids carried a code hash - the n-th occurrence
    gets ``<prefix>#<n>``, pairing the n-th vulnerable item with the n-th
    fixed twin (file order) instead of merging several pairs into one group.
    """
    seen: Counter = Counter()
    keys = []
    for item in items:
        kind = item_kind(item)
        base = _pair_key(item["id"]) if kind in PAIR_KINDS else item["id"]
        seen[(base, kind)] += 1
        n = seen[(base, kind)]
        keys.append(base if n == 1 else f"{base}#{n}")
    return keys


def sample_items(items: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministic, label-balanced subsample that keeps OSV pairs together.

    Items are grouped by ``pair_group_keys`` (a vuln/safe pair is one group; ids
    without the suffix are singletons — groups are atomic, so a pair is never
    split). Groups are shuffled with ``random.Random(seed)`` and taken greedily
    while (a) the total stays <= n and (b) |n_vulnerable - n_safe| stays <= 1.
    Tie-break: with an odd n the extra item's label is whichever singleton the
    shuffle reaches first; the result may fall short of n (by 1 when n is odd
    and no singleton of the needed label is left, more only if the pool can't
    supply a balanced sample). n >= len(items) returns every item unchanged.
    Selected items keep their input order.
    """
    if n >= len(items):
        return list(items)
    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(pair_group_keys(items)):
        groups.setdefault(key, []).append(idx)
    units = list(groups.values())
    random.Random(seed).shuffle(units)

    chosen: list[int] = []
    n_vuln = n_safe = 0
    for unit in units:
        if len(chosen) == n:
            break
        dv = sum(items[i]["label"] == "vulnerable" for i in unit)
        ds = len(unit) - dv
        if len(chosen) + len(unit) > n or abs((n_vuln + dv) - (n_safe + ds)) > 1:
            continue
        chosen.extend(unit)
        n_vuln += dv
        n_safe += ds
    return [items[i] for i in sorted(chosen)]


def parse_sample_kinds(spec: str) -> dict[str, int]:
    """'vulnerable=40,fixed_twin=40,ordinary=80' -> {kind: quota}. Raises
    ValueError on an unknown kind, a duplicate, or a non-positive quota."""
    quotas: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        kind, sep, value = part.partition("=")
        kind = kind.strip()
        if not sep or kind not in KINDS:
            raise ValueError(f"expected <kind>=<n> with kind in {', '.join(KINDS)}; got {part!r}")
        if kind in quotas:
            raise ValueError(f"kind {kind!r} given twice")
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"quota for {kind!r} is not an integer: {value!r}") from None
        if n <= 0:
            raise ValueError(f"quota for {kind!r} must be positive")
        quotas[kind] = n
    if not quotas:
        raise ValueError("no <kind>=<n> quotas given")
    return quotas


def sample_items_by_kind(items: list[dict], quotas: dict[str, int], seed: int = 42) -> list[dict]:
    """Deterministic stratified sample: at most ``quotas[kind]`` items per kind.

    Kinds not in ``quotas`` are left out. Items are grouped as in
    ``sample_items`` (a vulnerable/fixed-twin pair is one atomic group, so it is
    never split when both kinds are requested; with only one of them requested
    the other half is simply not in the pool). Complete pairs are drawn first
    (shuffled with ``random.Random(seed)``) so the pairwise metrics get as many
    pairs as the quotas allow, then everything else (shuffled with the same
    RNG); a group is taken if every kind stays within its quota. May fall short
    of a quota when the pool runs out. Selected items keep their input order.
    """
    groups: dict[str, list[int]] = {}
    for idx, (item, key) in enumerate(zip(items, pair_group_keys(items), strict=True)):
        if item_kind(item) in quotas:
            groups.setdefault(key, []).append(idx)
    pairs, others = [], []
    for unit in groups.values():
        kinds = {item_kind(items[i]) for i in unit}
        (pairs if set(PAIR_KINDS) <= kinds else others).append(unit)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rng.shuffle(others)

    counts: Counter = Counter()
    chosen: list[int] = []
    for unit in pairs + others:
        need = Counter(item_kind(items[i]) for i in unit)
        if all(counts[k] + v <= quotas[k] for k, v in need.items()):
            chosen.extend(unit)
            counts.update(need)
    return [items[i] for i in sorted(chosen)]


def gather(
    items, embedder, store, reranker, ann_candidates: int, progress: bool = False
) -> tuple[list[dict], dict]:
    """Run the ANN (+ rerank) pipeline once per item; cache scored candidates.

    ``reranker=None`` (reranker off) skips the cross-encoder. Returns
    ``(gathered, timing)``; ``timing`` keys (seconds, summed over items):
    ``n_items``, ``total_seconds`` (wall time of the whole call),
    ``seconds_per_item``, ``embed_seconds`` (embedder.embed_text),
    ``ann_seconds`` (store.search_cves), ``rerank_seconds`` (reranker.rerank;
    0 without a reranker). ``progress`` prints a line every ~5% of items.
    """
    gathered = []
    embed_s = ann_s = rerank_s = 0.0
    n_total = len(items)
    stride = max(1, n_total // 20)
    start = time.perf_counter()
    for done, item in enumerate(items, start=1):
        code = item["code"]
        t0 = time.perf_counter()
        vector = embedder.embed_text(code)
        t1 = time.perf_counter()
        candidates = store.search_cves(vector, limit=ann_candidates)
        t2 = time.perf_counter()
        embed_s += t1 - t0
        ann_s += t2 - t1
        if reranker is not None:
            # Mirror the retriever: rerank code-against-code (vulnerable_code),
            # not against the English description.
            for c in candidates:
                c["rerank_text"] = c.get("vulnerable_code") or c.get("description", "")
            if candidates:
                t3 = time.perf_counter()
                reranker.rerank(code, candidates, top_k=len(candidates))
                rerank_s += time.perf_counter() - t3
        scored = []
        for c in candidates:
            sim = c.get("similarity_score", 0.0)
            if reranker is None:
                # No cross-encoder: rank by similarity. score() picks the best
                # survivor via max(rerank_prob), so reusing that field keeps
                # score() and every metric unchanged — they just order by
                # similarity instead of cross-encoder probability. No
                # rerank_score is fabricated.
                prob = sim
            else:
                # Reranker already sets rerank_prob = sigmoid(raw logit); don't
                # re-apply sigmoid here (that double-squashed it into [0.5, 0.73]).
                prob = float(c.get("rerank_prob", 0.0))
            scored.append(
                {
                    # Untouched payload values: the LLM stage hands these to
                    # build_user_prompt exactly as the retriever's match dicts.
                    "payload": {k: c.get(k) for k in PROMPT_FIELDS},
                    "similarity_score": sim,
                    "rerank_prob": prob,
                    "category": c.get("category", "unknown"),
                    "cve_id": c.get("cve_id"),
                    "sim_fixed": c.get("sim_fixed"),
                    "twin_margin": c.get("twin_margin"),
                }
            )
        gathered.append(
            {
                "id": item["id"],
                "is_vulnerable": item["label"] == "vulnerable",
                "category": item.get("category"),
                "kind": item_kind(item),
                "label": item["label"],
                "language": item.get("language"),
                "length_matched": item.get("length_matched"),
                "code": code,
                "candidates": scored,
            }
        )
        if progress and (done % stride == 0 or done == n_total):
            elapsed = time.perf_counter() - start
            remaining = elapsed / done * (n_total - done)
            print(
                f"  [{done}/{n_total}] {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining",
                flush=True,
            )
    total = time.perf_counter() - start
    timing = {
        "n_items": n_total,
        "total_seconds": round(total, 4),
        "seconds_per_item": round(total / n_total, 4) if n_total else 0.0,
        "embed_seconds": round(embed_s, 4),
        "ann_seconds": round(ann_s, 4),
        "rerank_seconds": round(rerank_s, 4),
    }
    return gathered, timing


def score(
    gathered: list[dict], sim_t: float, rerank_t: float, margin_t: float | None = None
) -> dict:
    tp = fp = fn = tn = 0
    category_hits = 0
    top_with_twin = 0
    for g in gathered:
        sim_passing = [c for c in g["candidates"] if c["similarity_score"] >= sim_t]
        top = max(sim_passing, key=lambda c: c["rerank_prob"], default=None)
        if top is not None and top.get("twin_margin") is not None:
            top_with_twin += 1

        passing = [c for c in sim_passing if passes_twin_margin(c, margin_t)]
        best = max(passing, key=lambda c: c["rerank_prob"], default=None)
        predicted = best is not None and best["rerank_prob"] >= rerank_t

        if predicted and g["is_vulnerable"]:
            tp += 1
            if best["category"] == g["category"]:
                category_hits += 1
        elif predicted and not g["is_vulnerable"]:
            fp += 1
        elif not predicted and g["is_vulnerable"]:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "sim_threshold": sim_t,
        "rerank_threshold": rerank_t,
        "margin_threshold": margin_t,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "category_hit_rate": round(category_hits / tp, 4) if tp else 0.0,
        "twin_coverage": round(top_with_twin / len(gathered), 4) if gathered else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(gathered),
    }


# --- operating-point CVE matches (what the real pipeline hands the LLM) ------


def operating_cves(
    g: dict, sim_t: float, rerank_t: float, margin_t: float | None, top_k: int
) -> list[dict]:
    """The CVE matches CVERetriever.find_vulnerabilities would return for this
    item: similarity gate -> twin-margin gate -> rerank_prob gate -> ordered by
    relevance (rerank_prob; the similarity without a reranker, see gather) ->
    top ``top_k``. Returns the candidates' raw payload dicts, best first.

    Without a reranker ``rerank_t`` is NO_RERANK_THRESHOLD (a no-op, like the
    API ignoring RERANK_THRESHOLD). Feedback votes are not applied (the eval has
    none), and gather() searches without a language filter (a known deviation
    from the API, which filters by the request's language).
    """
    passing = [
        c for c in g["candidates"]
        if c["similarity_score"] >= sim_t
        and passes_twin_margin(c, margin_t)
        and c["rerank_prob"] >= rerank_t
    ]
    # Stable sort from ANN order, like rank_candidates + finalize_matches.
    passing.sort(key=lambda c: c["rerank_prob"], reverse=True)
    return [c["payload"] for c in passing[:top_k]]


# --- realistic metrics ------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for k successes out of n (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _rate(preds: list[bool]) -> dict:
    k, n = sum(preds), len(preds)
    return {
        "k": k, "n": n,
        "rate": round(k / n, 4) if n else None,
        "ci95": wilson_interval(k, n),
    }


def precision_at_base_rate(tpr: float | None, fpr: float | None, pi: float) -> float | None:
    """Expected precision when a fraction ``pi`` of scanned functions is
    vulnerable: TPR*pi / (TPR*pi + FPR*(1-pi)). None if undefined."""
    if tpr is None or fpr is None:
        return None
    denom = tpr * pi + fpr * (1 - pi)
    return round(tpr * pi / denom, 4) if denom else None


def realistic_metrics(records: list[dict], base_rates=BASE_RATES) -> dict:
    """Per-kind metrics from ``records`` ({id, kind, label, pred}); items with
    ``pred`` None (LLM error / not run) are excluded and counted.

    - tpr_vulnerable: flagged fraction of kind "vulnerable" (recall);
      fpr_fixed_twin / fpr_ordinary / fpr_handwritten_safe: flagged fraction of
      fixed twins, ordinary functions, handwritten safe items (tpr_handwritten
      for handwritten vulnerable ones). Each with a Wilson 95% interval.
      fpr_ordinary_length_matched: ordinary items with ``length_matched`` true
      (line counts matched to the vulnerable items; ordinary code is shorter).
    - precision_at_base_rate[pi] = TPR*pi / (TPR*pi + FPR_ordinary*(1-pi))
      (``_length_matched``: with the length-matched FPR).
    - precision_balanced_vs_{fixed_twin,ordinary}: the same at pi = 0.5.
    - pairs: over (vuln, fixed twin) pairs with both scored, the fraction with
      only the vuln flagged (vuln_only — the discrimination we want), only the
      twin flagged (twin_only), both, and neither.
    - observed: label-based confusion over every scored item (the old metric).
    """
    scored = [r for r in records if r["pred"] is not None]

    def preds(pred_filter) -> list[bool]:
        return [bool(r["pred"]) for r in scored if pred_filter(r)]

    tpr = _rate(preds(lambda r: r["kind"] == KIND_VULNERABLE))
    fpr_twin = _rate(preds(lambda r: r["kind"] == KIND_FIXED_TWIN))
    fpr_ord = _rate(preds(lambda r: r["kind"] == KIND_ORDINARY))
    fpr_ord_lm = _rate(
        preds(lambda r: r["kind"] == KIND_ORDINARY and r.get("length_matched") is True)
    )
    fpr_hw = _rate(preds(lambda r: r["kind"] == KIND_HANDWRITTEN and r["label"] != "vulnerable"))
    tpr_hw = _rate(preds(lambda r: r["kind"] == KIND_HANDWRITTEN and r["label"] == "vulnerable"))

    by_pair: dict[str, dict[str, bool]] = {}
    pair_records = [r for r in scored if r["kind"] in PAIR_KINDS]
    for r, key in zip(pair_records, pair_group_keys(pair_records), strict=True):
        by_pair.setdefault(key, {})[r["kind"]] = bool(r["pred"])
    complete = [p for p in by_pair.values() if len(p) == 2]
    n_pairs = len(complete)
    counts = Counter(
        ("vuln_only" if p[KIND_VULNERABLE] and not p[KIND_FIXED_TWIN]
         else "twin_only" if p[KIND_FIXED_TWIN] and not p[KIND_VULNERABLE]
         else "both" if p[KIND_VULNERABLE] else "neither")
        for p in complete
    )
    pairs = {"n": n_pairs}
    for key in ("vuln_only", "twin_only", "both", "neither"):
        pairs[key] = round(counts[key] / n_pairs, 4) if n_pairs else None
        pairs[f"n_{key}"] = counts[key]

    tp = sum(1 for r in scored if r["pred"] and r["label"] == "vulnerable")
    fp = sum(1 for r in scored if r["pred"] and r["label"] != "vulnerable")
    fn = sum(1 for r in scored if not r["pred"] and r["label"] == "vulnerable")
    tn = len(scored) - tp - fp - fn
    return {
        "n": len(records),
        "n_scored": len(scored),
        "n_excluded": len(records) - len(scored),
        "tpr_vulnerable": tpr,
        "fpr_fixed_twin": fpr_twin,
        "fpr_ordinary": fpr_ord,
        "fpr_ordinary_length_matched": fpr_ord_lm,
        "fpr_handwritten_safe": fpr_hw,
        "tpr_handwritten": tpr_hw,
        "precision_at_base_rate": {
            str(pi): precision_at_base_rate(tpr["rate"], fpr_ord["rate"], pi)
            for pi in base_rates
        },
        "precision_at_base_rate_length_matched": {
            str(pi): precision_at_base_rate(tpr["rate"], fpr_ord_lm["rate"], pi)
            for pi in base_rates
        },
        "precision_balanced_vs_fixed_twin": precision_at_base_rate(
            tpr["rate"], fpr_twin["rate"], 0.5
        ),
        "precision_balanced_vs_ordinary": precision_at_base_rate(
            tpr["rate"], fpr_ord["rate"], 0.5
        ),
        "pairs": pairs,
        "observed": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        },
    }


def _fmt_rate(r: dict) -> str:
    if r["rate"] is None:
        return "n/a (n=0)"
    lo, hi = r["ci95"]
    return f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}] ({r['k']}/{r['n']})"


def print_realistic(title: str, m: dict) -> None:
    print(f"\n{title} (n={m['n_scored']}, excluded={m['n_excluded']}):")
    print(f"  TPR vulnerable        {_fmt_rate(m['tpr_vulnerable'])}")
    print(f"  FPR fixed twin        {_fmt_rate(m['fpr_fixed_twin'])}")
    print(f"  FPR ordinary          {_fmt_rate(m['fpr_ordinary'])}")
    print(f"  FPR ordinary (len-m.) {_fmt_rate(m['fpr_ordinary_length_matched'])}")
    print(f"  FPR handwritten safe  {_fmt_rate(m['fpr_handwritten_safe'])}")
    print(f"  TPR handwritten       {_fmt_rate(m['tpr_handwritten'])}")
    prec = ", ".join(
        f"pi={pi}: {'n/a' if v is None else f'{v:.3f}'}"
        for pi, v in m["precision_at_base_rate"].items()
    )
    print(f"  precision @ base rate {prec}")
    print(
        f"  balanced precision    vs twin {m['precision_balanced_vs_fixed_twin']}, "
        f"vs ordinary {m['precision_balanced_vs_ordinary']}"
    )
    p = m["pairs"]
    if p["n"]:
        print(
            f"  pairs (n={p['n']})         vuln only {p['vuln_only']:.3f}, twin only "
            f"{p['twin_only']:.3f}, both {p['both']:.3f}, neither {p['neither']:.3f}"
        )


# --- LLM report stage -------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) — used for budgeting only when the
    provider response carries no ``usage``."""
    return math.ceil(len(text) / 4)


def prompt_sha256(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Cache key part: covers the system prompt too, so editing it invalidates."""
    return hashlib.sha256(f"{system_prompt}\x00{user_prompt}".encode()).hexdigest()


def llm_decision(llm_json, cves: list[dict]) -> dict:
    """``legacy`` arm: validate an LLM response as the old scan_runner did and
    derive the prediction. ``prediction`` = a validated finding references a retrieved CVE
    (a CVE finding in the report); ``any_finding`` = any validated finding
    (production's ``is_vulnerable``). Raises ValueError on a response shape the
    production pipeline would fail the scan on."""
    if not isinstance(llm_json, dict):
        raise ValueError(f"LLM returned a JSON {type(llm_json).__name__}, not an object")
    raw = llm_json.get("findings", [])
    if not isinstance(raw, list):
        raise ValueError(f"'findings' is a {type(raw).__name__}, not a list")
    allowed_cves = {c.get("cve_id") for c in cves if c.get("cve_id")}
    validated = validate_findings(raw, allowed_cves, set())
    flagged = [f["cve_id"] for f in validated if f.get("cve_id")]
    categories = {c.get("cve_id"): c.get("category") for c in cves}
    return {
        "prediction": bool(flagged),
        "any_finding": bool(validated),
        "validated_count": len(validated),
        "raw_finding_count": len(raw),
        "flagged_cve_ids": flagged,
        "flagged_categories": sorted({str(categories.get(c)) for c in flagged}),
    }


def review_decision(llm_json, batch: list[dict]) -> dict:
    """``current`` / ``no_retrieval`` arms: validate as scan_runner does now
    (the quote must be in the unit; a cve_id outside the shown matches is
    stripped). ``prediction`` = ``any_finding`` = at least one reviewed finding
    (production's ``is_vulnerable``); ``cve_finding`` = one cites a shown CVE.
    Raises ValueError on a response shape production would reject."""
    if not isinstance(llm_json, dict):
        raise ValueError(f"LLM returned a JSON {type(llm_json).__name__}, not an object")
    raw = llm_json.get("findings", [])
    if not isinstance(raw, list):
        raise ValueError(f"'findings' is a {type(raw).__name__}, not a list")
    shown = [c for u in batch for c in u["cves"]]
    allowed = {c.get("cve_id") for c in shown if c.get("cve_id")}
    validated = review_prompt.validate_findings(raw, batch, allowed, set())
    flagged = [f["cve_id"] for f in validated if f.get("cve_id")]
    categories = {c.get("cve_id"): c.get("category") for c in shown}
    return {
        "prediction": bool(validated),
        "any_finding": bool(validated),
        "cve_finding": bool(flagged),
        "validated_count": len(validated),
        "raw_finding_count": len(raw),
        "flagged_cve_ids": flagged,
        "flagged_categories": sorted({str(categories.get(c)) for c in flagged}),
        "cwes": sorted({f["cwe"] for f in validated if f.get("cwe")}),
    }


LLM_PROMPT_ARMS = ("current", "legacy", "no_retrieval")
ARM_RULES = {
    "current": "vulnerable iff >= 1 reviewed finding (quote in the code); production prompt "
               "with retrieved CVEs + Semgrep evidence, snippet mode (guard_diff N/A)",
    "no_retrieval": "vulnerable iff >= 1 reviewed finding; production prompt with Semgrep "
                    "evidence but zero CVEs",
    "legacy": "vulnerable iff >= 1 validated finding references a retrieved CVE (the old "
              "prompt; items without retrieved CVEs get no call)",
}


def eval_nonce(item_id: str) -> str:
    """Deterministic stand-in for untrusted.new_nonce(), so the prompt (and its
    cache key) is reproducible across runs."""
    return hashlib.sha256(f"eval-nonce:{item_id}".encode()).hexdigest()[:16]


def review_prompt_for(e: dict, arm: str) -> tuple[list[dict] | None, str | None]:
    """(batch, user prompt) exactly as scan_runner builds them for a snippet
    scan of ``e``: one unit with its Semgrep evidence (``e["semgrep"]``) and,
    for ``current``, its retrieved CVEs; budgeted by the production settings.
    (None, None) when the unit can't fit a prompt (production: not reviewed)."""
    unit = snippet_unit(e["code"], e.get("language"))
    raw = {"ghost_hunter_findings": e["cves"] if arm == "current" else [],
           "team_memory_findings": []}
    units = build_review_units([unit], raw, {unit_key(unit): e.get("semgrep") or []})
    batches, _ = plan_review_prompts(
        units,
        max_prompt_tokens=settings.LLM_MAX_PROMPT_TOKENS,
        max_units_per_prompt=settings.LLM_MAX_UNITS_PER_PROMPT,
        max_calls=settings.LLM_MAX_CALLS_PER_SCAN,
        max_refs_per_unit=settings.LLM_MAX_CVES_PER_UNIT,
    )
    if not batches:
        return None, None
    batch = assign_uids(batches[0])
    return batch, review_prompt.build_user_prompt(batch, eval_nonce(e["id"]))


def attach_semgrep(entries: list[dict], scanner) -> dict:
    """Scan every entry's code as a snippet (one engine run, no file context,
    like production snippet mode) and store evidence-grade hits in
    ``e["semgrep"]``. Returns a summary (items with evidence, by kind)."""
    units = [
        {"file_path": f"item{i:05d}", "function_name": None, "start_line": 1,
         "code": e["code"], "language": e.get("language")}
        for i, e in enumerate(entries)
    ]
    t0 = time.perf_counter()
    hits = semgrep_evidence(scanner, units, None, settings.SEMGREP_MIN_SEVERITY,
                            frozenset(settings.SEMGREP_EXCLUDED_RULES))
    for e, u in zip(entries, units, strict=True):
        e["semgrep"] = hits.get(unit_key(u), [])
    by_kind = Counter(e["kind"] for e in entries if e["semgrep"])
    return {
        "engine_available": bool(scanner is not None and scanner.available()),
        "min_severity": settings.SEMGREP_MIN_SEVERITY,
        "items_with_evidence": dict(by_kind),
        "items_by_kind": dict(Counter(e["kind"] for e in entries)),
        "rules": dict(Counter(h["rule_id"] for e in entries for h in e["semgrep"])),
        "seconds": round(time.perf_counter() - t0, 2),
    }


class LLMCache:
    """Append-only JSONL of successful LLM results keyed by (item id, prompt
    sha256, model). Errors are never cached, so a re-run retries them."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[tuple[str, str, str], dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._entries[(rec["id"], rec["prompt_sha256"], rec["model"])] = rec
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue  # a torn last line from an interrupted run

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, item_id: str, sha: str, model: str) -> dict | None:
        return self._entries.get((item_id, sha, model))

    def put(self, rec: dict) -> None:
        self._entries[(rec["id"], rec["prompt_sha256"], rec["model"])] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


def _http_event(resp: httpx.Response) -> dict:
    event = {"status": resp.status_code, "usage": None, "retry_after": None, "daily_limit": False}
    if resp.status_code == 200:
        try:
            data = resp.json()
            usage = data.get("usage")
            event["usage"] = usage if isinstance(usage, dict) else None
            # OpenRouter can put an error (incl. an upstream 429) in a 200 body.
            err = data.get("error")
            if isinstance(err, dict):
                event["error_code"] = err.get("code")
                if err.get("code") == 429:
                    message = str(err.get("message", ""))
                    event["daily_limit"] = any(m in message for m in DAILY_LIMIT_MARKERS)
        except (ValueError, AttributeError):
            pass
    else:
        try:
            event["retry_after"] = float(resp.headers.get("Retry-After"))
        except (TypeError, ValueError):
            pass
        if resp.status_code == 429:
            body = resp.text
            event["daily_limit"] = any(m in body for m in DAILY_LIMIT_MARKERS) or (
                event["retry_after"] is not None
                and event["retry_after"] >= DAILY_LIMIT_RETRY_AFTER_S
            )
    return event


class HttpUsageTap:
    """While active, records every httpx.AsyncClient.post response's token
    ``usage`` and rate-limit signals. LLMClient returns only the message text and
    folds a 429 into "HTTP 429", so this is how the eval sees real token counts
    (including reasoning tokens) and tells a daily limit from a per-minute one.
    Only the LLM stage runs inside it (no other httpx traffic in the eval)."""

    def __init__(self):
        self.events: list[dict] = []
        self._orig = None

    def __enter__(self):
        self._orig = orig = httpx.AsyncClient.post
        tap = self

        async def post(client, *args, **kwargs):
            resp = await orig(client, *args, **kwargs)
            tap.events.append(_http_event(resp))
            return resp

        httpx.AsyncClient.post = post
        return self

    def __exit__(self, *exc):
        httpx.AsyncClient.post = self._orig
        return False

    def take(self) -> list[dict]:
        events, self.events = self.events, []
        return events


def _usage_totals(events: list[dict]) -> dict | None:
    usages = [e["usage"] for e in events if e.get("usage")]
    if not usages:
        return None
    return {
        k: sum(int(u.get(k) or 0) for u in usages)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def classify_llm_error(exc: BaseException, events: list[dict]) -> str:
    """'daily_limit' | 'rate_limit' | 'other' for a failed router.generate."""
    text = str(exc)
    if any(e.get("daily_limit") for e in events) or any(m in text for m in DAILY_LIMIT_MARKERS):
        return "daily_limit"
    if (
        any(e.get("status") == 429 or e.get("error_code") == 429 for e in events)
        or "HTTP 429" in text
    ):
        return "rate_limit"
    return "other"


def llm_model_key(router) -> str:
    """Cache/model key: the router's primary "<provider>:<model>" (or "mock")."""
    return "mock" if getattr(router, "mock", False) else router.clients[0].label


def build_llm_entries(
    gathered: list[dict], sim_t: float, rerank_t: float, margin_t: float | None, top_k: int
) -> list[dict]:
    """Per item: what the LLM stage (and the retrieval-only prediction) needs."""
    return [
        {
            "id": g["id"],
            "kind": g["kind"],
            "label": g["label"],
            "category": g["category"],
            "language": g.get("language"),
            "length_matched": g.get("length_matched"),
            "code": g["code"],
            "cves": operating_cves(g, sim_t, rerank_t, margin_t, top_k),
        }
        for g in gathered
    ]


async def run_llm_stage(
    entries: list[dict],
    router,
    *,
    cache: LLMCache | None,
    max_calls: int,
    sleep_s: float = DEFAULT_LLM_SLEEP,
    tpm: int | None = DEFAULT_LLM_TPM,
    max_rate_limit_errors: int = DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS,
    token_budget: int | None = None,
    tap: HttpUsageTap | None = None,
    sleep=asyncio.sleep,
    progress: bool = True,
    arm: str = "current",
) -> dict:
    """Run the report stage over ``entries`` (see ``build_llm_entries``) with
    the ``arm``'s prompt (``LLM_PROMPT_ARMS``, rules in ``ARM_RULES``).

    Per item ``status``: ``no_candidates`` (legacy only: no retrieved CVE ->
    "safe", no call, as the old production did), ``too_large`` (the unit can't
    fit a prompt -> not reviewed, "safe", no call), ``ok`` / ``cached``
    (scored), ``error`` (the call failed — excluded from metrics), ``not_run``
    (a stop condition hit first — excluded).
    Stops calling (remaining items -> not_run) after ``max_calls`` real calls,
    when the next call would exceed ``token_budget``, on a daily-limit error, or
    after ``max_rate_limit_errors`` consecutive rate-limited failures. Between
    real calls it waits max(``sleep_s``, 60 * last call's tokens / ``tpm``).
    Cached items cost no call, no wait and no budget.
    """
    model = llm_model_key(router)
    n = len(entries)
    records: list[dict] = []
    calls = tokens_used = consecutive_rl = 0
    last_tokens = 0
    stopped: str | None = None
    for idx, e in enumerate(entries, start=1):
        rec = {
            "id": e["id"], "kind": e["kind"], "label": e["label"], "category": e["category"],
            "length_matched": e.get("length_matched"),
            "retrieved_cve_ids": [c.get("cve_id") for c in e["cves"]],
        }
        records.append(rec)
        if arm == "legacy":
            if not e["cves"]:
                rec.update(status="no_candidates", prediction=False, any_finding=False,
                           validated_count=0)
                continue
            system = SYSTEM_PROMPT
            prompt = build_user_prompt(e["code"], e["cves"], [])

            def decide(llm_json, e=e):
                return llm_decision(llm_json, e["cves"])
        else:
            batch, prompt = review_prompt_for(e, arm)
            if batch is None:
                rec.update(status="too_large", prediction=False, any_finding=False,
                           validated_count=0)
                continue
            system = review_prompt.SYSTEM_PROMPT
            rec["shown_cve_ids"] = [c.get("cve_id") for c in batch[0]["cves"]]
            rec["semgrep_rules"] = [h["rule_id"] for h in batch[0]["semgrep"]]

            def decide(llm_json, batch=batch):
                return review_decision(llm_json, batch)
        sha = prompt_sha256(prompt, system)
        rec["prompt_sha256"] = sha
        rec["prompt_tokens_est"] = estimate_tokens(system) + estimate_tokens(prompt)

        cached = cache.get(e["id"], sha, model) if cache is not None else None
        if cached is not None:
            try:
                rec.update(decide(cached["llm_json"]))
                rec.update(status="cached", provider_used=cached.get("provider_used"),
                           latency_s=cached.get("latency_s"), usage=cached.get("usage"))
                continue
            except (ValueError, KeyError):
                pass  # unusable cache entry: call again

        if stopped is None:
            if calls >= max_calls:
                stopped = "max_calls"
            elif token_budget is not None and tokens_used + rec["prompt_tokens_est"] > token_budget:
                stopped = "token_budget"
        if stopped is not None:
            rec.update(status="not_run", prediction=None, any_finding=None)
            continue

        if calls:
            wait = max(sleep_s, 60.0 * last_tokens / tpm if tpm else 0.0)
            if wait > 0:
                await sleep(wait)
        calls += 1
        if tap is not None:
            tap.take()  # drop anything not from this call
        t0 = time.perf_counter()
        error: BaseException | None = None
        try:
            llm_json, provider = await router.generate(system, prompt)
        except asyncio.CancelledError:
            # Ctrl-C under asyncio.run: keep what we have (the cache already
            # holds every finished item) and let main() write the partial --out.
            rec.update(status="not_run", prediction=None, any_finding=None)
            stopped = "interrupted"
            continue
        except Exception as exc:  # noqa: BLE001 — any failure is one errored item
            error = exc
        latency = round(time.perf_counter() - t0, 3)
        events = tap.take() if tap is not None else []
        usage = _usage_totals(events)
        rec.update(latency_s=latency, usage=usage,
                   http_statuses=[ev["status"] for ev in events] or None)

        if error is None:
            try:
                rec.update(decide(llm_json))
            except ValueError as exc:
                error = exc
            else:
                rec.update(status="ok", provider_used=provider)
                consecutive_rl = 0
                if cache is not None:
                    cache.put({
                        "id": e["id"], "prompt_sha256": sha, "model": model,
                        "provider_used": provider, "llm_json": llm_json,
                        "latency_s": latency, "usage": usage,
                        "prompt_tokens_est": rec["prompt_tokens_est"],
                    })
        if error is not None:
            kind = classify_llm_error(error, events)
            rec.update(status="error", prediction=None, any_finding=None,
                       error=f"{type(error).__name__}: {str(error)[:500]}", error_kind=kind)
            if kind == "daily_limit":
                stopped = "daily_limit"
            elif kind == "rate_limit":
                consecutive_rl += 1
                if consecutive_rl >= max_rate_limit_errors:
                    stopped = "rate_limit"
            else:
                consecutive_rl = 0

        last_tokens = usage["total_tokens"] if usage else (
            rec["prompt_tokens_est"] if error is None else 0
        )
        tokens_used += last_tokens
        if progress:
            pred = {True: "VULN", False: "safe", None: "-"}[rec.get("prediction")]
            detail = (
                f"{rec.get('error_kind')} error" if rec["status"] == "error"
                else f"pred={pred} findings={rec.get('validated_count')} "
                f"via {rec.get('provider_used')}"
            )
            print(
                f"  [LLM {idx}/{n}] {e['id']} ({e['kind']}): {detail}, {latency:.1f}s | "
                f"calls {calls}/{max_calls}, tokens so far {tokens_used}",
                flush=True,
            )
            if stopped in ("daily_limit", "rate_limit"):
                print(f"  Stopping LLM calls: {stopped}. Re-run later to resume from the cache.")

    status_counts = Counter(r["status"] for r in records)
    return {
        "model": model,
        "records": records,
        "calls": calls,
        "tokens_used": tokens_used,
        "stopped": stopped,
        "status_counts": dict(status_counts),
        "providers": dict(Counter(r["provider_used"] for r in records if r.get("provider_used"))),
    }


def summarize_llm(stage: dict, retrieval_preds: dict[str, bool]) -> dict:
    """Metrics for the LLM stage: the CVE-finding rule (``realistic``), the
    any-finding rule (``realistic_any_finding``), retrieval-only on the same
    scored items (``realistic_retrieval_same_items``), and the category hit rate
    among true positives."""
    recs = stage["records"]

    def rows(key: str) -> list[dict]:
        return [
            {"id": r["id"], "kind": r["kind"], "label": r["label"],
             "length_matched": r.get("length_matched"), "pred": r.get(key)}
            for r in recs
        ]

    same_items = [
        {"id": r["id"], "kind": r["kind"], "label": r["label"],
         "length_matched": r.get("length_matched"),
         "pred": retrieval_preds[r["id"]] if r.get("prediction") is not None else None}
        for r in recs
    ]
    tps = [r for r in recs if r.get("prediction") and r["label"] == "vulnerable"]
    hits = sum(str(r.get("category")) in r.get("flagged_categories", []) for r in tps)
    extra = {}
    if any(r.get("cve_finding") is not None for r in recs):
        # Review arms: how many reviewed findings also cite a shown CVE.
        extra["realistic_cve_finding"] = realistic_metrics(rows("cve_finding"))
    return {
        **extra,
        "realistic": realistic_metrics(rows("prediction")),
        "realistic_any_finding": realistic_metrics(rows("any_finding")),
        "realistic_retrieval_same_items": realistic_metrics(same_items),
        "category_hit_rate": round(hits / len(tps), 4) if tps else None,
    }


_UNSET = object()


def evaluate(
    embedder, store, reranker, dataset_path=DEFAULT_DATASET, sim_t=None, rerank_t=None,
    margin_t=_UNSET,
):
    """Score the pipeline at a single operating point (defaults to settings).

    ``dataset_path`` may be one path or a list. ``margin_t`` defaults to
    TWIN_MARGIN_MIN; pass None explicitly to score with the twin gate off.
    ``rerank_t`` defaults to RERANK_THRESHOLD with a reranker and to the no-op
    NO_RERANK_THRESHOLD without one (as the API ignores it when disabled).
    """
    sim_t = settings.SIM_THRESHOLD_CVE if sim_t is None else sim_t
    if rerank_t is None:
        rerank_t = NO_RERANK_THRESHOLD if reranker is None else settings.RERANK_THRESHOLD
    margin_t = settings.TWIN_MARGIN_MIN if margin_t is _UNSET else margin_t
    items = load_datasets(dataset_path)
    gathered, _timing = gather(items, embedder, store, reranker, settings.ANN_CANDIDATES)
    return score(gathered, sim_t, rerank_t, margin_t)


def evaluate_via_retriever(embedder, store, reranker, dataset_path=DEFAULT_DATASET) -> dict:
    """Operating-point metrics by running the REAL CVERetriever per snippet.

    Unlike the cosine sweep, this honours whatever retrieval mode is configured
    (dense gate or hybrid RRF fusion) — the fair way to compare hybrid vs dense.
    """
    from backend.app.core.cve_retriever import CVERetriever

    retriever = CVERetriever(embedder, store, reranker)
    tp = fp = fn = tn = cat_hits = 0
    for item in load_datasets(dataset_path):
        findings = retriever.find_vulnerabilities(item["code"], language=item.get("language"))
        predicted = len(findings) > 0
        is_vuln = item["label"] == "vulnerable"
        if predicted and is_vuln:
            tp += 1
            if findings[0].get("category") == item.get("category"):
                cat_hits += 1
        elif predicted and not is_vuln:
            fp += 1
        elif not predicted and is_vuln:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "mode": "hybrid" if settings.HYBRID_ENABLED else "dense",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "category_hit_rate": round(cat_hits / tp, 4) if tp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _fmt_margin(m: float | None) -> str:
    return "off" if m is None else f"{m:+.2f}"


def _print_table(rows: list[dict], top: int | None = 12):
    header = (
        f"{'sim_t':>6} {'rerank_t':>9} {'margin_t':>9} {'prec':>6} {'recall':>7} "
        f"{'f1':>6} {'cat_hit':>8} {'twin_cov':>9} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in rows if top is None else rows[:top]:
        print(
            f"{r['sim_threshold']:>6.2f} {r['rerank_threshold']:>9.2f} "
            f"{_fmt_margin(r.get('margin_threshold')):>9} "
            f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
            f"{r['category_hit_rate']:>8.3f} {r.get('twin_coverage', 0.0):>9.3f} "
            f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4}"
        )


def _rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CVE detection eval.")
    parser.add_argument(
        "--dataset", nargs="+", default=[str(DEFAULT_DATASET)],
        help="One or more JSONL eval sets (combined in order).",
    )
    parser.add_argument(
        "--sim-sweep", default=DEFAULT_SIM_SWEEP,
        help=f"start:stop:step for sim_t (default {DEFAULT_SIM_SWEEP}); "
        "SIM_THRESHOLD_CVE is always added.",
    )
    parser.add_argument(
        "--rerank-sweep", default=None,
        help=f"start:stop:step for rerank_t (default {DEFAULT_RERANK_SWEEP}). Without "
        "a reranker it collapses to a single no-op value; passing it is an error.",
    )
    parser.add_argument(
        "--margin-sweep", default=None,
        help="start:stop:step for the twin-margin gate (TWIN_MARGIN_MIN semantics), "
        "e.g. -0.10:0.20:0.02 ('off' is always included). Python < 3.14's argparse "
        "needs the --margin-sweep=-0.10:... form for a negative start.",
    )
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank", action="store_true",
        help="Force the cross-encoder on for this run (default: RERANKER_ENABLED, "
        f"currently {settings.RERANKER_ENABLED}).",
    )
    rerank_group.add_argument(
        "--no-rerank", action="store_true",
        help="Force the cross-encoder off (never loaded); rank candidates by "
        "similarity. Fast retrieval-only metrics.",
    )
    parser.add_argument(
        "--reranker-model", default=None,
        help="Override RERANKER_MODEL for this run (e.g. BAAI/bge-reranker-base).",
    )
    parser.add_argument(
        "--reranker-max-tokens", type=int, default=None,
        help="Override RERANKER_MAX_TOKENS for this run.",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Evaluate a label-balanced sample of ~N items (OSV vuln/safe pairs kept "
        "together) from the combined datasets. Not allowed with --write-baseline.",
    )
    parser.add_argument(
        "--sample-kinds", default=None, metavar="KIND=N,...",
        help="Stratified sample with per-kind quotas, e.g. "
        "vulnerable=40,fixed_twin=40,ordinary=80 (kinds: " + ", ".join(KINDS) + "; "
        "unlisted kinds are left out; vuln/twin pairs kept together). Deterministic "
        "with --seed. Not allowed with --sample or --write-baseline.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for --sample / --sample-kinds (default 42).",
    )
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--out", default=None, help="Write full sweep results JSON here.")

    llm = parser.add_argument_group(
        "LLM report stage (real provider calls; results go to --out only)"
    )
    llm.add_argument(
        "--llm", action="store_true",
        help="Also run the LLM review stage on every item (legacy: every item with "
        "retrieved CVEs). Requires --llm-max-calls and --out.",
    )
    llm.add_argument(
        "--llm-prompt", choices=LLM_PROMPT_ARMS, default="current",
        help="current (default): the production review prompt with retrieved CVEs and "
        "Semgrep evidence (snippet mode, so no guard_diff); no_retrieval: the same with "
        "zero CVEs; legacy: the pre-2026-09-24 retrieval-only prompt "
        "(ml/evaluation/legacy_prompt.py).",
    )
    llm.add_argument(
        "--llm-max-calls", type=int, default=None, metavar="N",
        help=f"Hard cap on real LLM calls this run (1..{LLM_MAX_CALLS_CAP}; cached items "
        "don't count). Required with --llm.",
    )
    llm.add_argument(
        "--llm-sleep", type=float, default=DEFAULT_LLM_SLEEP, metavar="S",
        help=f"Minimum seconds between real calls (default {DEFAULT_LLM_SLEEP}; Groq "
        "free tier is ~30 requests/min).",
    )
    llm.add_argument(
        "--llm-tpm", type=int, default=DEFAULT_LLM_TPM, metavar="T",
        help=f"Pace calls to stay under T tokens/minute: wait at least 60*tokens/T s "
        f"after each call (default {DEFAULT_LLM_TPM}, Groq free tier; 0 = off).",
    )
    llm.add_argument(
        "--llm-token-budget", type=int, default=None, metavar="T",
        help="Stop calling before total tokens this run would exceed T (Groq free "
        "tier: 200K tokens/day per model).",
    )
    llm.add_argument(
        "--llm-max-rate-limit-errors", type=int, default=DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS,
        metavar="K",
        help="Stop after K consecutive rate-limited failures (default "
        f"{DEFAULT_LLM_MAX_RATE_LIMIT_ERRORS}); a daily-limit error stops at once.",
    )
    llm.add_argument(
        "--llm-cache", default=str(DEFAULT_LLM_CACHE), metavar="PATH",
        help="JSONL cache of LLM results keyed by (item id, prompt sha256, model); "
        "a re-run resumes from it (default ml/evaluation/results/llm_cache.jsonl).",
    )
    llm.add_argument(
        "--llm-primary-only", action="store_true",
        help="Use only the primary provider/model (no same-provider fallback model "
        "such as GROQ_FALLBACK_MODEL / OPENROUTER_FALLBACK_MODEL, no "
        "LLM_FALLBACK_PROVIDER), so every scored item comes from one model.",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse + validate CLI args. Every refusal happens here, before any model load.

    Afterwards ``args.no_rerank`` is the resolved decision (True = no reranker),
    whether it came from --no-rerank, --rerank, an override, or the setting.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write_baseline and args.sample is not None:
        parser.error(
            "--write-baseline cannot be combined with --sample: a sample must never "
            "become the calibrated baseline (run on the full dataset instead)."
        )
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive integer.")
    if args.sample_kinds is not None:
        if args.sample is not None:
            parser.error("--sample and --sample-kinds are mutually exclusive.")
        if args.write_baseline:
            parser.error(
                "--write-baseline cannot be combined with --sample-kinds: a sample must "
                "never become the calibrated baseline."
            )
        try:
            args.sample_kinds = parse_sample_kinds(args.sample_kinds)
        except ValueError as exc:
            parser.error(f"--sample-kinds: {exc}")
    if args.llm:
        if args.write_baseline:
            parser.error(
                "--write-baseline cannot be combined with --llm: LLM results vary across "
                "model versions; they are recorded in --out only."
            )
        if args.llm_max_calls is None:
            parser.error("--llm requires --llm-max-calls N (a hard cap on real calls).")
        if not 1 <= args.llm_max_calls <= LLM_MAX_CALLS_CAP:
            parser.error(f"--llm-max-calls must be between 1 and {LLM_MAX_CALLS_CAP}.")
        if args.out is None:
            parser.error("--llm requires --out (where the LLM results are recorded).")
        if args.llm_sleep < 0 or args.llm_tpm < 0:
            parser.error("--llm-sleep and --llm-tpm must be >= 0.")
        if args.llm_token_budget is not None and args.llm_token_budget <= 0:
            parser.error("--llm-token-budget must be a positive integer.")
        if args.llm_max_rate_limit_errors < 1:
            parser.error("--llm-max-rate-limit-errors must be >= 1.")
    else:
        given = [
            flag for flag, value, default in (
                ("--llm-max-calls", args.llm_max_calls, None),
                ("--llm-token-budget", args.llm_token_budget, None),
                ("--llm-primary-only", args.llm_primary_only, False),
                ("--llm-prompt", args.llm_prompt, "current"),
            ) if value != default
        ]
        if given:
            parser.error(f"{', '.join(given)} requires --llm.")
    if args.reranker_max_tokens is not None and args.reranker_max_tokens <= 0:
        parser.error("--reranker-max-tokens must be a positive integer.")
    if args.no_rerank:
        if args.rerank_sweep is not None:
            parser.error(
                "--rerank-sweep requires a reranker: rerank_t is meaningless with "
                "--no-rerank (use --sim-sweep)."
            )
        if args.reranker_model is not None or args.reranker_max_tokens is not None:
            parser.error(
                "--reranker-model/--reranker-max-tokens cannot be combined with --no-rerank."
            )
    args.no_rerank = not use_reranker(
        True if args.no_rerank else (False if args.rerank else None),
        args.reranker_model, args.reranker_max_tokens,
    )
    if args.no_rerank:
        if args.rerank_sweep is not None:
            parser.error(
                "--rerank-sweep requires a reranker: RERANKER_ENABLED is off, so pass "
                "--rerank (or use --sim-sweep)."
            )
    elif args.rerank_sweep is None:
        args.rerank_sweep = DEFAULT_RERANK_SWEEP
    return args


def build_llm_router(primary_only: bool = False) -> LLMRouter:
    """The production router (fails fast on a missing key), optionally trimmed
    to its primary client."""
    # No per-minute token pacing in the router: the eval paces its own calls
    # (--llm-sleep / --llm-tpm), and pacing twice would double every wait.
    # Retry-After is still honoured.
    router = LLMRouter(pacer=TokenPacer({}))
    if primary_only and not router.mock:
        router.clients = router.clients[:1]
    return router


def build_semgrep_scanner():
    """The production scanner settings (None when SEMGREP_ENABLED is off)."""
    if not settings.SEMGREP_ENABLED:
        return None
    from backend.app.core.semgrep_scanner import SemgrepScanner

    return SemgrepScanner(timeout_s=max(settings.SEMGREP_TIMEOUT_S, 300.0),
                          exclude_rules=frozenset(settings.SEMGREP_EXCLUDED_RULES))


def run_llm(args, entries: list[dict], router, retrieval_preds: dict[str, bool]) -> dict:
    """Run + summarize the LLM stage for main(); returns the ``llm`` out-JSON block."""
    arm = args.llm_prompt
    cache = LLMCache(args.llm_cache)
    chain = ["mock"] if router.mock else [c.label for c in router.clients]
    semgrep_summary = None
    if arm != "legacy":
        semgrep_summary = attach_semgrep(entries, build_semgrep_scanner())
        print(f"Semgrep evidence (snippet mode): {semgrep_summary}")
    n_need = sum(bool(e["cves"]) for e in entries) if arm == "legacy" else len(entries)
    print(
        f"\nLLM stage ({arm} prompt): {n_need}/{len(entries)} items need a call; chain "
        f"{' -> '.join(chain)}; max {args.llm_max_calls} calls; cache {args.llm_cache} "
        f"({len(cache)} entries)."
    )

    async def _run():
        with HttpUsageTap() as tap:
            return await run_llm_stage(
                entries, router, cache=cache, max_calls=args.llm_max_calls,
                sleep_s=args.llm_sleep, tpm=args.llm_tpm,
                max_rate_limit_errors=args.llm_max_rate_limit_errors,
                token_budget=args.llm_token_budget, tap=tap, arm=arm,
            )

    stage = asyncio.run(_run())
    summary = summarize_llm(stage, retrieval_preds)
    counts = stage["status_counts"]
    print(
        f"LLM stage done: {stage['calls']} calls, {stage['tokens_used']} tokens; statuses "
        f"{counts}; providers {stage['providers']}"
        + (f"; stopped early: {stage['stopped']}" if stage["stopped"] else "")
    )
    if counts.get("not_run"):
        print(f"  {counts['not_run']} items not run — re-run the same command to resume.")
    print_realistic(f"LLM stage, {arm} prompt ({ARM_RULES[arm]})", summary["realistic"])
    if arm == "legacy":
        print_realistic("LLM stage, legacy prompt, any validated finding",
                        summary["realistic_any_finding"])
    print_realistic(
        "Retrieval-only on the same scored items", summary["realistic_retrieval_same_items"]
    )
    return {
        "config": {
            "model": stage["model"],
            "chain": chain,
            "primary_only": args.llm_primary_only,
            "max_calls": args.llm_max_calls,
            "sleep_s": args.llm_sleep,
            "tpm": args.llm_tpm,
            "token_budget": args.llm_token_budget,
            "cache": _rel(args.llm_cache),
            "top_k": settings.RETRIEVAL_TOP_K,
            "prompt": arm,
            "rule": ARM_RULES[arm],
            "cves_per_unit": None if arm == "legacy" else (
                0 if arm == "no_retrieval" else settings.LLM_MAX_CVES_PER_UNIT),
            "max_prompt_tokens": None if arm == "legacy" else settings.LLM_MAX_PROMPT_TOKENS,
            "semgrep": semgrep_summary,
            "guard_diff": "not applicable: snippet mode has no previous version of the code",
        },
        "calls": stage["calls"],
        "tokens_used": stage["tokens_used"],
        "stopped": stage["stopped"],
        "status_counts": counts,
        "providers": stage["providers"],
        **summary,
        "items": [
            {**r, "retrieval_pred": retrieval_preds[r["id"]]} for r in stage["records"]
        ],
    }


def main(argv=None):
    args = parse_args(argv)
    rr_cfg = reranker_config(args.no_rerank, args.reranker_model, args.reranker_max_tokens)
    sampled = args.sample is not None or args.sample_kinds is not None
    sample_meta = {
        "sample": args.sample,
        "sample_kinds": args.sample_kinds,
        "seed": args.seed if sampled else None,
    }
    # Before any model load: a missing LLM key fails here, not after the gather.
    router = build_llm_router(args.llm_primary_only) if args.llm else None

    print(f"Model: {settings.EMBEDDING_MODEL} (pooling={settings.EMBEDDING_POOLING})")
    if rr_cfg is None:
        print("Reranker: off (candidates ranked by similarity; --rerank to enable)")
    else:
        print(f"Reranker: {rr_cfg['model']} (max_tokens={rr_cfg['max_tokens']})")
    if args.sample is not None:
        print(f"Sample: {args.sample} (seed={args.seed})")
    if args.sample_kinds is not None:
        print(f"Stratified sample: {args.sample_kinds} (seed={args.seed})")
    embedder, store, reranker = build_components(
        args.no_rerank, args.reranker_model, args.reranker_max_tokens
    )

    if store.cve_schema_error:
        print(store.cve_schema_error)
        return
    seeded = store.count(store.cve_collection)
    if seeded == 0:
        print("cve_corpus is empty — run scripts/ingest_cve_corpus.py first.")
        return

    items = load_datasets(args.dataset)
    print(f"Loaded {len(items)} eval snippets; corpus has {seeded} points.")
    if args.sample is not None:
        pool = len(items)
        items = sample_items(items, args.sample, args.seed)
        n_vuln = sum(i["label"] == "vulnerable" for i in items)
        note = " (requested >= pool: using the whole pool)" if args.sample >= pool else ""
        print(
            f"Sampled {len(items)}/{pool} items: {n_vuln} vulnerable, "
            f"{len(items) - n_vuln} safe{note}."
        )
    if args.sample_kinds is not None:
        pool = len(items)
        items = sample_items_by_kind(items, args.sample_kinds, args.seed)
        got = Counter(item_kind(i) for i in items)
        sample_meta["sample_kinds_achieved"] = {k: got[k] for k in args.sample_kinds}
        short = {k: f"{got[k]}/{q}" for k, q in args.sample_kinds.items() if got[k] < q}
        print(
            f"Sampled {len(items)}/{pool} items by kind: {dict(got)}"
            + (f" (short of quota: {short})" if short else "") + "."
        )
    print(f"Items by kind: {dict(Counter(item_kind(i) for i in items))}")
    gathered, timing = gather(
        items, embedder, store, reranker, settings.ANN_CANDIDATES, progress=True
    )
    print(
        f"Gathered {timing['n_items']} items in {timing['total_seconds']:.1f}s "
        f"({timing['seconds_per_item']:.2f}s/item; embed {timing['embed_seconds']:.1f}s, "
        f"ANN {timing['ann_seconds']:.1f}s, rerank {timing['rerank_seconds']:.1f}s)."
    )
    run_meta = {"reranker": rr_cfg, **sample_meta, "timing": timing}

    n_cands = sum(len(g["candidates"]) for g in gathered)
    n_twins = sum(c["twin_margin"] is not None for g in gathered for c in g["candidates"])
    print(f"Candidates with a patched twin: {n_twins}/{n_cands}.")

    # Always score the operating point, whatever range was asked for.
    sim_values = sorted(set(frange(args.sim_sweep)) | {settings.SIM_THRESHOLD_CVE})
    # Without a reranker, rerank_prob == similarity, so a rerank sweep would only
    # re-apply the sim gate and multiply the table out: collapse it to one no-op.
    rerank_values = [NO_RERANK_THRESHOLD] if args.no_rerank else frange(args.rerank_sweep)
    op_rerank_t = NO_RERANK_THRESHOLD if args.no_rerank else settings.RERANK_THRESHOLD
    margin_values: list[float | None] = [None]
    if args.margin_sweep:
        margin_values += frange(args.margin_sweep)
    sweep = [
        score(gathered, s, r, m)
        for s in sim_values for r in rerank_values for m in margin_values
    ]
    sweep.sort(key=lambda m: (m["f1"], m["recall"]), reverse=True)

    print("\nTop threshold combinations by F1:")
    _print_table(sweep)

    operating = score(
        gathered, settings.SIM_THRESHOLD_CVE, op_rerank_t, settings.TWIN_MARGIN_MIN,
    )
    if args.margin_sweep:
        # The question the sweep answers: at today's operating point, what does
        # each margin gate buy/cost?
        print(
            f"\nTwin-margin sweep at the operating point (sim={settings.SIM_THRESHOLD_CVE}, "
            f"rerank={op_rerank_t}):"
        )
        _print_table(
            [
                score(gathered, settings.SIM_THRESHOLD_CVE, op_rerank_t, m)
                for m in margin_values
            ],
            top=None,
        )

    print(
        f"\nOperating point (settings sim={settings.SIM_THRESHOLD_CVE}, "
        f"rerank={op_rerank_t}, "
        f"twin_margin={_fmt_margin(settings.TWIN_MARGIN_MIN)}): "
        f"P={operating['precision']} R={operating['recall']} F1={operating['f1']} "
        f"twin_coverage={operating['twin_coverage']}"
    )
    print(
        f"Wall time: {timing['total_seconds']:.1f}s total, "
        f"{timing['seconds_per_item']:.2f}s/item."
    )

    # Realistic metrics at the operating point: retrieval-only prediction =
    # "at least one CVE match survives" (what the API would send to the LLM).
    entries = build_llm_entries(
        gathered, settings.SIM_THRESHOLD_CVE, op_rerank_t, settings.TWIN_MARGIN_MIN,
        settings.RETRIEVAL_TOP_K,
    )
    retrieval_preds = {e["id"]: bool(e["cves"]) for e in entries}
    realistic = realistic_metrics(
        [{"id": e["id"], "kind": e["kind"], "label": e["label"],
          "pred": retrieval_preds[e["id"]]} for e in entries]
    )
    print_realistic("Realistic metrics, retrieval only (operating point)", realistic)

    llm_block = run_llm(args, entries, router, retrieval_preds) if router is not None else None

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = {**run_meta, "operating": operating, "sweep": sweep, "realistic": realistic}
        if llm_block is not None:
            out["llm"] = llm_block
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote sweep results -> {out_path}")

    if args.write_baseline:
        baseline = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "pooling": settings.EMBEDDING_POOLING,
            "sim_threshold_cve": settings.SIM_THRESHOLD_CVE,
            "sim_threshold_team": settings.SIM_THRESHOLD_TEAM,
            "rerank_threshold": op_rerank_t,
            "twin_margin_min": settings.TWIN_MARGIN_MIN,
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "category_hit_rate": operating["category_hit_rate"],
            "twin_coverage": operating["twin_coverage"],
            "dataset_sha256": dataset_sha256(args.dataset),
            "datasets": [_rel(p) for p in args.dataset],
            "n": operating["n"],
            "best_operating_point": {
                k: sweep[0][k]
                for k in ("sim_threshold", "rerank_threshold", "margin_threshold", "f1")
            },
            **run_meta,
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Wrote baseline -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
