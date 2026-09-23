"""Detection eval harness.

Measures the Ghost Hunter (CVE) retrieval pipeline against a labeled dataset of
vulnerable/safe snippets. Embeds + ANN-searches + reranks each snippet ONCE, then
sweeps (similarity_threshold x rerank_threshold) in-memory over the cached
candidates — so a full sweep costs one model pass, not one per threshold combo.

Run from the project root (stop the API first — local Qdrant is single-process):

    python -m ml.evaluation.run_eval \
        --sim-sweep 0.50:0.95:0.05 --rerank-sweep 0.0:0.9:0.1 --write-baseline
    python -m ml.evaluation.run_eval --margin-sweep -0.10:0.20:0.02 \
        --dataset ml/evaluation/datasets/detection_eval.jsonl <other.jsonl ...>
    python -m ml.evaluation.run_eval --no-rerank --dataset <a.jsonl> <b.jsonl ...>
    python -m ml.evaluation.run_eval --sample 150 --seed 42 \
        --reranker-model BAAI/bge-reranker-base --out ml/evaluation/results/base.json

The prediction rule at a given (sim_t, rerank_t, margin_t): keep candidates with
similarity_score >= sim_t and (margin_t off, or no patched twin, or twin_margin
>= margin_t — TWIN_MARGIN_MIN semantics); if the best-reranked survivor has
rerank_prob >= rerank_t, predict "vulnerable" with that candidate's category.

``twin_coverage`` is the fraction of eval items whose top candidate (best
reranked after the similarity gate, before the margin gate) has a patched twin —
the margin sweep only means something when it is well above zero.

``--no-rerank`` skips the cross-encoder entirely (never loaded): candidates are
ranked by similarity (rerank_prob := similarity_score) and the rerank sweep
collapses to a no-op — fast retrieval-only metrics on the full set.
``--sample N --seed S`` evaluates a deterministic, label-balanced subsample that
keeps OSV ``<prefix>_vuln``/``<prefix>_safe`` pairs together — for comparing
configs quickly; it can never be written as the baseline.
``--reranker-model`` / ``--reranker-max-tokens`` override the settings for this
run. Output JSON records the reranker actually used (null with --no-rerank),
the sample/seed, and gather() timing (embed / ANN / rerank seconds).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_ROOT))

from backend.app.config import BASE_DIR, settings  # noqa: E402
from backend.app.core.cve_retriever import passes_twin_margin  # noqa: E402
from backend.app.core.embedder import Embedder  # noqa: E402
from backend.app.core.embedding_cache import EmbeddingCache  # noqa: E402
from backend.app.core.reranker import Reranker  # noqa: E402
from backend.app.core.vector_store import VectorStore  # noqa: E402

DEFAULT_DATASET = BASE_DIR / "ml" / "evaluation" / "datasets" / "detection_eval.jsonl"
BASELINE_PATH = BASE_DIR / "ml" / "evaluation" / "baseline.json"
RESULTS_DIR = BASE_DIR / "ml" / "evaluation" / "results"
DEFAULT_RERANK_SWEEP = "0.0:0.9:0.1"
# rerank_t used with --no-rerank: rerank_prob is the similarity score there, and
# every candidate that survives the similarity gate (sim_t >= 0) already has
# similarity >= 0, so rerank_t = 0.0 is a no-op rather than a second sim gate.
NO_RERANK_THRESHOLD = 0.0
PAIR_SUFFIXES = ("_vuln", "_safe")


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
    """Concatenate one or more JSONL eval sets, in the order given."""
    items: list[dict] = []
    for path in _as_paths(dataset_path):
        items.extend(load_dataset(path))
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


def reranker_config(
    no_rerank: bool = False, model: str | None = None, max_tokens: int | None = None
) -> dict | None:
    """The reranker a run uses (overrides resolved against settings), or None.

    This is both what build_components() constructs and what output JSON records.
    """
    if no_rerank:
        return None
    return {
        "model": settings.RERANKER_MODEL if model is None else model,
        "max_tokens": settings.RERANKER_MAX_TOKENS if max_tokens is None else max_tokens,
    }


def build_components(
    no_rerank: bool = False, reranker_model: str | None = None,
    reranker_max_tokens: int | None = None,
):
    """(embedder, store, reranker). With ``no_rerank`` the reranker is None and
    no cross-encoder is ever constructed (no model load)."""
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


def sample_items(items: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministic, label-balanced subsample that keeps OSV pairs together.

    Items are grouped by ``_pair_key`` (a vuln/safe pair is one group; ids
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
    for idx, item in enumerate(items):
        groups.setdefault(_pair_key(item["id"]), []).append(idx)
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


def gather(
    items, embedder, store, reranker, ann_candidates: int, progress: bool = False
) -> tuple[list[dict], dict]:
    """Run the ANN (+ rerank) pipeline once per item; cache scored candidates.

    ``reranker=None`` (--no-rerank) skips the cross-encoder. Returns
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


_UNSET = object()


def evaluate(
    embedder, store, reranker, dataset_path=DEFAULT_DATASET, sim_t=None, rerank_t=None,
    margin_t=_UNSET,
):
    """Score the pipeline at a single operating point (defaults to settings).

    ``dataset_path`` may be one path or a list. ``margin_t`` defaults to
    TWIN_MARGIN_MIN; pass None explicitly to score with the twin gate off.
    """
    sim_t = settings.SIM_THRESHOLD_CVE if sim_t is None else sim_t
    rerank_t = settings.RERANK_THRESHOLD if rerank_t is None else rerank_t
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
    parser.add_argument("--sim-sweep", default="0.50:0.95:0.05")
    parser.add_argument(
        "--rerank-sweep", default=None,
        help=f"start:stop:step for rerank_t (default {DEFAULT_RERANK_SWEEP}). With "
        "--no-rerank it collapses to a single no-op value; passing it is an error.",
    )
    parser.add_argument(
        "--margin-sweep", default=None,
        help="start:stop:step for the twin-margin gate (TWIN_MARGIN_MIN semantics), "
        "e.g. -0.10:0.20:0.02 ('off' is always included). Python < 3.14's argparse "
        "needs the --margin-sweep=-0.10:... form for a negative start.",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip the cross-encoder (never loaded); rank candidates by similarity. "
        "Fast retrieval-only metrics.",
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
        "--seed", type=int, default=42, help="RNG seed for --sample (default 42)."
    )
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--out", default=None, help="Write full sweep results JSON here.")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse + validate CLI args. Every refusal happens here, before any model load."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write_baseline and args.sample is not None:
        parser.error(
            "--write-baseline cannot be combined with --sample: a sample must never "
            "become the calibrated baseline (run on the full dataset instead)."
        )
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive integer.")
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
    elif args.rerank_sweep is None:
        args.rerank_sweep = DEFAULT_RERANK_SWEEP
    return args


def main(argv=None):
    args = parse_args(argv)
    rr_cfg = reranker_config(args.no_rerank, args.reranker_model, args.reranker_max_tokens)
    sample_meta = {
        "sample": args.sample,
        "seed": args.seed if args.sample is not None else None,
    }

    print(f"Model: {settings.EMBEDDING_MODEL} (pooling={settings.EMBEDDING_POOLING})")
    if rr_cfg is None:
        print("Reranker: off (--no-rerank; candidates ranked by similarity)")
    else:
        print(f"Reranker: {rr_cfg['model']} (max_tokens={rr_cfg['max_tokens']})")
    if args.sample is not None:
        print(f"Sample: {args.sample} (seed={args.seed})")
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

    sim_values = frange(args.sim_sweep)
    # With --no-rerank, rerank_prob == similarity, so a rerank sweep would only
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

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({**run_meta, "operating": operating, "sweep": sweep}, indent=2)
        )
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
