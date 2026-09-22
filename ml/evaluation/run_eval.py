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

The prediction rule at a given (sim_t, rerank_t, margin_t): keep candidates with
similarity_score >= sim_t and (margin_t off, or no patched twin, or twin_margin
>= margin_t — TWIN_MARGIN_MIN semantics); if the best-reranked survivor has
rerank_prob >= rerank_t, predict "vulnerable" with that candidate's category.

``twin_coverage`` is the fraction of eval items whose top candidate (best
reranked after the similarity gate, before the margin gate) has a patched twin —
the margin sweep only means something when it is well above zero.
"""
import argparse
import hashlib
import json
import math
import sys
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


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


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


def build_components():
    return Embedder(cache=EmbeddingCache()), VectorStore(), Reranker()


def gather(items, embedder, store, reranker, ann_candidates: int) -> list[dict]:
    """Run the ANN + rerank pipeline once per item; cache scored candidates."""
    gathered = []
    for item in items:
        code = item["code"]
        vector = embedder.embed_text(code)
        candidates = store.search_cves(vector, limit=ann_candidates)
        # Mirror the retriever: rerank code-against-code (vulnerable_code), not
        # against the English description.
        for c in candidates:
            c["rerank_text"] = c.get("vulnerable_code") or c.get("description", "")
        if candidates:
            reranker.rerank(code, candidates, top_k=len(candidates))
        scored = [
            {
                "similarity_score": c.get("similarity_score", 0.0),
                "rerank_prob": sigmoid(float(c["rerank_score"])) if "rerank_score" in c else 0.0,
                "category": c.get("category", "unknown"),
                "cve_id": c.get("cve_id"),
                "sim_fixed": c.get("sim_fixed"),
                "twin_margin": c.get("twin_margin"),
            }
            for c in candidates
        ]
        gathered.append(
            {
                "id": item["id"],
                "is_vulnerable": item["label"] == "vulnerable",
                "category": item.get("category"),
                "candidates": scored,
            }
        )
    return gathered


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
    gathered = gather(items, embedder, store, reranker, settings.ANN_CANDIDATES)
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


def main():
    parser = argparse.ArgumentParser(description="Run the CVE detection eval.")
    parser.add_argument(
        "--dataset", nargs="+", default=[str(DEFAULT_DATASET)],
        help="One or more JSONL eval sets (combined in order).",
    )
    parser.add_argument("--sim-sweep", default="0.50:0.95:0.05")
    parser.add_argument("--rerank-sweep", default="0.0:0.9:0.1")
    parser.add_argument(
        "--margin-sweep", default=None,
        help="start:stop:step for the twin-margin gate (TWIN_MARGIN_MIN semantics), "
        "e.g. -0.10:0.20:0.02 ('off' is always included). Python < 3.14's argparse "
        "needs the --margin-sweep=-0.10:... form for a negative start.",
    )
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--out", default=None, help="Write full sweep results JSON here.")
    args = parser.parse_args()

    print(f"Model: {settings.EMBEDDING_MODEL} (pooling={settings.EMBEDDING_POOLING})")
    embedder, store, reranker = build_components()

    if store.cve_schema_error:
        print(store.cve_schema_error)
        return
    seeded = store.count(store.cve_collection)
    if seeded == 0:
        print("cve_corpus is empty — run scripts/ingest_cve_corpus.py first.")
        return

    items = load_datasets(args.dataset)
    print(f"Loaded {len(items)} eval snippets; corpus has {seeded} points.")
    gathered = gather(items, embedder, store, reranker, settings.ANN_CANDIDATES)

    n_cands = sum(len(g["candidates"]) for g in gathered)
    n_twins = sum(c["twin_margin"] is not None for g in gathered for c in g["candidates"])
    print(f"Candidates with a patched twin: {n_twins}/{n_cands}.")

    sim_values = frange(args.sim_sweep)
    rerank_values = frange(args.rerank_sweep)
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
        gathered, settings.SIM_THRESHOLD_CVE, settings.RERANK_THRESHOLD,
        settings.TWIN_MARGIN_MIN,
    )
    if args.margin_sweep:
        # The question the sweep answers: at today's operating point, what does
        # each margin gate buy/cost?
        print(
            f"\nTwin-margin sweep at the operating point (sim={settings.SIM_THRESHOLD_CVE}, "
            f"rerank={settings.RERANK_THRESHOLD}):"
        )
        _print_table(
            [
                score(gathered, settings.SIM_THRESHOLD_CVE, settings.RERANK_THRESHOLD, m)
                for m in margin_values
            ],
            top=None,
        )

    print(
        f"\nOperating point (settings sim={settings.SIM_THRESHOLD_CVE}, "
        f"rerank={settings.RERANK_THRESHOLD}, "
        f"twin_margin={_fmt_margin(settings.TWIN_MARGIN_MIN)}): "
        f"P={operating['precision']} R={operating['recall']} F1={operating['f1']} "
        f"twin_coverage={operating['twin_coverage']}"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"operating": operating, "sweep": sweep}, indent=2))
        print(f"Wrote sweep results -> {out_path}")

    if args.write_baseline:
        baseline = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "pooling": settings.EMBEDDING_POOLING,
            "sim_threshold_cve": settings.SIM_THRESHOLD_CVE,
            "sim_threshold_team": settings.SIM_THRESHOLD_TEAM,
            "rerank_threshold": settings.RERANK_THRESHOLD,
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
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Wrote baseline -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
