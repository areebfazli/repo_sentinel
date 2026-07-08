"""Detection eval harness.

Measures the Ghost Hunter (CVE) retrieval pipeline against a labeled dataset of
vulnerable/safe snippets. Embeds + ANN-searches + reranks each snippet ONCE, then
sweeps (similarity_threshold x rerank_threshold) in-memory over the cached
candidates — so a full sweep costs one model pass, not one per threshold combo.

Run from the project root (stop the API first — local Qdrant is single-process):

    python -m ml.evaluation.run_eval \
        --sim-sweep 0.50:0.95:0.05 --rerank-sweep 0.0:0.9:0.1 --write-baseline

The prediction rule at a given (sim_t, rerank_t): keep candidates with
similarity_score >= sim_t; if the best-reranked survivor has rerank_prob >=
rerank_t, predict "vulnerable" with that candidate's category.
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


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def score(gathered: list[dict], sim_t: float, rerank_t: float) -> dict:
    tp = fp = fn = tn = 0
    category_hits = 0
    for g in gathered:
        passing = [c for c in g["candidates"] if c["similarity_score"] >= sim_t]
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
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "category_hit_rate": round(category_hits / tp, 4) if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(gathered),
    }


def evaluate(embedder, store, reranker, dataset_path=DEFAULT_DATASET, sim_t=None, rerank_t=None):
    """Score the pipeline at a single operating point (defaults to settings)."""
    sim_t = settings.SIM_THRESHOLD_CVE if sim_t is None else sim_t
    rerank_t = settings.RERANK_THRESHOLD if rerank_t is None else rerank_t
    items = load_dataset(Path(dataset_path))
    gathered = gather(items, embedder, store, reranker, settings.ANN_CANDIDATES)
    return score(gathered, sim_t, rerank_t)


def _print_table(rows: list[dict], top: int = 12):
    header = f"{'sim_t':>6} {'rerank_t':>9} {'prec':>6} {'recall':>7} {'f1':>6} {'cat_hit':>8}"
    print(header)
    print("-" * len(header))
    for r in rows[:top]:
        print(
            f"{r['sim_threshold']:>6.2f} {r['rerank_threshold']:>9.2f} "
            f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
            f"{r['category_hit_rate']:>8.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the CVE detection eval.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--sim-sweep", default="0.50:0.95:0.05")
    parser.add_argument("--rerank-sweep", default="0.0:0.9:0.1")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--out", default=None, help="Write full sweep results JSON here.")
    args = parser.parse_args()

    print(f"Model: {settings.EMBEDDING_MODEL} (pooling={settings.EMBEDDING_POOLING})")
    embedder, store, reranker = build_components()

    seeded = store.count(store.cve_collection)
    if seeded == 0:
        print("cve_corpus is empty — run scripts/ingest_cve_corpus.py first.")
        return

    items = load_dataset(Path(args.dataset))
    print(f"Loaded {len(items)} eval snippets; corpus has {seeded} points.")
    gathered = gather(items, embedder, store, reranker, settings.ANN_CANDIDATES)

    sim_values = frange(args.sim_sweep)
    rerank_values = frange(args.rerank_sweep)
    sweep = [score(gathered, s, r) for s in sim_values for r in rerank_values]
    sweep.sort(key=lambda m: (m["f1"], m["recall"]), reverse=True)

    print("\nTop threshold combinations by F1:")
    _print_table(sweep)

    operating = score(gathered, settings.SIM_THRESHOLD_CVE, settings.RERANK_THRESHOLD)
    print(
        f"\nOperating point (settings sim={settings.SIM_THRESHOLD_CVE}, "
        f"rerank={settings.RERANK_THRESHOLD}): "
        f"P={operating['precision']} R={operating['recall']} F1={operating['f1']}"
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
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "category_hit_rate": operating["category_hit_rate"],
            "dataset_sha256": dataset_sha256(Path(args.dataset)),
            "n": operating["n"],
            "best_operating_point": {
                k: sweep[0][k] for k in ("sim_threshold", "rerank_threshold", "f1")
            },
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Wrote baseline -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
