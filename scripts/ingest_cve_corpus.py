"""Ghost Hunter ingestion — seed the ``cve_corpus`` Qdrant collection.

Reads CVE records from ``data/cve_corpus/*.json`` (the handwritten
``sample_cves.json`` plus anything ``scripts/build_corpus_from_osv.py`` emits),
embeds each entry's vulnerable code — and its patched twin (``fixed_code``) when
present — with the configured embedding model, and upserts them.

Record schema (only ``cve_id`` + ``vulnerable_code`` are required; handwritten
entries omit the provenance fields and ``fixed_code``)::

    cve_id, category, cwe_id, description, severity, language,
    vulnerable_code, fixed_code, source, repo, commit, file_path, function_name

Each point has a named ``vuln`` vector and, for entries with a fix, a ``fixed``
vector (ROADMAP 1b: retrieval scores ``cos(q, vuln) - cos(q, fixed)``). This
vector layout replaced the single unnamed vector, so an existing collection must
be rebuilt once with ``--recreate`` (the API refuses to search the old layout).

Point IDs are deterministic so re-running upserts in place: ``uuid5(cve_id)``
for entries without a location (every handwritten entry — these IDs are what
feedback votes are keyed on, so they must not change), and
``uuid5("cve_id|file_path|function_name")`` for mined entries, since one advisory
can yield several fixed functions.

Note: the local Qdrant store (``./qdrant_data``) is single-process — stop the API
server before running this.
"""
import argparse
import glob
import json
import sys
import uuid
from pathlib import Path

# Add project root to path so we can import backend modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.config import BASE_DIR, settings  # noqa: E402
from backend.app.core.embedder import Embedder  # noqa: E402
from backend.app.core.embedding_cache import EmbeddingCache  # noqa: E402
from backend.app.core.sparse_encoder import encode as sparse_encode  # noqa: E402
from backend.app.core.vector_store import VectorStore  # noqa: E402

DEFAULT_DATA_DIR = BASE_DIR / "data" / "cve_corpus"

# Namespace for deterministic point IDs derived from the CVE id.
_ID_NAMESPACE = uuid.NAMESPACE_URL

# Provenance fields copied verbatim into the payload (None when absent).
_OPTIONAL_FIELDS = ("cwe_id", "repo", "commit", "file_path", "function_name")


def load_cves(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(str(data_dir / "*.json"))):
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def point_id_for(record: dict) -> str:
    """Deterministic Qdrant point id for a corpus record (see module docstring)."""
    file_path = record.get("file_path")
    function_name = record.get("function_name")
    if file_path or function_name:
        key = f"{record['cve_id']}|{file_path or ''}|{function_name or ''}"
    else:
        key = record["cve_id"]
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def _normalize_fixed(record: dict) -> str | None:
    """The patched twin, or None when absent/empty/identical to the vulnerable
    code (an identical twin would pin twin_margin at 0 and mean nothing)."""
    fixed = record.get("fixed_code")
    if not isinstance(fixed, str) or not fixed.strip():
        return None
    if fixed.split() == record["vulnerable_code"].split():
        return None
    return fixed


def prepare_records(
    records: list[dict],
) -> tuple[list[str], list[dict], list[str], list[str | None]]:
    """Validate records and build (ids, payloads, vulnerable_texts, fixed_texts).

    Records missing ``cve_id``/``vulnerable_code`` are skipped; a record whose id
    repeats an earlier one replaces it (last wins, like the upsert would).
    """
    by_id: dict[str, tuple[dict, str, str | None]] = {}
    skipped = 0
    for rec in records:
        if not rec.get("cve_id") or not (rec.get("vulnerable_code") or "").strip():
            skipped += 1
            continue
        fixed = _normalize_fixed(rec)
        payload = {
            "cve_id": rec["cve_id"],
            "category": rec.get("category") or "other",
            "description": rec.get("description") or "",
            "severity": rec.get("severity"),
            "language": rec.get("language"),
            # Store the vulnerable code so the reranker can compare code-against-
            # code, and the fix so the LLM prompt can show how it was patched.
            "vulnerable_code": rec["vulnerable_code"],
            "fixed_code": fixed,
            "source": rec.get("source") or "handwritten",
            **{k: rec.get(k) for k in _OPTIONAL_FIELDS},
            "embedding_model": settings.EMBEDDING_MODEL,
        }
        pid = point_id_for(rec)
        by_id.pop(pid, None)  # re-insert so a later duplicate also takes the later slot
        by_id[pid] = (payload, rec["vulnerable_code"], fixed)

    duplicates = len(records) - skipped - len(by_id)
    if skipped:
        print(f"Skipped {skipped} record(s) missing cve_id or vulnerable_code.")
    if duplicates:
        print(f"Collapsed {duplicates} duplicate record(s) sharing a point id (last wins).")

    ids = list(by_id)
    payloads = [v[0] for v in by_id.values()]
    vuln_texts = [v[1] for v in by_id.values()]
    fixed_texts = [v[2] for v in by_id.values()]
    return ids, payloads, vuln_texts, fixed_texts


def embed_pairs(
    embedder, vuln_texts: list[str], fixed_texts: list[str | None]
) -> tuple[list[list[float]], list[list[float] | None]]:
    """Embed vulnerable + fixed code in ONE cache-aware batch, then split back."""
    fixed_present = [t for t in fixed_texts if t is not None]
    vectors = embedder.embed_texts(vuln_texts + fixed_present)
    vuln_vectors = vectors[: len(vuln_texts)]
    fixed_iter = iter(vectors[len(vuln_texts):])
    fixed_vectors = [next(fixed_iter) if t is not None else None for t in fixed_texts]
    return vuln_vectors, fixed_vectors


def main():
    parser = argparse.ArgumentParser(description="Seed the Ghost Hunter CVE corpus.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory of *.json CVE record files.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the cve_corpus collection before inserting.",
    )
    args = parser.parse_args()

    print("Initializing Ghost Hunter Ingestion Pipeline...")
    embedder = Embedder(cache=EmbeddingCache())
    vector_store = VectorStore()

    if args.recreate:
        print("Recreating 'cve_corpus' collection...")
        vector_store.recreate_collection(vector_store.cve_collection)
    elif vector_store.cve_schema_error:
        print(f"Cannot ingest: {vector_store.cve_schema_error}")
        sys.exit(1)

    cves = load_cves(Path(args.data_dir))
    if not cves:
        print(f"No CVE records found in {args.data_dir}.")
        return
    print(f"Loaded {len(cves)} CVE records for ingestion.")

    ids, payloads, vuln_texts, fixed_texts = prepare_records(cves)
    twins = sum(t is not None for t in fixed_texts)
    print(f"{len(ids)} points; {twins} with a patched twin (fixed_code).")

    print(f"Embedding CVE code snippets via {settings.EMBEDDING_MODEL}...")
    vuln_vectors, fixed_vectors = embed_pairs(embedder, vuln_texts, fixed_texts)
    sparse = [sparse_encode(t) for t in vuln_texts] if settings.HYBRID_ENABLED else None

    print("Inserting embeddings into Qdrant 'cve_corpus' collection...")
    vector_store.insert_cves(
        vuln_vectors, payloads, ids=ids, sparse_vectors=sparse, fixed_vectors=fixed_vectors
    )

    total = vector_store.count(vector_store.cve_collection)
    print(f"Ingestion complete! cve_corpus now holds {total} points.")


if __name__ == "__main__":
    main()
