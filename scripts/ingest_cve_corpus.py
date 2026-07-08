"""Ghost Hunter ingestion — seed the ``cve_corpus`` Qdrant collection.

Reads illustrative CVE records from ``data/cve_corpus/*.json`` (see
``sample_cves.json``; in production this would parse NVD JSON feeds), embeds the
vulnerable code with the configured embedding model, and upserts them.

Point IDs are ``uuid5(cve_id)`` so re-running upserts in place instead of
duplicating points. Pass ``--recreate`` to drop and rebuild the collection.

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
from backend.app.core.vector_store import VectorStore  # noqa: E402

DEFAULT_DATA_DIR = BASE_DIR / "data" / "cve_corpus"

# Namespace for deterministic point IDs derived from the CVE id.
_ID_NAMESPACE = uuid.NAMESPACE_URL


def load_cves(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(str(data_dir / "*.json"))):
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


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

    cves = load_cves(Path(args.data_dir))
    if not cves:
        print(f"No CVE records found in {args.data_dir}.")
        return
    print(f"Loaded {len(cves)} CVE records for ingestion.")

    texts_to_embed = []
    payloads = []
    ids = []

    for cve in cves:
        # Embed the vulnerable code so we can match it against developer code.
        texts_to_embed.append(cve["vulnerable_code"])
        payloads.append(
            {
                "cve_id": cve["cve_id"],
                "category": cve.get("category", "unknown"),
                "description": cve["description"],
                "severity": cve["severity"],
                "language": cve["language"],
                # Store the vulnerable code so the reranker can compare
                # code-against-code instead of code-against-description.
                "vulnerable_code": cve["vulnerable_code"],
                "source": cve.get("source", "nvd_sample"),
                "embedding_model": settings.EMBEDDING_MODEL,
            }
        )
        ids.append(str(uuid.uuid5(_ID_NAMESPACE, cve["cve_id"])))

    print(f"Embedding CVE code snippets via {settings.EMBEDDING_MODEL}...")
    embeddings = embedder.embed_texts(texts_to_embed)

    print("Inserting embeddings into Qdrant 'cve_corpus' collection...")
    vector_store.insert_cves(embeddings, payloads, ids=ids)

    total = vector_store.count(vector_store.cve_collection)
    print(f"Ingestion complete! cve_corpus now holds {total} points.")


if __name__ == "__main__":
    main()
