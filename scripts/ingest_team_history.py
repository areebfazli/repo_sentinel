"""Team Memory ingestion — seed the ``team_history`` Qdrant collection.

Crawls a repo's closed-PR review comments (one memory per comment, with its diff
hunk) via GithubCrawler, or uses a small mock dataset for offline demos.

    python scripts/ingest_team_history.py --repo owner/name   # real PRs (needs GITHUB_TOKEN)
    python scripts/ingest_team_history.py --mock              # hardcoded demo data

An incremental cursor (ingestion_state table) means --repo only fetches PRs newer
than the last run; pass --full to ignore it. Point IDs are uuid5 of the comment
id, so re-ingestion upserts in place. Stop the API first (local Qdrant is
single-process).
"""
import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings  # noqa: E402
from backend.app.core.embedder import Embedder  # noqa: E402
from backend.app.core.embedding_cache import EmbeddingCache  # noqa: E402
from backend.app.core.github_crawler import GithubCrawler  # noqa: E402
from backend.app.core.vector_store import VectorStore  # noqa: E402
from backend.app.db.models import IngestionState  # noqa: E402
from backend.app.db.session import SessionLocal, engine  # noqa: E402

_ID_NAMESPACE = uuid.NAMESPACE_URL

EXT_LANGUAGE = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".java": "java",
}

# Offline demo data, shaped like GithubCrawler docs (one per review comment).
MOCK_TEAM_HISTORY = [
    {
        "id": "pr1042_rc1",
        "pr_number": 1042,
        "pr_title": "Fix memory leak in data loader",
        "pr_url": "https://github.com/internal/repo/pull/1042",
        "comment_id": 1,
        "comment_url": "https://github.com/internal/repo/pull/1042#discussion_r1",
        "author": "senior_dev",
        "author_association": "MEMBER",
        "file_path": "loader.py",
        "created_at": "2026-05-01T10:00:00+00:00",
        "diff_hunk": "- f = open(filepath, 'r')\n- data = f.read()\n"
                     "+ with open(filepath, 'r') as f:\n+     data = f.read()",
        "body": "Use a context manager (with open...) here. Last time we left a file open "
                "in the loop it crashed production after 3 days.",
    },
    {
        "id": "pr899_rc1",
        "pr_number": 899,
        "pr_title": "Add retry logic to payment gateway",
        "pr_url": "https://github.com/internal/repo/pull/899",
        "comment_id": 2,
        "comment_url": "https://github.com/internal/repo/pull/899#discussion_r2",
        "author": "lead_architect",
        "author_association": "OWNER",
        "file_path": "payments.py",
        "created_at": "2025-11-15T09:00:00+00:00",
        "diff_hunk": "- try:\n-     make_payment()\n- except:\n-     retry()\n"
                     "+ try:\n+     make_payment()\n+ except requests.exceptions.RequestException:\n"
                     "+     retry()",
        "body": "Never use a bare `except:`. It catches SystemExit and KeyboardInterrupt. We had a "
                "nightmare debugging this exact pattern in the auth service last year. Catch "
                "`requests.exceptions.RequestException` specifically.",
    },
]


def _text_content(doc: dict) -> str:
    diff = doc.get("diff_hunk", "")
    head = f"Review by {doc.get('author')} on {doc.get('file_path')}:\n{doc.get('body', '')}"
    return f"{diff}\n\n{head}" if diff else head


def _language(file_path: str | None) -> str:
    if not file_path:
        return "unknown"
    return EXT_LANGUAGE.get(Path(file_path).suffix.lower(), "unknown")


def _get_cursor(repo: str) -> int | None:
    IngestionState.__table__.create(bind=engine, checkfirst=True)
    with SessionLocal() as session:
        state = session.get(IngestionState, repo)
        return state.last_pr_number if state else None


def _set_cursor(repo: str, last_pr_number: int) -> None:
    with SessionLocal() as session:
        state = session.get(IngestionState, repo)
        if state is None:
            state = IngestionState(repo=repo, last_pr_number=last_pr_number)
            session.add(state)
        else:
            state.last_pr_number = max(state.last_pr_number, last_pr_number)
        state.last_run_at = datetime.now(UTC)
        session.commit()


def main():
    parser = argparse.ArgumentParser(description="Seed the Team Memory collection.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="GitHub repo as owner/name (uses GITHUB_TOKEN).")
    group.add_argument("--mock", action="store_true", help="Use the hardcoded demo dataset.")
    parser.add_argument("--limit", type=int, default=200, help="Max PRs to crawl.")
    parser.add_argument("--full", action="store_true", help="Ignore the incremental cursor.")
    parser.add_argument("--recreate", action="store_true", help="Drop + rebuild the collection.")
    args = parser.parse_args()

    print("Initializing Team Memory Ingestion Pipeline...")
    embedder = Embedder(cache=EmbeddingCache())
    vector_store = VectorStore()

    if args.recreate:
        print("Recreating 'team_history' collection...")
        vector_store.recreate_collection(vector_store.team_collection)

    if args.mock:
        docs = MOCK_TEAM_HISTORY
        print(f"Using mock team history ({len(docs)} review comments).")
    else:
        since = None if args.full else _get_cursor(args.repo)
        print(f"Crawling {args.repo} (since PR #{since}, limit {args.limit})...")
        crawler = GithubCrawler(token=settings.GITHUB_TOKEN)
        docs = crawler.fetch_team_history(args.repo, since_pr_number=since, limit=args.limit)

    if not docs:
        print("No new team history to ingest.")
        return

    texts, payloads, ids = [], [], []
    for doc in docs:
        text = _text_content(doc)
        texts.append(text)
        payloads.append(
            {
                "pr_id": str(doc["pr_number"]),
                "pr_number": doc["pr_number"],
                "comment_id": doc.get("comment_id"),
                "title": doc.get("pr_title"),
                "url": doc.get("comment_url") or doc.get("pr_url"),
                "author": doc.get("author"),
                "author_association": doc.get("author_association", "NONE"),
                "created_at": doc.get("created_at"),
                "file_path": doc.get("file_path"),
                "language": _language(doc.get("file_path")),
                "text": text,
                "diff_hunk": doc.get("diff_hunk", ""),
                "embedding_model": settings.EMBEDDING_MODEL,
            }
        )
        ids.append(str(uuid.uuid5(_ID_NAMESPACE, doc["id"])))

    print(f"Embedding {len(texts)} review comments via {settings.EMBEDDING_MODEL}...")
    embeddings = embedder.embed_texts(texts)

    print("Inserting embeddings into Qdrant 'team_history' collection...")
    vector_store.insert_team_history(embeddings, payloads, ids=ids)

    if not args.mock:
        max_pr = max(doc["pr_number"] for doc in docs)
        _set_cursor(args.repo, max_pr)
        print(f"Updated crawl cursor for {args.repo} -> PR #{max_pr}.")

    total = vector_store.count(vector_store.team_collection)
    print(f"Ingestion complete! team_history now holds {total} points.")


if __name__ == "__main__":
    main()
