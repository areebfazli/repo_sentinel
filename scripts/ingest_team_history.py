import sys
from pathlib import Path

# Add project root to path so we can import backend modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.core.embedder import Embedder
from backend.app.core.vector_store import VectorStore
from backend.app.core.github_crawler import GithubCrawler
from backend.app.config import settings
from github import GithubException

# Fallback dataset in case the unauthenticated GitHub API rate limits us (60 req/hr)
MOCK_TEAM_HISTORY = [
    {
        "id": "pr_1042",
        "title": "Fix memory leak in data loader",
        "url": "https://github.com/internal/repo/pull/1042",
        "author": "sarah_dev",
        "text_content": """
PR: Fix memory leak in data loader
Description: We were holding onto file handles in the background thread.
Review Comment by senior_dev on file loader.py:
Make sure you use a context manager (with open...) here. Last time we left a file open in the loop it crashed production after 3 days.
Code Diff:
- f = open(filepath, 'r')
- data = f.read()
+ with open(filepath, 'r') as f:
+     data = f.read()
        """
    },
    {
        "id": "pr_899",
        "title": "Add retry logic to payment gateway",
        "url": "https://github.com/internal/repo/pull/899",
        "author": "alex_junior",
        "text_content": """
PR: Add retry logic to payment gateway
Description: Payment API is flaky, adding a bare except to retry.
Review Comment by lead_architect on file payments.py:
Never use a bare `except:`. It catches SystemExit and KeyboardInterrupt. We had a nightmare debugging this exact pattern in the auth service last year. Catch `requests.exceptions.RequestException` specifically.
Code Diff:
- try:
-     make_payment()
- except:
-     retry()
+ try:
+     make_payment()
+ except requests.exceptions.RequestException:
+     retry()
        """
    }
]

def main():
    print("Initializing Team Memory Ingestion Pipeline...")
    embedder = Embedder()
    vector_store = VectorStore()
    crawler = GithubCrawler(token=settings.GITHUB_TOKEN)
    
    repo_url = "https://github.com/pallets/flask" # A popular repo to test against
    
    print("Using mock internal PR history (bypassing unauthenticated GitHub rate limits)...")
    team_docs = MOCK_TEAM_HISTORY

    if not team_docs:
        print("No team history found to ingest.")
        return
        
    print(f"Embedding {len(team_docs)} team discussions/PRs...")
    
    texts_to_embed = []
    payloads = []
    
    for doc in team_docs:
        # We embed the entire discussion (PR description + review comments + diffs)
        texts_to_embed.append(doc["text_content"])
        
        # Payload strips out the giant text blob to save DB space, keeping metadata
        payload = {
            "pr_id": doc["id"],
            "title": doc["title"],
            "url": doc["url"],
            "author": doc["author"],
            "snippet_preview": doc["text_content"][:200] + "..." # A preview for the final report
        }
        payloads.append(payload)
        
    embeddings = embedder.embed_texts(texts_to_embed)
    
    print("Inserting embeddings into Qdrant 'team_history' collection...")
    vector_store.insert_team_history(embeddings, payloads)
    
    print("Ingestion complete! Team Memory is ready for retrieval.")

if __name__ == "__main__":
    main()
