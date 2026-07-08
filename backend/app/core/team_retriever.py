from typing import Any

from backend.app.config import settings
from backend.app.core.embedder import Embedder
from backend.app.core.reranker import Reranker
from backend.app.core.vector_store import VectorStore


class TeamRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def find_team_history(
        self,
        code_snippet: str,
        language: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Embed developer code and search Team Memory for related past PR reviews.

        Same recall -> similarity gate -> rerank -> probability gate pipeline as the
        CVE side. Reranks against the stored discussion text. Thresholds default to
        settings.
        """
        limit = settings.RETRIEVAL_TOP_K if limit is None else limit
        threshold = settings.SIM_THRESHOLD_TEAM if threshold is None else threshold

        query_vector = self.embedder.embed_text(code_snippet)

        broad_matches = self.vector_store.search_team_history(
            query_vector, limit=settings.ANN_CANDIDATES, language=language
        )

        viable_matches = [
            res for res in broad_matches
            if res.get("similarity_score", 0.0) >= threshold
        ]

        # Rerank against the discussion text (full text once Phase 5 stores it;
        # snippet_preview is the current fallback).
        for match in viable_matches:
            match["rerank_text"] = match.get("text") or match.get("snippet_preview", "")

        reranked = self.reranker.rerank(code_snippet, viable_matches, top_k=limit)
        return [
            m for m in reranked
            if m.get("rerank_prob", 0.0) >= settings.RERANK_THRESHOLD
        ]


# Simple CLI test runner if executed directly
if __name__ == "__main__":
    print("Initializing components...")
    retriever = TeamRetriever(Embedder(), VectorStore(), Reranker())

    # A piece of code written by a hypothetical developer that we are reviewing
    test_code = """
def fetch_external_data():
    try:
        response = make_api_call()
        return response.json()
    except:
        # Just fail silently for now
        return None
    """

    print("\nAnalyzing developer code:")
    print(test_code)
    print("-" * 40)

    print("Team Memory is scanning past PR reviews...")
    findings = retriever.find_team_history(test_code)

    if findings:
        print("\n🧠 TEAM HAS SEEN THIS BEFORE!")
        for finding in findings:
            print(f"PR ID: {finding['pr_id']}")
            print(f"Author: {finding['author']}")
            print(f"Link: {finding['url']}")
            print(f"Preview: {finding.get('snippet_preview', finding.get('text', ''))[:120]}")
            print(f"Reranker Confidence: {finding.get('rerank_prob', 0.0):.4f}")
    else:
        print("\n✅ Clean. The team hasn't complained about this pattern before.")
