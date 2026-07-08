from typing import Any

from backend.app.core.embedder import Embedder
from backend.app.core.reranker import Reranker
from backend.app.core.vector_store import VectorStore


class TeamRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def find_team_history(
        self, code_snippet: str, limit: int = 1, threshold: float = 0.85
    ) -> list[dict[str, Any]]:
        """
        Embed a developer's code snippet and search the Team Memory for past PRs
        where the team discussed similar code patterns.
        """
        # 1. Convert developer's code to a vector
        query_vector = self.embedder.embed_text(code_snippet)
        
        # 2. Search Qdrant for top broad matches in team history
        broad_matches = self.vector_store.search_team_history(query_vector, limit=5)
        
        # Filter viable matches based on base similarity
        viable_matches = [
            res for res in broad_matches 
            if res.get("similarity_score", 0.0) >= threshold
        ]
        
        # 3. Rerank the viable matches using the Cross-Encoder
        # For the reranker, we'll map the "snippet_preview" as the description to evaluate against
        for match in viable_matches:
            match["description"] = match.get("snippet_preview", "")
            
        final_matches = self.reranker.rerank(code_snippet, viable_matches, top_k=limit)
        
        return final_matches

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
    findings = retriever.find_team_history(test_code, limit=1, threshold=0.75)
    
    if findings:
        print("\n🧠 TEAM HAS SEEN THIS BEFORE!")
        for finding in findings:
            print(f"PR ID: {finding['pr_id']}")
            print(f"Author: {finding['author']}")
            print(f"Link: {finding['url']}")
            print(f"Preview: {finding['snippet_preview']}")
            print(f"Reranker Confidence: {finding.get('rerank_score', 0.0):.4f}")
    else:
        print("\n✅ Clean. The team hasn't complained about this pattern before.")
