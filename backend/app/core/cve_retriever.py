from typing import Any

from backend.app.core.embedder import Embedder
from backend.app.core.reranker import Reranker
from backend.app.core.vector_store import VectorStore


class CVERetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def find_vulnerabilities(
        self, code_snippet: str, limit: int = 1, threshold: float = 0.85
    ) -> list[dict[str, Any]]:
        """
        Embed a developer's code snippet and search the CVE corpus for semantic matches.
        Uses a Cross-Encoder to rerank the top 5 base vector matches.
        """
        # 1. Convert developer's code to a vector
        query_vector = self.embedder.embed_text(code_snippet)
        
        # 2. Search Qdrant for top 5 broad matches (high recall)
        broad_matches = self.vector_store.search_cves(query_vector, limit=5)
        
        # Filter out absolute garbage matches first
        viable_matches = [
            res for res in broad_matches 
            if res.get("similarity_score", 0.0) >= threshold
        ]
        
        # 3. Rerank the viable matches using the Cross-Encoder (high precision)
        final_matches = self.reranker.rerank(code_snippet, viable_matches, top_k=limit)
        
        return final_matches

# Simple CLI test runner if executed directly
if __name__ == "__main__":
    print("Initializing components (this will load CodeBERT + MS-MARCO CrossEncoder)...")
    retriever = CVERetriever(Embedder(), VectorStore(), Reranker())
    
    # A piece of code written by a hypothetical developer that we are reviewing
    test_code = """
def delete_user_account(db, user_id):
    sql = "DELETE FROM users WHERE id = '" + user_id + "'"
    db.execute(sql)
    """
    
    print("\nAnalyzing developer code:")
    print(test_code)
    print("-" * 40)
    
    print("Ghost Hunter is scanning against CVE memory...")
    # Fetch top 1 after reranking
    findings = retriever.find_vulnerabilities(test_code, limit=1, threshold=0.75)
    
    if findings:
        print("\n🚨 VULNERABILITY FOUND!")
        for finding in findings:
            print(f"CVE ID: {finding['cve_id']}")
            print(f"Severity: {finding['severity']}/10.0")
            print(f"Description: {finding['description']}")
            print(f"Base Vector Score: {finding.get('similarity_score', 0.0):.4f}")
            print(f"Reranker Confidence: {finding.get('rerank_score', 0.0):.4f}")
    else:
        print("\n✅ Code looks safe. No CVE patterns matched.")

