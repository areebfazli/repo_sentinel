from typing import List, Dict, Any
from backend.app.core.embedder import Embedder
from backend.app.core.vector_store import VectorStore

class CVERetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def find_vulnerabilities(self, code_snippet: str, limit: int = 2, threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Embed a developer's code snippet and search the CVE corpus for semantic matches.
        Returns a list of matching CVEs above the similarity threshold.
        """
        # 1. Convert developer's code to a vector
        query_vector = self.embedder.embed_text(code_snippet)
        
        # 2. Search Qdrant for mathematically similar known vulnerabilities
        results = self.vector_store.search_cves(query_vector, limit=limit)
        
        # 3. Filter out low-confidence matches
        high_confidence_matches = [
            res for res in results 
            if res.get("similarity_score", 0.0) >= threshold
        ]
        
        return high_confidence_matches

# Simple CLI test runner if executed directly
if __name__ == "__main__":
    print("Initializing components...")
    retriever = CVERetriever(Embedder(), VectorStore())
    
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
    findings = retriever.find_vulnerabilities(test_code, limit=1, threshold=0.75)
    
    if findings:
        print("\n🚨 VULNERABILITY FOUND!")
        for finding in findings:
            print(f"CVE ID: {finding['cve_id']}")
            print(f"Severity: {finding['severity']}/10.0")
            print(f"Description: {finding['description']}")
            print(f"Similarity Score: {finding['similarity_score']:.4f}")
    else:
        print("\n✅ Code looks safe. No CVE patterns matched.")
