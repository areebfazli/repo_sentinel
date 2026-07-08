import asyncio
from typing import Any

from backend.app.core.cve_retriever import CVERetriever
from backend.app.core.embedder import Embedder
from backend.app.core.reranker import Reranker
from backend.app.core.team_retriever import TeamRetriever
from backend.app.core.vector_store import VectorStore


class RagMerger:
    def __init__(self):
        """
        Initializes the shared ML components and both retrieval pipelines.
        """
        # Load heavy ML models once and share them across both retrievers to save RAM/VRAM
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        
        self.cve_retriever = CVERetriever(self.embedder, self.vector_store, self.reranker)
        self.team_retriever = TeamRetriever(self.embedder, self.vector_store, self.reranker)

    async def _async_find_cves(self, code_snippet: str) -> list[dict[str, Any]]:
        # In a fully async system, qdrant-client async would be used. 
        # Wrapping synchronous calls for now.
        return await asyncio.to_thread(self.cve_retriever.find_vulnerabilities, code_snippet)

    async def _async_find_team_history(self, code_snippet: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.team_retriever.find_team_history, code_snippet)

    async def analyze_code(self, code_snippet: str) -> dict[str, Any]:
        """
        The core RepoSentinel function: Takes developer code, hits both databases concurrently,
        and merges the results into a unified data structure.
        """
        # Run both massive vector searches concurrently
        cve_findings, team_findings = await asyncio.gather(
            self._async_find_cves(code_snippet),
            self._async_find_team_history(code_snippet)
        )
        
        return {
            "ghost_hunter_findings": cve_findings,
            "team_memory_findings": team_findings,
            "is_vulnerable": bool(cve_findings or team_findings)
        }

# Simple CLI test runner if executed directly
if __name__ == "__main__":
    import json
    print("Initializing RepoSentinel Unified Brain...")
    merger = RagMerger()
    
    test_code = """
def fetch_user_data(db, user_id):
    # This combines two bad practices we've indexed!
    # 1. SQL Injection (Ghost Hunter should catch)
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    
    try:
        db.execute(query)
    except:
        # 2. Bare except block (Team Memory should catch)
        return None
    """
    
    print("\nAnalyzing merged threat context for new PR code...")
    results = asyncio.run(merger.analyze_code(test_code))
    
    print(json.dumps(results, indent=2))
