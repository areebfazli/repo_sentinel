import asyncio
from typing import Any

from backend.app.core.cve_retriever import CVERetriever
from backend.app.core.embedder import Embedder
from backend.app.core.embedding_cache import EmbeddingCache
from backend.app.core.reranker import Reranker
from backend.app.core.team_retriever import TeamRetriever
from backend.app.core.vector_store import VectorStore


class RagMerger:
    def __init__(self):
        """
        Initializes the shared ML components and both retrieval pipelines.
        """
        # Load heavy ML models once and share them across both retrievers to save RAM/VRAM
        self.embedder = Embedder(cache=EmbeddingCache())
        self.vector_store = VectorStore()
        self.reranker = Reranker()

        self.cve_retriever = CVERetriever(self.embedder, self.vector_store, self.reranker)
        self.team_retriever = TeamRetriever(self.embedder, self.vector_store, self.reranker)

    async def _async_find_cves(
        self, code_snippet: str, language: str | None
    ) -> list[dict[str, Any]]:
        # Retrieval is synchronous; run it off the event loop.
        return await asyncio.to_thread(
            self.cve_retriever.find_vulnerabilities, code_snippet, language
        )

    async def _async_find_team_history(self, code_snippet: str) -> list[dict[str, Any]]:
        # Team payloads don't carry a language field until Phase 5, so we don't
        # filter Team Memory by language yet (it would drop all mock results).
        return await asyncio.to_thread(self.team_retriever.find_team_history, code_snippet)

    async def analyze_code(
        self, code_snippet: str, language: str | None = None
    ) -> dict[str, Any]:
        """
        The core RepoSentinel function: Takes developer code, hits both databases concurrently,
        and merges the results into a unified data structure.
        """
        # Run both vector searches concurrently
        cve_findings, team_findings = await asyncio.gather(
            self._async_find_cves(code_snippet, language),
            self._async_find_team_history(code_snippet)
        )

        return {
            "ghost_hunter_findings": cve_findings,
            "team_memory_findings": team_findings,
            "is_vulnerable": bool(cve_findings or team_findings)
        }

    async def analyze_units(self, units: list[dict[str, Any]]) -> dict[str, Any]:
        """Files-mode analysis: one match set across per-function units.

        Embeds every unit's code in a single batch (cache-aware), then runs
        retrieval per unit and merges. Findings are anchored to their file/line
        and deduped by (point_id, file_path) keeping the best-scoring hit.
        """
        if not units:
            return {"ghost_hunter_findings": [], "team_memory_findings": [], "is_vulnerable": False}

        codes = [u["code"] for u in units]
        vectors = await asyncio.to_thread(self.embedder.embed_texts, codes)
        return await asyncio.to_thread(self._analyze_units_sync, units, vectors)

    def _analyze_units_sync(
        self, units: list[dict[str, Any]], vectors: list[list[float]]
    ) -> dict[str, Any]:
        cve_findings: list[dict[str, Any]] = []
        team_findings: list[dict[str, Any]] = []

        for unit, vector in zip(units, vectors, strict=False):
            language = unit.get("language") or "python"
            for match in self.cve_retriever.find_vulnerabilities(
                unit["code"], language=language, query_vector=vector
            ):
                cve_findings.append(_anchor(match, unit))
            for match in self.team_retriever.find_team_history(
                unit["code"], query_vector=vector
            ):
                team_findings.append(_anchor(match, unit))

        cve_findings = _dedupe(cve_findings)
        team_findings = _dedupe(team_findings)
        return {
            "ghost_hunter_findings": cve_findings,
            "team_memory_findings": team_findings,
            "is_vulnerable": bool(cve_findings or team_findings),
        }


def _anchor(match: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    match["file_path"] = unit["file_path"]
    match["start_line"] = unit["start_line"]
    match["end_line"] = unit["end_line"]
    match["function_name"] = unit["function_name"]
    return match


def _dedupe(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest adjusted_score per (point_id, file_path)."""
    best: dict[tuple, dict[str, Any]] = {}
    for f in findings:
        key = (f.get("point_id"), f.get("file_path"))
        current = best.get(key)
        if current is None or f.get("adjusted_score", 0.0) > current.get("adjusted_score", 0.0):
            best[key] = f
    return sorted(best.values(), key=lambda f: f.get("adjusted_score", 0.0), reverse=True)


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
    results = asyncio.run(merger.analyze_code(test_code, language="python"))

    print(json.dumps(results, indent=2))
