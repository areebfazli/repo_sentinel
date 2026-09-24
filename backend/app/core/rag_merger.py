import asyncio
from typing import Any

from backend.app.config import settings
from backend.app.core import feedback_store
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
        # The cross-encoder (~3 GB) is only built when enabled; off by default
        # (no measured gain, ROADMAP 1d). None -> retrievers keep similarity order.
        self.reranker = Reranker() if settings.RERANKER_ENABLED else None

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
        # Team Memory is intentionally NOT language-filtered: a past review of a
        # pattern is relevant across languages, and code-vs-discussion similarity
        # is already low, so filtering would mostly cost recall. (Payloads do
        # carry `language` now if we ever want to revisit this.)
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

        # Pass 1: retrieve + rank per unit WITHOUT hitting the feedback table, so
        # the vote lookup can be batched into a single query below (one round-trip
        # for the whole scan instead of two per unit).
        per_unit: list[tuple[dict[str, Any], list[dict], list[dict]]] = []
        point_ids: set[str] = set()
        for unit, vector in zip(units, vectors, strict=False):
            # None (not "python") when the unit's language is unknown/empty, so an
            # extensionless whole-file unit isn't wrongly filtered to Python CVEs.
            language = unit.get("language") or None
            cve_c = self.cve_retriever.rerank_candidates(
                unit["code"], language=language, query_vector=vector
            )
            team_c = self.team_retriever.rerank_candidates(
                unit["code"], query_vector=vector
            )
            per_unit.append((unit, cve_c, team_c))
            point_ids.update(m.get("point_id", "") for m in cve_c)
            point_ids.update(m.get("point_id", "") for m in team_c)

        # Pass 2: one feedback query for every candidate, then finalize per unit
        # against the shared vote map.
        net_votes = feedback_store.get_net_votes(list(point_ids))
        limit = settings.RETRIEVAL_TOP_K
        cve_base = self.cve_retriever.base_score
        team_base = self.team_retriever.base_score_factory()
        for unit, cve_c, team_c in per_unit:
            for match in feedback_store.finalize_matches(
                cve_c, limit, settings, base_score=cve_base, net_votes=net_votes
            ):
                cve_findings.append(_anchor(match, unit))
            for match in feedback_store.finalize_matches(
                team_c, limit, settings, base_score=team_base, net_votes=net_votes
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
    # Anchor keys are distinct from any payload fields (e.g. a team payload's own
    # file_path), so snippet-mode findings never inherit a historical location.
    match["anchor_file_path"] = unit["file_path"]
    match["anchor_start_line"] = unit["start_line"]
    match["anchor_end_line"] = unit["end_line"]
    match["anchor_function_name"] = unit["function_name"]
    return match


def _dedupe(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest adjusted_score per (point_id, file, function).

    Keyed by function too, so the same CVE/team match occurring in two different
    functions of one file stays as two findings (each needs its own anchor + gate
    entry) rather than collapsing to one.
    """
    best: dict[tuple, dict[str, Any]] = {}
    for f in findings:
        key = (f.get("point_id"), f.get("anchor_file_path"), f.get("anchor_function_name"))
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
