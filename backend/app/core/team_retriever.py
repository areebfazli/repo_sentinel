from datetime import UTC, datetime
from typing import Any

from backend.app.config import settings
from backend.app.core import feedback_store
from backend.app.core.embedder import Embedder
from backend.app.core.reranker import Reranker, rank_candidates
from backend.app.core.scoring import compute_adjusted_score, relevance
from backend.app.core.sparse_encoder import encode as sparse_encode
from backend.app.core.vector_store import VectorStore


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    # Treat naive timestamps as UTC so the age math stays consistent.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class TeamRetriever:
    def __init__(
        self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker | None
    ):
        """``reranker`` is None when RERANKER_ENABLED is off: candidates keep
        similarity order (see ``rank_candidates``)."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def base_score_factory(self):
        """Return a base_score(match) closure that weights ``relevance(match)``
        (rerank_prob, or similarity without a reranker) by recency +
        reviewer seniority, with ``now`` captured once so a batch of matches scores
        against a single reference time."""
        now = datetime.now(UTC)

        def base_score(m: dict[str, Any]) -> float:
            return compute_adjusted_score(
                relevance(m),
                _parse_dt(m.get("created_at")),
                m.get("author_association"),
                now,
                settings,
            )

        return base_score

    def rerank_candidates(
        self,
        code_snippet: str,
        language: str | None = None,
        threshold: float | None = None,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN top-N -> similarity gate -> rerank against stored discussion text
        (similarity order when the reranker is off), WITHOUT feedback. Split out
        so files mode can batch the feedback lookup across units (see
        ``find_team_history`` for the single-snippet path)."""
        threshold = settings.SIM_THRESHOLD_TEAM if threshold is None else threshold

        if query_vector is None:
            query_vector = self.embedder.embed_text(code_snippet)

        if settings.HYBRID_ENABLED:
            broad_matches = self.vector_store.search_team_history(
                query_vector,
                limit=settings.ANN_CANDIDATES,
                language=language,
                sparse_query=sparse_encode(code_snippet),
                dense_threshold=threshold,
            )
            viable_matches = broad_matches
        else:
            broad_matches = self.vector_store.search_team_history(
                query_vector, limit=settings.ANN_CANDIDATES, language=language
            )
            viable_matches = [
                res for res in broad_matches
                if res.get("similarity_score", 0.0) >= threshold
            ]

        # Rerank against the stored discussion text (diff hunk + review body).
        if self.reranker is not None:
            for match in viable_matches:
                match["rerank_text"] = match.get("text") or match.get("snippet_preview", "")

        return rank_candidates(self.reranker, code_snippet, viable_matches)

    def find_team_history(
        self,
        code_snippet: str,
        language: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed developer code and search Team Memory for related past PR reviews.

        Same recall -> similarity gate -> rerank -> probability gate pipeline as the
        CVE side (rerank + gate only when RERANKER_ENABLED). Reranks against the
        stored discussion text. Thresholds default to
        settings. ``query_vector`` lets callers pass a precomputed embedding.
        """
        limit = settings.RETRIEVAL_TOP_K if limit is None else limit
        reranked = self.rerank_candidates(code_snippet, language, threshold, query_vector)
        # Team score weights relevance by recency + reviewer seniority, then
        # feedback suppression/downweighting is applied on top.
        return feedback_store.finalize_matches(
            reranked, limit, settings, base_score=self.base_score_factory()
        )


# Simple CLI test runner if executed directly
if __name__ == "__main__":
    print("Initializing components...")
    retriever = TeamRetriever(
        Embedder(), VectorStore(), Reranker() if settings.RERANKER_ENABLED else None
    )

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
            prob = finding.get("rerank_prob")
            print(f"Reranker Confidence: {'n/a (reranker off)' if prob is None else f'{prob:.4f}'}")
    else:
        print("\n✅ Clean. The team hasn't complained about this pattern before.")
