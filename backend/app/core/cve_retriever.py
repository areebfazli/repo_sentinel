from typing import Any

from backend.app.config import settings
from backend.app.core import feedback_store
from backend.app.core.embedder import Embedder
from backend.app.core.reranker import Reranker
from backend.app.core.sparse_encoder import encode as sparse_encode
from backend.app.core.vector_store import VectorStore


class CVERetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def find_vulnerabilities(
        self,
        code_snippet: str,
        language: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed developer code and search the CVE corpus for matches.

        High recall (ANN top-N, optionally language-filtered) -> similarity gate ->
        cross-encoder rerank code-against-code -> rerank-probability gate. All
        thresholds default to settings (single source of truth). ``query_vector``
        lets callers pass a precomputed embedding (files mode batches embedding).
        """
        limit = settings.RETRIEVAL_TOP_K if limit is None else limit
        threshold = settings.SIM_THRESHOLD_CVE if threshold is None else threshold

        if query_vector is None:
            query_vector = self.embedder.embed_text(code_snippet)

        if settings.HYBRID_ENABLED:
            # RRF fuses a cosine-thresholded dense prefetch with a lexical sparse
            # prefetch; the prefetches already gate, so keep all fused candidates.
            broad_matches = self.vector_store.search_cves(
                query_vector,
                limit=settings.ANN_CANDIDATES,
                language=language,
                sparse_query=sparse_encode(code_snippet),
                dense_threshold=threshold,
            )
            viable_matches = broad_matches
        else:
            broad_matches = self.vector_store.search_cves(
                query_vector, limit=settings.ANN_CANDIDATES, language=language
            )
            viable_matches = [
                res for res in broad_matches
                if res.get("similarity_score", 0.0) >= threshold
            ]

        # Rerank against the stored vulnerable code (code-to-code), not the
        # English description.
        for match in viable_matches:
            match["rerank_text"] = match.get("vulnerable_code") or match.get("description", "")

        # Rerank all viable, then apply feedback suppression/downweight before
        # taking the top-k (CVE score is just the rerank probability).
        reranked = self.reranker.rerank(code_snippet, viable_matches, top_k=len(viable_matches))
        return feedback_store.finalize_matches(
            reranked, limit, settings, base_score=lambda m: m.get("rerank_prob", 0.0)
        )


# Simple CLI test runner if executed directly
if __name__ == "__main__":
    print("Initializing components (this will load the embedder + MS-MARCO CrossEncoder)...")
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
    findings = retriever.find_vulnerabilities(test_code, language="python")

    if findings:
        print("\n🚨 VULNERABILITY FOUND!")
        for finding in findings:
            print(f"CVE ID: {finding['cve_id']}")
            print(f"Severity: {finding['severity']}/10.0")
            print(f"Description: {finding['description']}")
            print(f"Base Vector Score: {finding.get('similarity_score', 0.0):.4f}")
            print(f"Reranker Confidence: {finding.get('rerank_prob', 0.0):.4f}")
    else:
        print("\n✅ Code looks safe. No CVE patterns matched.")
