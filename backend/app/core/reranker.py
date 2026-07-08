from typing import Any

import torch
from sentence_transformers import CrossEncoder

from backend.app.config import settings


class Reranker:
    def __init__(self):
        """
        Initialize the Cross-Encoder model.
        This model takes a pair of texts (query, document) and outputs a similarity score.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Cross-Encoder Reranker on device: {self.device}")
        
        # We use the ms-marco model optimized for semantic search relevance
        self.model = CrossEncoder(settings.RERANKER_MODEL, device=self.device)

    def rerank(
        self, query_code: str, candidate_cves: list[dict[str, Any]], top_k: int = 1
    ) -> list[dict[str, Any]]:
        """
        Rerank a list of candidate CVEs against the developer's query code.
        """
        if not candidate_cves:
            return []

        # The CrossEncoder expects pairs of sentences: [[query, candidate1], ...].
        # We use the CVE's description + vulnerable code as the candidate text to compare against
        pairs = []
        for cve in candidate_cves:
            # Reconstruct what the CVE represents
            # If 'vulnerable_code' isn't explicitly in the payload, we use description as fallback
            # (In our ingestion script we didn't save the raw code to the payload to save space,
            # but for a reranker it's better if we do. For now we rerank based on description)
            candidate_text = cve.get("description", "")
            pairs.append([query_code, candidate_text])
            
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Attach new reranker scores to the original candidates
        for i, cve in enumerate(candidate_cves):
            # Cross-encoder scores are logits, we can convert them or just use them for sorting
            # (Higher is better)
            cve["rerank_score"] = float(scores[i])
            
        # Sort by rerank score descending
        reranked_cves = sorted(candidate_cves, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_cves[:top_k]
