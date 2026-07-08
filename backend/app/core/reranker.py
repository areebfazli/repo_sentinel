import math
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from backend.app.config import settings


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


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
        self, query_code: str, candidates: list[dict[str, Any]], top_k: int = 1
    ) -> list[dict[str, Any]]:
        """Rerank candidates against the developer's query code.

        Each candidate is compared using its ``rerank_text`` (e.g. the CVE's
        vulnerable code, for code-to-code matching), falling back to
        ``description``. Attaches ``rerank_score`` (raw logit) and ``rerank_prob``
        (sigmoid) and returns the top_k by score.
        """
        if not candidates:
            return []

        pairs = [
            [query_code, c.get("rerank_text") or c.get("description", "")]
            for c in candidates
        ]
        scores = self.model.predict(pairs)

        for i, candidate in enumerate(candidates):
            score = float(scores[i])
            candidate["rerank_score"] = score
            candidate["rerank_prob"] = _sigmoid(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
