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
    def __init__(self, model_name: str | None = None, max_tokens: int | None = None):
        """
        Initialize the Cross-Encoder model.
        This model takes a pair of texts (query, document) and outputs a similarity score.

        ``model_name``/``max_tokens`` default to settings.RERANKER_MODEL /
        RERANKER_MAX_TOKENS; the eval harness overrides them to compare rerankers.
        """
        model_name = settings.RERANKER_MODEL if model_name is None else model_name
        max_tokens = settings.RERANKER_MAX_TOKENS if max_tokens is None else max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Cross-Encoder Reranker on device: {self.device}")

        # We use bge-reranker-v2-m3, a general-purpose multilingual cross-encoder reranker
        self.model = CrossEncoder(model_name, max_length=max_tokens, device=self.device)

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
        scores = self.model.predict(pairs, batch_size=settings.RERANKER_BATCH_SIZE)

        for i, candidate in enumerate(candidates):
            score = float(scores[i])
            candidate["rerank_score"] = score
            candidate["rerank_prob"] = _sigmoid(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
