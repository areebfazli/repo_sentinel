import math
from typing import Any

import torch
from sentence_transformers import CrossEncoder
from torch import nn

from backend.app.config import settings

# CrossEncoder.predict applies the model's activation_fn, which for a
# single-label model (num_labels=1: bge-reranker-v2-m3, ms-marco) defaults to
# nn.Sigmoid — so predict() returns probabilities, not logits. Sigmoid-ing that
# again squashed every rerank_prob into [0.5, 0.731] (the old "bge gives ~0.5"
# artefact). We pass nn.Identity so predict() returns the raw logit, store it as
# rerank_score, and apply _sigmoid exactly once for rerank_prob. The identity is
# passed both at construction and per predict() call, so a model whose saved
# config names another activation can't reintroduce it.
_LOGITS = nn.Identity()


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
        self.model = CrossEncoder(
            model_name, max_length=max_tokens, device=self.device, activation_fn=_LOGITS
        )

    def rerank(
        self, query_code: str, candidates: list[dict[str, Any]], top_k: int = 1
    ) -> list[dict[str, Any]]:
        """Rerank candidates against the developer's query code.

        Each candidate is compared using its ``rerank_text`` (e.g. the CVE's
        vulnerable code, for code-to-code matching), falling back to
        ``description``. Attaches ``rerank_score`` (raw logit, unbounded) and
        ``rerank_prob`` (sigmoid of that logit, in (0, 1)) and returns the top_k
        by score.
        """
        if not candidates:
            return []

        pairs = [
            [query_code, c.get("rerank_text") or c.get("description", "")]
            for c in candidates
        ]
        scores = self.model.predict(
            pairs, batch_size=settings.RERANKER_BATCH_SIZE, activation_fn=_LOGITS
        )

        for i, candidate in enumerate(candidates):
            score = float(scores[i])
            candidate["rerank_score"] = score
            candidate["rerank_prob"] = _sigmoid(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


def rank_candidates(
    reranker: Reranker | None, query_code: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Order a retriever's gated candidates, best first (none dropped).

    With a reranker (RERANKER_ENABLED): cross-encoder scores, attaching
    ``rerank_score``/``rerank_prob``. Without one: no score is fabricated —
    ``rerank_prob`` stays absent (``scoring.relevance`` then falls back to
    ``similarity_score`` and RERANK_THRESHOLD is not applied). Dense mode sorts
    by cosine ``similarity_score``; hybrid mode keeps the ANN order, since there
    ``similarity_score`` is Qdrant's RRF fusion score, which already ranks it.
    """
    if reranker is not None:
        return reranker.rerank(query_code, candidates, top_k=len(candidates))
    if settings.HYBRID_ENABLED:
        return list(candidates)
    return sorted(candidates, key=lambda c: c.get("similarity_score", 0.0), reverse=True)
