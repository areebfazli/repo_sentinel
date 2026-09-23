"""Reranker score semantics with a stubbed CrossEncoder (no model load).

CrossEncoder.predict applies nn.Sigmoid by default for single-label models, so
Reranker must ask for logits (identity activation) and sigmoid them exactly
once. Before this was fixed, rerank_prob was sigmoid(sigmoid(logit)) and could
never drop below 0.5.
"""
import math

import pytest
import torch
from torch import nn

from backend.app.core import reranker as reranker_mod
from backend.app.core.reranker import Reranker, _sigmoid


class StubCrossEncoder:
    """Mimics CrossEncoder: raw logits -> activation_fn (default Sigmoid, like
    a num_labels=1 model) -> numpy scores. Records the kwargs it was given."""

    last_init_kwargs: dict = {}

    def __init__(self, model_name, **kwargs):
        StubCrossEncoder.last_init_kwargs = kwargs
        self.activation_fn = kwargs.get("activation_fn") or nn.Sigmoid()
        self.logits: dict[str, float] = {}
        self.predict_kwargs: dict = {}

    def predict(self, pairs, batch_size=32, activation_fn=None, **kwargs):
        self.predict_kwargs = {"batch_size": batch_size, "activation_fn": activation_fn}
        act = activation_fn or self.activation_fn
        raw = torch.tensor([self.logits[doc] for _, doc in pairs], dtype=torch.float32)
        return act(raw).numpy()


@pytest.fixture
def reranker(monkeypatch):
    monkeypatch.setattr(reranker_mod, "CrossEncoder", StubCrossEncoder)
    return Reranker(model_name="stub/model", max_tokens=64)


def test_constructor_and_predict_request_identity_activation(reranker):
    assert isinstance(StubCrossEncoder.last_init_kwargs["activation_fn"], nn.Identity)
    assert StubCrossEncoder.last_init_kwargs["max_length"] == 64
    reranker.model.logits = {"a": 1.0}
    reranker.rerank("q", [{"rerank_text": "a"}])
    assert isinstance(reranker.model.predict_kwargs["activation_fn"], nn.Identity)


def test_rerank_score_is_logit_and_prob_is_single_sigmoid(reranker):
    reranker.model.logits = {"hi": 4.0, "zero": 0.0, "lo": -6.0}
    cands = [{"rerank_text": t, "id": t} for t in ("lo", "hi", "zero")]
    out = reranker.rerank("query", cands, top_k=3)

    assert [c["id"] for c in out] == ["hi", "zero", "lo"]
    by_id = {c["id"]: c for c in out}
    for key, logit in reranker.model.logits.items():
        c = by_id[key]
        assert c["rerank_score"] == pytest.approx(logit, abs=1e-6)
        assert c["rerank_prob"] == pytest.approx(_sigmoid(c["rerank_score"]))
        assert 0.0 < c["rerank_prob"] < 1.0
    # A strongly negative logit must give a low probability — impossible under
    # the old double sigmoid, whose floor was sigmoid(0) = 0.5.
    assert by_id["lo"]["rerank_prob"] < 0.01
    assert by_id["hi"]["rerank_prob"] > 0.98
    assert by_id["zero"]["rerank_prob"] == pytest.approx(0.5)


def test_sigmoid_is_stable_at_extremes():
    assert 0.0 <= _sigmoid(-1000.0) < 1e-300
    assert _sigmoid(1000.0) == 1.0
    assert _sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2.0)))
