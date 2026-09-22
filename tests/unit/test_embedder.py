"""Unit tests for Embedder's config-vs-model dim check (no real model loads)."""
from unittest.mock import MagicMock

import pytest

from backend.app.config import settings
from backend.app.core.embedder import Embedder


class FakeConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class FakeModel:
    """Stands in for the AutoModel.from_pretrained(...) return value.

    Embedder.__init__ does `AutoModel.from_pretrained(...).to(self.device)` then
    `.eval()` on the result, so `.to()` must return something with `.eval()` and
    `.config`; returning self from both keeps it simple.
    """

    def __init__(self, hidden_size):
        self.config = FakeConfig(hidden_size)

    def to(self, device):
        return self

    def eval(self):
        return self


def test_embedder_raises_on_hidden_size_mismatch(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_DIM", 768)

    fake_tokenizer = MagicMock()
    fake_model = FakeModel(hidden_size=1024)  # mismatches EMBEDDING_DIM=768

    monkeypatch.setattr(
        "backend.app.core.embedder.AutoTokenizer.from_pretrained",
        MagicMock(return_value=fake_tokenizer),
    )
    monkeypatch.setattr(
        "backend.app.core.embedder.AutoModel.from_pretrained",
        MagicMock(return_value=fake_model),
    )

    with pytest.raises(ValueError, match="hidden_size=1024"):
        Embedder()


def test_embedder_ok_when_hidden_size_matches(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_DIM", 768)

    fake_tokenizer = MagicMock()
    fake_model = FakeModel(hidden_size=768)  # matches EMBEDDING_DIM=768

    monkeypatch.setattr(
        "backend.app.core.embedder.AutoTokenizer.from_pretrained",
        MagicMock(return_value=fake_tokenizer),
    )
    monkeypatch.setattr(
        "backend.app.core.embedder.AutoModel.from_pretrained",
        MagicMock(return_value=fake_model),
    )

    embedder = Embedder()
    assert embedder.model is fake_model
    assert embedder.tokenizer is fake_tokenizer
