"""Unit tests for Embedder (no real model loads).

Covers the config-vs-model dim check and embed_texts batching/caching behaviour.
"""
import hashlib
from unittest.mock import MagicMock

import pytest

from backend.app.config import settings
from backend.app.core import embedder as embedder_module
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


# ---------------------------------------------------------------------------
# embed_texts: length-sorted batching, cache hits, incremental cache flush.
#
# These build the Embedder via Embedder.__new__ and set only the attributes
# embed_texts/_cache_key read (model_name, pooling, cache), then stub
# _embed_batch on the instance. That skips __init__ entirely (no torch device
# probe, no tokenizer/model construction), which is simpler and more robust
# than the from_pretrained monkeypatch pattern above for tests that never touch
# the model.
# ---------------------------------------------------------------------------


def _vec(text: str) -> list[float]:
    """Deterministic, text-identifying fake embedding (independent of batch position)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [float(len(text))] + [b / 255.0 for b in digest[:3]]


class FakeCache:
    """In-memory stand-in for EmbeddingCache (get_many / put_many only)."""

    def __init__(self, initial: dict[str, list[float]] | None = None):
        self.store: dict[str, list[float]] = dict(initial or {})
        self.get_calls: list[list[str]] = []
        self.put_calls: list[list[dict]] = []

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        self.get_calls.append(list(keys))
        return {k: self.store[k] for k in keys if k in self.store}

    def put_many(self, records: list[dict]) -> None:
        self.put_calls.append([dict(r) for r in records])
        for r in records:
            self.store[r["key"]] = r["vector"]


def _make_embedder(cache=None, fail_on_call: int | None = None):
    """Embedder with no model; _embed_batch records each batch and returns _vec()s."""
    emb = Embedder.__new__(Embedder)
    emb.model_name = "fake/model"
    emb.pooling = "mean"
    emb.cache = cache
    emb.batches = []

    def fake_embed_batch(batch_texts):
        emb.batches.append(list(batch_texts))
        if fail_on_call is not None and len(emb.batches) == fail_on_call:
            raise RuntimeError("simulated crash mid-run")
        return [_vec(t) for t in batch_texts]

    emb._embed_batch = fake_embed_batch
    return emb


def _mixed_texts() -> list[str]:
    # Wildly uneven lengths, deliberately NOT in length order; all distinct.
    return [
        "a",
        "b" * 500,
        "c" * 40,
        "d" * 2,
        "e" * 3000,
        "f" * 41,
        "g" * 7,
        "h" * 999,
        "i" * 3,
    ]


def test_embed_texts_preserves_input_order_for_mixed_lengths():
    texts = _mixed_texts()
    emb = _make_embedder(cache=None)

    out = emb.embed_texts(texts, batch_size=2)

    assert out == [_vec(t) for t in texts]
    # The test is only meaningful if batching actually reordered the inputs.
    input_order_chunks = [texts[i:i + 2] for i in range(0, len(texts), 2)]
    assert emb.batches != input_order_chunks


def test_embed_texts_batches_are_length_sorted():
    texts = _mixed_texts()
    emb = _make_embedder(cache=None)

    emb.embed_texts(texts, batch_size=2)

    # Batches, concatenated in call order, are exactly the length-sorted inputs,
    # chunked by batch_size (so each batch holds length-adjacent texts).
    flat = [t for batch in emb.batches for t in batch]
    assert flat == sorted(texts, key=len)
    assert [len(b) for b in emb.batches] == [2, 2, 2, 2, 1]


def test_embed_texts_duplicate_misses_all_filled():
    texts = ["x" * 10, "y", "x" * 10, "zz"]
    emb = _make_embedder(cache=None)

    out = emb.embed_texts(texts, batch_size=3)

    assert out == [_vec(t) for t in texts]


def test_embed_texts_serves_cache_hits_without_reembedding():
    texts = _mixed_texts()
    probe = _make_embedder()
    hit_texts = {texts[1], texts[4], texts[6]}  # long, very long, short
    sentinel = {t: [-1.0, float(i)] for i, t in enumerate(sorted(hit_texts))}
    cache = FakeCache({probe._cache_key(t): v for t, v in sentinel.items()})
    emb = _make_embedder(cache=cache)

    out = emb.embed_texts(texts, batch_size=2)

    embedded = [t for batch in emb.batches for t in batch]
    assert not hit_texts & set(embedded)
    assert sorted(embedded, key=len) == embedded
    assert set(embedded) == set(texts) - hit_texts
    for t, v in zip(texts, out, strict=True):
        assert v == (sentinel[t] if t in hit_texts else _vec(t))
    # Hits are never re-written to the cache.
    written = {r["key"] for call in cache.put_calls for r in call}
    assert written == {emb._cache_key(t) for t in set(texts) - hit_texts}


def test_embed_texts_flushes_cache_incrementally():
    flush_every = embedder_module._CACHE_FLUSH_EVERY_BATCHES
    n_miss = 2 * flush_every + 4  # batch_size=1 -> 2 full flush intervals + remainder
    miss_texts = [f"miss_{i}_" + "m" * (i % 5) for i in range(n_miss)]
    hit_text = "already cached"
    probe = _make_embedder()
    cache = FakeCache({probe._cache_key(hit_text): [9.0, 9.0]})
    emb = _make_embedder(cache=cache)

    out = emb.embed_texts([hit_text, *miss_texts], batch_size=1)

    assert out == [[9.0, 9.0]] + [_vec(t) for t in miss_texts]
    assert len(emb.batches) == n_miss
    assert [len(c) for c in cache.put_calls] == [flush_every, flush_every, 4]

    records = [r for call in cache.put_calls for r in call]
    by_key = {r["key"]: r for r in records}
    assert len(by_key) == len(records) == n_miss
    expected = {emb._cache_key(t): t for t in miss_texts}
    assert set(by_key) == set(expected)
    for key, text in expected.items():
        rec = by_key[key]
        assert rec["model"] == "fake/model"
        assert rec["vector"] == _vec(text)
        assert rec["dim"] == len(_vec(text))


def test_embed_texts_keeps_flushed_progress_when_interrupted():
    flush_every = embedder_module._CACHE_FLUSH_EVERY_BATCHES
    texts = [f"t{i:03d}" for i in range(3 * flush_every)]
    cache = FakeCache()
    emb = _make_embedder(cache=cache, fail_on_call=flush_every + 2)

    with pytest.raises(RuntimeError, match="simulated crash"):
        emb.embed_texts(texts, batch_size=1)

    # The first full flush interval was persisted before the crash.
    assert len(cache.put_calls) == 1
    assert len(cache.store) == flush_every


def test_embed_texts_empty_input():
    emb = _make_embedder(cache=None)
    assert emb.embed_texts([]) == []
    assert emb.batches == []

    cache = FakeCache()
    emb = _make_embedder(cache=cache)
    assert emb.embed_texts([]) == []
    assert emb.batches == []
    assert cache.put_calls == []


def test_embed_texts_progress_logging_only_for_large_runs(capsys):
    threshold = embedder_module._PROGRESS_LOG_MIN_MISSES

    _make_embedder().embed_texts([f"s{i}" for i in range(threshold)], batch_size=16)
    assert "Embedder: embedded" not in capsys.readouterr().out

    n = threshold + 50
    _make_embedder().embed_texts([f"L{i}" for i in range(n)], batch_size=16)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "Embedder: embedded" in ln]
    assert 5 <= len(lines) <= 12  # roughly every 10%
    assert lines[-1].startswith(f"Embedder: embedded {n}/{n} texts (100%)")
