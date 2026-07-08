"""Unit tests for the embedding cache (duplicate-key batches)."""
from backend.app.core.embedding_cache import EmbeddingCache


def test_put_many_deduplicates_batch_keys():
    cache = EmbeddingCache()
    rec = {"key": "dupkey-test", "model": "m", "dim": 2, "vector": [0.1, 0.2]}
    # Same key twice in one batch used to raise a UNIQUE-constraint IntegrityError.
    cache.put_many([dict(rec), dict(rec)])
    got = cache.get_many(["dupkey-test"])
    assert got["dupkey-test"] == [0.1, 0.2]


def test_get_many_empty():
    assert EmbeddingCache().get_many([]) == {}
