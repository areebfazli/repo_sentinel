import hashlib

import torch
from transformers import AutoModel, AutoTokenizer

from backend.app.config import settings

# embed_texts: write new embeddings to the cache every N model batches (not just
# once at the end), so an interrupted long run (e.g. ingest) keeps its progress.
_CACHE_FLUSH_EVERY_BATCHES = 8
# embed_texts: print progress (~every 10%) only for runs with more misses than this.
_PROGRESS_LOG_MIN_MISSES = 200


class Embedder:
    def __init__(self, model_name: str | None = None, pooling: str | None = None, cache=None):
        """Dense code embedder.

        model_name / pooling default to settings (EMBEDDING_MODEL, EMBEDDING_POOLING).
        pooling is "cls" (first-token) or "mean" (masked average of token states).
        cache is an optional EmbeddingCache; when set, embeddings are memoised by
        sha256(model:pooling:text) so unchanged text isn't re-embedded.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.pooling = (pooling or settings.EMBEDDING_POOLING).lower()
        self.cache = cache

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"Initializing Embedder ({self.model_name}, pooling={self.pooling}) "
            f"on device: {self.device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=settings.EMBEDDING_TRUST_REMOTE_CODE
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=settings.EMBEDDING_TRUST_REMOTE_CODE
        ).to(self.device)
        self.model.eval()

        hidden_size = self.model.config.hidden_size
        if hidden_size != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding model '{self.model_name}' has hidden_size={hidden_size}, "
                f"which does not match settings.EMBEDDING_DIM={settings.EMBEDDING_DIM}. "
                "Update EMBEDDING_DIM to match the model (and --recreate both Qdrant "
                "collections) before proceeding."
            )

    def _cache_key(self, text: str) -> str:
        digest = f"{self.model_name}:{self.pooling}:{text}"
        return hashlib.sha256(digest.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed a list of texts, returning one dense vector per input (in order)."""
        results: list[list[float] | None] = [None] * len(texts)

        # 1. Serve cache hits; collect the indices we still need to compute.
        keys = [self._cache_key(t) for t in texts] if self.cache else None
        misses = list(range(len(texts)))
        if self.cache:
            cached = self.cache.get_many(list(set(keys)))
            misses = []
            for i, k in enumerate(keys):
                if k in cached:
                    results[i] = cached[k]
                else:
                    misses.append(i)

        # 2. Run the model only for misses, in length-sorted batches.
        #
        # _embed_batch pads every batch to its longest member, so batching in input
        # order (short and 2000-token texts mixed) burns most of the compute on
        # padding. Sorting misses by length first puts similar-length texts in the
        # same batch. Each text's model input is unchanged (same tokenization and
        # truncation); only batch *composition* changes. Padding is masked via the
        # attention mask (and excluded from mean pooling), so a text's embedding
        # doesn't depend on which other texts share its batch, up to float noise
        # from different padded shapes (standard for transformers on CPU/GPU; the
        # old input-order batching had the same property).
        #
        # Length proxy: len(text) (characters), not token count. Tokenizing every
        # miss just to sort would be a second full tokenizer pass, while character
        # length tracks token length closely enough for code; a slightly imperfect
        # order only costs a little padding, never correctness. sorted() is stable,
        # so equal-length texts keep input order and the result is deterministic.
        order = sorted(misses, key=lambda i: len(texts[i]))

        n_misses = len(order)
        log_progress = n_misses > _PROGRESS_LOG_MIN_MISSES
        log_step = max(1, n_misses // 10)
        next_log = log_step
        processed = 0

        new_records: list[dict] = []
        for batch_no, start in enumerate(range(0, n_misses, batch_size), start=1):
            batch_idx = order[start:start + batch_size]
            batch_embeddings = self._embed_batch([texts[i] for i in batch_idx])
            # Write back to each text's ORIGINAL index so output order is preserved.
            for j, i in enumerate(batch_idx):
                results[i] = batch_embeddings[j]
                if self.cache:
                    new_records.append(
                        {
                            "key": keys[i],
                            "model": self.model_name,
                            "dim": len(batch_embeddings[j]),
                            "vector": batch_embeddings[j],
                        }
                    )

            # 3. Persist incrementally so an interrupted run keeps its progress.
            if self.cache and new_records and batch_no % _CACHE_FLUSH_EVERY_BATCHES == 0:
                self.cache.put_many(new_records)
                new_records = []

            if log_progress:
                processed += len(batch_idx)
                if processed >= next_log or processed == n_misses:
                    print(
                        f"Embedder: embedded {processed}/{n_misses} texts "
                        f"({100 * processed // n_misses}%)"
                    )
                    while next_log <= processed:
                        next_log += log_step

        # Flush whatever is left after the last full flush interval.
        if self.cache and new_records:
            self.cache.put_many(new_records)

        return results  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def _embed_batch(self, batch_texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=settings.EMBEDDING_MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        if self.pooling == "mean":
            # Masked mean over real tokens (ignore padding).
            mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
        else:  # "cls"
            embeddings = last_hidden[:, 0, :]

        # Normalize for cosine similarity.
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()
