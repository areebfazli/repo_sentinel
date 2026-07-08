import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.app.config import settings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


class VectorStore:
    def __init__(self):
        """
        Initialize the Qdrant client.
        Uses local file-based storage for dev, and connects to a server for prod.
        """
        if settings.is_dev:
            # File-based database (No Docker required)
            self.client = QdrantClient(path=settings.qdrant_local_path)
        else:
            # Server-based database
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )

        self.cve_collection = "cve_corpus"
        self.team_collection = "team_history"

        # CodeBERT/UniXcoder hidden size
        self.vector_size = 768
        self.hybrid = settings.HYBRID_ENABLED

        self._init_collections()

    def _init_collections(self):
        """Ensure both collections exist, creating them if they don't.

        Hybrid mode uses named dense + sparse(IDF) vectors; otherwise a single
        unnamed dense vector (the two shapes are not interchangeable — changing
        HYBRID_ENABLED requires re-ingesting with --recreate).
        """
        existing_collections = [c.name for c in self.client.get_collections().collections]

        for collection_name in [self.cve_collection, self.team_collection]:
            if collection_name not in existing_collections:
                if self.hybrid:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            DENSE_VECTOR: qmodels.VectorParams(
                                size=self.vector_size, distance=qmodels.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            SPARSE_VECTOR: qmodels.SparseVectorParams(
                                modifier=qmodels.Modifier.IDF
                            )
                        },
                    )
                else:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qmodels.VectorParams(
                            size=self.vector_size, distance=qmodels.Distance.COSINE
                        ),
                    )
            # Keyword index on `language` for fast server-mode filtering (local
            # Qdrant filters without one and warns). Idempotent / best-effort.
            if not settings.is_dev:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="language",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    def insert_cves(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
        sparse_vectors: list[dict] | None = None,
    ):
        """Insert embedded CVEs into the Ghost Hunter pipeline."""
        self._insert(self.cve_collection, vectors, payloads, ids, sparse_vectors)

    def insert_team_history(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
        sparse_vectors: list[dict] | None = None,
    ):
        """Insert embedded PRs/Commits into the Team Memory pipeline."""
        self._insert(self.team_collection, vectors, payloads, ids, sparse_vectors)

    def _insert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
        sparse_vectors: list[dict] | None = None,
    ):
        """Insert vectors into a collection.

        ``ids`` gives deterministic point IDs so re-ingestion upserts in place.
        ``sparse_vectors`` (required in hybrid mode) are {"indices", "values"}.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = []
        for i, (pid, dense, payload) in enumerate(
            zip(ids, vectors, payloads, strict=False)
        ):
            if self.hybrid:
                sparse = sparse_vectors[i] if sparse_vectors else {"indices": [], "values": []}
                vector = {
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: qmodels.SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    ),
                }
            else:
                vector = dense
            points.append(qmodels.PointStruct(id=pid, vector=vector, payload=payload))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=collection_name, points=points[i:i + batch_size])

    def recreate_collection(self, collection_name: str):
        """Drop and re-create a collection (used by ingestion --recreate)."""
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self._init_collections()

    def count(self, collection_name: str) -> int:
        """Return the number of points stored in a collection."""
        return self.client.count(collection_name).count

    def search_cves(
        self,
        query_vector: list[float],
        limit: int = 5,
        language: str | None = None,
        sparse_query: dict | None = None,
        dense_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find CVEs similar to the given code vector."""
        return self._search(
            self.cve_collection, query_vector, limit, language, sparse_query, dense_threshold
        )

    def search_team_history(
        self,
        query_vector: list[float],
        limit: int = 5,
        language: str | None = None,
        sparse_query: dict | None = None,
        dense_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find team history similar to the given code vector."""
        return self._search(
            self.team_collection, query_vector, limit, language, sparse_query, dense_threshold
        )

    def _language_filter(self, language: str | None):
        if not language:
            return None
        return qmodels.Filter(
            must=[qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=language))]
        )

    def _search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        language: str | None = None,
        sparse_query: dict | None = None,
        dense_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search. Hybrid mode fuses a (cosine-thresholded) dense prefetch and
        an IDF-thresholded sparse prefetch with Reciprocal Rank Fusion."""
        query_filter = self._language_filter(language)

        if self.hybrid and sparse_query is not None:
            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    qmodels.Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR,
                        filter=query_filter,
                        score_threshold=dense_threshold,
                        limit=settings.ANN_CANDIDATES,
                    ),
                    qmodels.Prefetch(
                        query=qmodels.SparseVector(
                            indices=sparse_query["indices"], values=sparse_query["values"]
                        ),
                        using=SPARSE_VECTOR,
                        filter=query_filter,
                        score_threshold=settings.HYBRID_SPARSE_MIN_SCORE,
                        limit=settings.ANN_CANDIDATES,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=limit,
            )
        else:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=DENSE_VECTOR if self.hybrid else None,
                limit=limit,
                query_filter=query_filter,
            )

        results = []
        for hit in response.points:
            res = hit.payload.copy() if hit.payload else {}
            res["similarity_score"] = hit.score
            res["point_id"] = str(hit.id)
            res["collection"] = collection_name
            results.append(res)
        return results
