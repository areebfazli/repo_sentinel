import math
import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.app.config import settings

# team_history's dense vector name in hybrid mode (unnamed otherwise).
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
# cve_corpus always uses named vectors: "vuln" (the searched dense vector, the
# CVE-side equivalent of DENSE_VECTOR) and "fixed" (the patched twin, omitted on
# points without a fix). Never ANN-searched; read back to score the twin margin.
VULN_VECTOR = "vuln"
FIXED_VECTOR = "fixed"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Stored and query vectors are already L2-normalised
    (Qdrant normalises on upsert for COSINE, the Embedder on output), but the
    norms are cheap and keep this correct for any caller."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def twin_scores(
    query_vector: list[float],
    vuln_vector: list[float] | None,
    fixed_vector: list[float] | None,
) -> dict[str, float | None]:
    """Patched-twin scores for one CVE hit (ROADMAP 1b).

    ``sim_fixed`` = cos(query, fixed); ``twin_margin`` = cos(query, vuln) -
    sim_fixed. A positive margin means the code looks more like the bug than the
    fix. Both are None when the entry has no stored fix (handwritten entries).
    Both cosines are computed here, identically, rather than taking the vuln side
    from the hit score — in hybrid mode that score is an RRF rank fusion, not a
    cosine.
    """
    if not fixed_vector or not vuln_vector:
        return {"sim_fixed": None, "twin_margin": None}
    sim_fixed = _cosine(query_vector, fixed_vector)
    return {"sim_fixed": sim_fixed, "twin_margin": _cosine(query_vector, vuln_vector) - sim_fixed}


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

        # Must match the embedder's hidden size (see Embedder's dim check); Qdrant
        # collection dims are fixed at creation, so an EMBEDDING_DIM mismatch
        # requires --recreate on both collections.
        self.vector_size = settings.EMBEDDING_DIM
        self.hybrid = settings.HYBRID_ENABLED
        # Set when an existing cve_corpus predates the vuln/fixed schema (or its
        # sparse config disagrees with HYBRID_ENABLED); CVE reads/writes raise it.
        self.cve_schema_error: str | None = None

        self._init_collections()

    def _dense_vector_name(self, collection_name: str) -> str | None:
        """The dense vector a collection is searched on (None = unnamed)."""
        if collection_name == self.cve_collection:
            return VULN_VECTOR
        return DENSE_VECTOR if self.hybrid else None

    def _vectors_config(self, collection_name: str):
        def dense(**kwargs):
            return qmodels.VectorParams(
                size=self.vector_size, distance=qmodels.Distance.COSINE, **kwargs
            )

        if collection_name == self.cve_collection:
            # m=0: no HNSW graph for "fixed" — it is only read back by point, never
            # ANN-searched, so building an index for it is wasted memory/time.
            return {
                VULN_VECTOR: dense(),
                FIXED_VECTOR: dense(hnsw_config=qmodels.HnswConfigDiff(m=0)),
            }
        return {DENSE_VECTOR: dense()} if self.hybrid else dense()

    def _init_collections(self):
        """Ensure both collections exist, creating them if they don't.

        cve_corpus always has named "vuln" + "fixed" dense vectors. team_history
        has a named dense vector in hybrid mode, else a single unnamed one. Hybrid
        mode adds a sparse(IDF) vector to both. The shapes are not
        interchangeable — changing HYBRID_ENABLED (or upgrading from the pre-twin
        cve_corpus) requires re-ingesting with --recreate.
        """
        existing_collections = [c.name for c in self.client.get_collections().collections]

        for collection_name in [self.cve_collection, self.team_collection]:
            if collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=self._vectors_config(collection_name),
                    sparse_vectors_config=(
                        {
                            SPARSE_VECTOR: qmodels.SparseVectorParams(
                                modifier=qmodels.Modifier.IDF
                            )
                        }
                        if self.hybrid
                        else None
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

        self.cve_schema_error = self._check_cve_schema()
        if self.cve_schema_error:
            logger.warning(self.cve_schema_error)

    def _check_cve_schema(self) -> str | None:
        """Detect a cve_corpus whose vector layout doesn't match this code.

        Qdrant would otherwise fail with an opaque "vector name not found" on
        the first search/upsert.
        """
        try:
            params = self.client.get_collection(self.cve_collection).config.params
        except Exception:
            return None  # best-effort; a real connectivity problem surfaces elsewhere
        vectors = params.vectors
        if not isinstance(vectors, dict) or not {VULN_VECTOR, FIXED_VECTOR} <= set(vectors):
            return (
                "cve_corpus predates the vuln/fixed (patched twin) vector schema — "
                "stop the API and run scripts/ingest_cve_corpus.py --recreate."
            )
        # (A collection built WITH sparse vectors still serves dense-only mode.)
        if self.hybrid and SPARSE_VECTOR not in (params.sparse_vectors or {}):
            return (
                "HYBRID_ENABLED is on but cve_corpus has no sparse vectors — "
                "run scripts/ingest_cve_corpus.py --recreate."
            )
        return None

    def _require_cve_schema(self):
        if self.cve_schema_error:
            raise RuntimeError(self.cve_schema_error)

    def insert_cves(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
        sparse_vectors: list[dict] | None = None,
        fixed_vectors: list[list[float] | None] | None = None,
    ):
        """Insert embedded CVEs into the Ghost Hunter pipeline.

        ``vectors`` embed the vulnerable code; ``fixed_vectors`` (aligned, None
        for entries without a fix) embed the patched twin.
        """
        self._require_cve_schema()
        self._insert(
            self.cve_collection, vectors, payloads, ids, sparse_vectors, fixed_vectors
        )

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
        fixed_vectors: list[list[float] | None] | None = None,
    ):
        """Insert vectors into a collection.

        ``ids`` gives deterministic point IDs so re-ingestion upserts in place.
        ``sparse_vectors`` (required in hybrid mode) are {"indices", "values"}.
        ``fixed_vectors`` (cve_corpus only) add the "fixed" named vector; a None
        entry simply omits it (Qdrant allows a point to lack a named vector).
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        dense_name = self._dense_vector_name(collection_name)
        points = []
        for i, (pid, dense, payload) in enumerate(
            zip(ids, vectors, payloads, strict=False)
        ):
            if dense_name is None:
                vector = dense
            else:
                vector = {dense_name: dense}
                if fixed_vectors and fixed_vectors[i] is not None:
                    vector[FIXED_VECTOR] = fixed_vectors[i]
                if self.hybrid:
                    sparse = sparse_vectors[i] if sparse_vectors else {"indices": [], "values": []}
                    vector[SPARSE_VECTOR] = qmodels.SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    )
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
        """Find CVEs similar to the given code vector (searched on "vuln").

        Every hit also carries ``sim_fixed`` / ``twin_margin`` (see twin_scores;
        None when the entry has no patched twin).
        """
        self._require_cve_schema()
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
        an IDF-thresholded sparse prefetch with Reciprocal Rank Fusion.

        On cve_corpus the "vuln" and "fixed" vectors are fetched back with each
        hit to score the patched twin; they are not left on the result.
        """
        query_filter = self._language_filter(language)
        dense_name = self._dense_vector_name(collection_name)
        twin = collection_name == self.cve_collection
        with_vectors = [VULN_VECTOR, FIXED_VECTOR] if twin else False

        if self.hybrid and sparse_query is not None:
            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    qmodels.Prefetch(
                        query=query_vector,
                        using=dense_name,
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
                with_vectors=with_vectors,
            )
        else:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=dense_name,
                limit=limit,
                query_filter=query_filter,
                with_vectors=with_vectors,
            )

        results = []
        for hit in response.points:
            res = hit.payload.copy() if hit.payload else {}
            res["similarity_score"] = hit.score
            res["point_id"] = str(hit.id)
            res["collection"] = collection_name
            if twin:
                vectors = hit.vector if isinstance(hit.vector, dict) else {}
                res.update(
                    twin_scores(
                        query_vector, vectors.get(VULN_VECTOR), vectors.get(FIXED_VECTOR)
                    )
                )
            results.append(res)
        return results
