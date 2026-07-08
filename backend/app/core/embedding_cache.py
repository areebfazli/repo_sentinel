"""SQLite-backed embedding cache.

Keyed by sha256(model:pooling:text). Ensures its table exists on construction so
standalone scripts (ingestion, eval) that don't call init_db() still work.
"""
import json

from sqlalchemy import select

from backend.app.db.models import EmbeddingCache as EmbeddingCacheModel
from backend.app.db.session import SessionLocal


class EmbeddingCache:
    def __init__(self, session_factory=SessionLocal):
        self._session_factory = session_factory
        # Idempotent, and on the injected factory's own bind (not the global
        # engine) so an injected/isolated DB gets the table it will actually use.
        with self._session_factory() as session:
            EmbeddingCacheModel.__table__.create(bind=session.get_bind(), checkfirst=True)

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        with self._session_factory() as session:
            rows = (
                session.execute(
                    select(EmbeddingCacheModel).where(EmbeddingCacheModel.key.in_(keys))
                )
                .scalars()
                .all()
            )
            return {r.key: json.loads(r.vector) for r in rows}

    def put_many(self, records: list[dict]) -> None:
        """Upsert records: each is {key, model, dim, vector: list[float]}.

        Collapse duplicate keys within the batch first — the same uncached text
        can appear twice in one embedding batch (e.g. two identical functions),
        and two merges of the same PK in one autoflush=False session would emit
        two INSERTs and raise a UNIQUE-constraint IntegrityError.
        """
        if not records:
            return
        deduped = {r["key"]: r for r in records}
        with self._session_factory() as session:
            for r in deduped.values():
                session.merge(
                    EmbeddingCacheModel(
                        key=r["key"],
                        model=r["model"],
                        dim=r["dim"],
                        vector=json.dumps(r["vector"]),
                    )
                )
            session.commit()
