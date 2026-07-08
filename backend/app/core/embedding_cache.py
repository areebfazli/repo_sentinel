"""SQLite-backed embedding cache.

Keyed by sha256(model:pooling:text). Ensures its table exists on construction so
standalone scripts (ingestion, eval) that don't call init_db() still work.
"""
import json

from sqlalchemy import select

from backend.app.db.models import EmbeddingCache as EmbeddingCacheModel
from backend.app.db.session import SessionLocal, engine


class EmbeddingCache:
    def __init__(self, session_factory=SessionLocal):
        self._session_factory = session_factory
        # Idempotent: create just this table if the DB hasn't been initialised.
        EmbeddingCacheModel.__table__.create(bind=engine, checkfirst=True)

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
        """Upsert records: each is {key, model, dim, vector: list[float]}."""
        if not records:
            return
        with self._session_factory() as session:
            for r in records:
                session.merge(
                    EmbeddingCacheModel(
                        key=r["key"],
                        model=r["model"],
                        dim=r["dim"],
                        vector=json.dumps(r["vector"]),
                    )
                )
            session.commit()
