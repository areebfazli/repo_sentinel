"""ORM models.

Tables are added in the phases that use them:
  Phase 3 -> EmbeddingCache
  Phase 4 -> Scan, Finding
  Phase 5 -> Feedback, IngestionState
"""
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EmbeddingCache(Base):
    """Content-addressed cache of embeddings, keyed by sha256(model:pooling:text).

    Lets ingestion, eval, and repeat scans skip re-embedding unchanged text.
    """

    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded list[float]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
