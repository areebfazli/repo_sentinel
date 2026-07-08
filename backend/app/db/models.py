"""ORM models.

Tables are added in the phases that use them:
  Phase 3 -> EmbeddingCache
  Phase 4 -> Scan, Finding
  Phase 5 -> Feedback, IngestionState
"""
from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
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


class Scan(Base):
    """One analysis job. Created queued, run in the background, polled by id."""

    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # uuid4 hex
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    mode: Mapped[str] = mapped_column(String, nullable=False, default="snippet")  # snippet | files
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    repo: Mapped[str | None] = mapped_column(String, nullable=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    author: Mapped[str | None] = mapped_column(String, nullable=True)
    is_vulnerable: Mapped[bool | None] = mapped_column(nullable=True)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_provider_used: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Finding(Base):
    """A retrieved match (CVE or team memory) tied to a scan.

    point_id lets feedback (Phase 5) target the exact vector this came from.
    file/line/function columns stay null until files mode (Phase 6).
    """

    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False)  # cve | team
    collection: Mapped[str] = mapped_column(String, nullable=False)
    point_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    cve_id: Mapped[str | None] = mapped_column(String, nullable=True)
    team_pr_id: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    function_name: Mapped[str | None] = mapped_column(String, nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rerank_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rerank_prob: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    adjusted_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Feedback(Base):
    """A developer's up/down vote on a finding.

    Unique per finding (upsert = change your vote). Suppression aggregates by
    point_id across scans, so downvoting the same memory in different scans
    accumulates.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    finding_id: Mapped[int] = mapped_column(
        ForeignKey("findings.id"), nullable=False, unique=True
    )
    scan_id: Mapped[str] = mapped_column(String, nullable=False)
    point_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String, nullable=False)
    vote: Mapped[int] = mapped_column(Integer, nullable=False)  # +1 or -1
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class IngestionState(Base):
    """Per-repo crawl cursor so Team Memory ingestion only fetches new PRs."""

    __tablename__ = "ingestion_state"

    repo: Mapped[str] = mapped_column(String, primary_key=True)  # owner/name
    last_pr_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_run_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
