"""ORM models.

Phase 1 ships the module so ``init_db()`` has something to import; concrete
tables (scans, findings, feedback, ingestion_state, embedding_cache) are added
in the phases that use them.
"""
from backend.app.db.session import Base  # noqa: F401

# Tables are declared in later phases:
#   Phase 3 -> EmbeddingCache
#   Phase 4 -> Scan, Finding
#   Phase 5 -> Feedback, IngestionState
