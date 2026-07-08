"""SQLAlchemy engine/session wiring.

Dev uses a local SQLite file (see ``settings.get_database_url``); production uses
whatever ``DATABASE_URL`` points at. ``init_db()`` creates all tables and is
called once from the FastAPI lifespan.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


_url = settings.get_database_url
# check_same_thread is only meaningful for SQLite; FastAPI BackgroundTasks may
# touch the session from a worker thread.
_connect_args = {"check_same_thread": False} if _url.startswith("sqlite") else {}

engine = create_engine(_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables registered on ``Base.metadata``."""
    # Import models for their side effect of registering tables on Base.metadata.
    from backend.app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
