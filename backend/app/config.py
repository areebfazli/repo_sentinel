from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "RepoSentinel"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development" # "development" or "production"

    # API Keys
    GITHUB_TOKEN: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # Service URLs (for production)
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: Optional[int] = 6333

    # Models
    CODEBERT_MODEL: str = "microsoft/codebert-base"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    SEVERITY_MODEL: str = "distilbert-base-uncased"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def get_database_url(self) -> str:
        """Return SQLite URL for dev, Postgres for prod"""
        if self.is_dev or not self.DATABASE_URL:
            # Local SQLite database in the root folder
            db_path = BASE_DIR / "repo_sentinel.sqlite"
            return f"sqlite:///{db_path}"
        return self.DATABASE_URL

    @property
    def qdrant_local_path(self) -> Optional[str]:
        """Return local path for Qdrant if in dev mode"""
        if self.is_dev:
            path = BASE_DIR / "qdrant_data"
            os.makedirs(path, exist_ok=True)
            return str(path)
        return None

settings = Settings()
