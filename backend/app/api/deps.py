"""Shared FastAPI dependencies."""

from fastapi import Header, HTTPException, status

from backend.app.config import settings


def require_api_key(
    x_reposentinel_key: str | None = Header(None, alias="X-RepoSentinel-Key"),
) -> None:
    """Validate the shared-secret header.

    - No key configured + dev  -> open (developer convenience).
    - No key configured + prod -> should be impossible (create_app guards at
      startup), but fail closed just in case.
    - Key configured           -> header must match exactly.
    """
    expected = settings.REPOSENTINEL_API_KEY
    if not expected:
        if settings.is_dev:
            return
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key not configured",
        )
    if x_reposentinel_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
