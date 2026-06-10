from pydantic import BaseModel, HttpUrl
from typing import Optional

class AnalyzeRequest(BaseModel):
    """
    The incoming webhook payload from a GitHub Action or a manual developer request.
    """
    repo_url: Optional[HttpUrl] = None
    pr_number: Optional[int] = None
    code_snippet: str
    language: str = "python"
    author: Optional[str] = "unknown"

class AnalyzeResponse(BaseModel):
    """
    The response sent back to the client/GitHub Action.
    """
    is_vulnerable: bool
    report_markdown: str
    ghost_hunter_matches: int
    team_memory_matches: int
