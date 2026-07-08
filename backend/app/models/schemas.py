
from pydantic import BaseModel, HttpUrl


class AnalyzeRequest(BaseModel):
    """
    The incoming webhook payload from a GitHub Action or a manual developer request.
    """
    repo_url: HttpUrl | None = None
    pr_number: int | None = None
    code_snippet: str
    language: str = "python"
    author: str | None = "unknown"

class AnalyzeResponse(BaseModel):
    """
    The response sent back to the client/GitHub Action.
    """
    is_vulnerable: bool
    report_markdown: str
    ghost_hunter_matches: int
    team_memory_matches: int
