from typing import Literal

from pydantic import BaseModel, HttpUrl


class AnalyzeRequest(BaseModel):
    """Incoming analysis request (manual dashboard or GitHub Action webhook)."""

    repo_url: HttpUrl | None = None
    pr_number: int | None = None
    code_snippet: str
    language: str = "python"
    author: str | None = "unknown"


class AnalyzeAccepted(BaseModel):
    """202 response: the scan was queued; poll poll_url for the result."""

    job_id: str
    status: str
    poll_url: str


class FindingOut(BaseModel):
    """A retrieved match surfaced to the client (and targetable by feedback)."""

    finding_id: int
    point_id: str
    source: str  # cve | team
    title: str
    severity: str | None = None
    cve_id: str | None = None
    team_pr_id: str | None = None
    similarity_score: float
    rerank_prob: float


class AnalyzeResult(BaseModel):
    """The completed analysis payload."""

    is_vulnerable: bool
    report_markdown: str
    findings: list[FindingOut]
    ghost_hunter_matches: int
    team_memory_matches: int
    llm_provider_used: str | None = None


class JobStatusResponse(BaseModel):
    """Polling response for a scan job."""

    job_id: str
    status: str  # queued | running | completed | failed
    created_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: AnalyzeResult | None = None


class FeedbackRequest(BaseModel):
    """A thumbs up/down on a finding (vote must be +1 or -1)."""

    finding_id: int
    vote: Literal[-1, 1]


class FeedbackResponse(BaseModel):
    status: str
    finding_id: int
    point_id: str
    net_votes: int
