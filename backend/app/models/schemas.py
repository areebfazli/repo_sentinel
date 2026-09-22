from typing import Literal

from pydantic import BaseModel, HttpUrl, model_validator


class FileInput(BaseModel):
    """A changed file in files mode. changed_lines (explicit) wins over patch;
    if neither is given, every function in the file is analyzed."""

    path: str
    content: str
    changed_lines: list[int] | None = None
    patch: str | None = None


class AnalyzeRequest(BaseModel):
    """Incoming analysis request (manual dashboard or GitHub Action webhook).

    Exactly one of ``code_snippet`` (dashboard) or ``files`` (diff-aware, from the
    GitHub Action) must be provided.
    """

    repo_url: HttpUrl | None = None
    pr_number: int | None = None
    code_snippet: str | None = None
    files: list[FileInput] | None = None
    # None = don't language-filter the CVE search (the dashboard can't reliably
    # know the language; a wrong guess causes false "clean" verdicts). Files mode
    # derives language per function and ignores this.
    language: str | None = None
    author: str | None = "unknown"

    @model_validator(mode="after")
    def _exactly_one_mode(self):
        # An empty files list counts as "not provided" — reject it rather than
        # letting it fall through to the snippet path with code_snippet=None.
        has_snippet = self.code_snippet is not None
        has_files = bool(self.files)
        if has_snippet == has_files:
            raise ValueError("provide exactly one of 'code_snippet' or a non-empty 'files'")
        return self


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
    file_path: str | None = None
    start_line: int | None = None
    similarity_score: float
    rerank_prob: float
    # Patched-twin scores (CVE matches with a stored fix only; else None):
    # cos(code, fixed version) and cos(code, vulnerable) - that.
    sim_fixed: float | None = None
    twin_margin: float | None = None


class ReportFinding(BaseModel):
    """An LLM-confirmed finding — what the GitHub Action gates on and comments.

    Unlike FindingOut (every raw retrieval match, for the dashboard), these have
    passed the LLM's applicability judgment + allowlist validation.
    """

    severity: str | None = None
    cve_id: str | None = None
    team_pr_id: str | None = None
    title: str
    explanation: str = ""
    fix_snippet: str = ""
    file_path: str | None = None
    start_line: int | None = None
    function_name: str | None = None
    finding_id: int | None = None
    point_id: str | None = None
    source: str | None = None


class AnalyzeResult(BaseModel):
    """The completed analysis payload."""

    is_vulnerable: bool
    report_markdown: str
    findings: list[FindingOut]
    report_findings: list[ReportFinding] = []
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
