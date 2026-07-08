import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from git import Repo
from github import Github
from github.Repository import Repository


class GithubCrawler:
    def __init__(self, token: str | None = None):
        """
        Initialize the GitHub crawler.
        If no token is provided, it will make unauthenticated requests (with lower rate limits).
        """
        self.gh = Github(token) if token else Github()
        # Base directory to clone repositories into
        self.clone_base_dir = Path("/tmp/repo_sentinel_clones")
        self.clone_base_dir.mkdir(parents=True, exist_ok=True)

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """Extract 'owner/repo' from a full GitHub URL."""
        # E.g., https://github.com/microsoft/vscode -> microsoft/vscode
        parts = repo_url.rstrip('/').split('/')
        return f"{parts[-2]}/{parts[-1]}"

    def clone_repository(self, repo_url: str) -> Path:
        """
        Clone a repository locally for parsing source code.
        Returns the path to the cloned repository.
        """
        repo_name = self._get_repo_name_from_url(repo_url)
        clone_dir = self.clone_base_dir / repo_name.replace("/", "_")
        
        # If it already exists, pull latest
        if clone_dir.exists() and (clone_dir / ".git").exists():
            repo = Repo(clone_dir)
            repo.remotes.origin.pull()
        else:
            # Clean up if it was a partial clone
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            Repo.clone_from(repo_url, clone_dir)
            
        return clone_dir

    def fetch_team_history(
        self,
        repo_full_name: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Crawl closed-PR review comments into Team Memory documents.

        Emits ONE document per review comment (not one blob per PR) so each
        memory is a focused code-review discussion tied to a diff hunk — much
        better for code-to-code matching. Ordered by most-recently-updated and
        stopped once a PR older than ``since`` (the last-run timestamp) is
        reached — so a long-lived PR that merges out of creation order is still
        picked up (re-ingestion is idempotent via uuid5 point ids).
        """
        repo: Repository = self.gh.get_repo(repo_full_name)
        prs = repo.get_pulls(state="closed", sort="updated", direction="desc")

        docs: list[dict[str, Any]] = []
        count = 0
        for pr in prs:
            if count >= limit:
                break
            updated = pr.updated_at
            if updated is not None and updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if since is not None and updated is not None and updated <= since:
                break  # everything newer than the last run has been seen
            count += 1

            for comment in pr.get_review_comments():
                body = comment.body or ""
                diff_hunk = comment.diff_hunk or ""
                if not body:
                    continue
                author = comment.user.login if comment.user else "unknown"
                text_content = (
                    f"{diff_hunk}\n\nReview by {author} on {comment.path}:\n{body}"
                    if diff_hunk
                    else f"Review by {author} on {comment.path}:\n{body}"
                )
                docs.append(
                    {
                        "id": f"pr{pr.number}_rc{comment.id}",
                        "pr_number": pr.number,
                        "pr_title": pr.title,
                        "pr_url": pr.html_url,
                        "comment_id": comment.id,
                        "comment_url": comment.html_url,
                        "author": author,
                        "author_association": comment.raw_data.get("author_association", "NONE"),
                        "file_path": comment.path,
                        "created_at": comment.created_at.isoformat(),
                        "body": body,
                        "diff_hunk": diff_hunk,
                        "text_content": text_content,
                    }
                )

        return docs

    def cleanup_clone(self, repo_url: str):
        """Remove the local clone of the repository."""
        repo_name = self._get_repo_name_from_url(repo_url)
        clone_dir = self.clone_base_dir / repo_name.replace("/", "_")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
