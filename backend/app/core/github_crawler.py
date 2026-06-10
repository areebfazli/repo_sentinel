import os
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from github import Github
from github.Repository import Repository
from git import Repo

class GithubCrawler:
    def __init__(self, token: Optional[str] = None):
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

    def fetch_team_history(self, repo_url: str) -> List[Dict[str, Any]]:
        """
        Crawl PRs, issues, and commits to build the 'Team Memory'.
        Returns a list of documents to be embedded.
        """
        repo_name = self._get_repo_name_from_url(repo_url)
        repo: Repository = self.gh.get_repo(repo_name)
        
        history_docs = []
        
        # 1. Fetch Pull Requests (closed/merged)
        # We look at closed PRs because they contain resolved discussions/bugs
        prs = repo.get_pulls(state='closed', sort='updated', direction='desc')
        
        # Limit to recent PRs for initial prototype to avoid massive rate limit hits
        count = 0
        for pr in prs:
            if count >= 50: # Temporary limit for MVP
                break
                
            # Skip if no body or comments
            if not pr.body and pr.comments == 0:
                continue
                
            doc = {
                "id": f"pr_{pr.number}",
                "type": "pull_request",
                "title": pr.title,
                "body": pr.body or "",
                "url": pr.html_url,
                "created_at": pr.created_at.isoformat(),
                "merged": pr.merged,
                "author": pr.user.login if pr.user else "unknown"
            }
            
            # Combine PR description and review comments into the text to embed
            text_content = f"PR: {pr.title}\nDescription: {pr.body}\n"
            
            # Get review comments (the actual code review discussions)
            review_comments = pr.get_review_comments()
            for comment in review_comments:
                text_content += f"\nReview Comment by {comment.user.login} on file {comment.path}:\n{comment.body}"
                if comment.diff_hunk:
                    text_content += f"\nCode Diff:\n{comment.diff_hunk}"
            
            doc["text_content"] = text_content
            history_docs.append(doc)
            count += 1
            
        # TODO: Add Issue and Commit fetching in the future
        
        return history_docs

    def cleanup_clone(self, repo_url: str):
        """Remove the local clone of the repository."""
        repo_name = self._get_repo_name_from_url(repo_url)
        clone_dir = self.clone_base_dir / repo_name.replace("/", "_")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
