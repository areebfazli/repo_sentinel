import requests
from typing import Dict, Any
from backend.app.config import settings

class ReportGenerator:
    def __init__(self):
        # Fall back to a dummy client if no key is provided, so the pipeline doesn't crash during dev
        self.api_key = settings.OPENROUTER_API_KEY
        self.use_api = bool(self.api_key and self.api_key != "your_openrouter_api_key_here")

    def generate_pr_comment(self, code_snippet: str, merged_findings: Dict[str, Any]) -> str:
        """
        Takes the raw data from the RAG Merger and uses Claude (via OpenRouter) 
        to format it into a highly actionable, context-aware PR comment.
        """
        cves = merged_findings.get("ghost_hunter_findings", [])
        team_history = merged_findings.get("team_memory_findings", [])
        
        if not cves and not team_history:
            return "✅ **RepoSentinel Scan Complete:** No known vulnerabilities or past team antipatterns detected in this code block."
            
        # Construct the context block to feed to Claude
        prompt = f"""
You are RepoSentinel, an advanced AI security reviewer. A developer has submitted the following code in a Pull Request:

```python
{code_snippet}
```

We have scanned this code against two databases.
1. The global CVE database (Ghost Hunter).
2. The internal team history database (Team Memory).

Here are the mathematical matches we found:
GHOST HUNTER (CVEs): {cves}
TEAM MEMORY (Past PRs): {team_history}

Write a clear, professional, and slightly urgent GitHub PR comment reviewing this code.
Use the exact structure requested below. Do not use generic warnings; explicitly reference the CVE IDs and the specific internal PR/author from the data provided.

Use this Markdown structure:
## 🔴 RepoSentinel Security Report

### 🌐 The World Has Seen This Break Before
[Explain the CVE findings here, linking the CVE ID and explaining how the developer's code triggers it. If none, say "Clean."]

### 🏠 Your Team Has Seen This Break Before
[Explain the Team Memory findings here. Mention the past PR ID and what the previous reviewer said. If none, say "Clean."]

### ✅ Recommended Fix
[Provide a concrete code snippet fixing the issues]
"""

        # If we don't have a real API key, return a mock response for testing the UI/Pipeline
        if not self.use_api:
            return f"""
## 🔴 RepoSentinel Security Report (MOCK LLM RESPONSE - Missing OpenRouter Key)

### 🌐 The World Has Seen This Break Before
Found {len(cves)} matching CVEs. 
First match: {cves[0]['cve_id'] if cves else 'None'}

### 🏠 Your Team Has Seen This Break Before
Found {len(team_history)} matching past PR discussions.
First match: {team_history[0]['pr_id'] if team_history else 'None'}

*(Please add your OPENROUTER_API_KEY to .env to generate the full report)*
"""

        # Actually call OpenRouter
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:8000", # Required by OpenRouter
                "X-Title": "RepoSentinel", # Required by OpenRouter
            },
            json={
                "model": "openai/gpt-oss-120b:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error connecting to OpenRouter LLM: {response.text}"
