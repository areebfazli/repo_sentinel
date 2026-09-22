"""LLM prompt construction, finding validation, and deterministic Markdown.

The LLM returns structured JSON findings; we validate them against the allowlist
of retrieved IDs (anti-hallucination) and render the Markdown ourselves so the
report format is deterministic and can't inject fake CVE IDs.
"""
import difflib
from typing import Any

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

SYSTEM_PROMPT = (
    "You are RepoSentinel, an AI security reviewer. You are given a developer's "
    "code plus matches retrieved from a CVE database (Ghost Hunter) and the team's "
    "past PR reviews (Team Memory). Decide which matches genuinely apply to the "
    "code and explain them.\n\n"
    "Return ONLY a JSON object of the form:\n"
    '{"findings": [{"severity": "critical|high|medium|low", "cve_id": "<id or null>", '
    '"team_pr_id": "<id or null>", "title": "...", "explanation": "...", '
    '"fix_snippet": "..."}]}\n\n'
    "Rules:\n"
    "- Only reference cve_id / team_pr_id values that appear in the provided context. "
    "Never invent identifiers.\n"
    "- If a retrieved match does not actually apply to the code, omit it.\n"
    "- If nothing genuinely applies, return {\"findings\": []}.\n"
    "- fix_snippet is a short corrected code example (may be empty)."
)


def severity_from_cvss(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"


# Prompt-size caps for retrieved corpus code. Mined (OSV) entries are whole
# functions and can run to hundreds of lines; the handwritten ones fit easily.
PROMPT_MAX_CODE_LINES = 40
PROMPT_MAX_DIFF_LINES = 40
PROMPT_MAX_LINE_CHARS = 200  # guards against minified one-line JS


def _clip_lines(lines: list[str], max_lines: int, what: str) -> list[str]:
    clipped = [
        ln if len(ln) <= PROMPT_MAX_LINE_CHARS else ln[:PROMPT_MAX_LINE_CHARS] + " ..."
        for ln in lines[:max_lines]
    ]
    if len(lines) > max_lines:
        clipped.append(f"... ({what} truncated, {len(lines) - max_lines} more lines)")
    return clipped


def fix_diff(vulnerable_code: str, fixed_code: str, max_lines: int = PROMPT_MAX_DIFF_LINES) -> str:
    """Compact unified diff vulnerable -> fixed (file headers dropped, 1 line of
    context, at most ``max_lines`` lines). Empty when the two don't differ."""
    diff = list(
        difflib.unified_diff(
            vulnerable_code.strip().splitlines(),
            fixed_code.strip().splitlines(),
            lineterm="",
            n=1,
        )
    )[2:]  # skip the ---/+++ header lines
    return "\n".join(_clip_lines(diff, max_lines, "diff"))


def _indented(text: str, prefix: str = "    ") -> list[str]:
    return [f"{prefix}{ln}" for ln in text.splitlines()]


def build_user_prompt(code_snippet: str, cves: list[dict], team: list[dict]) -> str:
    """Compose the user message: the code plus compact match context.

    CVE matches with a stored patched twin also carry the fix as a diff, so the
    model can tell code that matches the bug from code that already has the fix.
    """
    lines = ["Developer code under review:", "```", code_snippet.strip(), "```", ""]

    if cves:
        lines.append("Ghost Hunter — retrieved CVE matches:")
        any_diff = False
        for c in cves:
            lines.append(
                f"- cve_id={c.get('cve_id')} severity={c.get('severity')} "
                f"category={c.get('category')}: {c.get('description', '')}"
            )
            vulnerable = c.get("vulnerable_code") or ""
            if vulnerable.strip():
                lines.append("  vulnerable pattern:")
                code_lines = _clip_lines(
                    vulnerable.strip().splitlines(), PROMPT_MAX_CODE_LINES, "code"
                )
                lines.extend(_indented("\n".join(code_lines)))
                diff = fix_diff(vulnerable, c["fixed_code"]) if c.get("fixed_code") else ""
                if diff:
                    any_diff = True
                    lines.append("  how this CVE was fixed (unified diff, vulnerable -> fixed):")
                    lines.extend(_indented(diff))
        if any_diff:
            lines.append(
                "For each match that shows how it was fixed: judge whether the developer's "
                "code looks like the pre-fix version ('-' lines) or the post-fix version "
                "('+' lines). If it already has the fix, the match does not apply — omit it."
            )
        lines.append("")

    if team:
        lines.append("Team Memory — retrieved past PR discussions:")
        for t in team:
            preview = t.get("text") or t.get("snippet_preview", "")
            lines.append(
                f"- team_pr_id={t.get('pr_id')} title={t.get('title')} "
                f"author={t.get('author')}: {preview[:300]}"
            )
        lines.append("")

    lines.append("Write the findings JSON now.")
    return "\n".join(lines)


def validate_findings(
    llm_findings: list[dict], allowed_cves: set[str], allowed_prs: set[str]
) -> list[dict]:
    """Drop findings referencing IDs not in the retrieved allowlist.

    IDs are normalized to strings first: the prompt renders team_pr_id unquoted,
    so the LLM may return it as a JSON number, and a naive `1042 in {"1042"}`
    would wrongly discard a valid finding.
    """
    validated = []
    for f in llm_findings:
        if not isinstance(f, dict):
            continue
        cid = str(f["cve_id"]) if f.get("cve_id") is not None else None
        pid = str(f["team_pr_id"]) if f.get("team_pr_id") is not None else None
        if cid and cid not in allowed_cves:
            continue
        if pid and pid not in allowed_prs:
            continue
        validated.append(
            {
                "severity": (f.get("severity") or "").lower() or None,
                "cve_id": cid,
                "team_pr_id": pid,
                "title": f.get("title") or "Security finding",
                "explanation": f.get("explanation") or "",
                "fix_snippet": f.get("fix_snippet") or "",
            }
        )
    return validated


def render_markdown(findings: list[dict], cve_count: int, team_count: int) -> str:
    """Render the PR comment from validated structured findings (deterministic)."""
    if not findings:
        return (
            "## ✅ RepoSentinel Security Report\n\n"
            "No known vulnerabilities or past team antipatterns applied to this code."
        )

    findings = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.get("severity"), 4))

    out = ["## 🔴 RepoSentinel Security Report", ""]

    cve_findings = [f for f in findings if f.get("cve_id")]
    team_findings = [f for f in findings if f.get("team_pr_id")]
    other_findings = [f for f in findings if not f.get("cve_id") and not f.get("team_pr_id")]

    out.append("### 🌐 The World Has Seen This Break Before")
    if cve_findings:
        for f in cve_findings:
            out.extend(_render_finding(f, ref=f["cve_id"]))
    else:
        out.append("Clean.")
    out.append("")

    out.append("### 🏠 Your Team Has Seen This Break Before")
    if team_findings:
        for f in team_findings:
            out.extend(_render_finding(f, ref=f"PR {f['team_pr_id']}"))
    else:
        out.append("Clean.")
    out.append("")

    if other_findings:
        out.append("### ⚠️ Other Observations")
        for f in other_findings:
            out.extend(_render_finding(f, ref=None))
        out.append("")

    out.append(
        f"_Scanned against {cve_count} CVE match(es) and {team_count} team-memory match(es)._"
    )
    return "\n".join(out)


def _render_finding(f: dict[str, Any], ref: str | None) -> list[str]:
    sev = f.get("severity")
    badge = f"`{sev.upper()}` " if sev else ""
    heading = f"- **{badge}{f['title']}**"
    if ref:
        heading += f" ({ref})"
    block = [heading]
    if f.get("explanation"):
        block.append(f"  {f['explanation']}")
    if f.get("fix_snippet"):
        block.append("  ```")
        block.extend(f"  {line}" for line in f["fix_snippet"].strip().splitlines())
        block.append("  ```")
    return block
