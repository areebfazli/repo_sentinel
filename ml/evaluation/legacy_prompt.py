"""The pre-2026-09-24 report prompt, frozen for the eval's ``--llm-prompt legacy`` arm.

Verbatim copy of ``markdown_renderer``'s SYSTEM_PROMPT / build_user_prompt /
validate_findings as of commit 4567539: the LLM may only report findings tied to
a retrieved CVE / team match (anything else "omit"), the code and corpus text go
into the prompt raw, and an item without retrieved CVEs gets no call. Kept only
so the old and new prompts can be compared on the same model; not used by the
API.
"""
import difflib

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
