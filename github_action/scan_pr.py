"""RepoSentinel GitHub Action runner.

Runs in a PR workflow: collects the PR's changed files, sends them to the
RepoSentinel API (files mode), then posts findings as inline review comments —
deduped across pushes via hidden markers — plus a summary comment, and sets the
check status via a configurable severity gate.

Standalone (only `requests` required); backend modules are NOT importable here, so
the diff parser is duplicated. The decision-making logic (comment planning,
severity gate, line anchoring) is pure and unit-tested; main() does the I/O.
"""
import base64
import hashlib
import json
import os
import re
import sys
import time

import requests

GITHUB_API = "https://api.github.com"
SUPPORTED_EXTS = {".py", ".js", ".ts", ".go", ".java"}
MAX_FILE_BYTES = 1_000_000  # contents API caps ~1MB
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 15 * 60

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
# Only a marker at the very END of a body counts: ours is always appended last,
# so a marker-shaped string earlier in the body (e.g. inside quoted PR code or
# LLM text) can't hijack another finding's comment.
MARKER_RE = re.compile(r"<!-- (reposentinel:[^\s]+) -->\s*\Z")
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


# ---------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------
def parse_changed_lines(patch: str) -> list[int]:
    """New-file line numbers added by a unified-diff patch."""
    if not patch:
        return []
    changed, new_line, in_hunk = [], 0, False
    for line in patch.splitlines():
        header = HUNK_HEADER.match(line)
        if header:
            new_line, in_hunk = int(header.group(1)), True
            continue
        if not in_hunk:
            continue
        tag = line[0] if line else " "  # stripped-blank context line
        if tag == "+":
            changed.append(new_line)
            new_line += 1
        elif tag == "-" or tag == "\\":
            continue  # removals / no-newline markers don't advance the new file
        else:
            new_line += 1  # context line
    return changed


def parse_commentable_lines(patch: str) -> list[int]:
    """New-file lines a review comment can attach to: added AND context lines of
    every hunk (GitHub accepts RIGHT-side comments on any line shown in the diff)."""
    if not patch:
        return []
    lines, new_line, in_hunk = [], 0, False
    for line in patch.splitlines():
        header = HUNK_HEADER.match(line)
        if header:
            new_line, in_hunk = int(header.group(1)), True
            continue
        if not in_hunk:
            continue
        tag = line[0] if line else " "
        if tag == "-" or tag == "\\":
            continue
        lines.append(new_line)
        new_line += 1
    return lines


def severity_gate(findings: list[dict], threshold: str) -> int:
    """Exit code: 1 if any finding is at/above the threshold severity, else 0."""
    if not threshold or threshold == "none":
        return 0
    thr = SEVERITY_RANK.get(threshold, 99)
    for f in findings:
        sev = f.get("severity")
        if sev and SEVERITY_RANK.get(sev, 0) >= thr:
            return 1
    return 0


def anchor_line(
    finding: dict, changed_lines: list[int], commentable_lines: list[int] | None = None
) -> int | None:
    """Pick a diff line to attach the review comment to (must be in the diff).

    The finding's exact offending ``line`` (newer servers) wins when it is an
    added line or, with ``commentable_lines``, any line shown in the diff (a
    finding about a deleted guard points at unchanged code next to it). Else
    the function's ``start_line`` and the nearest added line, as before.
    """
    exact = finding.get("line")
    if exact is not None and (
        exact in changed_lines or (commentable_lines and exact in commentable_lines)
    ):
        return exact
    if not changed_lines:
        if exact is not None and commentable_lines:
            return min(commentable_lines, key=lambda ln: abs(ln - exact))
        return None
    start = finding.get("start_line") if exact is None else exact
    if start in changed_lines:
        return start
    # Prefer the first changed line at/after the function start, else the nearest.
    after = [ln for ln in changed_lines if start is None or ln >= start]
    if after:
        return min(after)
    return min(changed_lines, key=lambda ln: abs(ln - (start or changed_lines[0])))


def finding_marker(file_path: str, point_id: str, function_name: str | None) -> str:
    """Marker granularity matches the backend's (point_id, file, function) dedupe
    key, so it stays stable across pushes even when line numbers shift."""
    digest = hashlib.sha1(
        f"{file_path}|{point_id}|{function_name or ''}".encode()
    ).hexdigest()[:16]
    return f"reposentinel:f:{digest}"


def extract_marker(body: str) -> str | None:
    m = MARKER_RE.search(body or "")
    return m.group(1) if m else None


# --- Escaping of server-provided text ---------------------------------------
# Finding text is LLM-written (and quoted_code is PR code), so it is escaped
# here before it becomes Markdown: no links, images, raw HTML, @-mentions or
# marker-shaped comments. Mirrors backend/app/core/untrusted.py (not importable).
_URL_RE = re.compile(r"(?:\b[a-zA-Z][a-zA-Z0-9+.\-]{1,15}://|\bwww\.)[^\s<>]+")
_MENTION_RE = re.compile(r"(?<![\w`])@(?=[A-Za-z0-9])")
_MD_SPECIAL_RE = re.compile(r"([\\`*_\[\]()!#|~{}])")
_BACKTICKS_RE = re.compile(r"`+")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HIDDEN_CHARS_RE = re.compile(
    r"[\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064"
    r"\u2066-\u2069\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def _strip_hidden(text) -> str:
    """Prose: complete HTML comments and invisible / control characters removed
    (a dangling ``<!--`` is escaped by ``_escape_plain``)."""
    return _HIDDEN_CHARS_RE.sub("", _COMMENT_RE.sub("", str(text or "")))


def _code_text(text) -> str:
    """Code: nothing removed but invisible / control characters; an HTML-comment
    opener is defused in place (stripping it would delete code)."""
    return _HIDDEN_CHARS_RE.sub("", str(text or "")).replace("<!--", "<! --")


def md_code_span(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", _code_text(text))
    longest = max((len(m.group(0)) for m in _BACKTICKS_RE.finditer(text)), default=0)
    pad = " " if text.startswith("`") or text.endswith("`") or not text else ""
    fence = "`" * (longest + 1)
    return f"{fence}{pad}{text}{pad}{fence}"


def _escape_plain(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MENTION_RE.sub("@\u200b", _MD_SPECIAL_RE.sub(r"\\\1", text))


def md_inline(text) -> str:
    """One line of inert Markdown (URLs as code, mentions defused)."""
    text = re.sub(r"\s+", " ", _strip_hidden(text)).strip()
    out, pos = [], 0
    for m in _URL_RE.finditer(text):
        out.append(_escape_plain(text[pos:m.start()]))
        out.append(md_code_span(m.group(0)))
        pos = m.end()
    out.append(_escape_plain(text[pos:]))
    return re.sub(r"^([-+]|\d+\.)", r"\\\1", "".join(out))


def md_code_block(text) -> str:
    text = _code_text(text)
    longest = max((len(m.group(0)) for m in _BACKTICKS_RE.finditer(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def build_comment_body(finding: dict) -> str:
    refs = []
    if finding.get("cwe"):
        refs.append(md_inline(finding["cwe"]))
    if finding.get("cve_id"):
        refs.append(f"similar to {md_inline(finding['cve_id'])}")
    if finding.get("team_pr_id"):
        refs.append(f"team PR {md_inline(str(finding['team_pr_id']))}")
    sev = finding.get("severity")
    badge = f"**`{sev.upper()}`** " if sev in SEVERITY_RANK else ""
    if finding.get("deterministic"):
        icon = "🛡️"
    elif finding.get("cve_id"):
        icon = "🌐"
    elif finding.get("team_pr_id"):
        icon = "🏠"
    else:
        icon = "⚠️"
    header = f"{icon} {badge}{md_inline(finding.get('title') or 'Security finding')}"
    if refs:
        header += f" ({', '.join(refs)})"
    body = header
    if finding.get("explanation"):
        body += f"\n\n{md_inline(finding['explanation'])}"
    if finding.get("reasoning"):
        body += f"\n\n**Why:** {md_inline(finding['reasoning'])}"
    if finding.get("fix_snippet"):
        body += f"\n\n{md_code_block(finding['fix_snippet'])}"
    if finding.get("deterministic"):
        body += "\n\n<sub>Deterministic check (no LLM).</sub>"
    marker = finding_marker(
        finding.get("file_path", ""), _finding_identity(finding), finding.get("function_name")
    )
    return f"{body}\n\n<sub>RepoSentinel</sub>\n<!-- {marker} -->"


def _finding_identity(finding: dict) -> str:
    """dedupe_key (newer servers: one per finding within a function), else the
    retrieval point_id (older servers)."""
    return finding.get("dedupe_key") or finding.get("point_id") or ""


def plan_comment_ops(existing: list[dict], desired: list[dict]) -> dict:
    """Diff existing vs desired reposentinel comments by marker.

    existing: [{id, body}]; desired: [{marker, path, line, body}].
    Returns {create: [desired...], update: [{id, body}], delete: [ids]}.
    """
    existing_by_marker = {}
    for c in existing:
        marker = extract_marker(c.get("body", ""))
        if marker:
            existing_by_marker[marker] = c
    desired_by_marker = {d["marker"]: d for d in desired}

    create, update, delete = [], [], []
    for marker, d in desired_by_marker.items():
        ex = existing_by_marker.get(marker)
        if ex is None:
            create.append(d)
        elif (ex.get("body") or "").strip() != d["body"].strip():
            update.append({"id": ex["id"], "body": d["body"]})
    for marker, ex in existing_by_marker.items():
        if marker not in desired_by_marker:
            delete.append(ex["id"])
    return {"create": create, "update": update, "delete": delete}


def build_summary(report_markdown: str, gate_threshold: str) -> str:
    """The summary comment is the server-rendered report (deterministic Markdown,
    LLM text already escaped there), plus the gate footer and the dedup marker."""
    body = report_markdown.strip() or "## ✅ RepoSentinel\nNo findings."
    footer = f"\n\n<sub>Severity gate: `{gate_threshold}`</sub>"
    return f"{body}{footer}\n<!-- reposentinel:summary -->"


# ---------------------------------------------------------------------------
# GitHub + RepoSentinel I/O
# ---------------------------------------------------------------------------
def _gh_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def _gh_paginated(url: str, token: str) -> list[dict]:
    results, page = [], 1
    while True:
        resp = requests.get(
            url, headers=_gh_headers(token), params={"per_page": 100, "page": page}, timeout=30
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        results.extend(batch)
        page += 1
    return results


def collect_changed_files(repo: str, pr_number: int, head_sha: str, token: str) -> list[dict]:
    """Return [{path, content, patch}] for supported, non-removed changed files."""
    files = _gh_paginated(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files", token)
    collected = []
    for f in files:
        path = f["filename"]
        if f.get("status") == "removed":
            continue
        if not any(path.endswith(ext) for ext in SUPPORTED_EXTS):
            continue
        patch = f.get("patch")  # omitted by GitHub for very large diffs
        resp = requests.get(
            f"{GITHUB_API}/repos/{repo}/contents/{path}",
            headers=_gh_headers(token),
            params={"ref": head_sha},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  skip {path}: contents HTTP {resp.status_code}")
            continue
        data = resp.json()
        if data.get("size", 0) > MAX_FILE_BYTES or data.get("encoding") != "base64":
            print(f"  skip {path}: too large or non-text")
            continue
        content = base64.b64decode(data["content"]).decode("utf-8", "replace")
        if not patch:
            # No inline patch (huge diff): scan the whole file rather than
            # silently dropping it. Findings can only anchor to the summary.
            print(f"  {path}: no patch from GitHub (large diff) — scanning whole file")
        collected.append({"path": path, "content": content, "patch": patch})
    return collected


def run_analysis(api_base: str, api_key: str, files: list[dict], repo: str, pr_number: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-RepoSentinel-Key"] = api_key
    resp = requests.post(
        f"{api_base}/api/v1/analyze/",
        headers=headers,
        json={
            "files": files,
            "repo_url": f"https://github.com/{repo}",
            "pr_number": pr_number,
        },
        timeout=60,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_S)
        poll = requests.get(f"{api_base}/api/v1/analyze/{job_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        if data["status"] == "completed":
            return data["result"]
        if data["status"] == "failed":
            raise RuntimeError(f"Scan failed: {data.get('error')}")
    raise TimeoutError("Timed out waiting for the scan to complete")


def desired_comments(
    findings: list[dict],
    changed_by_file: dict[str, list[int]],
    commentable_by_file: dict[str, list[int]] | None = None,
) -> list[dict]:
    desired, unanchored = [], []
    commentable_by_file = commentable_by_file or {}
    for f in findings:
        path = f.get("file_path")
        line = (
            anchor_line(f, changed_by_file.get(path, []), commentable_by_file.get(path))
            if path else None
        )
        marker = finding_marker(path or "", _finding_identity(f), f.get("function_name"))
        body = build_comment_body(f)
        if line is None:
            unanchored.append(f)
            continue
        desired.append({"marker": marker, "path": path, "line": line, "body": body})
    return desired, unanchored


def _check(resp, what: str, *, ok_404: bool = False) -> bool:
    """Check a GitHub API response, logging a warning on failure. Never raises.

    ok_404 marks a 404 as success too (DELETE on an already-gone comment is the
    desired end state, not a failure). Returns whether the call succeeded, so
    callers can tally failures and keep processing the remaining operations.
    """
    if resp.ok or (ok_404 and resp.status_code == 404):
        return True
    snippet = (resp.text or "")[:200]
    print(f"  WARNING: {what} failed (HTTP {resp.status_code}): {snippet}")
    return False


def _apply_comment_ops(repo, pr_number, head_sha, token, ops):
    """Execute create (one review) / update (PATCH) / delete (DELETE).

    Returns (posted, failures): posted is False on 403 (fork PR — unchanged
    fallback behavior); failures is how many PATCH/DELETE calls failed. A bad
    update/delete is logged and skipped, not raised, so it can't abort the rest.
    """
    failures = 0
    try:
        if ops["create"]:
            comments = [
                {"path": c["path"], "line": c["line"], "side": "RIGHT", "body": c["body"]}
                for c in ops["create"]
            ]
            r = requests.post(
                f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews",
                headers=_gh_headers(token),
                json={"commit_id": head_sha, "event": "COMMENT", "comments": comments},
                timeout=30,
            )
            if r.status_code == 403:
                return False, failures
            r.raise_for_status()
        for up in ops["update"]:
            r = requests.patch(
                f"{GITHUB_API}/repos/{repo}/pulls/comments/{up['id']}",
                headers=_gh_headers(token),
                json={"body": up["body"]},
                timeout=30,
            )
            if not _check(r, f"PATCH inline comment {up['id']}"):
                failures += 1
        for cid in ops["delete"]:
            r = requests.delete(
                f"{GITHUB_API}/repos/{repo}/pulls/comments/{cid}",
                headers=_gh_headers(token),
                timeout=30,
            )
            if not _check(r, f"DELETE inline comment {cid}", ok_404=True):
                failures += 1
        return True, failures
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            return False, failures
        raise


def _upsert_summary(repo, pr_number, token, body) -> bool:
    """Create or update the summary comment. Returns whether it succeeded."""
    existing = _gh_paginated(f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments", token)
    for c in existing:
        if extract_marker(c.get("body", "")) == "reposentinel:summary":
            r = requests.patch(
                f"{GITHUB_API}/repos/{repo}/issues/comments/{c['id']}",
                headers=_gh_headers(token),
                json={"body": body},
                timeout=30,
            )
            return _check(r, f"PATCH summary comment {c['id']}")
    r = requests.post(
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
        headers=_gh_headers(token),
        json={"body": body},
        timeout=30,
    )
    return _check(r, "POST summary comment")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    token = os.environ.get("GITHUB_TOKEN", "")
    api_base = os.environ.get("REPOSENTINEL_URL", "").rstrip("/")
    api_key = os.environ.get("REPOSENTINEL_API_KEY", "")
    gate = os.environ.get("INPUT_FAIL_ON_SEVERITY", "none").lower()
    repo = os.environ["GITHUB_REPOSITORY"]

    with open(os.environ["GITHUB_EVENT_PATH"]) as f:
        event = json.load(f)
    pr = event["pull_request"]
    pr_number = pr["number"]
    head_sha = pr["head"]["sha"]

    print(f"RepoSentinel scanning {repo} PR #{pr_number} @ {head_sha[:8]}")
    files = collect_changed_files(repo, pr_number, head_sha, token)
    if not files:
        print("No supported changed files; nothing to scan.")
        return 0
    print(f"Analyzing {len(files)} changed file(s)...")

    result = run_analysis(api_base, api_key, files, repo, pr_number)
    # Gate + inline comments use the REVIEWED findings (LLM findings that quote
    # the code, plus deterministic guard checks) — not the raw retrieval matches.
    findings = result.get("report_findings", [])
    print(f"{len(findings)} confirmed finding(s); is_vulnerable={result.get('is_vulnerable')}")

    changed_by_file = {file["path"]: parse_changed_lines(file.get("patch") or "") for file in files}
    commentable_by_file = {
        file["path"]: parse_commentable_lines(file.get("patch") or "") for file in files
    }
    desired, unanchored = desired_comments(findings, changed_by_file, commentable_by_file)
    summary = build_summary(result.get("report_markdown", ""), gate)

    if dry_run:
        print("DRY RUN — planned inline comments:")
        for d in desired:
            print(f"  {d['path']}:{d['line']} {d['marker']}")
        print(f"  ({len(unanchored)} unanchored -> summary only)")
        print("Summary:\n" + summary)
    else:
        existing = _gh_paginated(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments", token)
        existing = [c for c in existing if extract_marker(c.get("body", ""))]
        ops = plan_comment_ops(existing, desired)
        posted, failures = _apply_comment_ops(repo, pr_number, head_sha, token, ops)
        if not posted:
            print("No write access (fork PR?); findings above stand in for inline comments.")
            for d in desired:
                print(f"  {d['path']}:{d['line']} — {d['body'].splitlines()[0]}")
        if not _upsert_summary(repo, pr_number, token, summary):
            failures += 1
        if failures:
            print(f"{failures} comment operation(s) failed")

    exit_code = severity_gate(findings, gate)
    print(f"Severity gate ({gate}) -> exit {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
