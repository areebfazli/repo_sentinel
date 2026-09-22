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
MARKER_RE = re.compile(r"<!-- (reposentinel:[^\s]+) -->")
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


def anchor_line(finding: dict, changed_lines: list[int]) -> int | None:
    """Pick a diff line to attach the review comment to (must be in the diff)."""
    if not changed_lines:
        return None
    start = finding.get("start_line")
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


def build_comment_body(finding: dict) -> str:
    ref = finding.get("cve_id") or (
        f"PR {finding['team_pr_id']}" if finding.get("team_pr_id") else ""
    )
    sev = finding.get("severity")
    badge = f"**`{sev.upper()}`** " if sev else ""
    if finding.get("cve_id"):
        icon = "🌐"
    elif finding.get("team_pr_id"):
        icon = "🏠"
    else:
        icon = "⚠️"
    header = f"{icon} {badge}{finding.get('title', 'Security finding')}"
    if ref:
        header += f" ({ref})"
    body = header
    if finding.get("explanation"):
        body += f"\n\n{finding['explanation']}"
    if finding.get("fix_snippet"):
        body += f"\n\n```\n{finding['fix_snippet']}\n```"
    marker = finding_marker(
        finding.get("file_path", ""), finding.get("point_id") or "", finding.get("function_name")
    )
    return f"{body}\n\n<sub>RepoSentinel</sub>\n<!-- {marker} -->"


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
    """The summary comment is the LLM-authored report (already applicability-judged
    and allowlist-validated), plus the gate footer and the dedup marker."""
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


def desired_comments(findings: list[dict], changed_by_file: dict[str, list[int]]) -> list[dict]:
    desired, unanchored = [], []
    for f in findings:
        path = f.get("file_path")
        line = anchor_line(f, changed_by_file.get(path, [])) if path else None
        marker = finding_marker(path or "", f.get("point_id", ""), f.get("function_name"))
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
    # Gate + inline comments use the LLM-VALIDATED findings (applicability-judged,
    # allowlist-checked) — not the raw high-recall retrieval matches.
    findings = result.get("report_findings", [])
    print(f"{len(findings)} confirmed finding(s); is_vulnerable={result.get('is_vulnerable')}")

    changed_by_file = {file["path"]: parse_changed_lines(file.get("patch") or "") for file in files}
    desired, unanchored = desired_comments(findings, changed_by_file)
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
