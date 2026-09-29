"""Fetch REAL benign commits for the PR eval from the GitHub API (network).

``scripts/build_pr_eval.py`` can only offer "bystander" benigns offline: files
from security fix commits, which is not a realistic benign sample. This script
fetches ordinary commits from the same repos and emits them in the PR eval item
format (``kind: "benign"``, ``source: "benign_commit"``), split by repo like the
rest of ``pr_eval_v1.jsonl`` (so no repo straddles dev and test).

Per repo (dev repos first, then test, then reserve; seeded order within each):

1. ``GET /repos/{repo}/commits?per_page=100&page=N`` (default branch, newest
   first; ``--pages`` pages).
2. Drop merge commits, bot authors, every known fix commit (the corpus builder's
   mined commits, the PR eval items' commits, and any commit referenced by an OSV
   advisory in the cached OSV zips) and any commit whose message matches
   ``SECURITY_RE`` (security / CVE / GHSA / vuln / injection / XSS / sanitize /
   traversal / ...). The keyword net is deliberately wide.
3. ``GET /repos/{repo}/commits/{sha}`` for up to ``--max-candidates-per-repo``
   survivors: keep commits that change 1..``--max-files`` code files
   (.py/.js/.ts, minus test/doc/example paths, like the builder) and at most
   ``--max-changed-lines`` lines, all with an inline patch.
4. ``GET /repos/{repo}/commits/{sha}/pulls`` (``--check-linked-prs``, on by
   default): drop the commit if a linked PR's title, body or labels match
   ``SECURITY_RE``.
5. File contents: the new side at the commit (``/contents/{path}?ref={sha}``),
   the old side by reverse-applying GitHub's patch (a second call at the parent
   only if that fails). Added / removed files come straight from the patch.

Every response is cached in ``--cache-dir`` (the corpus builder's cache and key
scheme), so a re-run makes no repeated calls. Rate limits: a used-up primary
limit (403 + ``X-RateLimit-Remaining: 0``) and secondary limits (403/429 +
``Retry-After``) either sleep (``--wait-on-rate-limit``) or stop cleanly;
``--max-api-calls`` is a hard cap and ``--min-interval`` spaces calls.

``--dry-run`` prints the plan and an estimated number of API calls and time and
makes **no** network call (it never builds a live client).

    python scripts/fetch_benign_commits.py --dry-run --max-commits 300 --per-repo 3
    python scripts/fetch_benign_commits.py --max-commits 300 --per-repo 3 --wait-on-rate-limit

Needs ``GITHUB_TOKEN`` (5,000 requests/hour); unauthenticated access (60/hour) is
refused unless ``--allow-unauthenticated``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_corpus_from_osv import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    GithubClient,
    RateLimitExceeded,
    Stats,
    _is_skipped_path,
    parse_fix_commit_url,
)
from scripts.build_pr_eval import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_ORDINARY,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLIT,
    MAX_CHANGED_LINES,
    MAX_FILE_BYTES,
    MAX_FILES,
    PatchError,
    apply_patch,
    changed_count,
    file_entry,
    item_language,
    neutral_pr_text,
    public_files,
    token,
)

API = "https://api.github.com"
DEFAULT_PR_EVAL = DEFAULT_OUT_DIR / "pr_eval_v1.jsonl"
DEFAULT_OUT = DEFAULT_OUT_DIR / "pr_eval_benign_commits_v1.jsonl"
SPLIT_ORDER = ("dev", "test", "reserve")

# Wide on purpose: dropping a benign commit is cheap, keeping a silent security
# fix as "benign" is not.
SECURITY_RE = re.compile(
    r"secur|vulnerab|\bvuln|\bcve\b|cve-\d|ghsa|pysec|\bcwe\b|cwe-\d|advisory|exploit|"
    r"malicious|attack|\bxss\b|cross.site|csrf|ssrf|\bxxe\b|\brce\b|remote code|inject|"
    r"sanitiz|sanitis|escap|travers|\bdos\b|denial.of.service|redos|catastrophic|"
    r"backtrack|prototype.pollution|__proto__|overflow|bypass|privilege|unauthori|"
    r"permission|\bauth\b|authenticat|authoriz|csp\b|content.security|leak|disclos|"
    r"sensitive|secret|credential|password|token|harden|unsafe|safe_load|pickle|"
    r"deseriali|symlink|zip.?slip|open.redirect|clickjack|spoof|tamper|timing|"
    r"constant.time|crypto|hmac|\bssl\b|\btls\b|certificate|sandbox|\beval\b",
    re.I,
)

# Assumptions for the --dry-run estimate (measured on the PR eval where possible).
EST_FILES_PER_COMMIT = 1.9  # mean files/PR of the PR eval's fix commits
EST_EXAMINED_PER_ACCEPTED = 2.5  # commit-detail calls per accepted commit
EST_PARENT_REFETCH = 0.05  # share of files whose reverse patch fails
EST_SECONDS_PER_CALL = 0.4


# ---------------------------------------------------------------------------
# Pure logic (unit-tested with stubbed HTTP in tests/unit/test_fetch_benign_commits.py)
# ---------------------------------------------------------------------------


def is_security_text(text: str | None) -> bool:
    return bool(text) and SECURITY_RE.search(text) is not None


def code_files(commit: dict) -> list[dict]:
    return [f for f in commit.get("files") or []
            if Path(f.get("filename", "")).suffix.lower() in SUPPORTED_EXTENSIONS
            and not _is_skipped_path(f.get("filename", ""))]


def known_prefixes(shas) -> set[str]:
    """7-char prefixes of known commits (OSV references may be abbreviated)."""
    return {s[:7].lower() for s in shas if s and len(s) >= 7}


def list_reject_reason(entry: dict, known: set[str], include_bots: bool = False) -> str | None:
    """Why a commit from the commit list is not a candidate (``None`` = keep).
    ``known`` holds ``known_prefixes`` of fix commits."""
    sha = entry.get("sha", "")
    if sha[:7].lower() in known:
        return "known_fix_commit"
    if len(entry.get("parents") or []) != 1:
        return "merge_or_root"
    author = entry.get("author") or {}
    login = (author.get("login") or "") if isinstance(author, dict) else ""
    if not include_bots and (login.endswith("[bot]") or (author or {}).get("type") == "Bot"):
        return "bot"
    if is_security_text((entry.get("commit") or {}).get("message")):
        return "security_keyword_message"
    return None


def detail_reject_reason(commit: dict, max_files: int, max_changed: int) -> str | None:
    """Why a fetched commit is not usable (``None`` = keep)."""
    files = code_files(commit)
    if not files:
        return "no_code_files"
    if len(files) > max_files:
        return "too_many_files"
    changed = 0
    for f in files:
        if f.get("status") == "renamed" and not f.get("patch"):
            continue
        if not f.get("patch"):
            return "file_without_patch"
        changed += int(f.get("additions") or 0) + int(f.get("deletions") or 0)
    if changed > max_changed:
        return "too_many_changed_lines"
    if changed == 0:
        return "no_changed_lines"
    return None


def prs_reject_reason(prs) -> str | None:
    if not isinstance(prs, list):
        return None
    for pr in prs:
        labels = " ".join(lb.get("name", "") for lb in pr.get("labels") or []
                          if isinstance(lb, dict))
        if is_security_text(" ".join([pr.get("title") or "", pr.get("body") or "", labels])):
            return "security_keyword_linked_pr"
    return None


def repo_order(repo_split: dict[str, str], seed: int) -> list[str]:
    """dev repos first, then test, then reserve; seeded order within a split."""
    out = []
    for split in SPLIT_ORDER:
        repos = sorted(r for r, s in repo_split.items() if s == split)
        random.Random(f"{seed}:benign_commits:{split}").shuffle(repos)
        out += repos
    return out


def estimate_calls(n_repos: int, max_commits: int, per_repo: int, pages: int,
                   check_prs: bool, cached_lists: int = 0) -> dict:
    """Rough API-call estimate for ``--dry-run`` (see the EST_* assumptions)."""
    repos_needed = min(n_repos, math.ceil(max_commits / max(per_repo, 1)))
    # Some repos yield fewer than per_repo commits: budget 25% more repos.
    repos_visited = min(n_repos, math.ceil(repos_needed * 1.25))
    lists = max(repos_visited * pages - cached_lists, 0)
    examined = math.ceil(max_commits * EST_EXAMINED_PER_ACCEPTED)
    pulls = math.ceil(examined * 0.6) if check_prs else 0
    contents = math.ceil(max_commits * EST_FILES_PER_COMMIT * (1 + EST_PARENT_REFETCH))
    total = lists + examined + pulls + contents
    return {"repos_needed": repos_needed, "repos_visited_est": repos_visited,
            "list_calls": lists, "commit_detail_calls": examined, "linked_pr_calls": pulls,
            "content_calls": contents, "total_calls": total,
            "est_minutes_at_0.4s_per_call": round(total * EST_SECONDS_PER_CALL / 60, 1),
            "hours_of_5000_per_hour_budget": round(total / 5000, 2)}


def build_benign_item(repo: str, commit: dict, entries: list[dict], split: str) -> dict:
    """A PR eval item (``kind: benign``, ``source: benign_commit``)."""
    sha = commit["sha"]
    paths = [e["path"] for e in entries]
    title, body = neutral_pr_text(paths)
    info = commit.get("commit") or {}
    return {
        "id": f"pr_benign_{token(repo.replace('/', '_'))}_{sha[:10]}",
        "kind": "benign", "language": item_language(entries, None), "repo": repo,
        "advisory_id": None, "cve_id": None, "category": None, "cwe": None, "split": split,
        "pr_title": title, "pr_body": body,
        "files": public_files(sorted(entries, key=lambda e: e["path"])),
        "target": {"path": None, "vuln_lines_new": [], "changed_lines_new": []},
        "source": "benign_commit",
        "notes": "Real commit on the default branch; its message and linked PRs passed a "
                 "security-keyword filter and it is not a known fix commit (not guaranteed "
                 "vulnerability-free).",
        "meta": {"commit": sha, "parent": (commit.get("parents") or [{}])[0].get("sha"),
                 "committed": ((info.get("committer") or {}).get("date")),
                 "changed_lines": sum(changed_count(e) for e in entries),
                 "truncated": False, "files_in_commit": len(commit.get("files") or [])},
    }


# ---------------------------------------------------------------------------
# Inputs (local files only)
# ---------------------------------------------------------------------------


def load_repo_splits(pr_eval: Path, ordinary: Path | None = None,
                     split_file: Path | None = None) -> tuple[dict[str, str], set[str]]:
    """``(repo -> split, known commit shas)`` from the PR eval items (streamed:
    the file holds full file contents). A repo is dev/test if v1 fixed its side
    (an item's split or ``meta.repo_side``), else reserve. With ``ordinary`` +
    ``split_file``, repos that only have v1 ordinary functions are added too."""
    sides: dict[str, set] = {}  # lower-cased repo -> v1 sides
    names: dict[str, str] = {}  # lower-cased repo -> first spelling seen
    known: set[str] = set()
    with open(pr_eval, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            it = json.loads(line)
            meta = it.get("meta") or {}
            names.setdefault(it["repo"].lower(), it["repo"])
            s = sides.setdefault(it["repo"].lower(), set())
            for side in (it["split"], meta.get("repo_side")):
                if side in ("dev", "test"):
                    s.add(side)
            if meta.get("commit"):
                known.add(meta["commit"])
            del it
    if ordinary and split_file and ordinary.exists() and split_file.exists():
        manifest = json.loads(split_file.read_text(encoding="utf-8"))
        side_of = {i: s for s in ("dev", "test") for i in manifest[s]["ids"]}
        with open(ordinary, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                if o["id"] in side_of and o.get("repo"):
                    names.setdefault(o["repo"].lower(), o["repo"])
                    sides.setdefault(o["repo"].lower(), set()).add(side_of[o["id"]])
    bad = sorted(r for r, s in sides.items() if len(s) > 1)
    if bad:
        raise ValueError(f"repos straddle dev and test: {bad[:5]}")
    return {names[r]: (next(iter(s)) if s else "reserve") for r, s in sides.items()}, known


def cache_path(cache_dir: Path, url: str) -> Path:
    """Where ``GithubClient`` caches ``url`` (same key scheme)."""
    return Path(cache_dir) / "api" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"


def known_fix_commits(cache_dir: Path, repos: set[str]) -> set[str]:
    """Every commit sha the corpus builder mined, plus every GitHub commit any
    cached OSV advisory references, for ``repos`` (lower-cased match)."""
    wanted = {r.lower() for r in repos}
    known: set[str] = set()
    state = cache_dir / "processed_advisories.json"
    if state.exists():
        for entry in json.loads(state.read_text(encoding="utf-8")).values():
            for p in entry.get("pairs") or []:
                known.add(p["commit"])
    for name in ("PyPI_all.zip", "npm_all.zip"):
        path = cache_dir / name
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if not member.endswith(".json"):
                    continue
                raw = zf.read(member)
                if b"/commit" not in raw:
                    continue
                for ref in json.loads(raw).get("references") or []:
                    parsed = parse_fix_commit_url(ref.get("url", ""))
                    if parsed and f"{parsed[0]}/{parsed[1]}".lower() in wanted:
                        known.add(parsed[2])
    return known


# ---------------------------------------------------------------------------
# Network side
# ---------------------------------------------------------------------------


class BudgetExceeded(Exception):
    pass


class BenignClient(GithubClient):
    """The builder's cached client plus secondary-rate-limit handling
    (``Retry-After``), a hard call budget and a minimum call interval. Returns
    lists as well as dicts (commit lists, linked PRs)."""

    def __init__(self, session, token, cache_dir, stats, *, wait_on_rate_limit=False,
                 max_calls=3000, min_interval=0.0, sleep=time.sleep, clock=time.monotonic):
        super().__init__(session, token, cache_dir, stats, wait_on_rate_limit, offline=False)
        self.max_calls = max_calls
        self.min_interval = min_interval
        self.sleep = sleep
        self.clock = clock
        self._last = None

    def get_json(self, url: str):
        cache_file = self._cache_file(url)
        if cache_file.exists():
            self.stats.cache_hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        while True:
            if self.stats.api_calls >= self.max_calls:
                raise BudgetExceeded(f"--max-api-calls={self.max_calls} reached")
            if self._last is not None and self.min_interval > 0:
                wait = self.min_interval - (self.clock() - self._last)
                if wait > 0:
                    self.sleep(wait)
            resp = self.session.get(url, headers=headers, timeout=30)
            self._last = self.clock()
            self.stats.api_calls += 1
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if remaining is not None:
                self.stats.rate_remaining = int(remaining)
            if reset is not None:
                self.stats.rate_reset = int(reset)
            retry_after = resp.headers.get("Retry-After")
            if resp.status_code in (403, 429) and (remaining == "0" or retry_after):
                if not self.wait_on_rate_limit:
                    raise RateLimitExceeded(int(reset) if reset else None)
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = max(int(reset) - time.time(), 0) + 1 if reset else 60
                print(f"  [rate limit] sleeping {delay:.0f}s")
                self.sleep(delay)
                continue
            break
        if resp.status_code == 404:
            data = {"__status__": 404}
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            return data
        if resp.status_code != 200:
            print(f"  [warn] GitHub API {resp.status_code} for {url}")
            return None
        data = resp.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data


def contents_url(repo: str, path: str, ref: str) -> str:
    return f"{API}/repos/{repo}/contents/{quote(path, safe='/')}?ref={ref}"


def fetch_entries(client: BenignClient, repo: str, commit: dict,
                  reasons: Counter) -> list[dict] | None:
    """Forward file entries for the commit's code files (``None`` = unusable)."""
    owner, name = repo.split("/", 1)
    sha = commit["sha"]
    parent = commit["parents"][0]["sha"]
    entries = []
    for f in code_files(commit):
        path, status, patch = f["filename"], f.get("status"), f.get("patch")
        if status == "renamed" and not patch:
            continue  # pure rename: no diff content
        old = new = None
        try:
            if status == "added":
                new = apply_patch("", patch)
            elif status == "removed":
                old = apply_patch("", patch, reverse=True)
            elif status in ("modified", "renamed", "changed"):
                new = client.get_file(owner, name, path, sha)
                if new is None:
                    reasons["content_missing"] += 1
                    return None
                try:
                    old = apply_patch(new, patch, reverse=True)
                except PatchError:
                    old_path = f.get("previous_filename") or path
                    old = client.get_file(owner, name, old_path, parent)
                    if old is None or apply_patch(old, patch) != new:
                        reasons["patch_does_not_apply"] += 1
                        return None
            else:
                reasons[f"status_{status}"] += 1
                return None
        except PatchError:
            reasons["patch_does_not_apply"] += 1
            return None
        if max(len((old or "").encode()), len((new or "").encode())) > MAX_FILE_BYTES:
            reasons["file_over_1MB"] += 1
            return None
        e = file_entry(path, old, new)
        if e is not None:
            entries.append(e)
    return entries or None


def fetch_repo(client: BenignClient, repo: str, split: str, known: set[str], args,
               reasons: Counter) -> list[dict]:
    items = []
    candidates = []
    for page in range(1, args.pages + 1):
        listing = client.get_json(f"{API}/repos/{repo}/commits?per_page=100&page={page}")
        if not isinstance(listing, list):
            reasons["commit_list_unavailable"] += 1
            break
        for entry in listing:
            why = list_reject_reason(entry, known, args.include_bots)
            if why:
                reasons[why] += 1
            else:
                candidates.append(entry)
        if len(listing) < 100:
            break
    for entry in candidates[: args.max_candidates_per_repo]:
        if len(items) >= args.per_repo:
            break
        commit = client.get_json(f"{API}/repos/{repo}/commits/{entry['sha']}")
        if not isinstance(commit, dict) or commit.get("__status__"):
            reasons["commit_unavailable"] += 1
            continue
        why = detail_reject_reason(commit, args.max_files, args.max_changed_lines)
        if why:
            reasons[why] += 1
            continue
        if args.check_linked_prs:
            why = prs_reject_reason(client.get_json(f"{API}/repos/{repo}/commits/"
                                                    f"{entry['sha']}/pulls"))
            if why:
                reasons[why] += 1
                continue
        entries = fetch_entries(client, repo, commit, reasons)
        if not entries:
            reasons["no_usable_files"] += 1
            continue
        items.append(build_benign_item(repo, commit, entries, split))
        reasons["accepted"] += 1
    return items


def run(args, session=None) -> int:
    cache_dir = Path(args.cache_dir)
    repo_split, known = load_repo_splits(
        Path(args.pr_eval), Path(args.ordinary) if args.ordinary else None,
        Path(args.split_file) if args.split_file else None)
    if args.repos:
        repo_split = {r: repo_split.get(r, "reserve") for r in args.repos}
    if args.splits:
        repo_split = {r: s for r, s in repo_split.items() if s in args.splits}
    order = repo_order(repo_split, args.seed)

    if args.dry_run:
        cached = sum(
            cache_path(cache_dir, f"{API}/repos/{repo}/commits?per_page=100&page={page}").exists()
            for repo in order for page in range(1, args.pages + 1))
        est = estimate_calls(len(order), args.max_commits, args.per_repo, args.pages,
                             args.check_linked_prs, cached)
        print("DRY RUN: no network calls are made.")
        print(f"repos available: {len(order)} "
              f"{dict(Counter(repo_split[r] for r in order))}; per repo <= {args.per_repo}; "
              f"target {args.max_commits} commits; {args.pages} list page(s)/repo; "
              f"<= {args.max_candidates_per_repo} commit details/repo; "
              f"linked-PR check {'on' if args.check_linked_prs else 'off'}")
        print(f"first repos: {order[:10]}")
        print("estimate: " + json.dumps(est))
        print(f"hard cap: --max-api-calls {args.max_api_calls}")
        return 0

    from backend.app.config import settings

    tok = settings.GITHUB_TOKEN
    if not tok and not args.allow_unauthenticated:
        sys.exit("GITHUB_TOKEN is not set (60 requests/hour unauthenticated); set it in .env "
                 "or pass --allow-unauthenticated.")
    if session is None:
        import requests

        session = requests.Session()
    stats = Stats()
    client = BenignClient(session, tok, cache_dir, stats,
                          wait_on_rate_limit=args.wait_on_rate_limit,
                          max_calls=args.max_api_calls, min_interval=args.min_interval)
    known = known_prefixes(known | known_fix_commits(cache_dir, set(order)))
    reasons: Counter = Counter()
    items: list[dict] = []
    stopped = None
    for repo in order:
        if len(items) >= args.max_commits:
            break
        try:
            got = fetch_repo(client, repo, repo_split[repo], known, args, reasons)
        except (RateLimitExceeded, BudgetExceeded) as exc:
            stopped = str(exc)
            break
        items += got[: args.max_commits - len(items)]
    items.sort(key=lambda it: it["id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} benign commits to {out}")
    print("by split: " + json.dumps(dict(Counter(it["split"] for it in items))))
    print("by language: " + json.dumps(dict(Counter(it["language"] for it in items))))
    print("reasons: " + json.dumps(dict(sorted(reasons.items()))))
    print(f"API calls {stats.api_calls}, cache hits {stats.cache_hits}, "
          f"rate remaining {stats.rate_remaining}")
    if stopped:
        print(f"stopped early: {stopped}. Responses so far are cached; re-run to continue.")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pr-eval", default=str(DEFAULT_PR_EVAL),
                    help="PR eval items: repos, their split, known commits")
    ap.add_argument("--ordinary", default=str(DEFAULT_ORDINARY),
                    help="v1 ordinary negatives: their repos are candidates too ('' = off)")
    ap.add_argument("--split-file", default=str(DEFAULT_SPLIT))
    ap.add_argument("--repos", nargs="*", help="only these owner/name repos")
    ap.add_argument("--splits", nargs="*", choices=SPLIT_ORDER, help="only repos of these splits")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-commits", type=int, default=300)
    ap.add_argument("--per-repo", type=int, default=3)
    ap.add_argument("--pages", type=int, default=1, help="commit-list pages (100) per repo")
    ap.add_argument("--max-candidates-per-repo", type=int, default=8,
                    help="commit-detail calls per repo at most")
    ap.add_argument("--max-files", type=int, default=MAX_FILES)
    ap.add_argument("--max-changed-lines", type=int, default=MAX_CHANGED_LINES)
    ap.add_argument("--check-linked-prs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-bots", action="store_true")
    ap.add_argument("--max-api-calls", type=int, default=3000)
    ap.add_argument("--min-interval", type=float, default=0.1, help="seconds between calls")
    ap.add_argument("--wait-on-rate-limit", action="store_true")
    ap.add_argument("--allow-unauthenticated", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and an API-call estimate; no network")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
