"""Build a PR-shaped eval set (unified diffs + full before/after files), offline.

The function-level eval sets ask "is this function vulnerable?". The product is
asked something else: a PR arrives as changed files (path, full new content,
unified-diff patch) and the question is "does this change introduce a
vulnerability?". This script rebuilds the held-out OSV eval advisories as PRs of
that shape, from the GitHub response cache of ``scripts/build_corpus_from_osv.py``
(``data/osv_cache/``: fix-commit JSON with ``files[].patch``, file contents at the
fix commit and at ``parents[0]``). No network, no models.

Item format (one JSON object per line)::

    {"id", "kind": "vuln_introducing" | "vuln_fix" | "benign",
     "language": "python" | "javascript", "repo": "owner/name",
     "advisory_id", "cve_id", "category", "cwe", "split": "dev" | "test" | "reserve",
     "pr_title", "pr_body",
     "files": [{"path", "old_content", "new_content", "patch"}],
     "target": {"path", "vuln_lines_new": [int], "changed_lines_new": [int]},
     "source": "reversed_fix" | "real_fix" | "bystander" | "real_commit",
     "notes": str, "meta": {...provenance...}}

Kinds, per held-out fix commit (one commit per advisory; aliases sharing a commit
give one PR):

- ``vuln_introducing`` (``reversed_fix``): the fix commit reversed. ``old_content``
  = post-fix file, ``new_content`` = pre-fix (vulnerable) file. ``target.path`` is
  the file holding the most eval-set vulnerable functions; ``vuln_lines_new`` are
  the new-file (vulnerable) lines inside those functions that the real fix deleted
  or modified. If the fix only inserted lines there, the insertion point +-2
  (``meta.vuln_lines_mode``).
- ``vuln_fix`` (``real_fix``): the fix commit as it happened (old = pre-fix, new =
  post-fix). ``vuln_lines_new`` is empty: the correct verdict is "nothing
  introduced".
- ``benign`` (``bystander``): code files of a fix commit that contain no function
  the corpus builder paired as changed by the fix (``processed_advisories.json``),
  in commits whose paired functions live in other files. Taken from the held-out
  eval commits first and from the other mined fix commits of the same repos (so
  they share the repos' split). **Caveat**: they come from security-adjacent
  commits and their distribution (size, kind of change) is not a realistic benign
  base rate; real benign commits come from ``scripts/fetch_benign_commits.py``.

Every code file the commit changed is included (.py/.js/.ts, skipping
test/doc/example paths like the builder), capped at ``--max-files`` files and
``--max-changed-lines`` added+removed lines per PR (vulnerable files first;
truncation is recorded in ``notes``). File content comes from the cache; a side
that is not cached is reconstructed from GitHub's patch where it applies exactly
(added/removed files, one-sided cache misses); renamed files are omitted (their
contents were never fetched). Patches are regenerated with difflib (standard
``@@`` headers, 3 context lines, ``a/<path>`` / ``b/<path>``, git's
``\\ No newline at end of file`` marker) so ``apply_patch(old, patch) == new``.

Labels: ``cwe`` is the first CWE of the advisory's OSV record and ``category``
the corpus category of its eval pair (``cwe_to_category``), except where
``LABEL_OVERRIDES`` corrects a CWE tag that contradicts the advisory's own text
(``meta.label_override`` records the original).

``pr_title`` / ``pr_body`` are synthetic and neutral ("Update <path>"): the
commit message and advisory text are never used (they would leak the label). A
variant file gives each vuln_introducing item a misleading benign description
(id suffix ``_misleading``) to test whether PR text blinds the reviewer.

``split`` maps each commit to ``ml/evaluation/splits/v1.json`` by its eval pair
ids (advisory + repo grouping, so nothing straddles dev and test); commits whose
pairs are unassigned in v1 are ``reserve``. Bystanders from other commits take
their repo's side.

    python scripts/build_pr_eval.py            # -> ml/evaluation/datasets/pr_eval/
    python scripts/build_pr_eval.py --check    # summary only, write nothing

v2 (``--v2``; seconds, offline, never rebuilds or rewrites v1): reads the BUILT
v1 files (sha256-checked against ``pr_eval_v1_manifest.json``) and the real
benign commits fetched by ``scripts/fetch_benign_commits.py``
(``pr_eval_benign_commits_v1.jsonl``, fetcher label ``benign_commit``) and writes

- ``pr_eval_v2_real_commits.jsonl``: every fetched commit, validated like the
  builder's own items (patch == the difflib diff of its old / new contents,
  caps, neutral PR text, split == its repo's v1 side, not a known fix / eval
  commit), ``source: real_commit`` (raw label in ``meta.fetched_source``);
- ``pr_eval_v2_sample_dev.jsonl``: the v1 dev sample's 60 vuln_introducing +
  60 vuln_fix items (same ids and content; ``LABEL_OVERRIDES`` applied) plus
  ``--v2-benign-dev`` (120) dev real commits, language mix of the sampled vuln
  items, round-robin over repos, at most ``--v2-benign-per-repo`` (2) per repo.
  No bystanders (they stay in v1 for reference);
- ``pr_eval_v2_test.jsonl``: every test vuln_introducing / vuln_fix item of v1
  plus every test real commit;
- ``pr_eval_v2_manifest.json`` (committed): input and output sha256s, counts,
  ids, the dev/test leak check (``split_leak_check`` against
  ``splits/v1.json``; the build fails on any leak) and real-commit vs
  bystander size stats.

    python scripts/build_pr_eval.py --v2
    python scripts/build_pr_eval.py --v2 --check
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_corpus_from_osv import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    _is_skipped_path,
    cwe_to_category,
    pair_code_hash,
)

DATASETS = ROOT / "ml" / "evaluation" / "datasets"
DEFAULT_OUT_DIR = DATASETS / "pr_eval"
DEFAULT_PAIR_FILES = [DATASETS / "detection_eval_osv_pypi.jsonl",
                      DATASETS / "detection_eval_osv_npm.jsonl"]
DEFAULT_ORDINARY = DATASETS / "detection_eval_ordinary.jsonl"
DEFAULT_SPLIT = ROOT / "ml" / "evaluation" / "splits" / "v1.json"
DEFAULT_CACHE_DIR = ROOT / "data" / "osv_cache"
VERSION = "v1"

MAX_FILES = 6
MAX_CHANGED_LINES = 2000
MAX_FILE_BYTES = 1_000_000  # the Action skips files over this (contents API limit)
CONTEXT_LINES = 3

MISLEADING_TITLE = "Refactor: simplify input handling, no behaviour change"
MISLEADING_BODY = (
    "Small cleanup: simplifies the input handling code paths and removes redundant "
    "checks. No functional or behaviour change intended; covered by the existing tests."
)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
NO_NEWLINE = "\\ No newline at end of file"

# Label corrections. An item's ``cwe`` is the first CWE of its OSV record
# (``database_specific.cwe_ids``, via the corpus builder's state) and its
# ``category`` is ``cwe_to_category`` of that list. These advisories' CWE tags
# contradict their own summary / details (checked offline against the OSV
# export in data/osv_cache/{npm,PyPI}_all.zip), so the label comes from here
# instead. Keyed by OSV id; an item matches if any of its ``meta.osv_ids`` does.
# The original label is kept in ``meta.label_override.from``, and ``sample_dev``
# stratifies on it so a correction never reshuffles the pre-registered sample.
LABEL_OVERRIDES: dict[str, dict] = {
    # CVE-2020-7771, "Prototype Pollution in asciitable.js": the advisory's PoC
    # sets ``__proto__`` through the main function; OSV tags it CWE-400.
    "GHSA-5pxj-mhwj-x5gv": {
        "cwe": "CWE-1321",
        "reason": "CVE-2020-7771 is prototype pollution (advisory title and __proto__ PoC); "
                  "OSV tags CWE-400",
    },
    # CVE-2013-0270, "OpenStack Keystone Denial of Service vulnerability via a large
    # HTTP request" ("CPU and memory consumption"); OSV tags CWE-119 / CWE-1284,
    # memory-safety numbers that do not describe a Python request-size DoS.
    "GHSA-4ppj-4p4v-jf4p": {
        "cwe": "CWE-400",
        "reason": "CVE-2013-0270 is a denial of service via a large HTTP request (CPU and "
                  "memory consumption); OSV tags CWE-119 / CWE-1284",
    },
}


def label_override(osv_ids) -> tuple[str, dict] | None:
    """``(osv id, override)`` of the first of ``osv_ids`` (sorted) listed in
    ``LABEL_OVERRIDES``, else None. The override's category is
    ``cwe_to_category`` of its CWE, like every other label."""
    for osv_id in sorted(osv_ids or ()):
        if osv_id in LABEL_OVERRIDES:
            o = LABEL_OVERRIDES[osv_id]
            return osv_id, {**o, "category": cwe_to_category([o["cwe"]])}
    return None


def sample_category(item: dict) -> str | None:
    """The category ``sample_dev`` stratifies on: the as-built (OSV-derived)
    one, also for an item whose label ``LABEL_OVERRIDES`` corrected."""
    o = (item.get("meta") or {}).get("label_override")
    return o["from"]["category"] if o else item.get("category")


# ---------------------------------------------------------------------------
# Unified diffs: generate, parse, apply (pure).
# ---------------------------------------------------------------------------


class PatchError(ValueError):
    """A patch that does not apply exactly."""


def split_lines(text: str | None) -> list[str]:
    """Lines with their ``\\n`` kept, split on ``\\n`` only (``str.splitlines``
    would also split on ``\\r``, form feeds, U+2028 ... and break round trips)."""
    if not text:
        return []
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


Opcode = tuple[str, int, int, int, int]


def diff_opcodes(a: list[str], b: list[str]) -> list[Opcode]:
    """``SequenceMatcher`` opcodes (no autojunk), with the common prefix and
    suffix trimmed first (as git does) so large files with small changes are
    fast."""
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    s = 0
    while s < len(a) - p and s < len(b) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    mid = difflib.SequenceMatcher(None, a[p:len(a) - s], b[p:len(b) - s],
                                  autojunk=False).get_opcodes()
    raw = [("equal", 0, p, 0, p)]
    raw += [(t, i1 + p, i2 + p, j1 + p, j2 + p) for t, i1, i2, j1, j2 in mid]
    raw.append(("equal", len(a) - s, len(a), len(b) - s, len(b)))
    ops: list[Opcode] = []
    for op in raw:
        if op[1] == op[2] and op[3] == op[4]:
            continue  # empty
        if ops and op[0] == "equal" and ops[-1][0] == "equal":
            ops[-1] = ("equal", ops[-1][1], op[2], ops[-1][3], op[4])
        else:
            ops.append(op)
    return ops


def swap_opcodes(ops: list[Opcode]) -> list[Opcode]:
    """The opcodes of the reverse diff (b -> a)."""
    flip = {"insert": "delete", "delete": "insert"}
    return [(flip.get(t, t), j1, j2, i1, i2) for t, i1, i2, j1, j2 in ops]


def group_opcodes(ops: list[Opcode], n: int = CONTEXT_LINES) -> list[list[Opcode]]:
    """Hunks with ``n`` lines of context (``SequenceMatcher.get_grouped_opcodes``)."""
    codes = list(ops) or [("equal", 0, 1, 0, 1)]
    if codes[0][0] == "equal":
        t, i1, i2, j1, j2 = codes[0]
        codes[0] = t, max(i1, i2 - n), i2, max(j1, j2 - n), j2
    if codes[-1][0] == "equal":
        t, i1, i2, j1, j2 = codes[-1]
        codes[-1] = t, i1, min(i2, i1 + n), j1, min(j2, j1 + n)
    groups, group = [], []
    for t, i1, i2, j1, j2 in codes:
        if t == "equal" and i2 - i1 > 2 * n:
            group.append((t, i1, min(i2, i1 + n), j1, min(j2, j1 + n)))
            groups.append(group)
            group = []
            i1, j1 = max(i1, i2 - n), max(j1, j2 - n)
        group.append((t, i1, i2, j1, j2))
    if group and not (len(group) == 1 and group[0][0] == "equal"):
        groups.append(group)
    return groups


def _range(start: int, length: int) -> str:
    """difflib's / git's hunk range: ``-0,0`` for an empty side at the top."""
    beginning = start + 1
    if length == 1:
        return f"{beginning}"
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


def unified_diff(old: str | None, new: str | None, path: str,
                 context: int = CONTEXT_LINES, opcodes: list[Opcode] | None = None) -> str:
    """Unified diff ``old -> new`` with ``a/<path>`` / ``b/<path>`` headers
    (``/dev/null`` for an added / removed file). ``""`` when nothing changed.
    ``opcodes`` (from ``diff_opcodes(old lines, new lines)``) skips the diff."""
    a, b = split_lines(old), split_lines(new)
    if a == b:
        return ""
    out = [f"--- {'/dev/null' if old is None else 'a/' + path}\n",
           f"+++ {'/dev/null' if new is None else 'b/' + path}\n"]

    def emit(prefix: str, line: str) -> None:
        if line.endswith("\n"):
            out.append(prefix + line)
        else:
            out.append(prefix + line + "\n" + NO_NEWLINE + "\n")

    for group in group_opcodes(diff_opcodes(a, b) if opcodes is None else opcodes, context):
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        out.append(f"@@ -{_range(i1, i2 - i1)} +{_range(j1, j2 - j1)} @@\n")
        for tag, a1, a2, b1, b2 in group:
            if tag == "equal":
                for line in a[a1:a2]:
                    emit(" ", line)
                continue
            if tag in ("replace", "delete"):
                for line in a[a1:a2]:
                    emit("-", line)
            if tag in ("replace", "insert"):
                for line in b[b1:b2]:
                    emit("+", line)
    return "".join(out)


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[tuple[str, str]] = field(default_factory=list)  # (tag, text incl. "\n")


def parse_patch(patch: str) -> list[Hunk]:
    """Hunks of a unified diff (headers before the first ``@@`` are skipped; so
    GitHub's header-less ``files[].patch`` parses too). A hunk ends when its
    ``@@`` counts are used up. Raises ``PatchError`` on a malformed hunk."""
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    need_old = need_new = 0
    for raw in (patch or "").split("\n"):
        m = _HUNK_RE.match(raw)
        if m:
            if cur is not None and (need_old or need_new):
                raise PatchError(f"truncated hunk before {raw!r}")
            cur = Hunk(int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1),
                       int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1))
            need_old, need_new = cur.old_len, cur.new_len
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if raw.startswith("\\"):
            if cur.lines and cur.lines[-1][1].endswith("\n"):
                tag, text = cur.lines[-1]
                cur.lines[-1] = (tag, text[:-1])
            continue
        if not (need_old or need_new):
            continue  # trailing text after a complete hunk (e.g. the final "")
        tag = raw[:1] or " "  # a blank context line may have lost its space
        body = raw[1:]
        if tag == " ":
            need_old -= 1
            need_new -= 1
        elif tag == "-":
            need_old -= 1
        elif tag == "+":
            need_new -= 1
        else:
            raise PatchError(f"bad hunk line {raw!r}")
        if need_old < 0 or need_new < 0:
            raise PatchError("hunk longer than its header")
        cur.lines.append((tag, body + "\n"))
    if cur is not None and (need_old or need_new):
        raise PatchError("truncated final hunk")
    return hunks


def apply_patch(old: str | None, patch: str, reverse: bool = False) -> str:
    """Apply a unified diff to ``old`` exactly (every context / removed line must
    match, no fuzz). ``reverse=True`` applies it backwards (new -> old)."""
    src = split_lines(old)
    out: list[str] = []
    pos = 0
    for h in parse_patch(patch):
        if reverse:
            start, length = h.new_start, h.new_len
            lines = [({"+": "-", "-": "+"}.get(t, t), x) for t, x in h.lines]
        else:
            start, length = h.old_start, h.old_len
            lines = h.lines
        idx = start - 1 if length else start  # "-5,0" inserts after line 5
        if idx < pos or idx > len(src):
            raise PatchError(f"hunk at line {start} out of order or past EOF")
        out.extend(src[pos:idx])
        k = idx
        for tag, text in lines:
            if tag in (" ", "-"):
                if k >= len(src) or src[k] != text:
                    raise PatchError(f"mismatch at line {k + 1}")
                if tag == " ":
                    out.append(text)
                k += 1
            else:
                out.append(text)
        pos = k
    out.extend(src[pos:])
    return "".join(out)


def patch_line_numbers(patch: str) -> tuple[list[int], int]:
    """``(new-file line numbers of '+' lines, number of '-' lines)``."""
    added: list[int] = []
    removed = 0
    for h in parse_patch(patch):
        n = h.new_start
        for tag, _ in h.lines:
            if tag == "+":
                added.append(n)
                n += 1
            elif tag == " ":
                n += 1
            else:
                removed += 1
    return added, removed


# ---------------------------------------------------------------------------
# Vulnerable lines, file selection, item assembly (pure).
# ---------------------------------------------------------------------------


def locate_code(content: str, code: str) -> list[tuple[int, int]]:
    """1-based inclusive line ranges where ``code`` occurs verbatim in ``content``."""
    ranges = []
    start = content.find(code) if code else -1
    while start != -1:
        first = content.count("\n", 0, start) + 1
        ranges.append((first, first + code.count("\n")))
        start = content.find(code, start + 1)
    return ranges


def vuln_lines(old: str | None, new: str | None, ranges: list[tuple[int, int]] | None,
               opcodes: list[Opcode] | None = None) -> tuple[list[int], str]:
    """Lines of ``new`` (the vulnerable side of a reversed fix) that the real fix
    deleted or modified, restricted to ``ranges`` (the vulnerable functions in
    ``new``; ``None`` = whole file). If there are none there (the fix only
    inserted lines), the insertion point(s) +-2. Returns ``(lines, mode)``."""
    a, b = split_lines(old), split_lines(new)
    n = len(b)

    def inside(line: int) -> bool:
        return ranges is None or any(s <= line <= e for s, e in ranges)

    changed: set[int] = set()
    gaps: list[int] = []  # b-line count before a gap where the fix inserted lines
    for tag, _i1, _i2, j1, j2 in diff_opcodes(a, b) if opcodes is None else opcodes:
        if tag in ("replace", "insert"):
            changed.update(j for j in range(j1 + 1, j2 + 1) if inside(j))
        elif tag == "delete":
            gaps.append(j1)
    if changed:
        return sorted(changed), "deleted_or_modified"
    near: set[int] = set()
    for g in gaps:
        if ranges is None or any(s - 1 <= g <= e for s, e in ranges):
            p = min(max(g, 1), max(n, 1))  # the line just before the gap
            near.update(x for x in range(p - 2, p + 3) if 1 <= x <= n)
    if near:
        return sorted(near), "insertion_point"
    return [], "none"


def file_entry(path: str, old: str | None, new: str | None,
               opcodes: list[Opcode] | None = None) -> dict | None:
    """``{"path", "old_content", "new_content", "patch"}`` plus private
    ``_added`` / ``_removed`` line counts and ``_ops``; ``None`` if nothing
    changed."""
    if opcodes is None:
        opcodes = diff_opcodes(split_lines(old), split_lines(new))
    patch = unified_diff(old, new, path, opcodes=opcodes)
    if not patch:
        return None
    added, removed = patch_line_numbers(patch)
    return {"path": path, "old_content": old, "new_content": new, "patch": patch,
            "_added": len(added), "_removed": removed, "_ops": opcodes}


def reverse_entry(entry: dict) -> dict:
    """The same file change backwards (new -> old), reusing the diff."""
    return file_entry(entry["path"], entry["new_content"], entry["old_content"],
                      swap_opcodes(entry["_ops"]))


def changed_count(entry: dict) -> int:
    return entry["_added"] + entry["_removed"]


def select_files(entries: list[dict], priority: list[str], max_files: int = MAX_FILES,
                 max_changed: int = MAX_CHANGED_LINES) -> tuple[list[dict], list[str]]:
    """Cap a PR: ``priority`` paths first (in order), then the rest by path; a
    file is dropped if it would exceed ``max_files`` or ``max_changed`` total
    added+removed lines. Returns ``(kept sorted by path, dropped paths)``. The
    caller decides what to do if a priority file is dropped."""
    by_path = {e["path"]: e for e in entries}
    order = [p for p in priority if p in by_path] + sorted(
        p for p in by_path if p not in set(priority))
    kept, dropped, total = [], [], 0
    for p in order:
        e = by_path[p]
        if len(kept) >= max_files or total + changed_count(e) > max_changed:
            dropped.append(p)
            continue
        kept.append(e)
        total += changed_count(e)
    return sorted(kept, key=lambda e: e["path"]), dropped


def public_files(entries: list[dict]) -> list[dict]:
    return [{k: e[k] for k in ("path", "old_content", "new_content", "patch")}
            for e in entries]


def neutral_pr_text(paths: list[str]) -> tuple[str, str]:
    """Synthetic, label-free PR title/body built from the file paths only."""
    paths = sorted(paths)
    if len(paths) == 1:
        title = f"Update {paths[0]}"
    else:
        title = f"Update {paths[0]} and {len(paths) - 1} other file" + (
            "s" if len(paths) > 2 else "")
    body = "Changes:\n" + "\n".join(f"- {p}" for p in paths)
    return title, body


def token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def pr_base_id(advisory_id: str, repo: str, commit: str) -> str:
    return f"pr_{token(advisory_id)}_{token(repo.split('/')[-1])}_{commit[:10]}"


def make_misleading(item: dict) -> dict:
    return {**item, "id": item["id"] + "_misleading", "pr_title": MISLEADING_TITLE,
            "pr_body": MISLEADING_BODY,
            "notes": (item["notes"] + " " if item["notes"] else "")
            + "Misleading benign PR description (variant)."}


def item_language(entries: list[dict], target: str | None) -> str:
    if target:
        return SUPPORTED_EXTENSIONS[Path(target).suffix.lower()]
    counts = Counter(SUPPORTED_EXTENSIONS[Path(e["path"]).suffix.lower()] for e in entries)
    return max(sorted(counts), key=lambda k: counts[k])


def split_for_ids(ids: list[str], side_of: dict[str, str]) -> str:
    """dev / test from the v1 side of the eval ids, else reserve. Raises if the
    ids straddle dev and test."""
    sides = {side_of[i] for i in ids if i in side_of}
    if len(sides) > 1:
        raise ValueError(f"eval ids straddle dev and test: {sorted(ids)[:4]}")
    return sides.pop() if sides else "reserve"


def straddling(items: list[dict]) -> dict[str, list[str]]:
    """Advisories / repos present in both dev and test (must be empty)."""
    seen: dict[str, dict[str, set]] = {"advisory": defaultdict(set), "repo": defaultdict(set)}
    for it in items:
        if it["split"] not in ("dev", "test"):
            continue
        seen["repo"][it["repo"].lower()].add(it["split"])
        advs = {it.get("advisory_id"), it.get("cve_id"), *(it.get("meta") or {}).get(
            "osv_ids", [])}
        for a in advs - {None}:
            seen["advisory"][a].add(it["split"])
    return {k: sorted(x for x, s in v.items() if len(s) > 1) for k, v in seen.items()}


LEAK_ID_RE = re.compile(r"\b(?:CVE-\d{4}-\d+|GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}|"
                        r"PYSEC-\d{4}-\d+)\b", re.I)


# ---------------------------------------------------------------------------
# Sampling (pure).
# ---------------------------------------------------------------------------


def _quotas(target: int, counts: dict) -> dict:
    """``min(target, total)`` split proportionally to ``counts`` (largest
    remainder, ties by key); never more than a stratum holds."""
    total = sum(counts.values())
    target = min(target, total)
    if not target:
        return {k: 0 for k in counts}
    raw = {k: target * n / total for k, n in counts.items()}
    q = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), str(k))):
        if sum(q.values()) >= target:
            break
        q[k] += 1
    return q


def sample_dev(items: list[dict], seed: int, n_vuln: int = 60, n_benign: int = 80,
               benign_per_repo: int = 3) -> list[dict]:
    """Stratified dev sample: ``n_vuln`` vuln_introducing items by (language,
    category), each with its vuln_fix twin, plus ``n_benign`` benign items with
    the sampled vuln language mix, at most ``benign_per_repo`` per repo, eval
    commits' bystanders before other commits'. Deterministic."""
    dev = [it for it in items if it["split"] == "dev"]
    fix_of = {it["meta"]["pair_base"]: it for it in dev if it["kind"] == "vuln_fix"}
    intro = sorted((it for it in dev if it["kind"] == "vuln_introducing"
                    and it["meta"]["pair_base"] in fix_of), key=lambda it: it["id"])
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for it in intro:
        strata[(it["language"], sample_category(it) or "none")].append(it)
    quotas = _quotas(n_vuln, {k: len(v) for k, v in strata.items()})
    picked: list[dict] = []
    for k in sorted(strata):
        pool = list(strata[k])
        random.Random(f"{seed}:intro:{'|'.join(k)}").shuffle(pool)
        picked.extend(pool[:quotas[k]])
    out = []
    for it in picked:
        out += [it, fix_of[it["meta"]["pair_base"]]]

    benign = sorted((it for it in dev if it["kind"] == "benign"), key=lambda it: it["id"])
    avail = Counter(it["language"] for it in benign)
    mix = Counter(it["language"] for it in picked)
    want = _quotas(n_benign, {k: mix.get(k, 0) or 0 for k in avail}) if mix else _quotas(
        n_benign, dict(avail))
    # Per language, eval commits' bystanders first, then the rest, each seeded.
    ordered: dict[str, list[dict]] = {}
    for lang in sorted(avail):
        pool = [it for it in benign if it["language"] == lang]
        rng = random.Random(f"{seed}:benign:{lang}")
        first = [it for it in pool if it["meta"].get("bystander_of") == "eval_fix_commit"]
        rest = [it for it in pool if it["meta"].get("bystander_of") != "eval_fix_commit"]
        rng.shuffle(first)
        rng.shuffle(rest)
        ordered[lang] = first + rest
    per_repo: Counter = Counter()
    chosen: list[dict] = []
    taken_ids: set[str] = set()

    def take(pool: list[dict], limit: int) -> None:
        n = 0
        for it in pool:
            if n >= limit or len(chosen) >= n_benign:
                break
            if it["id"] in taken_ids or per_repo[it["repo"]] >= benign_per_repo:
                continue
            per_repo[it["repo"]] += 1
            taken_ids.add(it["id"])
            chosen.append(it)
            n += 1

    for lang in sorted(ordered):
        take(ordered[lang], want.get(lang, 0))
    for lang in sorted(ordered):  # a language short of its quota: fill from the others
        take(ordered[lang], n_benign)
    return out + sorted(chosen, key=lambda it: it["id"])


# ---------------------------------------------------------------------------
# v2: real benign commits and the run samples (pure).
# ---------------------------------------------------------------------------

V2 = "v2"
REAL_COMMIT_SOURCE = "real_commit"
# ``scripts/fetch_benign_commits.py`` labels its items ``benign_commit``; v2
# relabels them ``real_commit`` (the raw label is kept in ``meta.fetched_source``).
FETCHED_COMMIT_SOURCES = ("benign_commit", REAL_COMMIT_SOURCE)
DEFAULT_REAL_COMMITS = DEFAULT_OUT_DIR / "pr_eval_benign_commits_v1.jsonl"
V2_BENIGN_DEV = 120
V2_BENIGN_PER_REPO = 2
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def apply_label_override(item: dict) -> bool:
    """Make sure a vuln item carries its ``LABEL_OVERRIDES`` correction
    (idempotent). Returns True when it had to be applied here; raises if an
    item claims an override that disagrees with the table."""
    meta = item.setdefault("meta", {})
    found = label_override(meta.get("osv_ids"))
    if found is None:
        if meta.get("label_override"):
            raise ValueError(f"{item['id']}: label_override not in LABEL_OVERRIDES")
        return False
    osv_id, o = found
    if meta.get("label_override"):
        if (item.get("cwe"), item.get("category")) != (o["cwe"], o["category"]):
            raise ValueError(f"{item['id']}: label {item.get('cwe')}/{item.get('category')} "
                             f"disagrees with LABEL_OVERRIDES[{osv_id}]")
        return False
    meta["label_override"] = {"advisory": osv_id, "reason": o["reason"],
                              "from": {"cwe": item.get("cwe"), "category": item.get("category")}}
    item["cwe"], item["category"] = o["cwe"], o["category"]
    return True


def real_commit_item(item: dict, repo_side: dict[str, str], *, max_files: int = MAX_FILES,
                     max_changed: int = MAX_CHANGED_LINES) -> dict:
    """A fetched benign commit as a v2 item, validated like the builder's own
    items: kind benign, a 40-hex ``meta.commit``, its split equal to its repo's
    v1 side (``repo_side``: lower-cased repo -> dev / test), 1..``max_files``
    code files whose patch is exactly ``unified_diff(old, new)`` (so
    ``apply_patch(old, patch) == new``), 1..``max_changed`` changed lines, the
    neutral path-only PR text and no target. Returns a copy with ``source:
    real_commit`` (the raw label in ``meta.fetched_source``) and
    ``meta.repo_side``. Raises ValueError."""
    iid = item.get("id", "?")

    def bad(why: str) -> ValueError:
        return ValueError(f"real commit {iid}: {why}")

    if item.get("kind") != "benign" or item.get("source") not in FETCHED_COMMIT_SOURCES:
        raise bad(f"kind/source {item.get('kind')}/{item.get('source')}")
    meta = dict(item.get("meta") or {})
    if not _SHA_RE.match(str(meta.get("commit") or "")):
        raise bad("meta.commit is not a 40-hex sha")
    side = repo_side.get(str(item.get("repo") or "").lower())
    if item.get("split") not in ("dev", "test") or item["split"] != side:
        raise bad(f"split {item.get('split')!r} but its repo's v1 side is {side!r}")
    if any(item.get(k) is not None for k in ("advisory_id", "cve_id", "category", "cwe")):
        raise bad("carries a vulnerability label")
    t = item.get("target") or {}
    if t.get("path") is not None or t.get("vuln_lines_new") or t.get("changed_lines_new"):
        raise bad("has a target")
    files = item.get("files") or []
    if not 1 <= len(files) <= max_files:
        raise bad(f"{len(files)} files (1..{max_files})")
    changed = 0
    for f in files:
        path, old, new, patch = f["path"], f.get("old_content"), f.get("new_content"), f["patch"]
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS or _is_skipped_path(path):
            raise bad(f"{path} is not a reviewed code path")
        if old is None and new is None:
            raise bad(f"{path} has no content")
        if not patch or patch != unified_diff(old, new, path):
            raise bad(f"{path}: patch is not the difflib diff of its contents")
        try:
            if apply_patch(old, patch) != (new or ""):
                raise bad(f"{path}: patch does not reproduce new_content")
        except PatchError as exc:
            raise bad(f"{path}: {exc}") from None
        added, removed = patch_line_numbers(patch)
        changed += len(added) + removed
    if not 0 < changed <= max_changed:
        raise bad(f"{changed} changed lines (1..{max_changed})")
    if meta.get("changed_lines") is not None and meta["changed_lines"] != changed:
        raise bad(f"meta.changed_lines {meta['changed_lines']} != {changed}")
    if (item.get("pr_title"), item.get("pr_body")) != neutral_pr_text([f["path"] for f in files]):
        raise bad("PR text is not the neutral path-only text")
    meta.update(changed_lines=changed, truncated=bool(meta.get("truncated")),
                fetched_source=item["source"], repo_side=side)
    return {**item, "source": REAL_COMMIT_SOURCE, "meta": meta}


def sample_real_commits(pool: list[dict], n: int, seed: int,
                        lang_mix: dict[str, int] | None = None,
                        per_repo: int = V2_BENIGN_PER_REPO) -> tuple[list[dict], int]:
    """``n`` items of ``pool`` spread over repos and languages, deterministic.

    Language quotas follow ``lang_mix`` (e.g. the sampled vuln items' languages;
    ``None`` = the pool's own mix), capped by what the pool holds. Within a
    language, repos are walked in seeded order, round-robin: every repo gives
    one item before any gives a second, and no repo gives more than
    ``per_repo`` (summed over languages). A language short of its quota is
    filled from the others (largest pool first); if the pool can't reach ``n``
    at ``per_repo``, the cap is raised one step at a time. Returns ``(items sorted by id, the
    per-repo cap used)``."""
    pool = sorted(pool, key=lambda it: it["id"])
    n = min(n, len(pool))
    avail = Counter(it["language"] for it in pool)
    mix = {k: (lang_mix or {}).get(k, 0) for k in avail} if lang_mix else dict(avail)
    if not any(mix.values()):
        mix = dict(avail)
    want = _scaled_quotas(n, mix)
    by_lang_repo: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for it in pool:
        by_lang_repo[it["language"]][it["repo"]].append(it)
    order: dict[str, list[str]] = {}
    for lang in sorted(by_lang_repo):
        repos = sorted(by_lang_repo[lang])
        random.Random(f"{seed}:real_commits:{lang}").shuffle(repos)
        order[lang] = repos
        for repo in repos:
            random.Random(f"{seed}:real_commits:{lang}:{repo}").shuffle(
                by_lang_repo[lang][repo])

    cap = max(per_repo, 1)
    while True:
        per: Counter = Counter()
        chosen: list[dict] = []
        for lang in sorted(order):
            _round_robin(order[lang], by_lang_repo[lang], want.get(lang, 0), n, cap, per, chosen)
        # A language short of its quota: fill from the others, largest pool first.
        for lang in sorted(order, key=lambda k: (-avail[k], k)):
            _round_robin(order[lang], by_lang_repo[lang], n - len(chosen), n, cap, per, chosen)
        if len(chosen) >= n or cap >= len(pool):
            return sorted(chosen, key=lambda it: it["id"]), cap
        cap += 1


def _scaled_quotas(n: int, weights: dict[str, int]) -> dict[str, int]:
    """``n`` split proportionally to ``weights`` (largest remainder, ties by
    key); unlike ``_quotas``, ``n`` may exceed the weights' sum."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()} if total else {}
    q = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), str(k))):
        if sum(q.values()) >= n:
            break
        q[k] += 1
    return q


def _round_robin(repos: list[str], items_of: dict[str, list[dict]], limit: int, n: int,
                 cap: int, per: Counter, chosen: list[dict]) -> None:
    """Take up to ``limit`` items (``chosen`` at most ``n``) from ``repos`` in
    order, in rounds: round r only takes from a repo that has given r items so
    far (all languages), so every repo gives one before any gives a second."""
    taken = {it["id"] for it in chosen}
    got = 0
    for rnd in range(cap):
        for repo in repos:
            if got >= limit or len(chosen) >= n:
                return
            if per[repo] > rnd or per[repo] >= cap:
                continue
            left = [it for it in items_of[repo] if it["id"] not in taken]
            if not left:
                continue
            taken.add(left[0]["id"])
            per[repo] += 1
            chosen.append(left[0])
            got += 1


def _eval_ids(item: dict) -> list[str]:
    """The v1 split-file ids of a vuln item's eval pairs."""
    return [f"{b}_{k}" for b in (item.get("meta") or {}).get("eval_pair_ids") or []
            for k in ("vuln", "safe")]


def _advisories(item: dict) -> set[str]:
    meta = item.get("meta") or {}
    return ({item.get("advisory_id"), item.get("cve_id"), *meta.get("osv_ids", []),
             *meta.get("advisory_ids", [])} - {None})


def split_leak_check(dev: list[dict], test: list[dict], side_of: dict[str, str],
                     repo_side: dict[str, str]) -> dict:
    """Assert that nothing of the test split reaches ``dev`` (and vice versa),
    against the split file (``side_of``: eval id -> dev / test) and the repos'
    v1 sides (``repo_side``: lower-cased repo -> dev / test). Every item must
    be of its file's split; a vuln item's eval pair ids must all be on that
    side of the split file; every repo must be on that side; and no repo,
    advisory, commit or id may appear in both. Raises ValueError; returns the
    summary for the manifest."""
    problems: list[str] = []
    for name, items in (("dev", dev), ("test", test)):
        for it in items:
            if it.get("split") != name:
                problems.append(f"{it['id']}: split {it.get('split')!r} in the {name} file")
            rs = repo_side.get(str(it.get("repo") or "").lower())
            if rs != name:
                problems.append(f"{it['id']}: repo {it.get('repo')} is on the {rs!r} side")
            if it["kind"] in ("vuln_introducing", "vuln_fix"):
                ids = _eval_ids(it)
                if not ids:
                    problems.append(f"{it['id']}: no eval pair ids")
                sides = {side_of.get(i) for i in ids}
                if sides != {name}:
                    problems.append(f"{it['id']}: eval ids on split-file side(s) "
                                    f"{sorted(map(str, sides))}")

    def shared_keys(fn) -> set:
        a, b = ({k for it in items for k in fn(it)} - {None} for items in (dev, test))
        return a & b

    shared = {
        "repos": shared_keys(lambda it: [it["repo"].lower()]),
        "advisories": shared_keys(_advisories),
        "commits": shared_keys(lambda it: [(it.get("meta") or {}).get("commit")]),
        "ids": shared_keys(lambda it: [it["id"]]),
    }
    for k, v in shared.items():
        if v:
            problems.append(f"{len(v)} {k} in both dev and test, e.g. {sorted(map(str, v))[:3]}")
    if any(straddling(dev + test).values()):
        problems.append(f"straddling: {straddling(dev + test)}")
    if problems:
        raise ValueError("dev/test leak: " + "; ".join(problems[:10]))
    return {"ok": True, "dev_items": len(dev), "test_items": len(test),
            "dev_repos": len({it["repo"].lower() for it in dev}),
            "test_repos": len({it["repo"].lower() for it in test}),
            **{f"shared_{k}": 0 for k in shared},
            "checked": "item split, repo v1 side, vuln eval ids vs the split file, shared "
                       "repos / advisories / commits / ids, straddling()"}


def size_row(item: dict) -> dict:
    """The size facts of one PR (``pr_size_stats`` aggregates them; small, so a
    streamed 130 MB file can be summarised without keeping its contents)."""
    files = item["files"]
    return {"language": item["language"], "repo": item["repo"].lower(), "files": len(files),
            "changed_lines": item["meta"]["changed_lines"],
            "new_lines": sum(len(split_lines(f.get("new_content"))) for f in files),
            "added": sum(f.get("old_content") is None for f in files),
            "removed": sum(f.get("new_content") is None for f in files)}


def pr_size_stats(rows: list[dict]) -> dict:
    """Size of a set of PRs (``size_row`` each): files, changed lines, new-file
    lines (what the reviewer reads), added / removed files, languages."""
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "by_language": _by(rows, lambda r: r["language"]),
        "files_per_pr": _dist([r["files"] for r in rows]),
        "changed_lines_per_pr": _dist([r["changed_lines"] for r in rows]),
        "new_file_lines_per_pr": _dist([r["new_lines"] for r in rows]),
        "files_added": sum(r["added"] for r in rows),
        "files_removed": sum(r["removed"] for r in rows),
        "repos": len({r["repo"] for r in rows}),
    }


# ---------------------------------------------------------------------------
# Cache side (reads data/osv_cache only; never the network).
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cache_client(cache_dir: Path):
    """The builder's ``GithubClient`` in offline mode: a cache miss is ``None``,
    never a request."""
    import requests

    from scripts.build_corpus_from_osv import GithubClient, Stats

    return GithubClient(requests.Session(), None, cache_dir, Stats(), offline=True)


def commit_url(repo: str, sha: str) -> str:
    return f"https://api.github.com/repos/{repo}/commits/{sha}"


def reconstruct(status: str, patch: str | None, pre: str | None,
                post: str | None) -> tuple[str | None, str | None, str]:
    """Both sides of a changed file from whatever is cached plus GitHub's patch.
    Returns ``(old, new, how)``; ``how`` starts with ``skip:`` when impossible."""
    try:
        if status == "added":
            if post is None and patch:
                post = apply_patch("", patch)
            return (None, post, "added") if post is not None else (None, None,
                                                                    "skip:added_not_cached")
        if status == "removed":
            if pre is None and patch:
                pre = apply_patch("", patch, reverse=True)
            return (pre, None, "removed") if pre is not None else (None, None,
                                                                   "skip:removed_not_cached")
        if status != "modified":
            return None, None, f"skip:{status}"
        if pre is not None and post is not None:
            if patch:
                try:
                    ok = apply_patch(pre, patch) == post
                except PatchError:
                    ok = False
                return pre, post, "cached" if ok else "cached_patch_mismatch"
            return pre, post, "cached"
        if not patch:
            return None, None, "skip:modified_not_cached"
        if pre is not None:
            return pre, apply_patch(pre, patch), "post_from_patch"
        if post is not None:
            return apply_patch(post, patch, reverse=True), post, "pre_from_patch"
        return None, None, "skip:modified_not_cached"
    except PatchError:
        return None, None, "skip:patch_does_not_apply"


def commit_files(client, repo: str, sha: str, commit_data: dict, stats: Counter,
                 exclude: set[str] | frozenset = frozenset()) -> tuple[list[dict], list[str]]:
    """Forward (pre -> post) file entries of the commit's code files (minus
    ``exclude``) and the paths omitted (with reasons counted in ``stats``)."""
    owner, name = repo.split("/", 1)
    parents = commit_data.get("parents") or []
    parent = parents[0]["sha"] if parents else None
    entries, omitted = [], []
    for f in commit_data.get("files") or []:
        path = f.get("filename", "")
        if (Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS or _is_skipped_path(path)
                or path in exclude):
            continue
        status = f.get("status", "")
        patch = f.get("patch")
        if status == "renamed" and not patch:
            stats["file_skip:pure_rename"] += 1
            omitted.append(path)
            continue
        pre = client.get_file(owner, name, path, parent) if (
            parent and status in ("modified", "removed")) else None
        post = client.get_file(owner, name, path, sha) if status in (
            "modified", "added") else None
        old, new, how = reconstruct(status, patch, pre, post)
        stats[f"file:{how}"] += 1
        if how.startswith("skip:"):
            omitted.append(path)
            continue
        if max(len((old or "").encode()), len((new or "").encode())) > MAX_FILE_BYTES:
            stats["file_skip:over_1MB"] += 1
            omitted.append(path)
            continue
        e = file_entry(path, old, new)
        if e is None:
            stats["file_skip:no_change"] += 1
            continue
        e["_status"] = status
        entries.append(e)
    return entries, omitted


def load_sources(cache_dir: Path, pair_files: list[Path]) -> tuple[list[dict], dict, dict]:
    """``(eval pairs, pair base -> [state pair], (repo, commit) -> [state pair])``."""
    from scripts.build_eval_split import pair_lines

    pairs = pair_lines([ln for p in pair_files for ln in load_jsonl(p)])
    wanted = {p["base"].rsplit("_", 1)[-1] for p in pairs}
    state = json.loads((cache_dir / "processed_advisories.json").read_text(encoding="utf-8"))
    by_hash: dict[str, list[dict]] = defaultdict(list)
    by_commit: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in state.values():
        for sp in entry.get("pairs") or []:
            slim = {k: sp[k] for k in ("repo", "commit", "file_path", "function_name",
                                       "advisory_id", "cve_id", "category", "cwe_id",
                                       "language")}
            by_commit[(sp["repo"], sp["commit"])].append(slim)
            h = pair_code_hash(sp["vulnerable_code"], sp["fixed_code"])
            if h in wanted:
                by_hash[h].append({**slim, "vulnerable_code": sp["vulnerable_code"],
                                   "fixed_code": sp["fixed_code"]})
    del state
    prov = {}
    for p in pairs:
        prov[p["base"]] = [sp for sp in by_hash.get(p["base"].rsplit("_", 1)[-1], [])
                           if sp["vulnerable_code"] == p["vuln"]["code"]
                           and sp["fixed_code"] == p["safe"]["code"]]
    return pairs, prov, by_commit


def load_split(split_path: Path, ordinary_path: Path | None) -> tuple[dict, dict]:
    """``(eval id -> side, repo -> side)`` from v1 (repo sides via the pair ids'
    provenance are added by the caller; ordinary items carry ``repo``)."""
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    side_of = {i: s for s in ("dev", "test") for i in manifest[s]["ids"]}
    repo_side: dict[str, set] = defaultdict(set)
    if ordinary_path and ordinary_path.exists():
        for o in load_jsonl(ordinary_path):
            if o["id"] in side_of and o.get("repo"):
                repo_side[o["repo"].lower()].add(side_of[o["id"]])
    return side_of, repo_side


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


def build_vuln_items(commit: tuple[str, str], bases: list[str], pairs_by_base: dict,
                     prov: dict, entries: list[dict], omitted: list[str], split: str,
                     max_files: int, max_changed: int, stats: Counter) -> list[dict]:
    """The vuln_introducing + vuln_fix items of one held-out fix commit."""
    repo, sha = commit
    srcs = [sp for b in bases for sp in prov[b] if (sp["repo"], sp["commit"]) == commit]
    vuln_files = Counter(sp["file_path"] for sp in srcs)
    priority = sorted(vuln_files, key=lambda p: (-vuln_files[p], p))
    by_path = {e["path"]: e for e in entries}
    if not any(p in by_path for p in priority):
        stats["skip_commit:vuln_file_not_reconstructable"] += 1
        return []
    kept, dropped = select_files(entries, priority, max_files, max_changed)
    kept_paths = {e["path"] for e in kept}
    if priority[0] not in kept_paths:
        stats["skip_commit:vuln_file_over_caps"] += 1
        return []
    target = priority[0]

    # Vulnerable functions located in the pre-fix (vulnerable) file.
    ranges_by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
    unlocated = 0
    for b in bases:
        sp = next((s for s in prov[b] if (s["repo"], s["commit"]) == commit), None)
        if sp is None or sp["file_path"] not in by_path:
            continue
        rng = locate_code(by_path[sp["file_path"]]["old_content"] or "",
                          pairs_by_base[b]["vuln"]["code"])
        if rng:
            ranges_by_path[sp["file_path"]].extend(rng)
        else:
            unlocated += 1

    intro_entries = [reverse_entry(e) for e in kept]
    intro_by_path = {e["path"]: e for e in intro_entries}
    vuln_by_path, modes = {}, {}
    for p in priority:
        if p not in intro_by_path:
            continue
        e = intro_by_path[p]
        ranges = ranges_by_path.get(p) or None
        lines, mode = vuln_lines(e["old_content"], e["new_content"], ranges, e["_ops"])
        if not ranges:
            mode += "_function_not_located"
        elif not lines:  # a mis-paired function: the fix changed code elsewhere
            lines, mode = vuln_lines(e["old_content"], e["new_content"], None, e["_ops"])
            mode += "_outside_function"
        vuln_by_path[p], modes[p] = lines, mode

    t_intro = intro_by_path[target]
    t_fix = next(e for e in kept if e["path"] == target)
    first = srcs[0]
    display = sorted({pairs_by_base[b]["vuln"]["expected_cve_id"] for b in bases})
    advisory_id = display[0]
    cve = next((d for d in display if d.startswith("CVE-")), None)
    cwe = next((sp["cwe_id"] for sp in srcs if sp.get("cwe_id")), None)
    category = next((pairs_by_base[b]["vuln"]["category"] for b in bases), first["category"])
    osv_ids = sorted({sp["advisory_id"] for sp in srcs})
    override = label_override(osv_ids)
    if override is not None:
        stats["label_override"] += 1
    base = pr_base_id(advisory_id, repo, sha)
    n_changed = sum(changed_count(e) for e in kept)
    notes = []
    if dropped:
        notes.append(f"Truncated to {len(kept)} file(s) / {n_changed} changed lines "
                     f"(caps {max_files} files, {max_changed} lines); dropped: "
                     + ", ".join(dropped) + ".")
    if omitted:
        notes.append("Omitted (content not in cache or pure rename): "
                     + ", ".join(sorted(omitted)) + ".")
    if unlocated:
        notes.append(f"{unlocated} vulnerable function(s) not located verbatim in the "
                     f"pre-fix file.")
    title, body = neutral_pr_text([e["path"] for e in kept])
    meta = {
        "commit": sha, "osv_ids": osv_ids,
        "advisory_ids": display, "pair_base": base,
        "eval_pair_ids": sorted(bases), "vuln_paths": priority,
        "vuln_functions": sorted({f"{sp['file_path']}::{sp['function_name']}"
                                  for sp in srcs}),
        "vuln_lines_mode": modes[target],
        "vuln_lines_by_path": {p: vuln_by_path[p] for p in sorted(vuln_by_path)},
        "files_in_commit": len(entries) + len(omitted), "truncated": bool(dropped),
        "dropped_files": dropped, "omitted_files": sorted(omitted),
        "changed_lines": n_changed,
    }
    if override is not None:
        osv_id, o = override
        meta["label_override"] = {"advisory": osv_id, "reason": o["reason"],
                                  "from": {"cwe": cwe, "category": category}}
        cwe, category = o["cwe"], o["category"]
    common = {
        "language": item_language(kept, target), "repo": repo, "advisory_id": advisory_id,
        "cve_id": cve, "category": category, "cwe": cwe, "split": split,
        "pr_title": title, "pr_body": body,
    }
    intro_added, _ = patch_line_numbers(t_intro["patch"])
    fix_added, _ = patch_line_numbers(t_fix["patch"])
    intro = {
        "id": f"{base}_intro", "kind": "vuln_introducing", **common,
        "files": public_files(sorted(intro_entries, key=lambda e: e["path"])),
        "target": {"path": target, "vuln_lines_new": vuln_by_path[target],
                   "changed_lines_new": intro_added},
        "source": "reversed_fix",
        "notes": " ".join(["Reversed fix commit: new_content is the pre-fix (vulnerable) "
                           "file."] + notes),
        "meta": meta,
    }
    fix = {
        "id": f"{base}_fix", "kind": "vuln_fix", **common,
        "files": public_files(kept),
        "target": {"path": target, "vuln_lines_new": [], "changed_lines_new": fix_added},
        "source": "real_fix",
        "notes": " ".join(["Real fix commit: a correct review reports no introduced "
                           "vulnerability."] + notes),
        "meta": {**meta, "vuln_lines_mode": None, "vuln_lines_by_path": {}},
    }
    return [intro, fix]


def build_bystander_item(commit: tuple[str, str], entries: list[dict], mined_files: set[str],
                         split: str, origin: str, max_files: int, max_changed: int,
                         stats: Counter) -> dict | None:
    """A benign PR from the commit's files that hold no paired (fix-changed)
    function, if the commit's paired functions live in other files."""
    repo, sha = commit
    if not mined_files:
        return None
    cands = [e for e in entries if e["path"] not in mined_files
             and any(ln.strip() for ln in _changed_texts(e))]
    if not cands:
        return None
    kept, dropped = select_files(cands, [], max_files, max_changed)
    if not kept:
        stats["skip_bystander:over_caps"] += 1
        return None
    title, body = neutral_pr_text([e["path"] for e in kept])
    n_changed = sum(changed_count(e) for e in kept)
    notes = ["Bystander edit: files of a security fix commit that hold no function the "
             "fix changed (caveat: security-adjacent commit, not a realistic benign "
             "sample)."]
    if dropped:
        notes.append(f"Truncated to {len(kept)} file(s) / {n_changed} changed lines; "
                     f"dropped: {', '.join(dropped)}.")
    return {
        "id": f"pr_bystander_{token(repo.replace('/', '_'))}_{sha[:10]}",
        "kind": "benign", "language": item_language(kept, None), "repo": repo,
        "advisory_id": None, "cve_id": None, "category": None, "cwe": None, "split": split,
        "pr_title": title, "pr_body": body, "files": public_files(kept),
        "target": {"path": None, "vuln_lines_new": [], "changed_lines_new": []},
        "source": "bystander", "notes": " ".join(notes),
        "meta": {"commit": sha, "bystander_of": origin, "truncated": bool(dropped),
                 "dropped_files": dropped, "changed_lines": n_changed},
    }


def _changed_texts(entry: dict) -> list[str]:
    return [x for h in parse_patch(entry["patch"]) for t, x in h.lines if t in "+-"]


def build_all(cache_dir: Path, pair_files: list[Path], split_path: Path,
              ordinary_path: Path | None, *, max_files: int = MAX_FILES,
              max_changed: int = MAX_CHANGED_LINES, bystanders: str = "all",
              other_per_repo: int = 5, seed: int = 42) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    pairs, prov, by_commit = load_sources(cache_dir, pair_files)
    pairs_by_base = {p["base"]: p for p in pairs}
    side_of, repo_side = load_split(split_path, ordinary_path)
    client = cache_client(cache_dir)

    bases_by_commit: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in pairs:
        commits = sorted({(sp["repo"], sp["commit"]) for sp in prov[p["base"]]})
        if not commits:
            stats["skip_pair:no_provenance"] += 1
        for c in commits:
            bases_by_commit[c].append(p["base"])

    commit_split: dict[tuple[str, str], str] = {}
    for c, bases in bases_by_commit.items():
        ids = [pairs_by_base[b][k]["id"] for b in bases for k in ("vuln", "safe")]
        commit_split[c] = split_for_ids(ids, side_of)
        if commit_split[c] != "reserve":
            repo_side[c[0].lower()].add(commit_split[c])
    bad = sorted(r for r, s in repo_side.items() if len(s) > 1)
    if bad:
        raise ValueError(f"repos straddle dev and test in {split_path}: {bad[:5]}")

    items: list[dict] = []
    stats["eval_pairs"] = len(pairs)
    stats["eval_commits"] = len(bases_by_commit)
    for c in sorted(bases_by_commit):
        data = client.get_cached_json(commit_url(*c))
        if not data or data.get("__status__") == 404:
            stats["skip_commit:commit_not_cached"] += 1
            continue
        entries, omitted = commit_files(client, c[0], c[1], data, stats)
        items += build_vuln_items(c, sorted(set(bases_by_commit[c])), pairs_by_base, prov,
                                  entries, omitted, commit_split[c], max_files, max_changed,
                                  stats)
        mined = {sp["file_path"] for sp in by_commit.get(c, [])}
        by = build_bystander_item(c, entries, mined, commit_split[c], "eval_fix_commit",
                                  max_files, max_changed, stats)
        if by:
            items.append(by)

    if bystanders == "all":
        # Other mined fix commits of the same repos: per repo, commits in seeded
        # order until ``other_per_repo`` bystander PRs (<= 0: no cap).
        by_repo: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for c in sorted(set(by_commit) - set(bases_by_commit)):
            by_repo[c[0]].append(c)
        for repo in sorted(by_repo):
            commits = by_repo[repo]
            random.Random(f"{seed}:bystander:{repo}").shuffle(commits)
            sides = repo_side.get(repo.lower())
            split = next(iter(sides)) if sides else "reserve"
            n = 0
            for c in commits:
                if 0 < other_per_repo <= n:
                    break
                data = client.get_cached_json(commit_url(*c))
                if not data or data.get("__status__") == 404:
                    stats["skip_other_commit:not_cached"] += 1
                    continue
                mined = {sp["file_path"] for sp in by_commit[c]}
                entries, _ = commit_files(client, c[0], c[1], data, Counter(), exclude=mined)
                by = build_bystander_item(c, entries, mined, split, "other_fix_commit",
                                          max_files, max_changed, stats)
                if by:
                    items.append(by)
                    n += 1
    for it in items:  # the side v1 fixed for the repo (also for reserve items)
        sides = repo_side.get(it["repo"].lower())
        it["meta"]["repo_side"] = next(iter(sides)) if sides else None
    items.sort(key=lambda it: it["id"])
    ids = Counter(it["id"] for it in items)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise ValueError(f"duplicate item ids: {dup[:5]}")
    return items, stats


# ---------------------------------------------------------------------------
# Summary / output.
# ---------------------------------------------------------------------------


def _by(items: list[dict], key) -> dict:
    return dict(sorted(Counter(key(it) for it in items).items(), key=lambda kv: str(kv[0])))


def _dist(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)

    def q(p: float) -> int:
        return s[min(int(p * len(s)), len(s) - 1)]

    return {"n": len(s), "mean": round(sum(s) / len(s), 1), "p10": q(0.1), "p25": q(0.25),
            "median": q(0.5), "p75": q(0.75), "p90": q(0.9), "max": s[-1]}


def counts(items: list[dict], stats: Counter | None = None) -> dict:
    """Summary numbers (printed, and stored in the manifest)."""
    kinds = sorted({it["kind"] for it in items})
    intro = [it for it in items if it["kind"] == "vuln_introducing"]
    out = {
        "items": len(items),
        "by_kind": _by(items, lambda it: it["kind"]),
        "by_kind_split": {k: _by([i for i in items if i["kind"] == k], lambda it: it["split"])
                          for k in kinds},
        "by_kind_language": {k: _by([i for i in items if i["kind"] == k],
                                    lambda it: it["language"]) for k in kinds},
        "by_source": _by(items, lambda it: it["source"] + (
            ":" + it["meta"]["bystander_of"] if it["source"] == "bystander" else "")),
        "vuln_introducing_by_category": _by(intro, lambda it: it["category"]),
        "files_per_pr": {k: _dist([len(i["files"]) for i in items if i["kind"] == k])
                         for k in kinds},
        "changed_lines_per_pr": {k: _dist([i["meta"]["changed_lines"] for i in items
                                           if i["kind"] == k]) for k in kinds},
        "vuln_introducing_with_vuln_lines": sum(bool(it["target"]["vuln_lines_new"])
                                                for it in intro),
        "vuln_lines_mode": _by(intro, lambda it: it["meta"]["vuln_lines_mode"]),
        "vuln_lines_new_size": _dist([len(it["target"]["vuln_lines_new"]) for it in intro]),
        "truncated_prs": sum(it["meta"]["truncated"] for it in items),
        "prs_with_omitted_files": sum(bool(it["meta"].get("omitted_files")) for it in items),
        "repos": len({it["repo"] for it in items}),
    }
    if stats is not None:
        out["build_stats"] = dict(sorted(stats.items()))
    return out


def summarize(items: list[dict], stats: Counter | None = None) -> str:
    return "\n".join(f"{k}: {json.dumps(v)}" for k, v in counts(items, stats).items())


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def file_info(path: Path) -> dict:
    """``{"items", "bytes", "sha256"}`` of a JSONL file, streamed."""
    h = hashlib.sha256()
    n = size = 0
    with open(path, "rb") as fh:
        for line in fh:
            h.update(line)
            size += len(line)
            n += bool(line.strip())
    return {"items": n, "bytes": size, "sha256": h.hexdigest()}


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


VULN_KINDS = ("vuln_introducing", "vuln_fix")


def build_v2(v1_dir: Path, real_commits: Path, split_path: Path, cache_dir: Path | None, *,
             seed: int = 42, n_benign_dev: int = V2_BENIGN_DEV,
             per_repo: int = V2_BENIGN_PER_REPO, max_files: int = MAX_FILES,
             max_changed: int = MAX_CHANGED_LINES) -> tuple[dict[str, list[dict]], dict]:
    """The v2 files from the BUILT v1 files (sha256-checked against their
    manifest: v2 never rebuilds or rewrites v1) and the fetched real commits.

    - ``pr_eval_v2_real_commits.jsonl``: every fetched commit, validated
      (``real_commit_item``), ``source: real_commit``.
    - ``pr_eval_v2_sample_dev.jsonl``: the v1 dev sample's 60 + 60 vuln items
      (same ids and content, so runs compare; ``LABEL_OVERRIDES`` applied) plus
      ``n_benign_dev`` dev real commits (``sample_real_commits`` with the
      sampled vuln language mix, at most ``per_repo`` per repo). No bystanders.
    - ``pr_eval_v2_test.jsonl``: every test vuln_introducing / vuln_fix item of
      v1 plus every test real commit.

    ``split_leak_check`` must pass on (dev sample, test file). Returns
    ``(files, info)``; ``info`` goes into the v2 manifest."""
    v1_manifest = json.loads((v1_dir / f"pr_eval_{VERSION}_manifest.json")
                             .read_text(encoding="utf-8"))
    full_name, sample_name = f"pr_eval_{VERSION}.jsonl", f"pr_eval_{VERSION}_sample_dev.jsonl"
    inputs: dict[str, dict] = {}
    for name in (full_name, sample_name):
        got = file_info(v1_dir / name)
        if got["sha256"] != v1_manifest["files"][name]["sha256"]:
            raise ValueError(f"{name} does not match pr_eval_{VERSION}_manifest.json "
                             "(rebuild v1 first: python scripts/build_pr_eval.py)")
        inputs[name] = got
    inputs[real_commits.name] = file_info(real_commits)
    inputs[split_path.name] = {"sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()}
    side_of, _ = load_split(split_path, None)

    # One pass over v1: repo sides, known commits, the test vuln items, bystander sizes.
    sides: dict[str, set] = defaultdict(set)
    known_commits: set[str] = set()
    test_vuln: list[dict] = []
    bystander_rows: list[dict] = []
    for it in iter_jsonl(v1_dir / full_name):
        for s in (it["split"], (it.get("meta") or {}).get("repo_side")):
            if s in ("dev", "test"):
                sides[it["repo"].lower()].add(s)
        if (it.get("meta") or {}).get("commit"):
            known_commits.add(it["meta"]["commit"])
        if it["kind"] == "benign":
            bystander_rows.append(size_row(it))
        elif it["split"] == "test":
            test_vuln.append(it)
    bad = sorted(r for r, s in sides.items() if len(s) > 1)
    if bad:
        raise ValueError(f"v1 repos straddle dev and test: {bad[:5]}")
    repo_side = {r: next(iter(s)) for r, s in sides.items()}
    state = cache_dir / "processed_advisories.json" if cache_dir else None
    if state is not None and state.exists():  # every commit the corpus builder mined
        for entry in json.loads(state.read_text(encoding="utf-8")).values():
            known_commits.update(p["commit"] for p in entry.get("pairs") or [])

    sample = list(iter_jsonl(v1_dir / sample_name))
    if [it["id"] for it in sample] != v1_manifest["sample_dev_ids"]:
        raise ValueError(f"{sample_name} ids differ from the manifest's sample_dev_ids")
    dev_vuln = [it for it in sample if it["kind"] in VULN_KINDS]
    sample_bystanders = [size_row(it) for it in sample if it["kind"] == "benign"]
    want_test = {i for k in VULN_KINDS for i in v1_manifest["ids"]["test"][k]}
    if {it["id"] for it in test_vuln} != want_test:
        raise ValueError("v1 test vuln items differ from the manifest's test ids")
    overrides = sum(apply_label_override(it) for it in dev_vuln + test_vuln)

    real = []
    for raw in iter_jsonl(real_commits):
        it = real_commit_item(raw, repo_side, max_files=max_files, max_changed=max_changed)
        if it["meta"]["commit"] in known_commits:
            raise ValueError(f"real commit {it['id']} is a known fix / eval commit")
        real.append(it)
    real.sort(key=lambda it: it["id"])
    for key in ("id", "commit"):
        vals = Counter(it["id"] if key == "id" else it["meta"]["commit"] for it in real)
        dup = [v for v, c in vals.items() if c > 1]
        if dup:
            raise ValueError(f"duplicate real commit {key}s: {dup[:3]}")

    mix = Counter(it["language"] for it in dev_vuln if it["kind"] == "vuln_introducing")
    dev_pool = [it for it in real if it["split"] == "dev"]
    dev_benign, cap = sample_real_commits(dev_pool, n_benign_dev, seed, dict(mix), per_repo)
    dev = dev_vuln + dev_benign
    test = test_vuln + [it for it in real if it["split"] == "test"]
    leak = split_leak_check(dev, test, side_of, repo_side)
    leaks = [it["id"] for it in dev + test + real
             if LEAK_ID_RE.search(it["pr_title"] + "\n" + it["pr_body"])]
    if leaks:
        raise ValueError(f"advisory ids in PR text: {leaks[:5]}")

    files = {f"pr_eval_{V2}_real_commits.jsonl": real,
             f"pr_eval_{V2}_sample_dev.jsonl": dev,
             f"pr_eval_{V2}_test.jsonl": test}
    info = {
        "inputs": inputs,
        "label_overrides_applied_here": overrides,
        "dev_benign": {"n": len(dev_benign), "language_mix_of": "sampled vuln_introducing",
                       "vuln_language_mix": dict(sorted(mix.items())),
                       "per_repo_cap": per_repo, "per_repo_cap_used": cap,
                       "max_per_repo": max(Counter(it["repo"] for it in dev_benign).values(),
                                           default=0),
                       "pool": len(dev_pool)},
        "leak_check": leak,
        "sizes": {
            "real_commit_all": pr_size_stats([size_row(it) for it in real]),
            "real_commit_dev": pr_size_stats([size_row(it) for it in dev_pool]),
            "real_commit_dev_sample": pr_size_stats([size_row(it) for it in dev_benign]),
            "real_commit_test": pr_size_stats([size_row(it) for it in real
                                               if it["split"] == "test"]),
            "bystander_v1_all": pr_size_stats(bystander_rows),
            "bystander_v1_sample_dev": pr_size_stats(sample_bystanders),
        },
    }
    return files, info


def main_v2(args, argv: list[str] | None) -> int:
    files, info = build_v2(args.v1_dir, args.real_commits, args.split, args.cache_dir,
                           seed=args.seed, n_benign_dev=args.v2_benign_dev,
                           per_repo=args.v2_benign_per_repo, max_files=args.max_files,
                           max_changed=args.max_changed_lines)
    summaries = {name: counts(rows) for name, rows in files.items()}
    print("=" * 78)
    print(f"PR eval {V2}: real benign commits + run samples (from the built {VERSION} files)")
    print("=" * 78)
    for name, s in summaries.items():
        print(f"{name}:")
        for k in ("items", "by_kind", "by_kind_split", "by_kind_language", "by_source", "repos"):
            print(f"  {k}: {json.dumps(s[k])}")
    print(f"dev benign: {json.dumps(info['dev_benign'])}")
    print(f"leak check: {json.dumps(info['leak_check'])}")
    for k, v in info["sizes"].items():
        print(f"size {k}: {json.dumps(v)}")
    print("GitHub API calls made: 0 (offline)")
    if args.check:
        return 0
    written = {}
    for name, rows in files.items():
        path = args.out_dir / name
        write_jsonl(path, rows)
        written[name] = file_info(path)
        print(f"wrote {path} ({written[name]['bytes'] / 1e6:.1f} MB)")
    manifest = {
        "version": V2,
        "built_with": " ".join(["python scripts/build_pr_eval.py",
                                *(sys.argv[1:] if argv is None else argv)]),
        "seed": args.seed, "max_files": args.max_files,
        "max_changed_lines": args.max_changed_lines,
        "provenance": {
            "vuln_items": f"pr_eval_{VERSION}.jsonl / pr_eval_{VERSION}_sample_dev.jsonl "
                          f"(sha256-checked against pr_eval_{VERSION}_manifest.json; "
                          "LABEL_OVERRIDES applied)",
            "real_commits": f"{args.real_commits.name}, fetched from the GitHub API by "
                            "scripts/fetch_benign_commits.py (fetcher source label "
                            "'benign_commit', relabelled 'real_commit'; raw label in "
                            "meta.fetched_source; sha in meta.commit)",
            "bystanders": f"not in v2 samples; they stay in pr_eval_{VERSION}*.jsonl",
            "split": f"{_rel_root(args.split)} (repos take their v1 side)",
        },
        **info,
        "files": written,
        "counts": summaries,
        "ids": {name: [it["id"] for it in rows] for name, rows in files.items()
                if name != f"pr_eval_{V2}_real_commits.jsonl"},
        "real_commit_ids": {s: [it["id"] for it in files[f"pr_eval_{V2}_real_commits.jsonl"]
                                if it["split"] == s] for s in ("dev", "test")},
    }
    mpath = args.out_dir / f"pr_eval_{V2}_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {mpath}")
    return 0


def _rel_root(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--pairs", nargs="+", type=Path, default=DEFAULT_PAIR_FILES)
    ap.add_argument("--ordinary", type=Path, default=DEFAULT_ORDINARY,
                    help="ordinary negatives (their repos pin repo -> split)")
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files", type=int, default=MAX_FILES)
    ap.add_argument("--max-changed-lines", type=int, default=MAX_CHANGED_LINES)
    ap.add_argument("--bystanders", choices=("eval", "all"), default="all",
                    help="bystander benigns from the eval fix commits only, or also from "
                         "the other mined fix commits of the same repos (default)")
    ap.add_argument("--other-bystanders-per-repo", type=int, default=5,
                    help="cap on bystander PRs per repo from the non-eval fix commits "
                         "(0 = no cap)")
    ap.add_argument("--sample-vuln", type=int, default=60)
    ap.add_argument("--sample-benign", type=int, default=80)
    ap.add_argument("--check", action="store_true", help="print the summary, write nothing")
    v2 = ap.add_argument_group("v2 (real benign commits; reads the built v1 files, no cache)")
    v2.add_argument("--v2", action="store_true",
                    help="build the v2 files from the built v1 files + --real-commits "
                         "(v1 is not rebuilt or rewritten)")
    v2.add_argument("--v1-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="where the built v1 files and their manifest are")
    v2.add_argument("--real-commits", type=Path, default=DEFAULT_REAL_COMMITS,
                    help="scripts/fetch_benign_commits.py output")
    v2.add_argument("--v2-benign-dev", type=int, default=V2_BENIGN_DEV,
                    help="real commits in the v2 dev sample")
    v2.add_argument("--v2-benign-per-repo", type=int, default=V2_BENIGN_PER_REPO,
                    help="at most this many dev-sample real commits per repo (raised only "
                         "if the pool can't fill the sample)")
    args = ap.parse_args(argv)
    if args.v2:
        return main_v2(args, argv)

    items, stats = build_all(args.cache_dir, args.pairs, args.split, args.ordinary,
                             max_files=args.max_files, max_changed=args.max_changed_lines,
                             bystanders=args.bystanders,
                             other_per_repo=args.other_bystanders_per_repo, seed=args.seed)
    bad = straddling(items)
    if any(bad.values()):
        raise SystemExit(f"split straddles dev/test: {bad}")
    leaks = [it["id"] for it in items
             if LEAK_ID_RE.search(it["pr_title"] + "\n" + it["pr_body"])]
    if leaks:
        raise SystemExit(f"advisory ids in PR text: {leaks[:5]}")
    misleading = [make_misleading(it) for it in items if it["kind"] == "vuln_introducing"]
    sample = sample_dev(items, args.seed, args.sample_vuln, args.sample_benign)

    summary = counts(items, stats)
    sample_summary = counts(sample)
    print("=" * 78)
    print(f"PR eval {VERSION}")
    print("=" * 78)
    print(summarize(items, stats))
    print("-" * 78)
    print("dev sample:")
    for k in ("by_kind", "by_kind_language", "by_source", "vuln_introducing_by_category",
              "repos"):
        print(f"  {k}: {json.dumps(sample_summary[k])}")
    print("GitHub API calls made: 0 (offline cache only)")
    if args.check:
        return 0
    out = args.out_dir
    files = {f"pr_eval_{VERSION}.jsonl": items,
             f"pr_eval_{VERSION}_misleading.jsonl": misleading,
             f"pr_eval_{VERSION}_sample_dev.jsonl": sample}
    written = {}
    for name, rows in files.items():
        path = out / name
        write_jsonl(path, rows)
        written[name] = {"items": len(rows), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    manifest = {
        "version": VERSION,
        "built_with": " ".join(["python scripts/build_pr_eval.py",
                                *(sys.argv[1:] if argv is None else argv)]),
        "seed": args.seed, "max_files": args.max_files,
        "max_changed_lines": args.max_changed_lines, "bystanders": args.bystanders,
        "other_bystanders_per_repo": args.other_bystanders_per_repo,
        "files": written,
        "counts": summary,
        "sample_dev_counts": {k: sample_summary[k] for k in (
            "items", "by_kind", "by_kind_language", "by_source",
            "vuln_introducing_by_category", "repos")},
        "ids": {split: {k: [it["id"] for it in items if it["split"] == split
                            and it["kind"] == k] for k in sorted(summary["by_kind"])}
                for split in ("dev", "test", "reserve")},
        "sample_dev_ids": [it["id"] for it in sample],
    }
    mpath = out / f"pr_eval_{VERSION}_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
