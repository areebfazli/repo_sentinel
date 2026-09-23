"""Build a paired vulnerable/fixed function corpus from OSV advisories.

Mines the OSV bulk export for an ecosystem (PyPI, npm) for advisories that carry
a GitHub fix-commit reference, fetches that commit via the GitHub API, and pairs
up pre-commit ("vulnerable") and post-commit ("fixed") functions whose bodies
changed inside the commit's diff hunks. Emits:

- ``{out-dir}/osv_{ecosystem}.json``: a corpus in the same shape as
  ``sample_cves.json`` (plus ``fixed_code``/``repo``/``commit``/... fields),
  ready for ``scripts/ingest_cve_corpus.py``.
- ``{eval-dir}/detection_eval_osv.jsonl``: a held-out eval set (vulnerable +
  fixed line per pair) in the same JSONL shape as ``detection_eval.jsonl``.
- ``{out-dir}/rejected/osv_{ecosystem}_rejected.json``: pairs dropped by the
  quality filter, each with a ``reject_reason`` (a subdirectory, because the
  ingest script loads every ``{out-dir}/*.json``).

The held-out split is done **by advisory**, not by individual function pair, so
no function in the eval set shares an advisory with anything left in the
training corpus (see ``split_by_advisory``).

Fix commits also touch bystander functions (renames, docstring edits, broad
refactors). Every pair gets a ``quality`` dict (``compute_diff_quality`` +
per-commit counts) and is filtered by ``--min-changed-stmt-lines``,
``--max-functions-per-commit`` and ``--drop-other`` *after* the split, so the
held-out advisories don't depend on those flags and a rejected pair never
reaches the eval set (see ``build_ecosystem_outputs``).

All pure logic (commit-URL parsing, hunk-header parsing, overlap checks, the
CWE->category table, whitespace-only-change detection, the advisory split, the
dedupe key, and the quality score/filter) lives in module-level functions with
no network access, so it can be imported and unit-tested directly. Everything
that talks to the network (downloading the OSV zip, calling the GitHub API) is
confined to ``main()`` and the ``GithubClient`` class below.

GitHub API usage is rate-limit aware: every raw API response is cached in
``--cache-dir`` keyed by a hash of the request URL, so a ``--resume`` run (or
simply re-running the script) makes zero redundant calls for anything already
fetched. Without ``--wait-on-rate-limit``, the script stops cleanly (not a
crash) the moment the rate limit is exhausted, always honouring
``--max-advisories`` as an upper bound on API usage. ``--offline`` reads the
cache only (no network at all), which is how to re-filter a finished run.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import difflib
import hashlib
import json
import keyword
import math
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import requests

# Add project root to path so this runs as a script (mirrors the other scripts/*.py).
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.config import BASE_DIR, settings  # noqa: E402
from backend.app.core.code_parser import CodeParser  # noqa: E402

OSV_ZIP_URL_TEMPLATE = "https://osv-vulnerabilities.storage.googleapis.com/{ecosystem}/all.zip"

# ext -> corpus "language" field. .ts is parsed with the JS grammar (see
# CodeParser), so it is emitted as "javascript" too, matching the rest of the repo.
SUPPORTED_EXTENSIONS: dict[str, str] = {".py": "python", ".js": "javascript", ".ts": "javascript"}

ECOSYSTEM_LANGUAGE_HINT: dict[str, str] = {"PyPI": "python", "npm": "javascript"}

SKIP_PATH_SUBSTRINGS = ("test", "spec", "__tests__", "fixtures")
SKIP_PATH_PREFIXES = ("docs/", "examples/")

DEFAULT_OUT_DIR = BASE_DIR / "data" / "cve_corpus"
DEFAULT_EVAL_DIR = BASE_DIR / "ml" / "evaluation" / "datasets"
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "osv_cache"

# CWE -> category table. First matching entry (in this order) wins; no CWE match
# (or no CWE at all) falls through to "other". Numbers are compared without the
# "CWE-" prefix.
CWE_CATEGORY_TABLE: list[tuple[tuple[str, ...], str]] = [
    (("89",), "sqli"),
    (("79", "80"), "xss"),
    (("77", "78", "94", "95"), "cmd_injection"),
    (("22", "23", "36"), "path_traversal"),
    (("798", "259", "321", "522"), "secrets"),
    (("918",), "ssrf"),
    (("611", "776"), "xxe"),
    (("502",), "deserialization"),
    (("327", "328", "326", "916"), "weak_crypto"),
    (("601",), "open_redirect"),
    (("862", "863", "639", "284"), "authz"),
    (("1321",), "prototype_pollution"),
    (("1333", "400"), "redos"),
]

_COMMIT_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/commit/(?P<sha>[0-9a-fA-F]{7,40})"
    r"(?:[/?#.].*)?$"
)
_PULL_COMMIT_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/\d+/commits/"
    r"(?P<sha>[0-9a-fA-F]{7,40})(?:[/?#.].*)?$"
)

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


# ---------------------------------------------------------------------------
# Pure logic (no network, no filesystem) — unit-tested directly.
# ---------------------------------------------------------------------------


def parse_fix_commit_url(url: str) -> tuple[str, str, str] | None:
    """Parse a GitHub commit URL into ``(owner, repo, sha)``.

    Accepts ``https://github.com/{owner}/{repo}/commit/{sha}`` and
    ``https://github.com/{owner}/{repo}/pull/{N}/commits/{sha}`` (optionally with
    a trailing path/query/fragment, e.g. ``#diff-...`` or ``.patch``). Returns
    ``None`` for anything else.
    """
    if not url:
        return None
    for pattern in (_COMMIT_URL_RE, _PULL_COMMIT_URL_RE):
        m = pattern.match(url.strip())
        if m:
            return m.group("owner"), m.group("repo"), m.group("sha")
    return None


def parse_hunk_ranges(patch: str) -> list[tuple[int, int]]:
    """Parse unified-diff ``@@ -a,b +c,d @@`` headers into post-commit line ranges.

    Returns a list of inclusive ``(start_line, end_line)`` ranges in the
    *post*-commit file. A hunk that adds zero lines (``+c,0``, a pure deletion)
    contributes no range.
    """
    ranges: list[tuple[int, int]] = []
    for line in patch.splitlines():
        m = _HUNK_HEADER_RE.match(line)
        if not m:
            continue
        start = int(m.group("start"))
        count = int(m.group("count")) if m.group("count") is not None else 1
        if count <= 0:
            continue
        ranges.append((start, start + count - 1))
    return ranges


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def function_overlaps_hunks(
    func_start: int, func_end: int, hunk_ranges: list[tuple[int, int]]
) -> bool:
    """True if a function's ``[func_start, func_end]`` (post-commit lines) overlaps
    any changed-line range from ``parse_hunk_ranges``."""
    return any(ranges_overlap(func_start, func_end, hs, he) for hs, he in hunk_ranges)


def cwe_to_category(cwe_ids: list[str] | None) -> str:
    """Map a list of CWE ids (e.g. ``["CWE-89"]``) to a corpus category.

    First entry in ``CWE_CATEGORY_TABLE`` (in listed order) whose CWE numbers
    intersect ``cwe_ids`` wins. No CWEs, or none that match, -> "other".
    """
    if not cwe_ids:
        return "other"
    normalized = {c.upper().replace("CWE-", "").strip() for c in cwe_ids if c}
    for numbers, category in CWE_CATEGORY_TABLE:
        if normalized & set(numbers):
            return category
    return "other"


def _normalize_code_for_diff(code: str, language: str) -> str:
    """Strip comments/blank lines and collapse whitespace for a whitespace/comment
    -only-change comparison. Whitespace is removed entirely (not just collapsed to
    a single space) so pure reformatting — reindentation, added/removed blank
    lines, spacing added around an operator — reads as identical. Not used for
    anything else (the real code is stored verbatim in the corpus)."""
    text = code
    if language == "javascript":
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if language == "python" and stripped.startswith("#"):
            continue
        if language == "javascript" and stripped.startswith("//"):
            continue
        lines.append(re.sub(r"\s+", "", stripped))
    return "\n".join(lines)


def is_whitespace_only_change(before: str, after: str, language: str) -> bool:
    """True if ``before`` -> ``after`` is only whitespace/comment reformatting
    (no change once comments, blank lines and whitespace runs are normalized)."""
    return _normalize_code_for_diff(before, language) == _normalize_code_for_diff(after, language)


def make_dedupe_key(repo: str, file_path: str, function_name: str, vulnerable_code: str) -> str:
    """Dedupe key: ``(repo, file_path, function_name, sha256(vulnerable_code))``."""
    digest = hashlib.sha256(vulnerable_code.encode("utf-8")).hexdigest()
    return f"{repo}|{file_path}|{function_name}|{digest}"


def split_by_advisory(
    pairs: list[dict], eval_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split ``pairs`` (each carrying an ``"advisory_id"`` key) into
    ``(corpus_pairs, eval_pairs)`` by holding out ``eval_fraction`` of the
    *advisory groups* (not individual pairs), shuffled deterministically by
    ``seed``. No advisory id appears on both sides."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        groups[p["advisory_id"]].append(p)

    advisory_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(advisory_ids)

    n_eval = round(len(advisory_ids) * eval_fraction)
    eval_ids = set(advisory_ids[:n_eval])

    corpus_pairs: list[dict] = []
    eval_pairs: list[dict] = []
    for aid in advisory_ids:
        (eval_pairs if aid in eval_ids else corpus_pairs).extend(groups[aid])
    return corpus_pairs, eval_pairs


def select_fix_reference(advisory: dict) -> tuple[str, str, str] | None:
    """Pick the first usable GitHub fix-commit reference from an advisory's
    ``references[]`` (FIX type preferred over WEB), or ``None``. Withdrawn
    advisories are always rejected."""
    if advisory.get("withdrawn"):
        return None
    refs = advisory.get("references") or []
    for wanted_type in ("FIX", "WEB"):
        for ref in refs:
            if ref.get("type") != wanted_type:
                continue
            parsed = parse_fix_commit_url(ref.get("url", ""))
            if parsed:
                return parsed
    return None


def get_display_id(advisory: dict) -> str:
    """Prefer a CVE alias if the advisory has one, else its own (GHSA) id."""
    for alias in advisory.get("aliases") or []:
        if isinstance(alias, str) and alias.startswith("CVE-"):
            return alias
    return advisory.get("id", "UNKNOWN")


def get_description(advisory: dict) -> str:
    text = advisory.get("summary") or advisory.get("details") or ""
    text = " ".join(text.split())  # collapse newlines/markdown whitespace
    return text[:300]


# -- CVSS v3.x base-score calculator (best-effort; OSV's severity[].score for a
# CVSS_V3/V3.1 entry is the vector string itself, not the numeric score). -------

_CVSS3_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_CVSS3_AC = {"L": 0.77, "H": 0.44}
_CVSS3_UI = {"N": 0.85, "R": 0.62}
_CVSS3_CIA = {"N": 0.0, "L": 0.22, "H": 0.56}
_CVSS3_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_CVSS3_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}


def _parse_cvss_vector(vector: str) -> dict[str, str] | None:
    if not vector or not vector.startswith("CVSS:3"):
        return None
    metrics: dict[str, str] = {}
    for part in vector.split("/")[1:]:
        if ":" in part:
            k, v = part.split(":", 1)
            metrics[k] = v
    return metrics or None


def _cvss3_base_score(metrics: dict[str, str]) -> float | None:
    try:
        av = _CVSS3_AV[metrics["AV"]]
        ac = _CVSS3_AC[metrics["AC"]]
        ui = _CVSS3_UI[metrics["UI"]]
        scope_changed = metrics["S"] == "C"
        pr = (_CVSS3_PR_CHANGED if scope_changed else _CVSS3_PR_UNCHANGED)[metrics["PR"]]
        c = _CVSS3_CIA[metrics["C"]]
        i = _CVSS3_CIA[metrics["I"]]
        a = _CVSS3_CIA[metrics["A"]]
    except KeyError:
        return None

    iss = 1 - ((1 - c) * (1 - i) * (1 - a))
    impact = (7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15) if scope_changed else 6.42 * iss
    if impact <= 0:
        return 0.0

    exploitability = 8.22 * av * ac * pr * ui
    base = min(1.08 * (impact + exploitability), 10.0) if scope_changed else min(
        impact + exploitability, 10.0
    )
    return math.ceil(base * 10) / 10.0


def parse_cvss_score(severity_list: list[dict] | None) -> float | None:
    """Best-effort numeric CVSS base score from an OSV ``severity[]`` list: a
    directly-numeric ``score`` wins, else a CVSS v3.x vector string is scored
    with the standard base-score formula. ``None`` if nothing usable is found."""
    for entry in severity_list or []:
        score = entry.get("score", "")
        try:
            return round(float(score), 1)
        except (TypeError, ValueError):
            pass
        metrics = _parse_cvss_vector(score)
        if metrics:
            computed = _cvss3_base_score(metrics)
            if computed is not None:
                return computed
    return None


# ---------------------------------------------------------------------------
# Per-pair quality score (pure). Fix commits often touch bystander functions:
# identifier renames, docstring edits, or a broad refactor around the real fix.
# ``compute_diff_quality`` measures how much *statement-level* code a pair's
# vulnerable -> fixed diff really changes; ``apply_quality_filter`` drops the
# noise. None of this feeds back into the split or the ids.
# ---------------------------------------------------------------------------

# Category -> regexes counted in a pair's changed lines (case-insensitive,
# non-overlapping; longer patterns are tried first at each position). "other"
# (and any category missing here) counts hits across the union of all lists.
# Deliberately loose: a rough "does this diff touch security-relevant code"
# signal for auditing, not a filter on its own. Tune freely.
SECURITY_TOKENS: dict[str, list[str]] = {
    "sqli": [r"execute", r"cursor", r"query", r"SELECT", r"%s", r"format", r"\bf[\"']"],
    "cmd_injection": [r"subprocess", r"os\.system", r"popen", r"shell=", r"exec", r"eval"],
    "path_traversal": [r"os\.path", r"open\(", r"join", r"realpath", r"normpath", r"\.\.",
                       r"safe_join"],
    "xss": [r"escape", r"Markup", r"innerHTML", r"sanitize", r"html"],
    "ssrf": [r"requests\.", r"urlopen", r"http", r"url", r"host"],
    "deserialization": [r"pickle", r"yaml\.load", r"marshal", r"unserialize"],
    "weak_crypto": [r"md5", r"sha1", r"random", r"\bDES\b", r"RC4", r"hashlib"],
    "open_redirect": [r"redirect", r"next", r"url"],
    "authz": [r"permission", r"is_authenticated", r"role", r"owner", r"user\.id", r"403"],
    "redos": [r"\bre\.", r"regex", r"compile", r"match"],
    "xxe": [r"etree", r"lxml", r"resolve_entities", r"XMLParser"],
    "secrets": [r"key", r"token", r"secret", r"password"],
    "prototype_pollution": [r"__proto__", r"prototype", r"constructor", r"hasOwnProperty",
                            r"Object\.assign", r"merge"],
}

# A rename-only pair is flagged ``security_rename`` when a renamed identifier
# (either side) hits the category's SECURITY_TOKENS or one of these generic
# markers: one-identifier security fixes (md5 -> sha256, yaml.load ->
# yaml.safe_load, innerHTML -> textContent, abspath -> realpath) are
# indistinguishable from cosmetic renames by token shape alone.
SECURITY_RENAME_MARKERS: list[str] = [r"safe", r"secur", r"saniti[sz]", r"escape", r"valid"]

_JS_KEYWORDS = frozenset(
    "async await break case catch class const continue debugger default delete do else "
    "export extends false finally for function if import in instanceof let new null of "
    "return static super switch this throw true try typeof undefined var void while with "
    "yield".split()
)
_KEYWORDS: dict[str, frozenset[str]] = {
    "python": frozenset(keyword.kwlist),
    "javascript": _JS_KEYWORDS,
}

# Token regexes. Group order matters: strings (with an optional Python prefix)
# before identifiers, so ``f"..."`` is one string token, not ``f`` + string.
# Operators are single characters — only consistency between the two sides
# matters, and whitespace is never part of a token.
_PY_TOKEN_RE = re.compile(
    r"(?P<cont>\\\r?\n)"
    r"|(?P<ws>\s+)"
    r"|(?P<comment>#[^\n]*)"
    r"|(?P<str>[rRbBuUfF]{0,2}(?:'''(?:\\.|[^\\])*?'''|\"\"\"(?:\\.|[^\\])*?\"\"\""
    r"|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"))"
    r"|(?P<num>\d[\w.]*|\.\d\w*)"
    r"|(?P<id>[^\W\d]\w*)"
    r"|(?P<op>\S)",
    re.S,
)
_JS_TOKEN_RE = re.compile(
    r"(?P<cont>\\\r?\n)"
    r"|(?P<ws>\s+)"
    r"|(?P<comment>//[^\n]*|/\*.*?(?:\*/|\Z))"
    r"|(?P<str>'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"|`(?:\\.|[^`\\])*`)"
    r"|(?P<num>\d[\w.]*|\.\d\w*)"
    r"|(?P<id>(?:[^\W\d]|\$)[\w$]*)"
    r"|(?P<op>\S)",
    re.S,
)
_STR_HEAD_RE = re.compile(r"([A-Za-z]*)('''|\"\"\"|'|\"|`)")


@dataclass(frozen=True)
class _Tok:
    kind: str  # "id" | "kw" | "str" | "num" | "op"
    text: str  # normalised text (strings: quote style dropped)
    line: int  # 0-based line of the token's first character
    start: int  # offsets into the source (for reconstructing line text)
    end: int


def _normalize_string_literal(raw: str) -> str:
    """``'a'``, ``"a"`` and ``'''a'''`` compare equal (quote style is cosmetic);
    the prefix (``f``, ``b``, ...) and the body are kept."""
    m = _STR_HEAD_RE.match(raw)
    if not m:
        return raw
    prefix, quote = m.group(1).lower(), m.group(2)
    body = raw[m.end():]
    if body.endswith(quote):
        body = body[: -len(quote)]
    return f"{prefix}|{body}"


def tokenize_code(code: str, language: str) -> list[_Tok]:
    """Tokenise ``code`` into statement tokens: comments are dropped, and in
    Python so are docstrings (any string literal that forms a whole statement —
    a no-op, so dropping every such string, not just the first, is safe).
    Multi-line strings are one token attributed to their first line."""
    token_re = _JS_TOKEN_RE if language == "javascript" else _PY_TOKEN_RE
    keywords = _KEYWORDS.get(language, frozenset())
    line_starts = [0] + [m.end() for m in re.finditer(r"\n", code)]

    def line_of(offset: int) -> int:
        return bisect.bisect_right(line_starts, offset) - 1

    raw: list[_Tok] = []
    continued: set[int] = set()  # indexes of tokens preceded by a backslash-newline
    for m in token_re.finditer(code):
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            continue
        if kind == "cont":
            continued.add(len(raw))
            continue
        text = m.group()
        if kind == "id" and text in keywords:
            kind = "kw"
        elif kind == "str":
            text = _normalize_string_literal(text)
        raw.append(_Tok(kind, text, line_of(m.start()), m.start(), m.end()))

    if language == "python":
        raw = _drop_python_docstrings(raw, continued, line_of)
    # Drop a formatter's "magic" trailing comma (a ``,`` whose closing bracket is
    # on a later line), so re-wrapping a call one-arg-per-line is not a change.
    return [
        t
        for i, t in enumerate(raw)
        if not (
            t.kind == "op"
            and t.text == ","
            and i + 1 < len(raw)
            and raw[i + 1].kind == "op"
            and raw[i + 1].text in ")]}"
            and raw[i + 1].line > t.line
        )
    ]


def _drop_python_docstrings(raw: list[_Tok], continued: set[int], line_of) -> list[_Tok]:
    """Drop Python string-statement "docstrings": a run of adjacent string tokens
    that starts a logical line (depth 0, previous token on an earlier line, no
    backslash continuation) and is followed by a new line, a ';', or EOF."""
    out: list[_Tok] = []
    depth = 0
    i = 0
    while i < len(raw):
        tok = raw[i]
        starts_line = depth == 0 and i not in continued and (
            i == 0 or line_of(raw[i - 1].end - 1) < tok.line
        )
        if tok.kind == "str" and starts_line:
            j = i
            while (
                j + 1 < len(raw)
                and raw[j + 1].kind == "str"
                and raw[j + 1].line == line_of(raw[j].end - 1)
            ):
                j += 1
            nxt = raw[j + 1] if j + 1 < len(raw) else None
            if nxt is None or (
                (j + 1) not in continued
                and (nxt.line > line_of(raw[j].end - 1) or nxt.text == ";")
            ):
                i = j + 1
                continue
        if tok.kind == "op":
            if tok.text in "([{":
                depth += 1
            elif tok.text in ")]}":
                depth = max(depth - 1, 0)
        out.append(tok)
        i += 1
    return out


def _code_lines(code: str, language: str) -> list[list[_Tok]]:
    """Group statement tokens by source line; lines with no statement tokens
    (blank, comment-only, docstring, or inside a multi-line string) vanish."""
    by_line: dict[int, list[_Tok]] = defaultdict(list)
    for tok in tokenize_code(code, language):
        by_line[tok.line].append(tok)
    return [by_line[n] for n in sorted(by_line)]


def _line_key(line: list[_Tok]) -> tuple:
    return tuple((t.kind, t.text) for t in line)


def _shape_key(line: list[_Tok]) -> tuple:
    """Like ``_line_key`` but with identifiers anonymised."""
    return tuple((t.kind, "" if t.kind == "id" else t.text) for t in line)


def _identifier_pairs(old: list[_Tok], new: list[_Tok]) -> list[tuple[str, str]] | None:
    """If ``old`` and ``new`` are the same token stream up to identifier names,
    return the aligned ``(old_id, new_id)`` pairs; else ``None``."""
    if len(old) != len(new):
        return None
    pairs = []
    for a, b in zip(old, new, strict=True):
        if a.kind != b.kind:
            return None
        if a.kind == "id":
            pairs.append((a.text, b.text))
        elif a.text != b.text:
            return None
    return pairs


def _line_text(code: str, line: list[_Tok]) -> str:
    return code[line[0].start: line[-1].end]


def count_security_tokens(text: str, category: str) -> int:
    """Hits of ``SECURITY_TOKENS[category]`` in ``text`` (union of every list
    for "other" or an unlisted category)."""
    patterns = SECURITY_TOKENS.get(category)
    if not patterns:
        patterns = list(dict.fromkeys(p for ps in SECURITY_TOKENS.values() for p in ps))
    ordered = sorted(patterns, key=len, reverse=True)
    regex = re.compile("|".join(f"(?:{p})" for p in ordered), re.I)
    return len(regex.findall(text))


def compute_diff_quality(
    vulnerable_code: str, fixed_code: str, language: str, category: str
) -> dict:
    """Statement-level diff metrics for one vulnerable -> fixed pair.

    1. Both sides are tokenised (comments, docstrings, whitespace and quote
       style dropped) and grouped into code lines; the line lists are diffed.
    2. Every changed block is checked for "same token stream up to identifier
       names" — the whole block first (catches re-wrapped lines), then line by
       line via a second diff on identifier-anonymised lines. Each match yields
       candidate ``old_id -> new_id`` pairs.
    3. A rename is accepted only if it is a consistent 1:1 mapping across the
       *whole function*: unchanged lines contribute ``x -> x`` identities, and
       an identifier mapped to two names (or two identifiers mapped to one
       name) invalidates every candidate that uses it. Accepted candidates are
       rename-only and don't count; everything else does.

    Returns ``changed_stmt_lines`` (removed + added code lines that survive
    the above), ``rename_only`` (nothing counted but at least one accepted
    rename), ``renamed_identifiers`` (the accepted ``{old: new}`` map),
    ``security_rename`` (a renamed identifier looks security-relevant, see
    ``SECURITY_RENAME_MARKERS``), and ``security_tokens``
    (``count_security_tokens`` over every changed code line, renamed lines
    included).
    """
    old_lines = _code_lines(vulnerable_code, language)
    new_lines = _code_lines(fixed_code, language)

    forward: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)

    def constrain(pairs) -> None:
        for a, b in pairs:
            forward[a].add(b)
            reverse[b].add(a)

    candidates: list[tuple[list[int], list[int], list[tuple[str, str]]]] = []
    real_old: list[int] = []
    real_new: list[int] = []
    changed_old: list[int] = []
    changed_new: list[int] = []

    outer = difflib.SequenceMatcher(
        None, [_line_key(ln) for ln in old_lines], [_line_key(ln) for ln in new_lines],
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in outer.get_opcodes():
        if tag == "equal":
            for ln in old_lines[i1:i2]:
                constrain((t.text, t.text) for t in ln if t.kind == "id")
            continue
        changed_old.extend(range(i1, i2))
        changed_new.extend(range(j1, j2))
        if tag != "replace":
            real_old.extend(range(i1, i2))
            real_new.extend(range(j1, j2))
            continue
        block_pairs = _identifier_pairs(
            [t for ln in old_lines[i1:i2] for t in ln],
            [t for ln in new_lines[j1:j2] for t in ln],
        )
        if block_pairs is not None:
            candidates.append((list(range(i1, i2)), list(range(j1, j2)), block_pairs))
            constrain(block_pairs)
            continue
        inner = difflib.SequenceMatcher(
            None,
            [_shape_key(ln) for ln in old_lines[i1:i2]],
            [_shape_key(ln) for ln in new_lines[j1:j2]],
            autojunk=False,
        )
        for itag, a1, a2, b1, b2 in inner.get_opcodes():
            if itag != "equal":
                real_old.extend(range(i1 + a1, i1 + a2))
                real_new.extend(range(j1 + b1, j1 + b2))
                continue
            for k in range(a2 - a1):
                oi, ni = i1 + a1 + k, j1 + b1 + k
                pairs = _identifier_pairs(old_lines[oi], new_lines[ni])
                if pairs is None:  # unreachable (equal shapes), but stay safe
                    real_old.append(oi)
                    real_new.append(ni)
                    continue
                candidates.append(([oi], [ni], pairs))
                constrain(pairs)

    renamed: dict[str, str] = {}
    for olds, news, pairs in candidates:
        if all(len(forward[a]) == 1 and len(reverse[b]) == 1 for a, b in pairs):
            renamed.update({a: b for a, b in pairs if a != b})
        else:
            real_old.extend(olds)
            real_new.extend(news)

    changed_stmt_lines = len(set(real_old)) + len(set(real_new))
    changed_text = "\n".join(
        [_line_text(vulnerable_code, old_lines[i]) for i in changed_old]
        + [_line_text(fixed_code, new_lines[j]) for j in changed_new]
    )
    renamed_text = " ".join([*renamed, *renamed.values()])
    security_rename = bool(renamed) and (
        count_security_tokens(renamed_text, category) > 0
        or re.search("|".join(SECURITY_RENAME_MARKERS), renamed_text, re.I) is not None
    )
    return {
        "changed_stmt_lines": changed_stmt_lines,
        "rename_only": changed_stmt_lines == 0 and bool(renamed),
        "renamed_identifiers": dict(sorted(renamed.items())),
        "security_rename": security_rename,
        "security_tokens": count_security_tokens(changed_text, category),
    }


def commit_key(pair: dict) -> tuple[str, str]:
    return pair["repo"], pair["commit"]


def count_functions_per_commit(pairs: list[dict]) -> dict[tuple[str, str], int]:
    """Distinct ``(file_path, function_name)`` pairs each commit produced (so a
    commit shared by a GHSA and a PYSEC advisory is not double-counted)."""
    seen: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for p in pairs:
        seen[commit_key(p)].add((p["file_path"], p["function_name"]))
    return {k: len(v) for k, v in seen.items()}


def code_files_in_commit(commit_data: dict | None) -> int | None:
    """Number of supported-language files the commit touched (any status),
    excluding test/docs/example paths — the same files the pairing considers."""
    if not commit_data or commit_data.get("__status__") == 404:
        return None
    return sum(
        1
        for f in commit_data.get("files") or []
        if Path(f.get("filename", "")).suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_skipped_path(f.get("filename", ""))
    )


def annotate_quality(
    pairs: list[dict], files_in_commit: dict[tuple[str, str], int | None]
) -> list[dict]:
    """Return copies of ``pairs`` with a ``quality`` dict attached (the input
    dicts, which are also the resumable cache state, are left untouched)."""
    funcs = count_functions_per_commit(pairs)
    out = []
    for p in pairs:
        q = compute_diff_quality(
            p["vulnerable_code"], p["fixed_code"], p["language"], p["category"]
        )
        q["functions_in_commit"] = funcs[commit_key(p)]
        q["files_in_commit"] = files_in_commit.get(commit_key(p))
        out.append({**p, "quality": q})
    return out


def quality_reject_reasons(
    entry: dict,
    min_changed_stmt_lines: int,
    max_functions_per_commit: int,
    drop_other: bool,
    keep_security_renames: bool = False,
) -> list[str]:
    """Every filter an annotated entry fails, most specific first (empty = keep).
    ``max_functions_per_commit <= 0`` disables the broad-commit cap;
    ``keep_security_renames`` exempts rename-only pairs flagged
    ``security_rename`` from the changed-lines minimum."""
    q = entry["quality"]
    reasons = []
    exempt = keep_security_renames and q["rename_only"] and q["security_rename"]
    if q["changed_stmt_lines"] < min_changed_stmt_lines and not exempt:
        if q["rename_only"]:
            reasons.append("rename_only")
        elif q["changed_stmt_lines"] == 0:
            reasons.append("comment_or_format_only")
        else:
            reasons.append("few_changed_stmt_lines")
    if 0 < max_functions_per_commit < q["functions_in_commit"]:
        reasons.append("broad_commit")
    if drop_other and entry.get("category") == "other":
        reasons.append("other_category")
    return reasons


def apply_quality_filter(
    entries: list[dict],
    min_changed_stmt_lines: int,
    max_functions_per_commit: int,
    drop_other: bool,
    keep_security_renames: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Split annotated entries into ``(kept, rejected)``; rejected copies carry
    ``reject_reason`` (the first failing filter) and ``reject_reasons`` (all)."""
    kept, rejected = [], []
    for e in entries:
        reasons = quality_reject_reasons(
            e, min_changed_stmt_lines, max_functions_per_commit, drop_other,
            keep_security_renames,
        )
        if reasons:
            rejected.append({**e, "reject_reason": reasons[0], "reject_reasons": reasons})
        else:
            kept.append(e)
    return kept, rejected


def build_ecosystem_outputs(
    pairs: list[dict],
    files_in_commit: dict[tuple[str, str], int | None],
    *,
    eval_fraction: float,
    seed: int,
    min_changed_stmt_lines: int,
    max_functions_per_commit: int,
    drop_other: bool,
    keep_security_renames: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Annotate, split by advisory, then filter each side: returns
    ``(corpus_kept, eval_kept, rejected)``.

    The split runs on the *unfiltered* deduped pairs, exactly as before the
    filter existed, so which advisories are held out never depends on the
    filter flags (tuning them only removes items from either side, which keeps
    evals under different settings comparable). Rejected pairs from either
    side go only to ``rejected`` (tagged ``held_out``), never to the eval set.
    """
    annotated = annotate_quality(pairs, files_in_commit)
    corpus_side, eval_side = split_by_advisory(annotated, eval_fraction, seed)
    flt = dict(
        min_changed_stmt_lines=min_changed_stmt_lines,
        max_functions_per_commit=max_functions_per_commit,
        drop_other=drop_other,
        keep_security_renames=keep_security_renames,
    )
    corpus_kept, corpus_rej = apply_quality_filter(corpus_side, **flt)
    eval_kept, eval_rej = apply_quality_filter(eval_side, **flt)
    rejected = [{**e, "held_out": False} for e in corpus_rej] + [
        {**e, "held_out": True} for e in eval_rej
    ]
    return corpus_kept, eval_kept, rejected


def build_eval_lines(eval_pairs: list[dict]) -> list[dict]:
    """Two ``detection_eval.jsonl``-shaped lines (vulnerable + safe) per pair."""
    lines = []
    for e in eval_pairs:
        fn_token = re.sub(r"[^A-Za-z0-9_-]+", "_", e["function_name"])
        adv_token = re.sub(r"[^A-Za-z0-9_-]+", "_", e["cve_id"])
        base_id = f"{adv_token}_{fn_token}"
        for suffix, label, code in (
            ("vuln", "vulnerable", e["vulnerable_code"]),
            ("safe", "safe", e["fixed_code"]),
        ):
            lines.append(
                {
                    "id": f"{base_id}_{suffix}",
                    "language": e["language"],
                    "label": label,
                    "category": e["category"],
                    "expected_cve_id": e["cve_id"],
                    "source": "osv",
                    "code": code,
                }
            )
    return lines


def rejected_path_for(out_dir: Path, ecosystem: str) -> Path:
    """``{out_dir}/rejected/osv_{eco}_rejected.json`` — in a subdirectory on
    purpose: ``ingest_cve_corpus.load_cves`` ingests every ``{out_dir}/*.json``
    (non-recursive), so a rejected file next to the corpus would be ingested."""
    return Path(out_dir) / "rejected" / f"osv_{ecosystem.lower()}_rejected.json"


def write_ecosystem_outputs(
    out_dir: Path, ecosystem: str, corpus_kept: list[dict], rejected: list[dict]
) -> tuple[Path, Path]:
    """Write the kept corpus (``advisory_id`` stripped, as before) and the
    rejected audit file (``advisory_id`` kept). Returns both paths."""
    out_path = Path(out_dir) / f"osv_{ecosystem.lower()}.json"
    corpus_out = [{k: v for k, v in e.items() if k != "advisory_id"} for e in corpus_kept]
    out_path.write_text(json.dumps(corpus_out, indent=2), encoding="utf-8")
    rej_path = rejected_path_for(out_dir, ecosystem)
    rej_path.parent.mkdir(parents=True, exist_ok=True)
    rej_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    return out_path, rej_path


# ---------------------------------------------------------------------------
# Network / filesystem side (not unit-tested; exercised by the tiny live run).
# ---------------------------------------------------------------------------


class RateLimitExceeded(Exception):
    def __init__(self, reset_epoch: int | None):
        self.reset_epoch = reset_epoch
        msg = "GitHub API rate limit exhausted"
        if reset_epoch:
            msg += f"; resets at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reset_epoch))}"
        super().__init__(msg)


@dataclass
class Stats:
    advisories_scanned: int = 0
    advisories_with_fix: int = 0
    commits_fetched: int = 0
    functions_paired: int = 0
    by_language: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_calls: int = 0
    cache_hits: int = 0
    offline_misses: int = 0
    offline_incomplete_advisories: int = 0
    kept: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    kept_by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rate_remaining: int | None = None
    rate_reset: int | None = None


class GithubClient:
    """Thin GitHub REST client: caches every response by URL hash, and turns a
    used-up rate limit into either a sleep-and-retry (``wait_on_rate_limit``) or
    a clean ``RateLimitExceeded``. With ``offline=True`` it never touches the
    network: a cache miss returns ``None`` and is counted in
    ``stats.offline_misses``."""

    def __init__(
        self,
        session: requests.Session,
        token: str | None,
        cache_dir: Path,
        stats: Stats,
        wait_on_rate_limit: bool = False,
        offline: bool = False,
    ):
        self.session = session
        self.token = token
        self.cache_dir = Path(cache_dir) / "api"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = stats
        self.wait_on_rate_limit = wait_on_rate_limit
        self.offline = offline

    def _cache_file(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def get_cached_json(self, url: str) -> dict | None:
        """The cached response for ``url``, or ``None`` — never a network call."""
        cache_file = self._cache_file(url)
        if not cache_file.exists():
            return None
        return json.loads(cache_file.read_text(encoding="utf-8"))

    def get_json(self, url: str) -> dict | None:
        cache_file = self._cache_file(url)
        if cache_file.exists():
            self.stats.cache_hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))
        if self.offline:
            self.stats.offline_misses += 1
            return None

        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        while True:
            try:
                resp = self.session.get(url, headers=headers, timeout=30)
            except requests.RequestException as exc:
                print(f"  [warn] network error for {url}: {exc}")
                return None

            self.stats.api_calls += 1
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if remaining is not None:
                self.stats.rate_remaining = int(remaining)
            if reset is not None:
                self.stats.rate_reset = int(reset)

            if resp.status_code == 403 and remaining == "0":
                if self.wait_on_rate_limit and reset:
                    sleep_s = max(int(reset) - time.time(), 0) + 1
                    print(f"  [rate limit] sleeping {sleep_s:.0f}s until reset...")
                    time.sleep(sleep_s)
                    continue
                raise RateLimitExceeded(int(reset) if reset else None)
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

    def get_file(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/"
            f"{quote(path, safe='/')}?ref={ref}"
        )
        data = self.get_json(url)
        if not data or data.get("__status__") == 404:
            return None
        content = data.get("content")
        if not content or data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return None


def download_osv_zip(
    ecosystem: str, cache_dir: Path, session: requests.Session, offline: bool = False
) -> Path:
    zip_path = Path(cache_dir) / f"{ecosystem}_all.zip"
    if zip_path.exists():
        print(f"Using cached {zip_path}")
        return zip_path
    if offline:
        sys.exit(f"--offline: {zip_path} is not cached; run once without --offline first.")
    url = OSV_ZIP_URL_TEMPLATE.format(ecosystem=ecosystem)
    print(f"Downloading {url} ...")
    resp = session.get(url, timeout=180, stream=True)
    resp.raise_for_status()
    tmp_path = zip_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    tmp_path.rename(zip_path)
    print(f"  saved to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def iter_osv_advisories(zip_path: Path):
    """Yield each advisory dict from an OSV bulk-export zip. Handles both the
    observed "one JSON file per advisory" layout and a single combined-list
    JSON file, in case that ever changes."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            with zf.open(name) as fh:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                yield data


def _is_skipped_path(path: str) -> bool:
    lower = path.lower()
    if any(s in lower for s in SKIP_PATH_SUBSTRINGS):
        return True
    return lower.startswith(SKIP_PATH_PREFIXES)


def pair_functions_in_file(
    parser: CodeParser,
    pre_content: str,
    post_content: str,
    ext: str,
    hunk_ranges: list[tuple[int, int]],
) -> list[dict]:
    """Pair pre/post functions by name (the parser doesn't expose an enclosing
    class, so name-only) and keep only pairs whose body changed for real
    (non-whitespace) and whose post-commit range overlaps the diff hunks. A
    duplicate name on either side (e.g. same-named methods in different
    classes) is ambiguous with a name-only key and is skipped rather than
    risking a mis-pair; a name with no counterpart on the other side (a
    rename/move) is likewise skipped, never mis-paired."""
    language = SUPPORTED_EXTENSIONS[ext]
    pre_funcs = parser.extract_functions(pre_content, ext)
    post_funcs = parser.extract_functions(post_content, ext)

    pre_by_name: dict[str, list[dict]] = defaultdict(list)
    for f in pre_funcs:
        pre_by_name[f["name"]].append(f)

    pairs = []
    for post_f in post_funcs:
        candidates = pre_by_name.get(post_f["name"])
        if not candidates or len(candidates) != 1:
            continue
        pre_f = candidates[0]
        if pre_f["code"] == post_f["code"]:
            continue
        if not function_overlaps_hunks(post_f["start_line"], post_f["end_line"], hunk_ranges):
            continue
        if is_whitespace_only_change(pre_f["code"], post_f["code"], language):
            continue
        pairs.append({"pre": pre_f, "post": post_f, "language": language})
    return pairs


def print_summary(stats: Stats, rate_limited: bool) -> None:
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Advisories scanned:           {stats.advisories_scanned}")
    print(f"Advisories with usable fix:   {stats.advisories_with_fix}")
    print(f"Commits fetched:               {stats.commits_fetched}")
    print(f"Functions paired (deduped):   {stats.functions_paired}")

    print("\nBy language (paired):")
    for lang, count in sorted(stats.by_language.items()):
        print(f"  {lang:12s} {count}")

    print("\nBy category (paired, before the quality filter):")
    for cat, count in sorted(stats.by_category.items()):
        print(f"  {cat:20s} {count}")

    rejected = sum(stats.rejected_by_reason.values())
    print(f"\nQuality filter: kept {stats.kept}, rejected {rejected}")
    for reason, count in sorted(stats.rejected_by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {reason:24s} {count}")

    print("\nBy category (kept):")
    for cat, count in sorted(stats.kept_by_category.items()):
        print(f"  {cat:20s} {count}")
    if stats.kept:
        other = stats.kept_by_category.get("other", 0)
        print(f"  'other' share: {other}/{stats.kept} ({other / stats.kept:.1%})")

    print(f"\nGitHub API calls made:         {stats.api_calls}")
    print(f"Cache hits (no network call):  {stats.cache_hits}")
    if stats.offline_misses or stats.offline_incomplete_advisories:
        print(f"Offline cache misses:          {stats.offline_misses}")
        print(f"Advisories skipped (incomplete cache): {stats.offline_incomplete_advisories}")
    if stats.rate_remaining is not None:
        print(f"Rate limit remaining:          {stats.rate_remaining}")
    if stats.rate_reset is not None:
        reset_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.rate_reset))
        print(f"Rate limit resets at:          {reset_str}")
    if rate_limited:
        print(
            "\nStopped early: GitHub API rate limit exhausted. Every response fetched so\n"
            "far is cached, so re-running with --resume (ideally with GITHUB_TOKEN set)\n"
            "will continue without re-fetching anything already fetched."
        )
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a paired vulnerable/fixed function corpus from OSV advisories "
        "and their GitHub fix commits."
    )
    p.add_argument(
        "--ecosystem",
        action="append",
        choices=["PyPI", "npm"],
        help="OSV ecosystem to mine (repeatable). Default: both PyPI and npm.",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Corpus output directory.")
    p.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Eval-set output directory.")
    p.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE_DIR), help="OSV zip + GitHub API response cache."
    )
    p.add_argument(
        "--max-advisories",
        type=int,
        default=200,
        help="Upper bound on advisories with a parseable fix reference actually fetched "
        "(hard cap on GitHub API usage).",
    )
    p.add_argument("--eval-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-lines", type=int, default=3)
    p.add_argument("--max-lines", type=int, default=120)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip advisories already fully processed in a previous run (from --cache-dir's "
        "processed_advisories.json) instead of reprocessing them.",
    )
    p.add_argument(
        "--wait-on-rate-limit",
        action="store_true",
        help="Sleep until the GitHub rate limit resets instead of stopping cleanly.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Never touch the network: read the OSV zip and GitHub responses from "
        "--cache-dir only (a miss skips that advisory). With --resume, this re-filters "
        "a previous run's pairs with zero network calls.",
    )
    p.add_argument(
        "--min-changed-stmt-lines",
        type=int,
        default=1,
        help="Reject pairs whose diff changes fewer statement lines than this, after "
        "dropping blank/comment/docstring lines and consistent identifier renames "
        "(default 1 rejects rename-only and comment/format-only pairs; 0 disables).",
    )
    p.add_argument(
        "--max-functions-per-commit",
        type=int,
        default=6,
        help="Reject every pair from a commit that produced more paired functions than "
        "this (broad refactors; default 6; 0 disables).",
    )
    p.add_argument(
        "--drop-other",
        action="store_true",
        help="Also reject pairs whose category is 'other' (no or unmapped CWE).",
    )
    p.add_argument(
        "--keep-security-renames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rename-only pairs whose renamed identifier looks security-relevant "
        "(quality.security_rename, e.g. md5 -> sha256, load -> safe_load) instead of "
        "rejecting them as rename_only. On by default; --no-keep-security-renames "
        "to reject them too.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    eval_dir = Path(args.eval_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    token = settings.GITHUB_TOKEN
    if not token and not args.offline:
        print(
            "No GITHUB_TOKEN configured — using unauthenticated GitHub API access "
            "(60 requests/hour). Set GITHUB_TOKEN in .env for a much bigger budget."
        )

    ecosystems = args.ecosystem or ["PyPI", "npm"]
    session = requests.Session()
    stats = Stats()
    parser = CodeParser()
    client = GithubClient(
        session, token, cache_dir, stats, args.wait_on_rate_limit, offline=args.offline
    )

    state_path = cache_dir / "processed_advisories.json"
    processed_state: dict[str, dict] = {}
    if state_path.exists():
        try:
            processed_state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            processed_state = {}

    all_eval_lines: list[dict] = []
    advisories_attempted = 0
    rate_limited = False

    for ecosystem in ecosystems:
        if rate_limited:
            break

        zip_path = download_osv_zip(ecosystem, cache_dir, session, offline=args.offline)
        ecosystem_pairs: list[dict] = []
        # (repo, commit) -> code files the commit touched; from the cached commit
        # response, so resumed advisories need no network call either.
        files_in_commit: dict[tuple[str, str], int | None] = {}

        for advisory in iter_osv_advisories(zip_path):
            if advisories_attempted >= args.max_advisories:
                print(f"Reached --max-advisories={args.max_advisories}, stopping intake.")
                break

            adv_id = advisory.get("id")
            if not adv_id:
                continue
            stats.advisories_scanned += 1

            if args.resume and adv_id in processed_state:
                cached_pairs = processed_state[adv_id].get("pairs", [])
                if cached_pairs:
                    stats.advisories_with_fix += 1
                for key in {commit_key(p) for p in cached_pairs} - files_in_commit.keys():
                    commit_url = f"https://api.github.com/repos/{key[0]}/commits/{key[1]}"
                    files_in_commit[key] = code_files_in_commit(
                        client.get_cached_json(commit_url)
                    )
                ecosystem_pairs.extend(cached_pairs)
                continue

            ref = select_fix_reference(advisory)
            if ref is None:
                continue
            owner, repo, sha = ref
            advisories_attempted += 1
            misses_before = stats.offline_misses

            try:
                commit_data = client.get_json(
                    f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
                )
            except RateLimitExceeded as exc:
                print(
                    f"\n[rate limit] {exc}. Stopping cleanly after {stats.api_calls} API "
                    f"calls ({advisories_attempted} advisories attempted this run)."
                )
                rate_limited = True
                break

            if not commit_data or commit_data.get("__status__") == 404:
                continue

            stats.advisories_with_fix += 1
            stats.commits_fetched += 1
            files_in_commit[(f"{owner}/{repo}", sha)] = code_files_in_commit(commit_data)

            parents = commit_data.get("parents") or []
            if not parents:
                continue
            parent_sha = parents[0]["sha"]

            cwe_ids = (advisory.get("database_specific") or {}).get("cwe_ids") or []
            category = cwe_to_category(cwe_ids)
            cwe_id_out = cwe_ids[0] if cwe_ids else None
            severity = parse_cvss_score(advisory.get("severity"))
            description = get_description(advisory)
            display_id = get_display_id(advisory)

            advisory_pairs: list[dict] = []
            try:
                for file_entry in commit_data.get("files") or []:
                    if file_entry.get("status") != "modified":
                        continue
                    path = file_entry.get("filename", "")
                    ext = Path(path).suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS or _is_skipped_path(path):
                        continue
                    patch = file_entry.get("patch")
                    if not patch:
                        continue
                    hunk_ranges = parse_hunk_ranges(patch)
                    if not hunk_ranges:
                        continue

                    post_content = client.get_file(owner, repo, path, sha)
                    pre_content = client.get_file(owner, repo, path, parent_sha)
                    if post_content is None or pre_content is None:
                        continue

                    for pair in pair_functions_in_file(
                        parser, pre_content, post_content, ext, hunk_ranges
                    ):
                        pre_f, post_f = pair["pre"], pair["post"]
                        line_count = len(pre_f["code"].splitlines())
                        if not (args.min_lines <= line_count <= args.max_lines):
                            continue
                        advisory_pairs.append(
                            {
                                "cve_id": display_id,
                                "category": category,
                                "cwe_id": cwe_id_out,
                                "description": description,
                                "severity": severity,
                                "language": pair["language"],
                                "vulnerable_code": pre_f["code"],
                                "fixed_code": post_f["code"],
                                "source": "osv",
                                "repo": f"{owner}/{repo}",
                                "commit": sha,
                                "file_path": path,
                                "function_name": post_f["name"],
                                "advisory_id": adv_id,
                            }
                        )
            except RateLimitExceeded as exc:
                print(f"\n[rate limit] {exc}. Stopping cleanly after {stats.api_calls} API calls.")
                rate_limited = True
                break

            if stats.offline_misses > misses_before:
                # Some file was missing from the cache: the pairs are incomplete, so
                # neither use them nor record the advisory as processed (an online
                # --resume must still fetch it).
                stats.offline_incomplete_advisories += 1
                continue
            processed_state[adv_id] = {"pairs": advisory_pairs}
            ecosystem_pairs.extend(advisory_pairs)

        # Persist resumable state after every ecosystem (and on an early rate-limit exit).
        state_path.write_text(json.dumps(processed_state, indent=2), encoding="utf-8")

        # Global dedupe within this ecosystem's pairs.
        seen_keys: set[str] = set()
        deduped_pairs: list[dict] = []
        for p in ecosystem_pairs:
            key = make_dedupe_key(
                p["repo"], p["file_path"], p["function_name"], p["vulnerable_code"]
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_pairs.append(p)

        for p in deduped_pairs:
            stats.functions_paired += 1
            stats.by_language[p["language"]] += 1
            stats.by_category[p["category"]] += 1

        if not deduped_pairs:
            print(f"No pairs produced for ecosystem {ecosystem}.")
            continue

        corpus_kept, eval_kept, rejected = build_ecosystem_outputs(
            deduped_pairs,
            files_in_commit,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            min_changed_stmt_lines=args.min_changed_stmt_lines,
            max_functions_per_commit=args.max_functions_per_commit,
            drop_other=args.drop_other,
            keep_security_renames=args.keep_security_renames,
        )
        for e in corpus_kept + eval_kept:
            stats.kept += 1
            stats.kept_by_category[e["category"]] += 1
        for e in rejected:
            stats.rejected_by_reason[e["reject_reason"]] += 1

        out_path, rej_path = write_ecosystem_outputs(out_dir, ecosystem, corpus_kept, rejected)
        print(f"Wrote {len(corpus_kept)} corpus entries to {out_path}")
        print(f"Wrote {len(rejected)} rejected pairs (with reject_reason) to {rej_path}")

        all_eval_lines.extend(build_eval_lines(eval_kept))

    if all_eval_lines:
        eval_path = eval_dir / "detection_eval_osv.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for line in all_eval_lines:
                f.write(json.dumps(line) + "\n")
        print(f"Wrote {len(all_eval_lines)} eval lines to {eval_path}")

    print_summary(stats, rate_limited)


if __name__ == "__main__":
    main()
