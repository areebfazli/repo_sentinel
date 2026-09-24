"""Handling of untrusted text on the way into and out of the LLM (ROADMAP 7).

Everything the LLM sees except our own instructions is attacker-influenceable:
the PR's code and file names, and the third-party corpus (OSV fix commits'
``vulnerable_code`` / ``fixed_code``, advisory descriptions, team review text).
Everything the LLM writes ends up in PR comments. So:

- **In:** each untrusted block is wrapped in a tag carrying a per-prompt random
  nonce (``<untrusted_<nonce> kind=...>`` ... ``</untrusted_<nonce>>``), which the
  text inside can't forge because it can't know the nonce. Before wrapping, text
  is neutralised (``sanitize_untrusted``) WITHOUT deleting anything: HTML-comment
  openers (which hide text in rendered Markdown) are defused in place,
  zero-width / bidi control characters (Trojan Source) become visible
  placeholders, runs of three or more backticks can't open or close a Markdown
  fence, and anything that looks like one of our tags is defused. The code the
  model reviews is exactly the PR's code, and line count is preserved, so line
  numbers map 1:1 to the real file. Only an explicit ``max_chars`` cap (used
  for short evidence snippets, never for the code under review) cuts text, and
  it says so.
- **Out:** LLM text fields are cleaned (``clean_llm_text``: zero-width / control
  characters removed, length-capped; prose fields also lose complete HTML
  comments, code fields keep everything) and rendered through ``md_inline`` /
  ``md_code_span`` / ``md_code_block``, which escape Markdown and HTML (code
  spans / blocks defuse ``<!--``) so a finding can't inject links, images, raw
  HTML, @-mentions or ``<!-- reposentinel:... -->`` dedupe markers into a PR
  comment.

Pure functions; ``github_action/scan_pr.py`` carries its own copy of the output
escaping (it can't import the backend).
"""
import html
import re
import secrets

# Zero-width and invisible formatting characters, and the Unicode bidi controls
# used by "Trojan Source" (CVE-2021-42574). Replaced by a visible [U+XXXX] in
# prompts (their presence in code is itself suspicious) and removed from output.
_INVISIBLE = (
    "\u00ad\u061c\u180e\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069\ufeff"
)
_INVISIBLE_RE = re.compile(f"[{_INVISIBLE}]")
# C0/C1 controls except tab / newline / carriage return.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Characters str.splitlines() treats as line breaks although "\n"-counting
# (tree-sitter, GitHub, editors) does not: U+2028 / U+2029 and a lone "\r".
# In prompts they become visible placeholders, so a unit's line count - and so
# every line number shown to the model - matches the real file.
_EXTRA_BREAK_RE = re.compile(r"[\u2028\u2029]|\r(?!\n)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE_RUN_RE = re.compile(r"`{3,}")
_TILDE_FENCE_RE = re.compile(r"~{3,}")
# Anything shaped like our block tags (open or close, any nonce).
_TAG_LIKE_RE = re.compile(r"<\s*(/?)\s*(untrusted)", re.IGNORECASE)

# An HTML-comment opener, defused: nothing after it can be hidden by a Markdown
# renderer, and nothing is deleted (the text stays readable as-is).
COMMENT_OPEN = "<!--"
COMMENT_OPEN_DEFUSED = "&lt;!--"
TRUNCATION_NOTE = "[... truncated: {n} more characters not shown]"
# Stand-ins sanitize_untrusted puts in place of fence characters.
FENCE_BACKTICK = "\u02cb"  # modifier letter grave: looks like ` but never forms a fence
FENCE_TILDE = "\u02dc"  # small tilde


def new_nonce() -> str:
    """A fresh per-prompt tag token (64 random bits, hex)."""
    return secrets.token_hex(8)


def _visible_codepoint(match: re.Match) -> str:
    return f"[U+{ord(match.group(0)):04X}]"


def defuse_html_comments(text: str) -> str:
    """Neutralise every HTML-comment opener in place (``<!--`` -> ``&lt;!--``):
    nothing can be hidden behind it and nothing is removed."""
    return text.replace(COMMENT_OPEN, COMMENT_OPEN_DEFUSED)


def sanitize_untrusted(text: str | None, max_chars: int | None = None) -> str:
    """Neutralise untrusted text (code or prose) before it goes into a prompt.

    Nothing is ever deleted: every transform is a visible, 1:1 substitution
    that keeps the line structure, so code the model reviews is exactly the
    code in the PR, and line N of the output is line N of the input:

    - HTML-comment openers are defused (``&lt;!--``), not stripped: stripping
      would hide the code between ``# <!--`` and ``# -->`` (or, with an
      unclosed ``"<!--"`` string literal, the rest of the function) from the
      reviewer. A stray ``-->`` is harmless and left alone.
    - zero-width / bidi / control characters and the extra line breaks
      ``str.splitlines`` knows about become ``[U+XXXX]``;
    - runs of 3+ backticks / tildes use look-alike characters, so they can't
      open or close a Markdown fence;
    - anything shaped like our block tags is defused (``&lt;untrusted``).

    Only the ``max_chars`` cap removes text, at a line boundary, and it says
    so (``TRUNCATION_NOTE``). ``match_form`` undoes these substitutions when a
    finding's quote is checked against the code.
    """
    text = text or ""
    text = defuse_html_comments(text)
    text = _INVISIBLE_RE.sub(_visible_codepoint, text)
    text = _CONTROL_RE.sub(_visible_codepoint, text)
    text = _EXTRA_BREAK_RE.sub(_visible_codepoint, text)
    text = _FENCE_RUN_RE.sub(lambda m: FENCE_BACKTICK * len(m.group(0)), text)
    text = _TILDE_FENCE_RE.sub(lambda m: FENCE_TILDE * len(m.group(0)), text)
    text = _TAG_LIKE_RE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)
    if max_chars is not None and len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        cut = cut if cut > max_chars // 2 else max_chars
        text = text[:cut] + "\n" + TRUNCATION_NOTE.format(n=len(text) - cut)
    return text


_PLACEHOLDER_RE = re.compile(r"\[U\+[0-9A-F]{4,6}\]")
_WHITESPACE_RE = re.compile(r"\s+")


def match_form(text: str) -> str:
    """Canonical form for comparing an LLM's quote with the code it reviewed,
    whichever side of ``sanitize_untrusted`` either one comes from: fence
    look-alikes back to backticks / tildes, HTML entities (``&lt;!--``,
    ``&lt;untrusted``) unescaped, ``[U+XXXX]`` placeholders and the invisible /
    control characters they stand for dropped, and ALL whitespace removed (so
    re-indented or re-wrapped statements compare equal)."""
    text = (text or "").replace(FENCE_BACKTICK, "`").replace(FENCE_TILDE, "~")
    text = html.unescape(text)
    text = _PLACEHOLDER_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _EXTRA_BREAK_RE.sub("", text)
    return _WHITESPACE_RE.sub("", text)


def safe_label(value, max_chars: int = 160) -> str:
    """A short single-line label from untrusted input (file path, function name,
    CVE id, category) for use OUTSIDE untrusted blocks: no newlines, no tag or
    fence characters, capped."""
    text = sanitize_untrusted(str(value if value is not None else ""))
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w./\-+@:$#<>\[\] ]", "_", text).replace("<", "‹").replace(">", "›")
    return text[:max_chars]


def wrap_untrusted(nonce: str, kind: str, text: str | None, max_chars: int | None = None,
                   **attrs) -> str:
    """``text`` (sanitised) inside a nonce-tagged block. The nonce is also
    scrubbed from the text, so even a leaked nonce can't close the block."""
    body = sanitize_untrusted(text, max_chars).replace(nonce, "[nonce]")
    attr_text = "".join(f' {k}="{safe_label(v, 60)}"' for k, v in attrs.items() if v is not None)
    return f'<untrusted_{nonce} kind="{kind}"{attr_text}>\n{body}\n</untrusted_{nonce}>'


# ---------------------------------------------------------------------------
# LLM output -> Markdown
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:\b[a-zA-Z][a-zA-Z0-9+.\-]{1,15}://|\bwww\.)[^\s<>]+")
_MENTION_RE = re.compile(r"(?<![\w`])@(?=[A-Za-z0-9])")
# ASCII punctuation that can start Markdown syntax inline (CommonMark allows a
# backslash escape before any ASCII punctuation).
_MD_SPECIAL_RE = re.compile(r"([\\`*_\[\]()!#|~{}])")
_BACKTICKS_RE = re.compile(r"`+")


def clean_llm_text(value, max_chars: int, *, code: bool = False) -> str:
    """An LLM output field as plain text: zero-width / bidi and control
    characters removed, capped. Prose fields (``code=False``) also lose complete
    HTML comments (hidden text has no place in a title or explanation; a
    dangling ``<!--`` stays and is escaped when rendered). Code fields
    (``quoted_code``, ``fix_snippet``) keep every character: the renderers
    defuse ``<!--`` in code spans and blocks instead. Still to be escaped when
    rendered."""
    if value is None:
        return ""
    text = str(value)
    if not code:
        text = _HTML_COMMENT_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _escape_plain(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _MD_SPECIAL_RE.sub(r"\\\1", text)
    return _MENTION_RE.sub("@​", text)


def md_code_span(text: str) -> str:
    """Inline code that can't be broken out of: the delimiter is longer than any
    backtick run inside; newlines become spaces; an HTML-comment opener is
    defused (a code span shows it literally, but raw-body scanners such as the
    Action's marker parser would see it)."""
    text = re.sub(r"\s*\n\s*", " ", text or "").replace(COMMENT_OPEN, "<! --")
    longest = max((len(m.group(0)) for m in _BACKTICKS_RE.finditer(text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") or not text else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def md_inline(text: str) -> str:
    """Untrusted text as one line of inert Markdown: HTML and Markdown syntax
    escaped, URLs shown as code (not links), @-mentions defused, newlines
    collapsed (no headings / lists / block quotes can start)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    out: list[str] = []
    pos = 0
    for m in _URL_RE.finditer(text):
        out.append(_escape_plain(text[pos:m.start()]))
        out.append(md_code_span(m.group(0)))
        pos = m.end()
    out.append(_escape_plain(text[pos:]))
    rendered = "".join(out)
    # A leading "-", "+" or "1." would start a list item.
    return re.sub(r"^([-+]|\d+\.)", r"\\\1", rendered)


def md_code_block(text: str, indent: str = "") -> list[str]:
    """Fenced code block lines; the fence is longer than any backtick run in the
    text, and an HTML-comment opener is defused (a code block never renders it,
    but raw-body scanners such as the Action's marker parser would see it)."""
    text = (text or "").replace(COMMENT_OPEN, "<! --")
    longest = max((len(m.group(0)) for m in _BACKTICKS_RE.finditer(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{indent}{fence}", *(f"{indent}{ln}" for ln in text.splitlines()), f"{indent}{fence}"]
