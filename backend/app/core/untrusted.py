"""Handling of untrusted text on the way into and out of the LLM (ROADMAP 7).

Everything the LLM sees except our own instructions is attacker-influenceable:
the PR's code and file names, and the third-party corpus (OSV fix commits'
``vulnerable_code`` / ``fixed_code``, advisory descriptions, team review text).
Everything the LLM writes ends up in PR comments. So:

- **In:** each untrusted block is wrapped in a tag carrying a per-prompt random
  nonce (``<untrusted_<nonce> kind=...>`` ... ``</untrusted_<nonce>>``), which the
  text inside can't forge because it can't know the nonce. Before wrapping, text
  is neutralised (``sanitize_untrusted``): HTML comments (hidden in rendered
  Markdown, a classic carrier for injected instructions) and zero-width / bidi
  control characters (Trojan Source) are replaced by visible placeholders, runs
  of three or more backticks can't open or close a Markdown fence, anything that
  looks like one of our tags is defused, and the block is length-capped. Line
  count is preserved, so line numbers still map 1:1 to the real code.
- **Out:** LLM text fields are cleaned (``clean_llm_text``: comments and control
  characters stripped, length-capped) and rendered through ``md_inline`` /
  ``md_code_span`` / ``md_code_block``, which escape Markdown and HTML so a
  finding can't inject links, images, raw HTML, @-mentions or
  ``<!-- reposentinel:... -->`` dedupe markers into a PR comment.

Pure functions; ``github_action/scan_pr.py`` carries its own copy of the output
escaping (it can't import the backend).
"""
import re
import secrets

# Zero-width and invisible formatting characters, and the Unicode bidi controls
# used by "Trojan Source" (CVE-2021-42574). Replaced by a visible [U+XXXX] in
# prompts (their presence in code is itself suspicious) and removed from output.
_INVISIBLE = (
    "­؜᠎​‌‍‎‏‪‫‬‭‮"
    "⁠⁡⁢⁣⁤⁦⁧⁨⁩﻿"
)
_INVISIBLE_RE = re.compile(f"[{_INVISIBLE}]")
# C0/C1 controls except tab / newline / carriage return.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_FENCE_RUN_RE = re.compile(r"`{3,}")
_TILDE_FENCE_RE = re.compile(r"~{3,}")
# Anything shaped like our block tags (open or close, any nonce).
_TAG_LIKE_RE = re.compile(r"<\s*(/?)\s*(untrusted)", re.IGNORECASE)

HTML_COMMENT_PLACEHOLDER = "[html comment removed]"
TRUNCATION_NOTE = "[... truncated: {n} more characters not shown]"


def new_nonce() -> str:
    """A fresh per-prompt tag token (64 random bits, hex)."""
    return secrets.token_hex(8)


def _visible_codepoint(match: re.Match) -> str:
    return f"[U+{ord(match.group(0)):04X}]"


def _comment_placeholder(match: re.Match) -> str:
    # Keep the comment's newlines so line numbers after it stay correct.
    return HTML_COMMENT_PLACEHOLDER + "\n" * match.group(0).count("\n")


def sanitize_untrusted(text: str | None, max_chars: int | None = None) -> str:
    """Neutralise untrusted text before it goes into a prompt. Line count is
    preserved (except by the ``max_chars`` cap, which cuts at a line boundary
    and says so)."""
    text = text or ""
    text = _HTML_COMMENT_RE.sub(_comment_placeholder, text)
    text = _INVISIBLE_RE.sub(_visible_codepoint, text)
    text = _CONTROL_RE.sub(_visible_codepoint, text)
    # U+02CB (modifier letter grave) looks like a backtick but never forms a fence.
    text = _FENCE_RUN_RE.sub(lambda m: "ˋ" * len(m.group(0)), text)
    text = _TILDE_FENCE_RE.sub(lambda m: "˜" * len(m.group(0)), text)
    text = _TAG_LIKE_RE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)
    if max_chars is not None and len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        cut = cut if cut > max_chars // 2 else max_chars
        text = text[:cut] + "\n" + TRUNCATION_NOTE.format(n=len(text) - cut)
    return text


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


def clean_llm_text(value, max_chars: int) -> str:
    """An LLM output field as plain text: HTML comments, zero-width / bidi and
    control characters removed, capped. Still to be escaped when rendered."""
    if value is None:
        return ""
    text = str(value)
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
    backtick run inside; newlines become spaces."""
    text = re.sub(r"\s*\n\s*", " ", text or "")
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
    text = (text or "").replace("<!--", "<! --")
    longest = max((len(m.group(0)) for m in _BACKTICKS_RE.finditer(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{indent}{fence}", *(f"{indent}{ln}" for ln in text.splitlines()), f"{indent}{fence}"]
