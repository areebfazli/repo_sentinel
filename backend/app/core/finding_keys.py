"""Stable identities of report findings (the Action's comment dedupe keys).

Shared by the per-unit review (``services.scan_runner``) and the PR-level
review (``core.pr_review``) so a finding on the same line keeps the same key
(and so the same inline comment) whichever review mode produced it. Pure.
"""
import hashlib

from backend.app.core.markdown_renderer import SEVERITY_ORDER
from backend.app.core.untrusted import match_form


def dedupe_key(prefix: str, *parts) -> str:
    """Stable id of a report finding within its function, for the Action's
    comment markers (which also hash file + function)."""
    text = "|".join(" ".join(str(p or "").split()) for p in parts)
    return f"{prefix}:{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def anchored_line_text(unit: dict, line: int | None) -> str:
    """The unit's code line at real line ``line`` (as shown to the model), or ""."""
    if line is None:
        return ""
    lines = (unit.get("prompt_code") or "").splitlines()
    numbers = unit.get("line_numbers")
    if numbers:
        idx = numbers.index(line) if line in numbers else -1
    else:
        idx = line - int(unit.get("start_line") or 1)
    return lines[idx] if 0 <= idx < len(lines) else ""


def llm_dedupe_key(unit: dict, finding: dict) -> str:
    """Comment identity of an LLM finding: the source plus the code line it is
    anchored to (``untrusted.match_form``: whitespace-insensitive), falling back
    to the quote's first line. Only stable fields: the Action's marker adds the
    file path and function name, and nothing the LLM words differently from run
    to run (title, CWE, explanation) is part of it, so a re-run with the same
    code updates the same comment. Survives line shifts and re-indentation;
    changes only when the offending line itself changes.

    ``unit`` is anything with ``prompt_code`` / ``start_line`` (and optionally
    ``line_numbers``): a review unit, or a whole new file starting at line 1."""
    text = anchored_line_text(unit, finding.get("line"))
    if not match_form(text):
        text = ((finding.get("quoted_code") or "").strip().splitlines() or [""])[0]
    return dedupe_key("llm", match_form(text))


def legacy_llm_dedupe_key(finding: dict) -> str:
    """The key older servers used (quote's first line + CWE or title). Sent as
    ``legacy_dedupe_keys`` so the Action can adopt comments posted by them
    instead of deleting and re-posting them."""
    quote_first = (finding.get("quoted_code") or "").strip().splitlines()[:1]
    return dedupe_key("llm", quote_first[0] if quote_first else "",
                      finding.get("cwe") or finding.get("title"))


def disambiguate_dedupe_keys(findings: list[dict]) -> None:
    """Two findings on the same line of the same function (say SQLi and XSS)
    share a key; number the later ones ("#2", ...) in a stable order
    (severity, CWE, title), so each keeps its own comment."""
    groups: dict[tuple, list[dict]] = {}
    for f in findings:
        groups.setdefault((f.get("file_path"), f.get("function_name"), f.get("dedupe_key")),
                          []).append(f)
    for group in groups.values():
        group.sort(key=lambda f: (SEVERITY_ORDER.get(f.get("severity"), 4), f.get("cwe") or "",
                                  f.get("title") or ""))
        for n, f in enumerate(group[1:], start=2):
            f["dedupe_key"] = f"{f['dedupe_key']}#{n}"
