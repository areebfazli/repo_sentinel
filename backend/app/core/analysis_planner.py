"""Turn changed files into per-function analysis units.

For each file we extract functions (tree-sitter), keep only those overlapping the
changed lines (explicit ``changed_lines`` wins over ``patch``; neither -> all
functions), skip any marked ``# reposentinel-ignore``, and cap the total. Files in
an unsupported language become a single whole-file unit.
"""
from pathlib import Path

from backend.app.core.diff_utils import parse_patch_changed_lines

IGNORE_DIRECTIVE = "reposentinel-ignore"

# The parser reports a short language (ext without the dot, e.g. "py"); the vector
# corpus stores full names ("python"). Normalize so the language filter matches.
_LANGUAGE_NAMES = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "go": "go",
    "java": "java",
}


def _language_name(short: str) -> str:
    return _LANGUAGE_NAMES.get(short, short)


def _changed_line_set(file) -> set[int] | None:
    """Explicit changed_lines, else parsed patch, else None (= analyze all functions)."""
    if getattr(file, "changed_lines", None):
        return set(file.changed_lines)
    if getattr(file, "patch", None):
        return set(parse_patch_changed_lines(file.patch))
    return None


def _overlaps(start_line: int, end_line: int, changed: set[int]) -> bool:
    return not changed.isdisjoint(range(start_line, end_line + 1))


def _has_ignore(func: dict, source_lines: list[str]) -> bool:
    # Directive inside the function, or on the line immediately above it.
    if IGNORE_DIRECTIVE in func["code"]:
        return True
    above_idx = func["start_line"] - 2  # 0-based line above start_line
    if 0 <= above_idx < len(source_lines) and IGNORE_DIRECTIVE in source_lines[above_idx]:
        return True
    return False


def plan_units(files, parser, max_units: int) -> tuple[list[dict], int]:
    """Return (units, dropped) where dropped is how many were cut by the cap."""
    units: list[dict] = []

    for file in files:
        ext = Path(file.path).suffix.lower()

        if not parser.supports(ext):
            # Unsupported language -> one whole-file unit (unless ignored).
            if IGNORE_DIRECTIVE not in file.content:
                units.append(
                    {
                        "file_path": file.path,
                        "function_name": None,
                        "start_line": 1,
                        "end_line": len(file.content.splitlines()) or 1,
                        "code": file.content,
                        "language": ext.lstrip(".") or "text",
                    }
                )
            continue

        changed = _changed_line_set(file)
        source_lines = file.content.splitlines()
        for func in parser.extract_functions(file.content, ext):
            if changed is not None and not _overlaps(func["start_line"], func["end_line"], changed):
                continue
            if _has_ignore(func, source_lines):
                continue
            units.append(
                {
                    "file_path": file.path,
                    "function_name": func["name"],
                    "start_line": func["start_line"],
                    "end_line": func["end_line"],
                    "code": func["code"],
                    "language": _language_name(func["language"]),
                }
            )

    dropped = 0
    if len(units) > max_units:
        dropped = len(units) - max_units
        units = units[:max_units]
    return units, dropped
