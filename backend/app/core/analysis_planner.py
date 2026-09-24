"""Turn changed files into per-function analysis units.

For each file we extract functions (tree-sitter), keep only those overlapping the
changed lines (explicit ``changed_lines`` wins over ``patch``; neither -> all
functions), skip any marked ``# reposentinel-ignore``, and cap the total. Files in
an unsupported language become a single whole-file unit. Each unit carries its
``changed_lines`` (None without change information).
"""
from pathlib import Path

from backend.app.core.diff_utils import parse_patch_changed_lines

IGNORE_DIRECTIVE = "reposentinel-ignore"

# The parser reports a short language (ext without the dot, e.g. "py"); the vector
# corpus stores full names. TypeScript is parsed with the JS grammar and the corpus
# labels those entries "javascript", so .ts maps to javascript (not typescript,
# which the corpus never contains).
_LANGUAGE_NAMES = {
    "py": "python",
    "js": "javascript",
    "ts": "javascript",
    "go": "go",
    "java": "java",
}


def _language_name(short: str) -> str:
    return _LANGUAGE_NAMES.get(short, short)


def _changed_line_set(file) -> set[int] | None:
    """Explicit changed_lines wins (even when empty), else parsed patch, else None
    (= analyze all functions). An explicit empty list means 'no changed lines',
    NOT 'not provided'."""
    if getattr(file, "changed_lines", None) is not None:
        return set(file.changed_lines)
    if getattr(file, "patch", None):
        return set(parse_patch_changed_lines(file.patch))
    return None


def _unit_changed_lines(changed: set[int] | None, start: int, end: int) -> list[int] | None:
    """The changed lines inside [start, end] (sorted), or None when the file
    came without change information. Used to keep the changed part of a unit
    in view when it has to be trimmed for the LLM prompt."""
    if changed is None:
        return None
    return sorted(ln for ln in changed if start <= ln <= end)


def _whole_file_unit(file, ext: str, changed: set[int] | None = None) -> dict:
    end = len(file.content.splitlines()) or 1
    return {
        "file_path": file.path,
        "function_name": None,
        "start_line": 1,
        "end_line": end,
        "code": file.content,
        "language": _language_name(ext.lstrip(".")),
        "changed_lines": _unit_changed_lines(changed, 1, end),
    }


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
                units.append(_whole_file_unit(file, ext, _changed_line_set(file)))
            continue

        changed = _changed_line_set(file)
        source_lines = file.content.splitlines()
        functions = parser.extract_functions(file.content, ext)
        before = len(units)
        for func in functions:
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
                    "changed_lines": _unit_changed_lines(
                        changed, func["start_line"], func["end_line"]
                    ),
                }
            )

        # Coverage fallback: scan the whole file when changed code lives OUTSIDE
        # any function (module-level statements, no-function scripts) — even if the
        # file also has a function change. Without this, a hardcoded secret added
        # at module level in a file that also edits a function is silently missed.
        if IGNORE_DIRECTIVE in file.content:
            continue
        covered: set[int] = set()
        for func in functions:
            covered |= set(range(func["start_line"], func["end_line"] + 1))
        if changed is None:
            needs_whole_file = len(units) == before  # no functions at all
        else:
            needs_whole_file = bool(changed - covered)  # changed lines outside functions
        if needs_whole_file:
            units.append(_whole_file_unit(file, ext, changed))

    dropped = 0
    if len(units) > max_units:
        dropped = len(units) - max_units
        units = units[:max_units]
    return units, dropped
