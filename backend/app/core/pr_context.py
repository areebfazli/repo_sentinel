"""PR bundle for the PR-level review (``core.pr_review``): the change as a whole.

From a files-mode request (per file: path, new content, and the old content
and / or the unified ``patch``) this builds:

- ``PRFile`` per changed file: old content reconstructed by reverse-applying
  the patch (``guard_diff.reverse_apply_patch``) when only the patch is given,
  or a patch synthesised from old + new; the new-side lines the change touches
  (``guard_diff.patch_touched_lines``: added lines plus deletion points).
- the analysis units (``analysis_planner.plan_units``, planned with deletion
  points so a function whose only change is a removed guard is included);
- ``ChangedFunction`` per changed function: its after version (new-file line
  numbers) and before version (old-file line numbers), matched by name like
  ``guard_diff.old_code_for_units``;
- ``SymbolIndex`` over the new versions of the PR's files (tree-sitter):
  function / method / class / module-level variable definitions and call
  sites, so "definition of X", "callers of X" and "callees of X" resolve
  within the PR's files (the context loop and the auto-included neighbours);
- token-budgeted rendering of one file's section of the audit prompt: the
  diff with new-file line numbers, the changed functions after / before, then
  their direct callers / callees in the PR, shrunk in that order of
  importance when over budget (``render_file_section`` / ``fit_file_section``).

Everything here is pure / CPU-bound (tree-sitter); call from async code via
``asyncio.to_thread``. All code, paths and names go into prompts through
``untrusted.wrap_untrusted`` / ``safe_label`` only.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Parser

from backend.app.core.analysis_planner import plan_units
from backend.app.core.guard_diff import (
    PatchMismatch,
    _old_line_for,
    _parse_hunks,
    patch_touched_lines,
    reverse_apply_patch,
)
from backend.app.core.review_plan import SIZING_NONCE, elide, estimate_tokens
from backend.app.core.untrusted import safe_label, wrap_untrusted

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "javascript", ".tsx": "javascript",
    ".go": "go",
    ".java": "java",
}


def _ext(path: str) -> str:
    return Path(path or "").suffix.lower()


def language_for(path: str, default: str | None = None) -> str | None:
    return LANGUAGE_BY_EXT.get(_ext(path), default)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@dataclass
class PRFile:
    """One changed file. Also the file object ``plan_units`` / ``guard_diff``
    take (``path``, ``content``, ``patch``, ``changed_lines``, ``base_content``).

    ``status``: "added" (no old version), "modified", "deleted" (no new
    version) or "unknown_base" (old version unknown: no patch, or it doesn't
    apply). ``changed_lines``: new-file lines touched (added lines plus
    deletion points), or None when unknown (every function is reviewed).
    """

    path: str
    new_content: str | None
    old_content: str | None
    patch: str | None
    language: str | None
    status: str
    changed_lines: list[int] | None
    note: str | None = None

    @property
    def content(self) -> str:
        return self.new_content or ""

    @property
    def base_content(self) -> str | None:
        return self.old_content


def synthesize_patch(old: str, new: str, context: int = 3) -> str:
    """A GitHub-style patch (hunks only, no file headers) turning ``old`` into ``new``."""
    lines = list(difflib.unified_diff(old.split("\n"), new.split("\n"), lineterm="",
                                      n=context))
    return "\n".join(lines[2:])


def added_file_patch(new: str) -> str:
    """The patch of a newly added file: every line added."""
    lines = new.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + ln for ln in lines)


def _get(item: Any, *names: str):
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if value is not None:
            return value
    return None


def normalize_files(files: list, language: str | None = None) -> list[PRFile]:
    """``files``: dicts or objects with ``path`` and ``new_content`` (or
    ``content``; None = deleted), optionally ``old_content`` (or
    ``base_content``; "" = added file), ``patch``, ``changed_lines``,
    ``language``, ``status``. Old content is rebuilt from the patch when
    missing; a patch is synthesised from old + new when missing."""
    out: list[PRFile] = []
    for item in files:
        path = str(_get(item, "path", "filename") or "")
        new = _get(item, "new_content", "content")
        old = _get(item, "old_content", "base_content")
        patch = _get(item, "patch") or None
        explicit = _get(item, "changed_lines")
        status = _get(item, "status")
        lang = _get(item, "language") or language_for(path, language)
        note = None
        if old is None and status == "added" and new is not None:
            old = ""
        if old is None and patch and new is not None:
            try:
                old = reverse_apply_patch(new, patch)
            except PatchMismatch as exc:
                note = f"patch_mismatch:{exc}"
        if patch is None and old is not None and new is not None:
            patch = synthesize_patch(old, new) if old else added_file_patch(new)
        if new is None:
            status = "deleted"
        elif old is None:
            status = "unknown_base"
        elif old == "" and new:
            status = "added"
        else:
            status = "modified"
        if explicit is not None:
            changed = sorted({int(x) for x in explicit})
        elif status == "added":
            changed = list(range(1, len((new or "").splitlines()) + 1))
        elif patch and new is not None:
            try:
                changed = patch_touched_lines(patch)
            except Exception:  # malformed patch: review everything
                changed = None
        elif old is not None and new is not None:
            changed = []  # both versions known and identical: nothing changed
        else:
            changed = None
        out.append(PRFile(path, new, old, patch, lang, status, changed, note))
    return out


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------


def diff_lines(patch: str | None) -> list[tuple[int | None, str]]:
    """The patch's hunks as (new-file line or None, rendered line): a hunk
    header, then each line as ``"<+|-| ><text>"``; removed lines have no
    new-file number."""
    out: list[tuple[int | None, str]] = []
    if not patch:
        return out
    try:
        hunks = _parse_hunks(patch)
    except Exception:
        return out
    for a, b, c, d, lines in hunks:
        out.append((None, f"@@ -{a},{b} +{c},{d} @@"))
        new_line = c if d > 0 else c + 1
        for line in lines:
            tag, text = line[0], line[1:]
            if tag == "-":
                out.append((None, f"-{text}"))
            else:
                out.append((new_line, f"{tag}{text}"))
                new_line += 1
    return out


def deleted_old_lines(patch: str | None) -> set[int]:
    """Old-file (1-based) line numbers the patch removes (a modified line is
    removed and re-added)."""
    out: set[int] = set()
    if not patch:
        return out
    try:
        hunks = _parse_hunks(patch)
    except Exception:
        return out
    for a, b, _c, _d, lines in hunks:
        old = a if b > 0 else a + 1
        for line in lines:
            tag = line[0]
            if tag == "-":
                out.add(old)
                old += 1
            elif tag != "+":
                old += 1
    return out


def old_line_for(patch: str | None, new_line: int) -> int:
    """Approximate old-file line of new-file ``new_line`` (``new_line`` itself
    without a patch)."""
    if not patch:
        return new_line
    try:
        return _old_line_for(patch, new_line)
    except Exception:
        return new_line


def new_line_for(patch: str | None, old_line: int) -> int:
    """New-file line of old-file ``old_line`` (``old_line`` itself without a
    patch): a kept line's own new number; a deleted line maps to the deletion
    point, the new-file line that follows it (as ``patch_touched_lines``)."""
    if not patch:
        return old_line
    try:
        hunks = _parse_hunks(patch)
    except Exception:
        return old_line
    delta = 0
    for a, b, c, d, lines in hunks:
        old = a if b > 0 else a + 1
        new = c if d > 0 else c + 1
        if old > old_line:
            break
        for line in lines:
            tag = line[0]
            if tag == "+":
                new += 1
                continue
            if old == old_line:
                return new
            old += 1
            if tag != "-":
                new += 1
        delta = new - old
    return old_line + delta


def render_diff(rows: list[tuple[int | None, str]], max_rows: int | None = None) -> str:
    """Numbered diff text: ``"   12| +code"`` (new-file line numbers; blank
    for removed lines and hunk headers). ``max_rows`` clips, with a marker."""
    shown = rows if max_rows is None else rows[:max_rows]
    text = [f"{'' if n is None else n:>5}| {t}" for n, t in shown]
    if max_rows is not None and len(rows) > max_rows:
        text.append(f"[... {len(rows) - max_rows} more diff line(s) not shown ...]")
    return "\n".join(text)


def numbered(code: str, start_line: int, line_numbers: list | None = None) -> str:
    lines = code.splitlines()
    numbers = line_numbers or [start_line + i for i in range(len(lines))]
    return "\n".join(f"{'...' if n is None else n:>5}| {ln}"
                     for n, ln in zip(numbers, lines, strict=False))


# ---------------------------------------------------------------------------
# Symbol index
# ---------------------------------------------------------------------------

_DEF_NODES = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {"function_declaration": "function", "generator_function_declaration":
                   "function", "method_definition": "method", "class_declaration": "class"},
    "go": {"function_declaration": "function", "method_declaration": "method",
           "type_spec": "type"},
    "java": {"method_declaration": "method", "constructor_declaration": "method",
             "class_declaration": "class", "interface_declaration": "class"},
}
_CALL_NODES = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "go": {"call_expression"},
    "java": {"method_invocation", "object_creation_expression"},
}
_JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function",
                       "generator_function"}
_TS_EXT = {".py": ".py", ".js": ".js", ".jsx": ".js", ".mjs": ".js", ".cjs": ".js",
           ".ts": ".ts", ".tsx": ".ts", ".go": ".go", ".java": ".java"}


@dataclass
class Definition:
    path: str
    name: str
    qualname: str
    kind: str  # function | method | class | type | variable
    start_line: int
    end_line: int
    code: str


@dataclass
class CallSite:
    path: str
    callee: str
    line: int
    caller: Definition | None


def _node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace") if node else ""


def _last_name(node, src: bytes) -> str:
    """Last identifier of a callee / declarator expression (``a.b.c`` -> ``c``)."""
    if node is None:
        return ""
    for fld in ("attribute", "property", "field", "name"):
        child = node.child_by_field_name(fld)
        if child is not None and node.type in (
            "attribute", "member_expression", "selector_expression", "field_access",
            "scoped_identifier", "qualified_type",
        ):
            return _node_text(child, src)
    if node.type in ("identifier", "property_identifier", "field_identifier",
                     "type_identifier", "private_property_identifier"):
        return _node_text(node, src)
    text = _node_text(node, src)
    return text.rsplit(".", 1)[-1].strip() if len(text) < 200 else ""


def _callee(node, src: bytes, lang: str) -> str:
    if lang == "java":
        name = node.child_by_field_name("name") or node.child_by_field_name("type")
        return _last_name(name, src)
    target = (node.child_by_field_name("function") or node.child_by_field_name("constructor"))
    return _last_name(target, src)


def _def_name(node, src: bytes, lang: str) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return _node_text(name, src)
    return None


def _js_assigned_function(node, src: bytes) -> tuple[str, Any] | None:
    """``const f = () => ...`` / ``{p: function () {}}`` / ``exports.f = ...``:
    (name, function node)."""
    if node.type == "variable_declarator":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            return _node_text(node.child_by_field_name("name"), src), value
    if node.type == "pair":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            return _node_text(node.child_by_field_name("key"), src).strip("'\""), value
    if node.type == "assignment_expression":
        value = node.child_by_field_name("right")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            return _last_name(node.child_by_field_name("left"), src), value
    return None


class SymbolIndex:
    """Definitions and call sites of the new versions of a PR's files."""

    def __init__(self):
        self.defs: list[Definition] = []
        self.calls: list[CallSite] = []

    @classmethod
    def build(cls, files: list[PRFile], parser) -> SymbolIndex:
        index = cls()
        for f in files:
            if f.new_content is None:
                continue
            try:
                index._add_file(f.path, f.new_content, parser)
            except Exception:  # an index gap must never break a scan
                continue
        return index

    def _add_file(self, path: str, content: str, parser) -> None:
        ext = _TS_EXT.get(_ext(path))
        lang = language_for(path)
        if ext is None or lang is None or ext not in getattr(parser, "languages", {}):
            return
        src = content.encode("utf-8")
        tree = Parser(parser.languages[ext]).parse(src)
        def_types = _DEF_NODES.get(lang, {})
        call_types = _CALL_NODES.get(lang, set())
        # (node, enclosing definition, enclosing class qualname, at module level)
        stack = [(tree.root_node, None, None, True)]
        while stack:
            node, owner, klass, top = stack.pop()
            new_owner, new_class = owner, klass
            definition = None
            kind = def_types.get(node.type)
            if kind is not None:
                name = _def_name(node, src, lang)
                if name:
                    if kind == "function" and klass and lang == "python":
                        kind = "method"
                    qual = f"{klass}.{name}" if klass and kind == "method" else name
                    definition = self._add_def(path, name, qual, kind, node, src)
                    if kind == "class":
                        new_class = name
            elif lang == "javascript":
                assigned = _js_assigned_function(node, src)
                if assigned and assigned[0]:
                    name, fn = assigned
                    definition = self._add_def(path, name, name, "function", node, src)
            if definition is None and top and self._is_module_variable(node, lang):
                self._add_variables(path, node, src, lang)
            if definition is not None and definition.kind != "class":
                new_owner = definition
            if node.type in call_types:
                callee = _callee(node, src, lang)
                if callee:
                    self.calls.append(CallSite(path, callee, node.start_point[0] + 1, owner))
            child_top = top and node.type in ("module", "program", "source_file",
                                              "export_statement", "decorated_definition")
            for child in reversed(node.named_children):
                stack.append((child, new_owner, new_class, child_top))

    def _add_def(self, path, name, qual, kind, node, src) -> Definition:
        d = Definition(path, name, qual, kind, node.start_point[0] + 1, node.end_point[0] + 1,
                       _node_text(node, src))
        self.defs.append(d)
        return d

    @staticmethod
    def _is_module_variable(node, lang: str) -> bool:
        if lang == "python":
            return node.type == "expression_statement" and any(
                c.type == "assignment" for c in node.named_children)
        if lang == "javascript":
            return node.type in ("lexical_declaration", "variable_declaration")
        return False

    def _add_variables(self, path, node, src, lang) -> None:
        if lang == "python":
            for c in node.named_children:
                left = c.child_by_field_name("left") if c.type == "assignment" else None
                if left is not None and left.type == "identifier":
                    self._add_def(path, _node_text(left, src), _node_text(left, src),
                                  "variable", node, src)
            return
        for c in node.named_children:
            if c.type != "variable_declarator":
                continue
            value = c.child_by_field_name("value")
            if value is not None and value.type in _JS_FUNCTION_VALUES:
                continue  # indexed as a function
            name = c.child_by_field_name("name")
            if name is not None and name.type == "identifier":
                self._add_def(path, _node_text(name, src), _node_text(name, src), "variable",
                              node, src)

    # --- lookups ---------------------------------------------------------------

    @staticmethod
    def _norm(symbol: str) -> tuple[str, str]:
        """(full, last segment) of a requested symbol ("mod.Cls.m()" -> "m")."""
        full = (symbol or "").strip().strip("`'\"").removesuffix("()").strip()
        last = full
        for sep in ("::", "#", "."):
            last = last.rsplit(sep, 1)[-1]
        return full, last.strip()

    def _in_file(self, items: list, file: str | None) -> list:
        if not file:
            return items
        file = file.strip().lstrip("./")
        scoped = [d for d in items if d.path == file or d.path.endswith("/" + file)
                  or file.endswith("/" + d.path)]
        return scoped or items  # a wrong file hint doesn't hide the symbol

    def definitions(self, symbol: str, file: str | None = None) -> list[Definition]:
        full, last = self._norm(symbol)
        if not last:
            return []
        found = [d for d in self.defs if d.qualname == full or d.name == last]
        return self._in_file(found, file)

    def callers(self, symbol: str, file: str | None = None) -> list[Definition]:
        """Functions in the PR's files that call ``symbol`` (by name)."""
        _, last = self._norm(symbol)
        seen: list[Definition] = []
        for c in self.calls:
            if c.callee == last and c.caller is not None and c.caller not in seen:
                seen.append(c.caller)
        return seen  # ``file`` names where the symbol lives, not its callers

    def callees(self, definition: Definition) -> list[Definition]:
        """Definitions (in the PR's files) of what ``definition`` calls."""
        names = []
        for c in self.calls:
            if c.caller is definition and c.callee not in names:
                names.append(c.callee)
        out: list[Definition] = []
        for name in names:
            for d in self.defs:
                if d.name == name and d.kind != "variable" and d is not definition \
                        and d not in out:
                    out.append(d)
        return out

    def enclosing(self, path: str, line: int) -> Definition | None:
        """Innermost function / method definition of ``path`` containing ``line``."""
        best = None
        for d in self.defs:
            if d.path == path and d.kind in ("function", "method") \
                    and d.start_line <= line <= d.end_line:
                if best is None or d.end_line - d.start_line < best.end_line - best.start_line:
                    best = d
        return best

    def for_unit(self, unit: dict) -> Definition | None:
        """The definition a planner unit (function) corresponds to."""
        for d in self.defs:
            if d.path == unit.get("file_path") and d.start_line == unit.get("start_line") \
                    and d.kind in ("function", "method"):
                return d
        return None


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass
class ChangedFunction:
    unit: dict  # the planner unit (file_path, function_name, start_line, end_line, code, ...)
    old_code: str | None
    old_start: int | None
    status: str  # modified | added | unknown_base

    @property
    def path(self) -> str:
        return self.unit["file_path"]


@dataclass
class PRBundle:
    files: list[PRFile]
    units: list[dict]
    dropped_units: int
    functions: list[ChangedFunction]
    index: SymbolIndex
    by_path: dict[str, PRFile] = field(default_factory=dict)
    # CodeParser functions per new file (``extract_functions``): names and
    # ranges exactly as the planner / guard_diff / Action markers use them.
    spans: dict[str, list[dict]] = field(default_factory=dict)

    def function_at(self, path: str, line: int) -> dict | None:
        """Innermost CodeParser function of ``path`` containing ``line``."""
        best = None
        for f in self.spans.get(path, []):
            if f["start_line"] <= line <= f["end_line"] and (
                    best is None or f["end_line"] - f["start_line"]
                    < best["end_line"] - best["start_line"]):
                best = f
        return best

    def outermost_function_at(self, path: str, line: int) -> dict | None:
        """Outermost CodeParser function of ``path`` containing ``line``: a
        closure's or callback's enclosing top-level function, or a method (its
        class is not a function). Climbing from the innermost function stops
        where the next enclosing function would cross a class (a method of a
        class defined inside a function stays the method; ``SymbolIndex``
        classes: Python ``class``, JS ``class`` declarations)."""
        chain = sorted((f for f in self.spans.get(path, [])
                        if f["start_line"] <= line <= f["end_line"]),
                       key=lambda f: f["end_line"] - f["start_line"])
        if not chain:
            return None
        classes = [(d.start_line, d.end_line) for d in self.index.defs
                   if d.path == path and d.kind == "class"]
        best = chain[0]
        for outer in chain[1:]:
            if any(outer["start_line"] <= a and b <= outer["end_line"]
                   and a <= best["start_line"] and best["end_line"] <= b
                   for a, b in classes):
                break
            best = outer
        return best

    def units_of(self, path: str) -> list[dict]:
        return [u for u in self.units if u.get("file_path") == path]

    def functions_of(self, path: str) -> list[ChangedFunction]:
        return [f for f in self.functions if f.path == path]

    def resolve_path(self, name: str | None) -> str | None:
        """The PR file path the model meant (exact, or a unique suffix match)."""
        if not name:
            return None
        name = str(name).strip().strip("`'\"").lstrip("./")
        if name in self.by_path:
            return name
        hits = [p for p in self.by_path if p.endswith("/" + name) or name.endswith("/" + p)]
        return hits[0] if len(hits) == 1 else None


def _old_functions(pr_file: PRFile, parser) -> list[dict]:
    if not pr_file.old_content:
        return []
    ext = _TS_EXT.get(_ext(pr_file.path))
    if ext is None or not parser.supports(ext):
        return []
    try:
        return parser.extract_functions(pr_file.old_content, ext)
    except Exception:
        return []


def build_pr_bundle(files: list, parser, *, language: str | None = None,
                    max_units: int = 50) -> PRBundle:
    """Normalise ``files`` (see ``normalize_files``) and build the bundle."""
    pr_files = files if files and isinstance(files[0], PRFile) else normalize_files(
        files, language)
    live = [f for f in pr_files if f.new_content is not None]
    units, dropped = plan_units(live, parser, max_units)
    functions: list[ChangedFunction] = []
    spans: dict[str, list[dict]] = {}
    for f in live:
        ext = _TS_EXT.get(_ext(f.path))
        if ext is not None and parser.supports(ext):
            try:
                spans[f.path] = parser.extract_functions(f.new_content, ext)
            except Exception:
                spans[f.path] = []
        olds = _old_functions(f, parser)
        for unit in units:
            if unit["file_path"] != f.path or unit.get("function_name") is None:
                continue
            if f.old_content is None:
                functions.append(ChangedFunction(unit, None, None, "unknown_base"))
                continue
            same = [o for o in olds if o["name"] == unit["function_name"]]
            if not same:
                functions.append(ChangedFunction(unit, None, None, "added"))
                continue
            target = _old_line_for(f.patch, unit["start_line"]) if f.patch else unit["start_line"]
            best = min(same, key=lambda o: abs(o["start_line"] - target))
            functions.append(ChangedFunction(unit, best["code"], best["start_line"], "modified"))
    return PRBundle(pr_files, units, dropped, functions, SymbolIndex.build(live, parser),
                    {f.path: f for f in pr_files}, spans)


# ---------------------------------------------------------------------------
# Rendering (one file's section of the audit prompt)
# ---------------------------------------------------------------------------


@dataclass
class SectionPlan:
    """How much of a file's section to render (``fit_file_section`` shrinks it)."""

    include_context: bool = True
    include_before: bool = True
    after_radius: int | None = None  # None = whole functions; else lines around changes
    include_after: bool = True
    diff_rows: int | None = None  # None = whole diff

    @property
    def partial(self) -> bool:
        """Changed lines went unseen (the diff was clipped)."""
        return self.diff_rows is not None

    def describe(self) -> str | None:
        cut = []
        if not self.include_context:
            cut.append("neighbouring functions left out")
        if not self.include_before:
            cut.append("before-versions left out")
        if not self.include_after:
            cut.append("after-versions left out (diff only)")
        elif self.after_radius is not None:
            cut.append(f"after-versions shown only within {self.after_radius} line(s) of the "
                       "changes")
        if self.diff_rows is not None:
            cut.append(f"diff clipped to {self.diff_rows} line(s)")
        return "; ".join(cut) or None


def context_definitions(bundle: PRBundle, path: str, limit: int = 4) -> list[Definition]:
    """Direct callers and callees (within the PR's files) of the file's changed
    functions, excluding the changed functions themselves."""
    changed = [bundle.index.for_unit(fn.unit) for fn in bundle.functions_of(path)]
    changed = [d for d in changed if d is not None]
    out: list[Definition] = []
    for d in changed:
        for other in bundle.index.callers(d.name) + bundle.index.callees(d):
            if other not in changed and other not in out and other.kind != "class":
                out.append(other)
    return out[:limit]


def _after_code(fn: ChangedFunction, radius: int | None) -> tuple[str, list | None]:
    code = fn.unit.get("code") or ""
    start = int(fn.unit.get("start_line") or 1)
    if radius is None:
        return code, None
    lines = code.splitlines()
    focus = [ln - start for ln in fn.unit.get("changed_lines") or ()
             if 0 <= ln - start < len(lines)]
    keep = {0}
    for c in focus:
        keep.update(range(max(c - radius, 0), min(c + radius, len(lines) - 1) + 1))
    return elide(lines, keep, start)


def definition_block(nonce: str, d: Definition, kind: str = "context_code",
                     max_lines: int | None = None) -> str:
    """A PR definition as a numbered, untrusted block with a trusted header."""
    lines = d.code.splitlines()
    numbers = None
    code = d.code
    if max_lines is not None and len(lines) > max_lines:
        code, numbers = elide(lines, set(range(max_lines)), d.start_line)
    header = (f"`{safe_label(d.path)}` {d.kind} `{safe_label(d.qualname, 80)}` "
              f"(lines {d.start_line}-{d.end_line}):")
    return header + "\n" + wrap_untrusted(nonce, kind, numbered(code, d.start_line, numbers),
                                          file=d.path)


def render_file_section(bundle: PRBundle, pr_file: PRFile, nonce: str,
                        plan: SectionPlan | None = None, leads: list[str] | None = None) -> str:
    """One changed file's part of the audit prompt: header, numbered diff,
    changed functions (after, then before), leads, then direct callers /
    callees in the PR."""
    plan = plan or SectionPlan()
    path = pr_file.path
    lang = f" ({safe_label(pr_file.language, 20)})" if pr_file.language else ""
    out = [f"### File `{safe_label(path)}`{lang}: {pr_file.status}"]
    rows = diff_lines(pr_file.patch)
    if rows:
        out.append("Diff (unified; the column before '|' is the line number in the NEW file, "
                   "empty for removed lines; then the +/-/space marker):")
        out.append(wrap_untrusted(nonce, "diff", render_diff(rows, plan.diff_rows), file=path))
    elif pr_file.status == "unknown_base":
        changed = pr_file.changed_lines
        where = (f"changed lines: {', '.join(map(str, changed[:40]))}" if changed
                 else "changed lines unknown")
        out.append(f"(No diff available for this file; its previous version is unknown - "
                   f"{where}.)")
    if plan.include_after:
        for fn in bundle.functions_of(path):
            code, numbers = _after_code(fn, plan.after_radius)
            start = int(fn.unit.get("start_line") or 1)
            end = int(fn.unit.get("end_line") or start)
            name = safe_label(fn.unit.get("function_name"), 80)
            out.append(f"Changed function `{name}` ({fn.status}) - AFTER the change, new-file "
                       f"lines {start}-{end}:")
            out.append(wrap_untrusted(nonce, "code_after", numbered(code, start, numbers),
                                      file=path, function=fn.unit.get("function_name")))
            if plan.include_before and fn.old_code is not None:
                out.append(f"Same function BEFORE the change, old-file lines {fn.old_start}-"
                           f"{fn.old_start + len(fn.old_code.splitlines()) - 1}:")
                out.append(wrap_untrusted(nonce, "code_before",
                                          numbered(fn.old_code, fn.old_start or 1),
                                          file=path, function=fn.unit.get("function_name")))
    if leads:
        out.append("Leads for this file (hints worth checking, NOT findings):")
        out.extend(leads)
    if plan.include_context:
        ctx = context_definitions(bundle, path)
        if ctx:
            out.append("Direct callers / callees of the changed functions in this PR (context, "
                       "unchanged unless listed above):")
            out.extend(definition_block(nonce, d, max_lines=40) for d in ctx)
    return "\n".join(out)


def fit_file_section(bundle: PRBundle, pr_file: PRFile, available: int,
                     leads: list[str] | None = None) -> SectionPlan | None:
    """The richest ``SectionPlan`` whose section fits ``available`` estimated
    tokens: drop neighbouring context, then before-versions, then narrow the
    after-versions around the changed lines, then drop them (the diff shows
    every change), then clip the diff (partial). None if not even that fits."""

    def size(plan: SectionPlan) -> int:
        return estimate_tokens(render_file_section(bundle, pr_file, SIZING_NONCE, plan, leads))

    plan = SectionPlan()
    if size(plan) <= available:
        return plan
    for attr in ("include_context", "include_before"):
        setattr(plan, attr, False)
        if size(plan) <= available:
            return plan
    longest = max((len((fn.unit.get("code") or "").splitlines())
                   for fn in bundle.functions_of(pr_file.path)), default=0)
    lo, hi = 0, longest
    if longest:
        plan.after_radius = 0
        if size(plan) <= available:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                plan.after_radius = mid
                if size(plan) <= available:
                    lo = mid
                else:
                    hi = mid - 1
            plan.after_radius = lo
            return plan
    plan.after_radius = None
    plan.include_after = False
    if size(plan) <= available:
        return plan
    rows = len(diff_lines(pr_file.patch))
    lo, hi = 1, rows
    plan.diff_rows = 1
    if rows == 0 or size(plan) > available:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        plan.diff_rows = mid
        if size(plan) <= available:
            lo = mid
        else:
            hi = mid - 1
    plan.diff_rows = lo
    return plan
