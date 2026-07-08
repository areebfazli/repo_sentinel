"""Tree-sitter function extraction.

Rewritten for the tree-sitter >= 0.23 API (``Parser(language)``, ``Query`` +
``QueryCursor.captures`` returning ``dict[str, list[Node]]``). The primary entry
point is ``extract_functions(source, ext)`` — a pure string-in/list-out call the
request path uses to split a changed file into per-function units. ``parse_file``
/ ``parse_directory`` remain for ingestion-side directory walks.
"""
import os
from pathlib import Path
from typing import Any

import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor


class CodeParser:
    def __init__(self):
        """Initialize the AST parsers for supported languages."""
        self.languages: dict[str, Language] = {
            ".py": Language(tree_sitter_python.language()),
            ".js": Language(tree_sitter_javascript.language()),
            ".ts": Language(tree_sitter_javascript.language()),  # TS -> JS parser for now
            ".go": Language(tree_sitter_go.language()),
            ".java": Language(tree_sitter_java.language()),
        }

        # Queries to extract functions/methods for each language. We capture the
        # whole function node; the name is read via child_by_field_name("name").
        self.queries: dict[str, str] = {
            ".py": "(function_definition) @function",
            ".js": """
                (function_declaration) @function
                (method_definition) @function
                (arrow_function) @function
            """,
            ".ts": """
                (function_declaration) @function
                (method_definition) @function
                (arrow_function) @function
            """,
            ".go": """
                (function_declaration) @function
                (method_declaration) @function
            """,
            ".java": "(method_declaration) @function",
        }

        # Compile queries once per language.
        self._compiled: dict[str, Query] = {
            ext: Query(self.languages[ext], src) for ext, src in self.queries.items()
        }

    def supports(self, ext: str) -> bool:
        return ext.lower() in self.languages

    def extract_functions(self, source: str, ext: str) -> list[dict[str, Any]]:
        """Extract every function/method block from ``source``.

        ``ext`` is the file extension (with dot, e.g. ".py"). Returns a list of
        dicts with keys: language, name, start_line, end_line (1-based), code.
        Unsupported extensions return an empty list.
        """
        ext = ext.lower()
        if ext not in self.languages:
            return []

        lang = self.languages[ext]
        parser = Parser(lang)
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)

        cursor = QueryCursor(self._compiled[ext])
        captures = cursor.captures(tree.root_node)  # dict[str, list[Node]]

        functions: list[dict[str, Any]] = []
        for node in captures.get("function", []):
            func_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")
            name_node = node.child_by_field_name("name")
            name = (
                source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
                if name_node is not None
                else "<anonymous>"
            )
            functions.append(
                {
                    "language": ext[1:],
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "code": func_text,
                }
            )

        # Deterministic order by position.
        functions.sort(key=lambda f: (f["start_line"], f["end_line"]))
        return functions

    def parse_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse a single file and extract all functions (with file_path attached)."""
        ext = file_path.suffix.lower()
        if ext not in self.languages:
            return []
        try:
            source_code = file_path.read_text(encoding="utf-8")
        except Exception:
            return []  # skip unreadable or binary files

        functions = self.extract_functions(source_code, ext)
        for func in functions:
            func["file_path"] = str(file_path)
        return functions

    def parse_directory(self, dir_path: Path) -> list[dict[str, Any]]:
        """Recursively walk a directory and extract functions from supported files."""
        all_functions: list[dict[str, Any]] = []
        ignore_dirs = {".git", "node_modules", "venv", "env", "__pycache__", "build", "dist"}

        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.languages:
                    funcs = self.parse_file(file_path)
                    for f in funcs:
                        f["file_path"] = str(file_path.relative_to(dir_path))
                    all_functions.extend(funcs)

        return all_functions
