"""Unit tests for the tree-sitter code parser (no ML models involved)."""
from backend.app.core.code_parser import CodeParser

PY_SOURCE = '''\
import os


def safe_add(a, b):
    return a + b


class Service:
    def handle(self, request):
        return request.ok


def outer():
    def inner():
        return 1
    return inner
'''


def test_extract_python_functions_names_and_lines():
    parser = CodeParser()
    funcs = parser.extract_functions(PY_SOURCE, ".py")
    names = {f["name"] for f in funcs}

    # Top-level, method, and nested functions are all captured.
    assert {"safe_add", "handle", "outer", "inner"}.issubset(names)

    safe_add = next(f for f in funcs if f["name"] == "safe_add")
    assert safe_add["language"] == "py"
    assert safe_add["start_line"] == 4  # 1-based
    assert "return a + b" in safe_add["code"]


def test_functions_sorted_by_position():
    parser = CodeParser()
    funcs = parser.extract_functions(PY_SOURCE, ".py")
    starts = [f["start_line"] for f in funcs]
    assert starts == sorted(starts)


def test_unsupported_extension_returns_empty():
    parser = CodeParser()
    assert parser.extract_functions("SELECT 1;", ".sql") == []


def test_supports_helper():
    parser = CodeParser()
    assert parser.supports(".py")
    assert parser.supports(".JS")  # case-insensitive
    assert not parser.supports(".rb")


def test_javascript_functions():
    parser = CodeParser()
    js = "function greet(name) { return 'hi ' + name; }\nconst add = (a, b) => a + b;\n"
    funcs = parser.extract_functions(js, ".js")
    assert any(f["name"] == "greet" for f in funcs)
    # Arrow function has no name field -> anonymous, but is still captured.
    assert any("=>" in f["code"] for f in funcs)
