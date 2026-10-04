"""PR bundle building, diff rendering, symbol index and section budgeting
(core/pr_context.py). Pure: tree-sitter only, no models, no network."""
import pytest

from backend.app.core.code_parser import CodeParser
from backend.app.core.pr_context import (
    SectionPlan,
    build_pr_bundle,
    diff_lines,
    fit_file_section,
    normalize_files,
    render_diff,
    render_file_section,
    synthesize_patch,
)
from backend.app.core.review_plan import SIZING_NONCE, estimate_tokens

PY_OLD = (
    "import os\n"
    "\n"
    "LIMIT = 10\n"
    "\n"
    "def clean(p):\n"
    "    return p.replace('..', '')\n"
    "\n"
    "def read(p):\n"
    "    p = clean(p)\n"
    "    return open(p).read()\n"
    "\n"
    "def view(req):\n"
    "    return read(req.args['f'])\n"
)
# read() loses its guard; a helper is added above it, so new line numbers shift.
PY_NEW = (
    "import os\n"
    "\n"
    "LIMIT = 10\n"
    "\n"
    "def clean(p):\n"
    "    return p.replace('..', '')\n"
    "\n"
    "def log(msg):\n"
    "    print(msg)\n"
    "\n"
    "def read(p):\n"
    "    return open(p).read()\n"
    "\n"
    "def view(req):\n"
    "    return read(req.args['f'])\n"
    "\n"
    "class Store:\n"
    "    def fetch(self, key):\n"
    "        return read(key)\n"
)

JS_OLD = (
    "const helper = (x) => x.trim();\n"
    "class Api {\n"
    "  get(q) { return helper(q); }\n"
    "}\n"
    "module.exports.handle = function (req, res) {\n"
    "  res.send(new Api().get(req.query.q));\n"
    "};\n"
)
JS_NEW = JS_OLD.replace("x.trim()", "x")


@pytest.fixture(scope="module")
def parser():
    return CodeParser()


@pytest.fixture(scope="module")
def bundle(parser):
    return build_pr_bundle(
        [{"path": "app/files.py", "old_content": PY_OLD, "new_content": PY_NEW},
         {"path": "web/api.js", "content": JS_NEW, "patch": synthesize_patch(JS_OLD, JS_NEW)}],
        parser)


# --- files --------------------------------------------------------------------


def test_old_content_is_rebuilt_from_the_patch_and_patch_from_old():
    patch = synthesize_patch(PY_OLD, PY_NEW)
    [from_patch] = normalize_files([{"path": "a.py", "content": PY_NEW, "patch": patch}])
    assert from_patch.old_content == PY_OLD and from_patch.status == "modified"
    [from_old] = normalize_files([{"path": "a.py", "new_content": PY_NEW,
                                   "old_content": PY_OLD}])
    assert from_old.patch == patch
    # Touched lines include the deletion point of the removed guard (new line 11/12).
    assert {8, 9, 11, 12} <= set(from_old.changed_lines)


def test_added_deleted_and_unknown_base_files():
    added, deleted, unknown, mismatch = normalize_files([
        {"path": "n.py", "new_content": "def f():\n    return 1\n", "old_content": ""},
        {"path": "d.py", "new_content": None, "old_content": "x = 1\n",
         "patch": "@@ -1 +0,0 @@\n-x = 1"},
        {"path": "u.py", "content": "def f():\n    pass\n", "changed_lines": [2]},
        {"path": "m.py", "content": "a\n", "patch": "@@ -1 +1 @@\n-b\n+c"},
    ])
    assert added.status == "added" and added.changed_lines == [1, 2]
    assert added.patch.startswith("@@ -0,0 +1,2 @@") and "+    return 1" in added.patch
    assert deleted.status == "deleted" and deleted.content == ""
    assert unknown.status == "unknown_base" and unknown.changed_lines == [2]
    assert mismatch.status == "unknown_base" and mismatch.note.startswith("patch_mismatch")


def test_diff_rows_carry_new_file_line_numbers():
    rows = diff_lines(synthesize_patch(PY_OLD, PY_NEW))
    by_text = {t: n for n, t in rows}
    assert by_text["+def log(msg):"] == 8
    assert by_text["-    p = clean(p)"] is None  # removed: no new-file line
    assert by_text[" def read(p):"] == 11
    text = render_diff(rows, max_rows=3)
    assert "more diff line(s) not shown" in text


# --- bundle -------------------------------------------------------------------


def test_bundle_has_before_and_after_with_real_line_numbers(bundle):
    fns = {fn.unit["function_name"]: fn for fn in bundle.functions_of("app/files.py")}
    assert set(fns) >= {"log", "read"}
    read = fns["read"]
    assert read.status == "modified"
    assert read.unit["start_line"] == 11 and read.old_start == 8  # new vs old numbering
    assert "p = clean(p)" in read.old_code and "clean" not in read.unit["code"]
    assert fns["log"].status == "added" and fns["log"].old_code is None
    # The JS file came with a patch only: old version rebuilt, arrow unit changed.
    js = bundle.by_path["web/api.js"]
    assert js.old_content == JS_OLD
    assert [fn.status for fn in bundle.functions_of("web/api.js")] == ["modified"]


def test_file_section_renders_diff_after_before_and_neighbours(bundle):
    text = render_file_section(bundle, bundle.by_path["app/files.py"], "N0NCE")
    assert '<untrusted_N0NCE kind="diff" file="app/files.py">' in text
    assert "   12| def read(p):" not in text  # after-code numbering is "   11| def read"
    assert "   11| def read(p):" in text and "    9|     p = clean(p)" in text
    assert "BEFORE the change, old-file lines 8-10" in text
    # view() calls read() and is unchanged: included as context. Store.fetch()
    # also calls it, but was added by the PR, so it is shown as a changed function.
    assert "function `view`" in text and "Changed function `fetch` (added)" in text
    assert "method `Store.fetch`" not in text


# --- symbol index -------------------------------------------------------------


def test_python_symbols_definitions_callers_callees(bundle):
    idx = bundle.index
    kinds = {(d.qualname, d.kind) for d in idx.defs if d.path == "app/files.py"}
    assert {("clean", "function"), ("read", "function"), ("Store", "class"),
            ("Store.fetch", "method"), ("LIMIT", "variable")} <= kinds
    assert [d.qualname for d in idx.definitions("Store.fetch")] == ["Store.fetch"]
    assert [d.start_line for d in idx.definitions("files.clean()")] == [5]
    assert {d.qualname for d in idx.callers("read")} >= {"view", "Store.fetch"}
    [read] = idx.definitions("read", "app/files.py")
    assert "clean" not in [d.name for d in idx.callees(read)]  # the call was removed
    assert idx.enclosing("app/files.py", 19).qualname == "Store.fetch"


def test_javascript_symbols(bundle):
    idx = bundle.index
    names = {(d.qualname, d.kind) for d in idx.defs if d.path == "web/api.js"}
    assert {("helper", "function"), ("Api", "class"), ("Api.get", "method"),
            ("handle", "function")} <= names
    assert [d.qualname for d in idx.callers("helper")] == ["Api.get"]
    assert [d.qualname for d in idx.callers("get")] == ["handle"]
    [get] = idx.definitions("get")
    assert [d.qualname for d in idx.callees(get)] == ["helper"]


def test_file_hint_narrows_but_never_hides_a_symbol(parser):
    b = build_pr_bundle([
        {"path": "a.py", "new_content": "def run(x):\n    return 1\n", "old_content": ""},
        {"path": "b.py", "new_content": "def run(y):\n    return 2\n", "old_content": ""},
    ], parser)
    assert [d.path for d in b.index.definitions("run", "b.py")] == ["b.py"]
    assert len(b.index.definitions("run", "missing.py")) == 2
    assert b.index.definitions("nothing") == []


# --- budget -------------------------------------------------------------------


def _size(bundle, f, plan):
    return estimate_tokens(render_file_section(bundle, f, SIZING_NONCE, plan))


def test_section_shrinks_context_then_before_then_after_then_clips_diff(bundle):
    f = bundle.by_path["app/files.py"]
    full = _size(bundle, f, SectionPlan())
    assert fit_file_section(bundle, f, full) == SectionPlan()
    no_ctx = SectionPlan(include_context=False)
    assert fit_file_section(bundle, f, _size(bundle, f, no_ctx)) == no_ctx
    no_before = SectionPlan(include_context=False, include_before=False)
    assert fit_file_section(bundle, f, _size(bundle, f, no_before)) == no_before
    diff_only = SectionPlan(include_context=False, include_before=False, include_after=False)
    plan = fit_file_section(bundle, f, _size(bundle, f, diff_only))
    assert plan.include_after is False and not plan.partial
    clipped = fit_file_section(bundle, f, _size(bundle, f, diff_only) - 20)
    assert clipped.partial and clipped.diff_rows < len(diff_lines(f.patch))
    assert fit_file_section(bundle, f, 10) is None


def test_after_code_is_windowed_around_changes(parser):
    body = "".join(f"    x{i} = {i}\n" for i in range(60))
    old = "def big(a):\n" + body + "    return a\n"
    new = old.replace("    x30 = 30\n", "    x30 = eval(a)\n")
    b = build_pr_bundle([{"path": "big.py", "old_content": old, "new_content": new}], parser)
    f = b.by_path["big.py"]
    tight = SectionPlan(include_context=False, include_before=False, after_radius=2)
    plan = fit_file_section(b, f, _size(b, f, tight))
    assert plan.include_after and plan.after_radius is not None and plan.after_radius >= 2
    text = render_file_section(b, f, "N", plan)
    assert "x30 = eval(a)" in text and "line(s) omitted" in text


def test_deleted_old_lines_and_old_line_mapping():
    from backend.app.core.pr_context import deleted_old_lines, old_line_for

    old = "a\nb\nc\nd\ne\n"
    new = "a\nc\nD\ne\n"  # b removed, d modified
    patch = synthesize_patch(old, new)
    assert deleted_old_lines(patch) == {2, 4}
    assert old_line_for(patch, 2) == 3 and old_line_for(patch, 4) == 5
    assert deleted_old_lines(None) == set() and old_line_for(None, 7) == 7
