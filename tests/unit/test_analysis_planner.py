"""Unit tests for the files -> analysis-unit planner."""
from backend.app.core.analysis_planner import plan_units
from backend.app.core.code_parser import CodeParser
from backend.app.models.schemas import FileInput

PARSER = CodeParser()

TWO_FUNCS = '''\
def alpha(x):
    return x + 1


def beta(y):
    return y * 2
'''


def test_no_changed_lines_analyzes_all_functions():
    files = [FileInput(path="m.py", content=TWO_FUNCS)]
    units, dropped = plan_units(files, PARSER, max_units=50)
    names = {u["function_name"] for u in units}
    assert names == {"alpha", "beta"}
    assert dropped == 0


def test_changed_lines_selects_overlapping_function_only():
    files = [FileInput(path="m.py", content=TWO_FUNCS, changed_lines=[6])]  # inside beta
    units, _ = plan_units(files, PARSER, max_units=50)
    assert [u["function_name"] for u in units] == ["beta"]


def test_patch_used_when_no_explicit_changed_lines():
    # Patch marks line 2 (inside alpha) as added.
    patch = "@@ -1,2 +1,2 @@\n def alpha(x):\n-    return x\n+    return x + 1\n"
    files = [FileInput(path="m.py", content=TWO_FUNCS, patch=patch)]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert [u["function_name"] for u in units] == ["alpha"]


def test_ignore_directive_inside_function():
    content = "def gamma():\n    # reposentinel-ignore\n    eval(user_input)\n"
    files = [FileInput(path="m.py", content=content)]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert units == []


def test_ignore_directive_line_above():
    content = "# reposentinel-ignore\ndef gamma():\n    eval(user_input)\n"
    files = [FileInput(path="m.py", content=content)]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert units == []


def test_unsupported_language_is_whole_file_unit():
    files = [FileInput(path="notes.txt", content="just some text\nmore text\n")]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert len(units) == 1
    assert units[0]["function_name"] is None
    assert units[0]["file_path"] == "notes.txt"


def test_max_units_cap_reports_dropped():
    files = [FileInput(path="m.py", content=TWO_FUNCS)]
    units, dropped = plan_units(files, PARSER, max_units=1)
    assert len(units) == 1
    assert dropped == 1
