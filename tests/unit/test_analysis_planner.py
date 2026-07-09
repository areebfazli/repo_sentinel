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


def test_explicit_empty_changed_lines_analyzes_nothing():
    # changed_lines=[] means "nothing changed" (explicit), not "analyze all".
    files = [FileInput(path="m.py", content=TWO_FUNCS, changed_lines=[])]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert units == []


def test_module_level_only_falls_back_to_whole_file():
    # No functions, just a top-level vulnerable statement — must still be scanned.
    content = "import subprocess\nsubprocess.run(cmd, shell=True)\n"
    files = [FileInput(path="script.py", content=content)]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert len(units) == 1
    assert units[0]["function_name"] is None
    assert units[0]["language"] == "python"
    assert "shell=True" in units[0]["code"]


def test_changed_module_level_line_falls_back_to_whole_file():
    # A changed line outside any function (module-level) still gets scanned.
    content = "API_KEY = 'sk_live_hardcoded'\n\ndef unrelated():\n    return 1\n"
    files = [FileInput(path="m.py", content=content, changed_lines=[1])]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert any(u["function_name"] is None for u in units)  # whole-file fallback fired


def test_changed_module_level_alongside_function_change():
    # A module-level secret changed in the SAME file as a function change must
    # still be scanned (whole-file fallback), not silently skipped.
    content = "API_KEY = 'sk_live_hardcoded'\n\ndef bar():\n    return 1\n"
    files = [FileInput(path="m.py", content=content, changed_lines=[1, 3])]
    units, _ = plan_units(files, PARSER, max_units=50)
    names = [u["function_name"] for u in units]
    assert "bar" in names   # the changed function
    assert None in names    # whole-file fallback covers the module-level line


def test_function_only_change_no_whole_file_noise():
    # A change entirely inside a function should NOT add a whole-file unit.
    content = "def bar():\n    x = 1\n    return x\n"
    files = [FileInput(path="m.py", content=content, changed_lines=[2])]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert [u["function_name"] for u in units] == ["bar"]


def test_ts_maps_to_javascript_language():
    content = "function render(u) {\n  el.innerHTML = u;\n}\n"
    files = [FileInput(path="app.ts", content=content)]
    units, _ = plan_units(files, PARSER, max_units=50)
    assert units and units[0]["language"] == "javascript"
