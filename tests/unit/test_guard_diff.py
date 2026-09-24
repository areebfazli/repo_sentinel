"""Unit tests for the PR diff-direction check (guard_diff)."""
import difflib
import random
from types import SimpleNamespace

import pytest

from backend.app.core.analysis_planner import plan_units
from backend.app.core.code_parser import CodeParser
from backend.app.core.guard_diff import (
    PatchMismatch,
    flip_direction,
    guard_diff,
    guard_diff_for_file,
    old_code_for_units,
    patch_touched_lines,
    reverse_apply_patch,
)


def _kinds(result, direction):
    return {c.kind for c in result.changes if c.direction == direction}


# ---------------------------------------------------------------------------
# Detector families: (language, old, new, expected kind, expected direction).
# Old -> new is always the guard-REMOVING direction.
# ---------------------------------------------------------------------------

PY_CASES = [
    ("sanitiser_call",
     "def render(name):\n    return '<b>' + escape(name) + '</b>'\n",
     "def render(name):\n    return '<b>' + name + '</b>'\n",
     "sanitiser", "removed"),
    ("auth_call",
     "def delete(request, obj):\n    check_permission(request.user, obj)\n    obj.delete()\n",
     "def delete(request, obj):\n    obj.delete()\n",
     "auth_check", "removed"),
    ("auth_guard_block",
     "def view(request):\n    if not request.user.is_staff:\n"
     "        raise PermissionDenied()\n    return render(request)\n",
     "def view(request):\n    return render(request)\n",
     "auth_check", "removed"),
    ("path_containment_block",
     "def read(base, name):\n    p = os.path.join(base, name)\n"
     "    if '..' in name:\n        raise ValueError('bad path')\n    return open(p).read()\n",
     "def read(base, name):\n    p = os.path.join(base, name)\n    return open(p).read()\n",
     "path_containment", "removed"),
    ("bounds_block",
     "def parse(buf, n):\n    if n > len(buf):\n        raise ValueError('short')\n"
     "    return buf[:n]\n",
     "def parse(buf, n):\n    return buf[:n]\n",
     "bounds_check", "removed"),
    ("yaml_swap",
     "def load(s):\n    return yaml.safe_load(s)\n",
     "def load(s):\n    return yaml.load(s)\n",
     "safe_api", "weakened"),
    ("yaml_safeloader_arg",
     "def load(s):\n    return yaml.load(s, Loader=yaml.SafeLoader)\n",
     "def load(s):\n    return yaml.load(s, Loader=yaml.Loader)\n",
     "safe_api", "weakened"),
    ("pickle_swap",
     "def load(b):\n    return json.loads(b)\n",
     "def load(b):\n    return pickle.loads(b)\n",
     "safe_api", "weakened"),
    ("eval_swap",
     "def conv(s):\n    return ast.literal_eval(s)\n",
     "def conv(s):\n    return eval(s)\n",
     "safe_api", "weakened"),
    ("shell_flag",
     "def run(cmd):\n    subprocess.run(cmd, shell=False)\n",
     "def run(cmd):\n    subprocess.run(cmd, shell=True)\n",
     "shell", "weakened"),
    ("tls_verify_flag",
     "def fetch(url):\n    return requests.get(url)\n",
     "def fetch(url):\n    return requests.get(url, verify=False)\n",
     "tls_verify", "weakened"),
    ("autoescape_flag",
     "def env():\n    return Environment(autoescape=True)\n",
     "def env():\n    return Environment(autoescape=False)\n",
     "sanitiser", "weakened"),
    ("sql_fstring",
     "def get(cur, uid):\n    cur.execute('SELECT * FROM users WHERE id = %s', (uid,))\n",
     "def get(cur, uid):\n    cur.execute(f'SELECT * FROM users WHERE id = {uid}')\n",
     "sql_param", "weakened"),
    ("sql_percent_format",
     "def get(cur, uid):\n    cur.execute('SELECT * FROM users WHERE id = %s', (uid,))\n",
     "def get(cur, uid):\n    cur.execute('SELECT * FROM users WHERE id = %s' % uid)\n",
     "sql_param", "weakened"),
    ("realpath_to_abspath",
     "def p(x):\n    return os.path.realpath(x)\n",
     "def p(x):\n    return os.path.abspath(x)\n",
     "path_containment", "weakened"),
    ("mark_safe_introduced",
     "def show(v):\n    return format_html('{}', v)\n",
     "def show(v):\n    return mark_safe(v)\n",
     "sanitiser", "weakened"),
]

JS_CASES = [
    ("inner_html_swap",
     "function show(el, s) {\n  el.textContent = s;\n}\n",
     "function show(el, s) {\n  el.innerHTML = s;\n}\n",
     "sanitiser", "weakened"),
    ("proto_guard_block",
     "function set(o, key, v) {\n  if (key === '__proto__') return;\n  o[key] = v;\n}\n",
     "function set(o, key, v) {\n  o[key] = v;\n}\n",
     "proto_guard", "removed"),
    ("has_own_property_call",
     "function get(o, k) {\n  return Object.prototype.hasOwnProperty.call(o, k) ? o[k] : null;\n}",
     "function get(o, k) {\n  return o[k];\n}",
     "proto_guard", "removed"),
    ("auth_403_block_in_method",
     "handle(req, res) {\n  if (!req.user.isAdmin) { return res.status(403).end(); }\n"
     "  doIt(req);\n}",
     "handle(req, res) {\n  doIt(req);\n}",
     "auth_check", "removed"),
    ("exec_swap",
     "function run(file) {\n  child_process.execFile('ls', [file]);\n}",
     "function run(file) {\n  child_process.exec('ls ' + file);\n}",
     "shell", "weakened"),
    ("reject_unauthorized",
     "function get(u) { return https.get(u, {rejectUnauthorized: true}); }",
     "function get(u) { return https.get(u, {rejectUnauthorized: false}); }",
     "tls_verify", "weakened"),
    ("sql_template_literal",
     "function q(db, id) {\n  return db.query('SELECT * FROM t WHERE id = ?', [id]);\n}",
     "function q(db, id) {\n  return db.query(`SELECT * FROM t WHERE id = ${id}`);\n}",
     "sql_param", "weakened"),
    ("escape_call_arrow",
     "const f = (s) => `<p>${escapeHtml(s)}</p>`;",
     "const f = (s) => `<p>${s}</p>`;",
     "sanitiser", "removed"),
    ("eval_swap_js",
     "function p(s) { return JSON.parse(s); }",
     "function p(s) { return eval(s); }",
     "safe_api", "weakened"),
]


@pytest.mark.parametrize("name,old,new,kind,direction", PY_CASES, ids=[c[0] for c in PY_CASES])
def test_python_detectors(name, old, new, kind, direction):
    r = guard_diff(old, new, "python")
    assert r.risk == "guard_removed", (name, r)
    assert kind in _kinds(r, direction), (name, r.changes)


@pytest.mark.parametrize("name,old,new,kind,direction", JS_CASES, ids=[c[0] for c in JS_CASES])
def test_javascript_detectors(name, old, new, kind, direction):
    r = guard_diff(old, new, "javascript")
    assert r.risk == "guard_removed", (name, r)
    assert kind in _kinds(r, direction), (name, r.changes)


@pytest.mark.parametrize("case", PY_CASES + JS_CASES, ids=[c[0] for c in PY_CASES + JS_CASES])
def test_direction_symmetry(case):
    name, old, new, kind, direction = case
    lang = "python" if case in PY_CASES else "javascript"
    fwd = guard_diff(old, new, lang)
    rev = guard_diff(new, old, lang)
    assert rev.risk == "guard_added"
    assert (fwd.removed_score, fwd.added_score) == (rev.added_score, rev.removed_score)
    assert sorted((flip_direction(c.direction), c.kind) for c in fwd.changes) == \
        sorted((c.direction, c.kind) for c in rev.changes)
    assert kind in _kinds(rev, flip_direction(direction))


def test_swap_reports_both_sides_and_is_alert():
    r = guard_diff("def f(s):\n    return yaml.safe_load(s)\n",
                   "def f(s):\n    return yaml.load(s)\n", "python")
    (c,) = r.changes
    assert "safe_load" in c.old_text and "yaml.load" in c.new_text
    assert c.confidence >= 0.8 and r.alert


def test_dropped_explicit_safe_flag_reports_old_side():
    r = guard_diff("def env():\n    return Environment(autoescape=True)\n",
                   "def env():\n    return Environment()\n", "python")
    (c,) = r.changes
    assert c.direction == "weakened" and "autoescape=True" in c.old_text and c.new_text == ""
    (c2,) = guard_diff("def env():\n    return Environment()\n",
                       "def env():\n    return Environment(autoescape=True)\n", "python").changes
    assert c2.direction == "strengthened" and "autoescape=True" in c2.new_text


def test_unescape_is_not_a_sanitiser():
    r = guard_diff("def f(s):\n    return html_unescape(s)\n", "def f(s):\n    return s\n",
                   "python")
    assert r.changes == ()


def test_removed_check_alone_is_not_alert():
    r = guard_diff(PY_CASES[0][1], PY_CASES[0][2], "python")
    assert r.risk == "guard_removed" and not r.alert


def test_line_numbers_point_into_the_right_side():
    old = "def f(x):\n    y = 1\n    if not valid(x):\n        raise ValueError()\n    return x\n"
    new = "def f(x):\n    y = 1\n    return x\n"
    r = guard_diff(old, new, "python")
    assert {c.line for c in r.changes if c.direction == "removed"} == {3}


# ---------------------------------------------------------------------------
# Rename / format / move immunity
# ---------------------------------------------------------------------------

GUARDED_PY = (
    "def save(request, filename, data):\n"
    "    target = os.path.join(UPLOAD_DIR, filename)\n"
    "    if not os.path.realpath(target).startswith(UPLOAD_DIR):\n"
    "        raise PermissionError('outside upload dir')\n"
    "    if len(data) > MAX_SIZE:\n"
    "        return None\n"
    "    check_permission(request.user)\n"
    "    body = escape(data)\n"
    "    with open(target, 'w') as fh:\n"
    "        fh.write(body)\n"
)


@pytest.mark.parametrize("edited", [
    # local rename
    GUARDED_PY.replace("target", "dest_path").replace("body", "payload"),
    # reformat: wrapping, quote style, blank lines, comments
    GUARDED_PY.replace("'outside upload dir'", '"outside upload dir"')
    .replace("    body = escape(data)\n",
             "\n    # escape it\n    body = escape(\n        data\n    )\n"),
    # docstring mentioning dangerous things
    GUARDED_PY.replace("):\n    target", "):\n    \"\"\"Never use shell=True or verify=False."
                       "\"\"\"\n    target", 1),
    # move a guard (reorder independent statements)
    GUARDED_PY.replace("    check_permission(request.user)\n", "")
    .replace("    body = escape(data)\n",
             "    body = escape(data)\n    check_permission(request.user)\n"),
    # added log line
    GUARDED_PY.replace("    body = escape(data)\n",
                       "    logger.info('saving %s', filename)\n    body = escape(data)\n"),
])
def test_benign_edits_are_ignored(edited):
    r = guard_diff(GUARDED_PY, edited, "python")
    assert r.risk == "none", r.changes
    assert not [c for c in r.changes if c.confidence >= 0.3], r.changes


def test_renamed_taint_name_does_not_flip_guard():
    old = "def f(x):\n    headers = x.h\n    if 'Pragma' in headers:\n        return None\n"
    new = "def f(x):\n    hdrs2 = x.h\n    if 'Pragma' in hdrs2:\n        return None\n"
    assert guard_diff(old, new, "python").changes == ()


def test_js_rename_and_reformat_ignored():
    old = ("function set(obj, key, value) {\n  if (key === '__proto__') { return; }\n"
           "  obj[key] = sanitize(value);\n}")
    new = ("function set(target, k, v) {\n  if (k === \"__proto__\") {\n    return;\n  }\n"
           "  // assign\n  target[k] = sanitize(v);\n}")
    assert guard_diff(old, new, "javascript").risk == "none"


def test_return_of_computed_value_is_not_a_guard():
    old = "def f(config):\n    if 'prompt' in config:\n        p = load(config)\n        return p\n"
    new = "def f(config):\n    return None\n"
    r = guard_diff(old, new, "python")
    assert not [c for c in r.changes if c.kind == "input_check"]


def test_guard_moved_into_new_helper_is_discounted():
    old = ("def f(blob):\n    if not isinstance(blob, dict):\n        raise ValueError('x')\n"
           "    return blob\n")
    new = "def f(blob):\n    _require_dict_blob(blob)\n    return blob\n"
    r = guard_diff(old, new, "python")
    assert r.risk == "none"
    assert all(c.confidence < 0.3 for c in r.changes)


# ---------------------------------------------------------------------------
# Graceful handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old,new,lang", [
    ("def f(:\n  if if if\n  raise", "def f(:\n", "python"),
    ("}{ ) function ( => => ;;; <<<", "", "javascript"),
    ("", "", "python"),
    ("\x00\x01\xff binary �", "def f(): pass", "python"),
])
def test_unparsable_code_never_raises(old, new, lang):
    r = guard_diff(old, new, lang)
    assert r.risk in ("none", "guard_removed", "guard_added")


def test_missing_old_or_unsupported_language():
    assert guard_diff(None, "def f(): pass", "python").note == "no_old_code"
    r = guard_diff("x", "y", "go")
    assert r.risk == "none" and r.note.startswith("unsupported_language")
    assert guard_diff("a", "b", None).risk == "none"


def test_typescript_goes_through_js_grammar():
    r = guard_diff(JS_CASES[0][1], JS_CASES[0][2], "ts")
    assert r.risk == "guard_removed"


# ---------------------------------------------------------------------------
# Patch reverse-application and files mode
# ---------------------------------------------------------------------------


def _github_patch(old: str, new: str, n: int = 3) -> str:
    """Unified diff without ---/+++ headers, like GitHub's per-file ``patch``."""
    lines = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True), n=n))
    return "".join(lines[2:])


def test_reverse_apply_simple_and_multi_hunk():
    old = "".join(f"line {i}\n" for i in range(1, 41))
    new = old.replace("line 3\n", "line 3 changed\n").replace("line 30\n", "")
    new = new.replace("line 20\n", "line 20\nextra\n")
    assert reverse_apply_patch(new, _github_patch(old, new)) == old


def test_reverse_apply_added_file_and_headers():
    new = "a\nb\n"
    patch = "--- /dev/null\n+++ b/x.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
    assert reverse_apply_patch(new, patch) == ""


def test_reverse_apply_pure_deletion_at_start():
    old = "x\ny\nz\n"
    new = "z\n"
    assert reverse_apply_patch(new, _github_patch(old, new, n=0)) == old


def test_reverse_apply_random_roundtrip():
    rng = random.Random(0)
    for _ in range(50):
        old_lines = [f"v{rng.randint(0, 9)}" for _ in range(rng.randint(0, 30))]
        new_lines = list(old_lines)
        for _ in range(rng.randint(1, 5)):
            op = rng.choice("adr")
            i = rng.randint(0, len(new_lines))
            if op == "a":
                new_lines.insert(i, f"n{rng.randint(0, 99)}")
            elif new_lines and i < len(new_lines):
                if op == "d":
                    del new_lines[i]
                else:
                    new_lines[i] = f"r{rng.randint(0, 99)}"
        old = "".join(line + "\n" for line in old_lines)
        new = "".join(line + "\n" for line in new_lines)
        if old == new:
            continue
        assert reverse_apply_patch(new, _github_patch(old, new, n=rng.choice([0, 1, 3]))) == old


def test_reverse_apply_mismatch_and_truncation_raise():
    old, new = "a\nb\nc\n", "a\nB\nc\n"
    patch = _github_patch(old, new)
    with pytest.raises(PatchMismatch):
        reverse_apply_patch("a\nX\nc\n", patch)
    with pytest.raises(PatchMismatch):
        reverse_apply_patch(new, patch.rsplit("\n", 2)[0] + "\n")  # truncated hunk
    with pytest.raises(PatchMismatch):
        reverse_apply_patch(new, "not a patch")


OLD_FILE = (
    "import os\n\n\n"
    "def read(base, name):\n"
    "    p = os.path.join(base, name)\n"
    "    if not os.path.realpath(p).startswith(base):\n"
    "        raise ValueError('outside base')\n"
    "    return open(p).read()\n\n\n"
    "def untouched(x):\n"
    "    return x + 1\n"
)
NEW_FILE = (
    "import os\n\n\n"
    "def read(base, name):\n"
    "    p = os.path.join(base, name)\n"
    "    return open(p).read()\n\n\n"
    "def untouched(x):\n"
    "    return x + 1\n\n\n"
    "def brand_new(y):\n"
    "    return y\n"
)


@pytest.fixture(scope="module")
def parser():
    return CodeParser()


def test_patch_only_marks_added_lines_so_deletions_need_touched_lines(parser):
    patch = _github_patch(OLD_FILE, NEW_FILE)
    file = SimpleNamespace(path="app/io.py", content=NEW_FILE, patch=patch, changed_lines=None)
    units, _ = plan_units([file], parser, max_units=50)
    # The planner's added-lines-only view misses the function whose guard was deleted.
    assert "read" not in [u["function_name"] for u in units]
    touched = patch_touched_lines(patch)
    assert {5, 6} <= set(touched)  # both sides of the deletion inside read()


def test_patch_touched_lines_deletion_at_file_start():
    assert patch_touched_lines("@@ -1,2 +0,0 @@\n-a\n-b\n") == [1]
    assert patch_touched_lines("@@ -1,1 +1,1 @@\n-a\n+b\n") == [1]


def test_files_mode_reconstructs_old_units(parser):
    patch = _github_patch(OLD_FILE, NEW_FILE)
    file = SimpleNamespace(path="app/io.py", content=NEW_FILE, patch=patch,
                           changed_lines=patch_touched_lines(patch))
    units, _ = plan_units([file], parser, max_units=50)
    names = [u["function_name"] for u in units]
    assert "read" in names and "brand_new" in names and "untouched" not in names
    pairs = dict(zip(names, old_code_for_units(file, units, parser), strict=True))
    assert "realpath" in pairs["read"][0]
    assert pairs["brand_new"] == (None, "new_function")
    results = dict(zip(names, guard_diff_for_file(file, units, parser), strict=True))
    assert results["read"].risk == "guard_removed"
    assert results["brand_new"].risk == "none" and results["brand_new"].note == "new_function"


def test_files_mode_without_patch_or_bad_patch(parser):
    units = [{"function_name": "read", "start_line": 4, "end_line": 6, "code": "x",
              "language": "python"}]
    no_patch = SimpleNamespace(path="a.py", content=NEW_FILE, patch=None)
    assert guard_diff_for_file(no_patch, units, parser)[0].note == "no_patch"
    bad = SimpleNamespace(path="a.py", content="totally different\n",
                          patch=_github_patch(OLD_FILE, NEW_FILE))
    assert guard_diff_for_file(bad, units, parser)[0].note.startswith("patch_mismatch")


def test_files_mode_prefers_base_content(parser):
    file = SimpleNamespace(path="a.py", content=NEW_FILE, patch=None, base_content=OLD_FILE)
    units, _ = plan_units([SimpleNamespace(path="a.py", content=NEW_FILE, patch=None,
                                           changed_lines=[5])], parser, max_units=50)
    (res,) = guard_diff_for_file(file, units, parser)
    assert res.risk == "guard_removed"
