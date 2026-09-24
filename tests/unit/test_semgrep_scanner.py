"""SemgrepScanner: target building, JSON parsing, batching, graceful degradation.

No engine is run: subprocess is faked, so these tests need neither semgrep nor
network.
"""
import json
import subprocess
from pathlib import Path

import pytest

from backend.app.core import semgrep_scanner as ss
from backend.app.core.semgrep_scanner import (
    SemgrepScanner,
    build_target_source,
    parse_results,
    short_rule_id,
    unit_key,
)


def _unit(code, start=1, name="f", path="app/x.py", language="python"):
    return {"file_path": path, "function_name": name, "start_line": start,
            "end_line": start + len(code.splitlines()) - 1, "code": code,
            "language": language}


def _result(path, line, check_id="a.b.rule-x", end=None, message="msg", **extra):
    return {
        "check_id": check_id,
        "path": path,
        "start": {"line": line, "col": 1},
        "end": {"line": end or line, "col": 5},
        "extra": {"message": message, "severity": "WARNING",
                  "metadata": {"cwe": "CWE-78: OS command"}, "lines": "requires login",
                  **extra},
    }


# --- target building / line mapping -------------------------------------------

def test_python_unit_keeps_real_line_numbers():
    unit = _unit("def f(x):\n    return eval(x)", start=10)
    src = build_target_source(unit, ".py").splitlines()
    assert src[9] == "def f(x):"
    assert src[10] == "    return eval(x)"
    assert all(line == "" for line in src[:9])


def test_js_method_wrapped_on_line_above():
    unit = _unit("get(req) {\n  eval(req.body)\n}", start=5, path="a.js", language="javascript")
    src = build_target_source(unit, ".js").splitlines()
    assert src[3].startswith("class ") and src[3].endswith("{")
    assert src[4] == "get(req) {"
    assert src[-1] == "}"


def test_js_method_at_line_one_wrapper_prefixed_same_line():
    unit = _unit("get(req) {\n  eval(req.body)\n}", start=1, path="a.js", language="javascript")
    src = build_target_source(unit, ".js").splitlines()
    assert src[0].startswith("class ") and src[0].endswith("get(req) {")
    assert src[1] == "  eval(req.body)"  # line 2 stays line 2


@pytest.mark.parametrize("code", [
    "function f(a) {\n  return a\n}",
    "async function f(a) {\n  return a\n}",
    "(a) => {\n  return a\n}",
    "async (a) => {\n  return a\n}",
    "async ({a}) => {\n  return a\n}",
    "x => {\n  return x\n}",
])
def test_js_functions_and_arrows_not_wrapped(code):
    unit = _unit(code, start=3, path="a.ts", language="javascript")
    src = build_target_source(unit, ".ts").splitlines()
    assert src[:2] == ["", ""]
    assert src[2:] == code.splitlines()  # line 3 is line 3, nothing appended
    assert "class" not in "\n".join(src)


def test_java_method_wrapped_and_go_gets_package():
    java = build_target_source(_unit("void q() {\n}", start=3, path="A.java"), ".java")
    assert java.splitlines()[1].startswith("class ")
    go = build_target_source(_unit("func H() {\n}", start=1, path="m.go"), ".go")
    assert go.splitlines()[0] == "package main; func H() {"


def test_whole_file_unit_never_wrapped():
    unit = _unit("get(req) {\n}", start=1, name=None, path="a.js")
    assert build_target_source(unit, ".js") == "get(req) {\n}\n"


# --- parsing -----------------------------------------------------------------

def test_parse_results_into_hits():
    unit = _unit("def f(c):\n    import subprocess\n    subprocess.call(c, shell=True)\n",
                 start=20)
    long_msg = "word " * 100
    payload = {"results": [_result("u00000.py", 22, check_id="home.me.rules.py.rule-x",
                                   message=long_msg, end=23)]}
    out = parse_results(payload, {"u00000.py": unit}, {"rule-x"})
    [hit] = out[unit_key(unit)]
    assert hit["rule_id"] == "rule-x"
    assert hit["line"] == 22 and hit["end_line"] == 22  # clamped to the unit's last line
    assert hit["cwe"] == ["CWE-78"]
    assert hit["severity"] == "medium"
    assert len(hit["message"]) <= 200 and "\n" not in hit["message"]
    assert hit["snippet"] == "    subprocess.call(c, shell=True)"


def test_security_severity_and_cwe_list_normalised():
    unit = _unit("a\nb\nc\nd\ne")
    res = _result("u.py", 1, end=5)
    res["extra"]["metadata"] = {"cwe": ["CWE-79: XSS", "cwe-80"], "security-severity": "High"}
    [hit] = parse_results({"results": [res]}, {"u.py": unit})[unit_key(unit)]
    assert hit["cwe"] == ["CWE-79", "CWE-80"]
    assert hit["severity"] == "high"
    assert hit["snippet"] == "a\nb\nc"  # at most 3 lines


def test_hits_outside_unit_dropped_and_duplicates_collapse():
    unit = _unit("x()\ny()", start=5)
    payload = {"results": [
        _result("u.py", 4),  # wrapper / padding line
        _result("u.py", 7),  # past the unit
        _result("u.py", 5), _result("u.py", 5),
        _result("other.py", 5),  # unknown target
        {"garbage": True},
    ]}
    out = parse_results(payload, {"u.py": unit})
    assert [h["line"] for h in out[unit_key(unit)]] == [5]


def test_file_target_assigns_innermost_unit():
    outer = _unit("\n".join(f"l{i}" for i in range(1, 11)), start=1, name="outer")
    inner = _unit("l3\nl4\nl5", start=3, name="inner")
    whole_tail = _unit("l9\nl10", start=9, name=None)
    payload = {"results": [_result("f.py", 4), _result("f.py", 7), _result("f.py", 10),
                           _result("f.py", 30)]}
    out = parse_results(payload, {"f.py": [outer, inner, whole_tail]})
    assert [h["line"] for h in out[unit_key(inner)]] == [4]
    assert [h["line"] for h in out[unit_key(outer)]] == [7]
    assert [h["line"] for h in out[unit_key(whole_tail)]] == [10]


def test_short_rule_id_handles_dotted_ids():
    known = {"python_crypto_rule-crypto.hazmat-hash-md5", "python_eval_rule-eval"}
    assert short_rule_id("home.a.rules.python.crypto.python_crypto_rule-crypto.hazmat-hash-md5",
                         known) == "python_crypto_rule-crypto.hazmat-hash-md5"
    assert short_rule_id("x.y.python_eval_rule-eval", known) == "python_eval_rule-eval"
    assert short_rule_id("x.y.unknown-rule", known) == "unknown-rule"


def test_unit_key():
    assert unit_key({"file_path": "a.py", "function_name": None, "start_line": 3}) == \
        ("a.py", None, 3)


def test_vendored_rules_present_and_ids_loaded():
    ids = ss.load_rule_ids(ss.DEFAULT_RULES_DIR)
    assert len(ids) > 200
    assert "python_eval_rule-eval" in ids


# --- scanner with a fake engine ------------------------------------------------

class _FakeProc:
    def __init__(self, stdout="", returncode=0, stderr="", timeout=False):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self._timeout = timeout
        self.pid = 999999
        self.killed = False

    def communicate(self, timeout=None):
        if self._timeout and not self.killed:
            self.killed = True
            raise subprocess.TimeoutExpired("semgrep", timeout)
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True


@pytest.fixture
def scanner(tmp_path):
    engine = tmp_path / "semgrep"
    engine.write_text("#!/bin/sh\n")
    engine.chmod(0o755)
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "r.yml").write_text('rules:\n- id: "rule-x"\n')
    return SemgrepScanner(rules_dir=rules, engine_path=str(engine), timeout_s=5)


def _patch_popen(monkeypatch, proc_factory):
    calls = []

    def fake_popen(cmd, cwd, **kwargs):
        files = {name: Path(cwd, name).read_text() for name in cmd
                 if "/" not in name and Path(cwd, name).is_file()}
        calls.append({"cmd": cmd, "files": files, "env": kwargs.get("env")})
        return proc_factory(files)

    monkeypatch.setattr(ss.subprocess, "Popen", fake_popen)
    return calls


def test_batch_is_one_engine_call(scanner, monkeypatch):
    units = [_unit("def a():\n    eval(x)", start=3, path="p/a.py", name="a"),
             _unit("function b() {\n  eval(y)\n}", start=1, path="p/b.js", name="b",
                   language="javascript"),
             _unit("no ext", path="README", name=None, language="text")]

    def factory(files):
        results = []
        for name, text in files.items():
            for i, line in enumerate(text.splitlines(), start=1):
                if "eval" in line:
                    results.append(_result(name, i, check_id="p.q.rule-x"))
        return _FakeProc(json.dumps({"results": results, "errors": []}))

    calls = _patch_popen(monkeypatch, factory)
    out = scanner.scan_units(units)
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert "--json" in cmd and "--metrics=off" in cmd and "--timeout" in cmd
    assert sorted(calls[0]["files"]) == ["u00000.py", "u00001.js"]  # no-ext unit skipped
    assert out[("p/a.py", "a", 3)][0]["line"] == 4
    assert out[("p/b.js", "b", 1)][0]["line"] == 2
    assert out[("p/a.py", "a", 3)][0]["rule_id"] == "rule-x"
    assert "GROQ_API_KEY" not in calls[0]["env"]


def test_sources_scan_file_once(scanner, monkeypatch):
    source = "import os\n\ndef a():\n    os.system(x)\n\ndef b():\n    os.system(y)\n"
    units = [_unit("def a():\n    os.system(x)", start=3, path="m.py", name="a"),
             _unit("def b():\n    os.system(y)", start=6, path="m.py", name="b")]

    def factory(files):
        [(name, text)] = files.items()
        assert text == source
        return _FakeProc(json.dumps({"results": [_result(name, 4), _result(name, 7),
                                                 _result(name, 1)]}))

    calls = _patch_popen(monkeypatch, factory)
    out = scanner.scan_units(units, sources={"m.py": source})
    assert len(calls) == 1
    assert [h["line"] for h in out[("m.py", "a", 3)]] == [4]
    assert [h["line"] for h in out[("m.py", "b", 6)]] == [7]


def test_disabled_or_empty_returns_empty(scanner, monkeypatch):
    calls = _patch_popen(monkeypatch, lambda files: _FakeProc("{}"))
    assert scanner.scan_units([]) == {}
    scanner.enabled = False
    assert scanner.scan_units([_unit("x")]) == {}
    assert calls == []


def test_engine_missing_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "_find_engine", lambda: None)
    sc = SemgrepScanner(rules_dir=tmp_path)
    assert sc.available() is False
    assert sc.scan_units([_unit("def f():\n    pass")]) == {}
    missing = SemgrepScanner(rules_dir=tmp_path, engine_path=str(tmp_path / "nope"))
    assert missing.available() is False
    assert missing.scan_units([_unit("x")]) == {}


def test_rules_dir_missing_degrades(scanner, tmp_path):
    scanner.rules_dir = tmp_path / "absent"
    assert scanner.available() is False
    assert scanner.scan_units([_unit("x")]) == {}


def test_popen_oserror_degrades(scanner, monkeypatch):
    def boom(*a, **k):
        raise OSError("exec format error")

    monkeypatch.setattr(ss.subprocess, "Popen", boom)
    assert scanner.scan_units([_unit("x")]) == {}


def test_timeout_kills_process_group(scanner, monkeypatch):
    proc = _FakeProc(timeout=True)
    _patch_popen(monkeypatch, lambda files: proc)
    killed = []
    monkeypatch.setattr(ss.os, "killpg", lambda pid, sig: killed.append(pid))
    assert scanner.scan_units([_unit("x")]) == {}
    assert killed == [proc.pid]


@pytest.mark.parametrize("stdout,rc", [
    ("not json", 0),
    ("[1, 2]", 0),
    ("", 2),
    ("fatal: bad config", 7),
])
def test_bad_output_degrades(scanner, monkeypatch, stdout, rc):
    _patch_popen(monkeypatch, lambda files: _FakeProc(stdout, returncode=rc, stderr="boom"))
    assert scanner.scan_units([_unit("x")]) == {}


def test_nonzero_exit_with_json_still_parsed(scanner, monkeypatch):
    payload = {"results": [_result("u00000.py", 1)], "errors": [{"type": "ParseError"}]}
    _patch_popen(monkeypatch, lambda files: _FakeProc(json.dumps(payload), returncode=2))
    out = scanner.scan_units([_unit("eval(x)", path="a.py")])
    assert out[("a.py", "f", 1)][0]["line"] == 1


def test_opengrep_command_omits_semgrep_only_flags(tmp_path):
    sc = SemgrepScanner(rules_dir=tmp_path, engine_path="/opt/bin/opengrep")
    cmd = sc._command(["u.py"])
    assert cmd[:2] == ["/opt/bin/opengrep", "scan"]
    assert "--metrics=off" not in cmd and cmd[-1] == "u.py"


def test_excluded_rules_dropped(scanner, monkeypatch):
    payload = {"results": [_result("u00000.py", 1, check_id="x.rule-x"),
                           _result("u00000.py", 1, check_id="x.noisy-rule")]}
    _patch_popen(monkeypatch, lambda files: _FakeProc(json.dumps(payload)))
    scanner.exclude_rules = frozenset({"noisy-rule"})
    [hit] = scanner.scan_units([_unit("eval(x)", path="a.py")])[("a.py", "f", 1)]
    assert hit["rule_id"] == "rule-x"
    scanner.exclude_rules = frozenset({"noisy-rule", "rule-x"})
    assert scanner.scan_units([_unit("eval(x)", path="a.py")]) == {}


def test_language_dirs_scope_configs_and_skip_unruled_targets(scanner, monkeypatch):
    for sub in ("python", "javascript", "javascript-lgpl3"):
        (scanner.rules_dir / sub).mkdir()
    scanner.language_dirs = ss.DEFAULT_LANGUAGE_DIRS
    calls = _patch_popen(monkeypatch, lambda files: _FakeProc('{"results": []}'))
    scanner.scan_units([_unit("x = 1", path="a.py"),
                        _unit("def x\nend", path="b.rb", language="ruby")])
    cmd = calls[0]["cmd"]
    configs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--config"]
    assert configs == [str(scanner.rules_dir / "python")]
    assert sorted(calls[0]["files"]) == ["u00000.py"]  # .rb: no rules -> not scanned
    # A batch with nothing the rules cover never starts the engine.
    assert scanner.scan_units([_unit("def x\nend", path="b.rb", language="ruby")]) == {}
    assert len(calls) == 1


def test_default_rules_dir_uses_vendored_layout():
    assert SemgrepScanner().language_dirs == ss.DEFAULT_LANGUAGE_DIRS
    for subs in ss.DEFAULT_LANGUAGE_DIRS.values():
        for sub in subs:
            assert (ss.DEFAULT_RULES_DIR / sub).is_dir()
