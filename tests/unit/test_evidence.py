"""Semgrep + guard_diff evidence wired into scans (stub scanner, stub merger,
stub LLM router: no engine, no models, no network)."""
import asyncio

import pytest

from backend.app.core.analysis_planner import plan_units
from backend.app.core.code_parser import CodeParser
from backend.app.core.evidence import (
    filter_semgrep_hits,
    guard_alert_findings,
    guard_evidence,
    guess_language,
    plan_with_touched_lines,
    semgrep_evidence,
)
from backend.app.core.semgrep_scanner import SemgrepScanner
from backend.app.models.schemas import AnalyzeResult, FileInput
from backend.app.services import scan_runner

# A guard deleted, nothing added: the function has no ADDED line at all.
PATH_OLD = (
    "def read(request, path):\n"
    "    if '..' in path:\n"
    "        raise ValueError('bad path')\n"
    "    return open(path).read()\n"
)
PATH_NEW = "def read(request, path):\n    return open(path).read()\n"
PATH_PATCH = (
    "@@ -1,4 +1,2 @@\n"
    " def read(request, path):\n"
    "-    if '..' in path:\n"
    "-        raise ValueError('bad path')\n"
    "     return open(path).read()\n"
)

# An unsafe-API swap: the alert tier.
YAML_NEW = (
    "import yaml\n"
    "\n"
    "def load(path):\n"
    "    with open(path) as f:\n"
    "        return yaml.load(f)\n"
    "\n"
    "def other():\n"
    "    return 1\n"
)
YAML_PATCH = (
    "@@ -3,4 +3,4 @@\n"
    " def load(path):\n"
    "     with open(path) as f:\n"
    "-        return yaml.safe_load(f)\n"
    "+        return yaml.load(f)\n"
    " \n"
)


@pytest.fixture(scope="module")
def parser():
    return CodeParser()


def _hit(rule_id, severity, line=5):
    return {"rule_id": rule_id, "severity": severity, "cwe": ["CWE-94"], "line": line,
            "end_line": line, "message": "m", "snippet": "s"}


def test_filter_semgrep_hits_severity_floor_exclusions_and_low_confidence():
    key = ("a.py", "f", 1)
    hits = {key: [_hit("python_eval_rule-eval", "high"), _hit("x_rule-medium", "medium"),
                  _hit("python_random_rule-random", "critical"),
                  _hit("javascript_dos_rule-non-literal-regexp", "critical")],
            ("b.py", "g", 1): [_hit("low_rule", "low")]}
    out = filter_semgrep_hits(hits, "high", {"python_random_rule-random"})
    assert list(out) == [key]
    assert [h["rule_id"] for h in out[key]] == [
        "python_eval_rule-eval", "javascript_dos_rule-non-literal-regexp"]
    assert [h["low_confidence"] for h in out[key]] == [False, True]
    assert len(filter_semgrep_hits(hits, "medium")[key]) == 4


def test_semgrep_evidence_degrades_without_engine(tmp_path):
    scanner = SemgrepScanner(engine_path=str(tmp_path / "no-semgrep"))
    unit = {"file_path": "a.py", "function_name": "f", "start_line": 1, "code": "eval(x)\n",
            "language": "python"}
    assert semgrep_evidence(scanner, [unit], None, "high") == {}
    assert semgrep_evidence(None, [unit], None, "high") == {}


def test_guess_language():
    assert guess_language("def f(x):\n    return x\n") == "python"
    assert guess_language("const f = (x) => x;\n") == "javascript"
    assert guess_language("SELECT 1") is None


def test_deletion_only_guard_removal_is_planned_and_detected(parser):
    f = FileInput(path="views.py", content=PATH_NEW, patch=PATH_PATCH)
    units, _ = plan_units([f], parser, 50)
    assert units == []  # added-lines-only planning misses it
    units, _ = plan_units(plan_with_touched_lines([f]), parser, 50)
    assert [u["function_name"] for u in units] == ["read"]
    guard = guard_evidence([f], units, parser)
    [g] = guard.values()
    assert g["risk"] == "guard_removed" and not g["alert"]
    assert any(c["direction"] == "removed" for c in g["changes"])
    assert guard_alert_findings(guard) == []  # removed checks alone: evidence, not alert


def test_alert_tier_becomes_deterministic_finding(parser):
    f = FileInput(path="cfg.py", content=YAML_NEW, patch=YAML_PATCH)
    units, _ = plan_units(plan_with_touched_lines([f]), parser, 50)
    assert [u["function_name"] for u in units] == ["load"]
    guard = guard_evidence([f], units, parser)
    [finding] = guard_alert_findings(guard)
    assert finding["source"] == "guard_diff" and finding["deterministic"] is True
    assert finding["severity"] == "medium" and finding["line"] == 5  # GUARD_ALERT_SEVERITY
    assert finding["corroborated_by"] == []
    assert guard_alert_findings(guard, "high")[0]["severity"] == "high"
    assert guard_alert_findings(guard, "bogus")[0]["severity"] == "medium"
    assert finding["dedupe_key"].startswith("guard:")
    assert "yaml.load" in finding["quoted_code"]


def test_explicit_changed_lines_are_left_alone():
    f = FileInput(path="a.py", content="x", changed_lines=[3], patch=PATH_PATCH)
    assert plan_with_touched_lines([f])[0].changed_lines == [3]


# --- through run_scan ----------------------------------------------------------


class StubScanner:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def scan_units(self, units, sources=None):
        self.calls.append((units, sources))
        return {k: v for k, v in self.hits.items()
                if any((u["file_path"], u["function_name"], u["start_line"]) == k
                       for u in units)}


class NoRetrievalMerger:
    async def analyze_code(self, code, language=None):
        return {"ghost_hunter_findings": [], "team_memory_findings": [], "is_vulnerable": False}

    async def analyze_units(self, units):
        return {"ghost_hunter_findings": [], "team_memory_findings": [], "is_vulnerable": False}


class CapturingRouter:
    mock = False

    def __init__(self):
        self.prompts = []

    async def generate(self, system, user, **kwargs):
        self.prompts.append(user)
        return {"findings": []}, "groq:stub"


class FailingRouter:
    mock = False

    async def generate(self, system, user, **kwargs):
        from backend.app.core.llm_client import LLMError

        raise LLMError("All LLM clients failed (stub): HTTP 500")


def _run(request, monkeypatch, scanner, router=None):
    """Run one scan end to end through run_scan and return (result, router)."""
    import json
    import uuid

    from backend.app.db.models import Scan
    from backend.app.db.session import SessionLocal, init_db

    init_db()
    monkeypatch.setattr(scan_runner, "_get_semgrep", lambda: scanner)
    job_id = uuid.uuid4().hex
    with SessionLocal() as session:
        session.add(Scan(id=job_id, status="queued", mode="files",
                         request_json=json.dumps(request)))
        session.commit()
    router = router or CapturingRouter()
    scan_runner.reset_scan_semaphore()
    asyncio.run(scan_runner.run_scan(job_id, NoRetrievalMerger(), router))
    with SessionLocal() as session:
        scan = session.get(Scan, job_id)
        assert scan.status == "completed", scan.error
        return json.loads(scan.result_json), router


def test_files_scan_feeds_semgrep_and_guard_evidence(monkeypatch):
    scanner = StubScanner({("cfg.py", "load", 3): [_hit("python_deserialization_rule-yaml",
                                                         "high", line=5)]})
    request = {"files": [{"path": "cfg.py", "content": YAML_NEW, "patch": YAML_PATCH},
                         {"path": "views.py", "content": PATH_NEW, "patch": PATH_PATCH}]}
    result, router = _run(request, monkeypatch, scanner)
    AnalyzeResult(**result)  # schema-valid

    [(units, sources)] = scanner.calls
    assert sources == {"cfg.py": YAML_NEW, "views.py": PATH_NEW}  # full files, one run
    [prompt] = router.prompts
    assert "Static-analysis evidence" in prompt and "python_deserialization_rule-yaml" in prompt
    assert "Change-direction evidence" in prompt  # both units removed/weakened a guard
    assert "removed" in prompt and "weakened" in prompt

    # The LLM found nothing, yet the alert-tier change is reported.
    [finding] = result["report_findings"]
    assert finding["source"] == "guard_diff" and finding["file_path"] == "cfg.py"
    assert result["is_vulnerable"] is True
    assert "deterministic check, no LLM" in result["report_markdown"]
    assert [s["rule_id"] for s in result["static_analysis"]] == ["python_deserialization_rule-yaml"]
    assert {g["function_name"]: g["alert"] for g in result["guard_diff"]} == {
        "load": True, "read": False}
    # The Semgrep hit on the same line corroborates the deterministic finding.
    assert finding["severity"] == "medium" and finding["corroborated_by"] == ["semgrep"]
    assert "corroborated by semgrep" in result["report_markdown"]


def test_snippet_scan_runs_semgrep_on_the_snippet_without_guard(monkeypatch):
    scanner = StubScanner({(None, None, 1): [_hit("python_eval_rule-eval", "high", line=2)]})
    result, router = _run({"code_snippet": "def f(x):\n    return eval(x)\n"}, monkeypatch,
                          scanner)
    [(units, sources)] = scanner.calls
    assert sources is None and units[0]["language"] == "python"  # guessed
    assert "python_eval_rule-eval" in router.prompts[0]
    assert "Change-direction evidence" not in router.prompts[0]
    assert result["guard_diff"] == [] and result["is_vulnerable"] is False
    assert len(result["static_analysis"]) == 1


def test_llm_failure_still_reports_semgrep_and_guard_evidence(monkeypatch):
    scanner = StubScanner({("cfg.py", "load", 3): [_hit("python_deserialization_rule-yaml",
                                                         "high", line=5)]})
    request = {"files": [{"path": "cfg.py", "content": YAML_NEW, "patch": YAML_PATCH}]}
    result, _ = _run(request, monkeypatch, scanner, FailingRouter())
    AnalyzeResult(**result)
    assert result["review_status"] == "failed" and result["llm_provider_used"] is None
    [finding] = result["report_findings"]  # the deterministic alert survives
    assert finding["source"] == "guard_diff" and result["is_vulnerable"] is True
    assert [s["rule_id"] for s in result["static_analysis"]] == ["python_deserialization_rule-yaml"]
    assert result["guard_diff"] and "LLM review failed" in result["report_markdown"]
    assert [u["reason"] for u in result["units_not_reviewed"]] == ["llm_error"]


def test_corroboration_needs_the_same_unit_and_a_nearby_line():
    from backend.app.core.evidence import corroborate_deterministic

    det = {"source": "guard_diff", "deterministic": True, "file_path": "a.py",
           "function_name": "f", "start_line": 1, "line": 10}
    llm_near = {"source": "llm", "file_path": "a.py", "function_name": "f", "start_line": 1,
                "line": 12, "end_line": 12}
    llm_far = {**llm_near, "line": 40, "end_line": 41}
    llm_other = {**llm_near, "function_name": "g"}
    findings = [dict(det), llm_far, llm_other]
    corroborate_deterministic(findings, {})
    assert findings[0]["corroborated_by"] == []
    findings = [dict(det), llm_near]
    corroborate_deterministic(findings, {("a.py", "f", 1): [
        {"line": 9, "low_confidence": False}]})
    assert findings[0]["corroborated_by"] == ["llm", "semgrep"]
    findings = [dict(det)]
    corroborate_deterministic(findings, {("a.py", "f", 1): [{"line": 9, "low_confidence": True}]})
    assert findings[0]["corroborated_by"] == []  # regex heuristics don't count
