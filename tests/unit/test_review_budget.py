"""Per-scan LLM token budget: evidence-first packing into as few prompts as fit,
truncation of oversized units, and reporting of units left out."""
import asyncio
import json
import uuid

import pytest

from backend.app.config import settings
from backend.app.core.llm_client import LLMError
from backend.app.core.markdown_renderer import (
    SYSTEM_PROMPT,
    build_user_prompt,
    validate_findings,
)
from backend.app.core.review_plan import (
    assign_uids,
    build_review_units,
    estimate_tokens,
    plan_review_prompts,
    unit_priority,
)
from backend.app.services import scan_runner

BUDGET = {"max_prompt_tokens": 6000, "max_units_per_prompt": 6, "max_calls": 6,
          "max_refs_per_unit": 2}


def _unit(name, lines=5, start=1, path="a.py"):
    code = f"def {name}(x):\n" + "".join(f"    y{i} = x + {i}\n" for i in range(lines - 1))
    return {"file_path": path, "function_name": name, "start_line": start,
            "end_line": start + lines - 1, "code": code, "language": "python"}


def _cve(cve_id, sim, anchor):
    return {"cve_id": cve_id, "similarity_score": sim, "description": "d" * 300,
            "vulnerable_code": "\n".join(f"v{i} = 1" for i in range(40)),
            "anchor_file_path": anchor["file_path"],
            "anchor_function_name": anchor["function_name"],
            "anchor_start_line": anchor["start_line"]}


def _review(units, cves=(), semgrep=None, guard=None):
    raw = {"ghost_hunter_findings": list(cves), "team_memory_findings": []}
    return build_review_units(units, raw, semgrep, guard)


def _prompt_tokens(batch):
    return estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(
        build_user_prompt(assign_uids(batch), "f" * 16))


def test_small_scan_is_one_call_with_capped_references():
    u = _unit("f")
    cves = [_cve(f"CVE-{i}", 0.3 + i / 10, u) for i in range(3)]
    batches, dropped = plan_review_prompts(_review([u, _unit("g", start=20)], cves), **BUDGET)
    assert len(batches) == 1 and dropped == []
    by_name = {x["function_name"]: x for x in batches[0]}
    assert [c["cve_id"] for c in by_name["f"]["cves"]] == ["CVE-0", "CVE-1"]  # cap 2, in order


def test_units_are_ordered_by_evidence():
    units = [_unit(n, start=10 * i) for i, n in enumerate("abcde")]
    key = {n: ("a.py", n, 10 * i) for i, n in enumerate("abcde")}
    semgrep = {key["b"]: [{"rule_id": "r", "line": 11}]}
    guard = {key["c"]: {"risk": "guard_removed", "alert": False, "changes": []},
             key["d"]: {"risk": "guard_removed", "alert": True, "changes": []}}
    cves = [_cve("CVE-9", 0.9, units[4]), _cve("CVE-1", 0.3, units[0])]
    review = _review(units, cves, semgrep, guard)
    order = [u["function_name"] for u in sorted(review, key=unit_priority, reverse=True)]
    assert order == ["d", "b", "c", "e", "a"]
    [batch], _ = plan_review_prompts(review, **BUDGET)
    assert [u["function_name"] for u in batch] == order


def test_packing_respects_units_per_prompt_and_call_cap():
    units = [_unit(f"f{i}", start=10 * i) for i in range(20)]
    batches, dropped = plan_review_prompts(
        _review(units), **{**BUDGET, "max_units_per_prompt": 4, "max_calls": 3})
    assert [len(b) for b in batches] == [4, 4, 4]
    assert len(dropped) == 8 and {u["not_reviewed_reason"] for u in dropped} == {"budget"}
    for b in batches:
        assert _prompt_tokens(b) <= BUDGET["max_prompt_tokens"]


def test_token_budget_splits_prompts():
    units = [_unit(f"f{i}", lines=120, start=200 * i) for i in range(4)]
    batches, dropped = plan_review_prompts(_review(units), **{**BUDGET, "max_prompt_tokens": 3000})
    assert len(batches) > 1 and not dropped
    for b in batches:
        assert _prompt_tokens(b) <= 3000


def test_oversized_unit_loses_references_then_code_is_elided():
    big = _unit("big", lines=900)
    batches, dropped = plan_review_prompts(
        _review([big], [_cve("CVE-1", 0.8, big)]), **{**BUDGET, "max_prompt_tokens": 4000})
    [[unit]] = batches
    assert not dropped and unit["cves"] == []
    assert "of 900 lines" in unit["truncated"] and unit["partial"] is True
    assert _prompt_tokens([unit]) <= 4000
    assert "Only part of this unit is shown" in build_user_prompt([unit], "f" * 16)


def _sink_at_end(n=902, start=100):
    """A 902-line function whose only dangerous call is its last line."""
    code = "def handler(request):\n" + "".join(f"    v{i} = request.args.get('k{i}')\n"
                                              for i in range(n - 2))
    code += "    os.system(request.args['cmd'])\n"
    return {"file_path": "big.py", "function_name": "handler", "start_line": start,
            "end_line": start + n - 1, "code": code, "language": "python"}


def test_snippet_mode_keeps_head_and_tail_with_real_line_numbers():
    unit = _sink_at_end()  # no changed_lines: snippet / no-diff mode
    [[shown]], _ = plan_review_prompts(_review([unit]), **{**BUDGET, "max_prompt_tokens": 4000})
    prompt = build_user_prompt(assign_uids([shown]), "f" * 16)
    assert "1001|     os.system(request.args['cmd'])" in prompt  # last line, real number
    assert "  100| def handler(request):" in prompt
    assert "  ...| [... " in prompt and "line(s) omitted: lines " in prompt
    assert "first and last lines" in shown["truncated"] and shown["partial"] is True
    assert "## Unit U1: `big.py` function `handler` lines 100-1001" in prompt
    # A finding quoting the sink anchors to its real line.
    [f] = validate_findings([{"unit": "U1", "title": "cmd", "severity": "high",
                              "quoted_code": "os.system(request.args['cmd'])"}],
                            [shown], set(), set())
    assert f["line"] == 1001
    # Nothing matches inside an omission marker.
    assert validate_findings([{"unit": "U1", "title": "x",
                               "quoted_code": "line(s) omitted: lines"}],
                             [shown], set(), set()) == []


def test_files_mode_window_is_centred_on_the_changed_lines():
    unit = {**_sink_at_end(), "changed_lines": [600, 601]}
    [[shown]], _ = plan_review_prompts(_review([unit]), **{**BUDGET, "max_prompt_tokens": 4000})
    numbers = [n for n in shown["line_numbers"] if n is not None]
    assert numbers[0] == 100  # the signature is kept
    assert 600 in numbers and 601 in numbers
    window = numbers[1:]
    assert window == list(range(window[0], window[-1] + 1))  # one contiguous window
    assert abs((600 - window[0]) - (window[-1] - 601)) <= 1  # centred on the change
    assert shown["partial"] is False  # every changed line was shown
    assert "around the changed lines" in shown["truncated"]
    assert _prompt_tokens([shown]) <= 4000


def test_changed_lines_too_far_apart_get_one_window_each():
    unit = {**_sink_at_end(), "changed_lines": [150, 1001]}
    [[shown]], _ = plan_review_prompts(_review([unit]), **{**BUDGET, "max_prompt_tokens": 4000})
    numbers = [n for n in shown["line_numbers"] if n is not None]
    assert {100, 150, 1001} <= set(numbers)
    assert shown["line_numbers"].count(None) == 1  # two windows (signature + 150, and 1001)
    assert "1001|     os.system(request.args['cmd'])" in build_user_prompt(
        assign_uids([shown]), "f" * 16)


def test_unit_that_cannot_fit_at_all_is_reported():
    batches, dropped = plan_review_prompts(
        _review([_unit("f")]), **{**BUDGET, "max_prompt_tokens": 50})
    assert batches == [] and dropped[0]["not_reviewed_reason"] == "too_large"


# --- through run_scan ----------------------------------------------------------


class NoRetrievalMerger:
    async def analyze_units(self, units):
        return {"ghost_hunter_findings": [], "team_memory_findings": [], "is_vulnerable": False}


class ScriptedRouter:
    """Fails the calls whose index is in ``fail``; otherwise flags line 2 of U1."""

    mock = False

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.prompts = []

    async def generate(self, system, user):
        idx = len(self.prompts)
        self.prompts.append(user)
        if idx in self.fail:
            raise LLMError("groq:stub HTTP 500")
        return {"findings": [{"unit": "U1", "severity": "low", "title": f"t{idx}",
                              "quoted_code": "y0 = x + 0"}]}, "groq:stub"


def _run_files(n_units, router, monkeypatch, **limits):
    from backend.app.db.models import Scan
    from backend.app.db.session import SessionLocal, init_db

    for k, v in limits.items():
        monkeypatch.setattr(settings, k, v)
    init_db()
    content = "".join(_unit(f"f{i}")["code"] + "\n" for i in range(n_units))
    job_id = uuid.uuid4().hex
    with SessionLocal() as session:
        session.add(Scan(id=job_id, status="queued", mode="files", request_json=json.dumps(
            {"files": [{"path": "m.py", "content": content}]})))
        session.commit()
    scan_runner.reset_scan_semaphore()
    asyncio.run(scan_runner.run_scan(job_id, NoRetrievalMerger(), router))
    with SessionLocal() as session:
        scan = session.get(Scan, job_id)
        return scan.status, json.loads(scan.result_json) if scan.result_json else None


def test_scan_reports_units_beyond_the_call_cap(monkeypatch):
    router = ScriptedRouter()
    status, result = _run_files(5, router, monkeypatch, LLM_MAX_UNITS_PER_PROMPT=2,
                                LLM_MAX_CALLS_PER_SCAN=2)
    assert status == "completed"
    assert result["llm_calls"] == 2 == len(router.prompts)
    assert [u["reason"] for u in result["units_not_reviewed"]] == ["budget"]
    assert "1 unit(s) NOT reviewed by the LLM (LLM budget: 2 call(s)" in result["report_markdown"]
    assert len(result["report_findings"]) == 2  # one per successful call
    assert result["llm_provider_used"] == "groq:stub"


def test_partial_llm_failure_keeps_the_rest_and_says_so(monkeypatch):
    status, result = _run_files(4, ScriptedRouter(fail={0}), monkeypatch,
                                LLM_MAX_UNITS_PER_PROMPT=2)
    assert status == "completed"
    assert [u["reason"] for u in result["units_not_reviewed"]] == ["llm_error", "llm_error"]
    assert "LLM call failed" in result["report_markdown"]
    assert len(result["report_findings"]) == 1


def test_all_llm_calls_failing_fails_the_scan(monkeypatch):
    status, _ = _run_files(2, ScriptedRouter(fail={0, 1}), monkeypatch,
                           LLM_MAX_UNITS_PER_PROMPT=1)
    assert status == "failed"


@pytest.mark.parametrize("n_units", [1, 6])
def test_scan_that_fits_is_one_call(monkeypatch, n_units):
    router = ScriptedRouter()
    status, result = _run_files(n_units, router, monkeypatch)
    assert status == "completed" and result["llm_calls"] == 1
    assert result["units_not_reviewed"] == []
