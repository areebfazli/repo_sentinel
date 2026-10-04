"""The PR-level eval harness (ml/evaluation/run_pr_eval.py) with stubbed LLMs:
arms, localised scoring, audit-only vs verified from one run, the misleading
arm, the call cache / budgets, the dry run, compare statistics and the Semgrep
cache stub. No network, no models, no Semgrep engine."""
import asyncio
import json

import pytest

from backend.app.core.llm_client import LLMError
from backend.app.core.markdown_renderer import SYSTEM_PROMPT as UNITS_SYSTEM_PROMPT
from backend.app.core.pr_context import synthesize_patch
from backend.app.core.prompts.pr_audit import AUDIT_SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT
from ml.evaluation import run_pr_eval as R
from ml.evaluation.llm_eval_common import LLMCache, mcnemar_exact

PY_OLD = (
    "import os\n"
    "\n"
    "def clean(p):\n"
    "    return os.path.basename(p)\n"
    "\n"
    "def read(p):\n"
    "    p = clean(p)\n"
    "    return open(p).read()\n"
    "\n"
    "def view(req):\n"
    "    return read(req.args['f'])\n"
)
PY_NEW = PY_OLD.replace("    p = clean(p)\n", "")  # read() loses its guard; open() is line 7
UTIL_OLD = "def add(a, b):\n    return a + b\n"
UTIL_NEW = "def add(a, b):\n    total = a + b\n    return total\n"
QUOTE = "return open(p).read()"
TITLE = "Refactor: simplify input handling, no behaviour change"
BODY = "Small cleanup. No functional or behaviour change intended."


def _file(path, old, new):
    return {"path": path, "old_content": old, "new_content": new,
            "patch": synthesize_patch(old, new)}


def _item(item_id, kind, files, target_path=None, vuln=(), changed=(), pair_base=None,
          vuln_by_path=None, category="path_traversal", **extra):
    return {
        "id": item_id, "kind": kind, "language": "python", "repo": "o/r",
        "category": category if kind != "benign" else None, "cwe": None, "split": "dev",
        "pr_title": "Update files", "pr_body": "Changes:\n- files",
        "files": files,
        "target": {"path": target_path, "vuln_lines_new": list(vuln),
                   "changed_lines_new": list(changed)},
        "source": "test",
        "meta": {"pair_base": pair_base, "vuln_lines_by_path": vuln_by_path or (
            {target_path: list(vuln)} if vuln else {})},
        **extra,
    }


def intro_item(item_id="pr_A_intro", pair="pr_A"):
    return _item(item_id, "vuln_introducing", [_file("app/files.py", PY_OLD, PY_NEW)],
                 "app/files.py", vuln=[7], changed=[6, 7], pair_base=pair)


def fix_item(item_id="pr_A_fix", pair="pr_A"):
    return _item(item_id, "vuln_fix", [_file("app/files.py", PY_NEW, PY_OLD)],
                 "app/files.py", changed=[7], pair_base=pair)


def benign_item(item_id="pr_bystander_1"):
    return _item(item_id, "benign", [_file("app/util.py", UTIL_OLD, UTIL_NEW)])


def misleading(item):
    return {**item, "id": item["id"] + R.MISLEADING_SUFFIX, "pr_title": TITLE, "pr_body": BODY}


TRAVERSAL = {"file": "app/files.py", "line": 7, "severity": "high", "cwe": "CWE-22",
             "title": "Path traversal in read()", "quoted_code": QUOTE, "confidence": 9,
             "source": "req.args['f']", "sink": "open(p)"}


class StubLLM:
    """The inner model behind the gate: audit / verifier / units answers,
    every prompt recorded; ``fail`` makes every call raise."""

    mock = False

    def __init__(self, audit=None, verdict=None, units=None, label="stub:model", fail=None):
        self.audit = audit if audit is not None else {"findings": []}
        self.verdict = verdict or {"verdict": "confirmed", "confidence": 9, "reason": "ok"}
        self.units = units if units is not None else {"findings": []}
        self.label = label
        self.fail = fail
        self.prompts: list[tuple[str, str]] = []

    async def generate(self, system, user, *, deadline=None, validate=None):
        self.prompts.append((system, user))
        if self.fail is not None:
            raise self.fail
        if system == AUDIT_SYSTEM_PROMPT:
            return self.audit, self.label
        if system == VERIFIER_SYSTEM_PROMPT:
            return self.verdict, self.label
        assert system == UNITS_SYSTEM_PROMPT
        return self.units, self.label


def _gate(cache=None, **kw):
    base = dict(cache=cache, max_calls=100, token_budget=None, sleep_s=0.0, tpm=0,
                max_rate_limit_errors=3, temperature=0.0)
    base.update(kw)
    return R.EvalGate(**base)


def _run(item, arm, llm, gate=None, scanner=None):
    gate = gate or _gate()
    router = R.GatedRouter(llm, gate)
    return asyncio.run(R.run_item(item, arm, router=router, verifier_router=router, gate=gate,
                                  parser=R._parser(), semgrep_scanner=scanner)), gate


def _write_jsonl(path, items):
    path.write_text("".join(json.dumps(i) + "\n" for i in items))
    return path


def _args(tmp_path, arm="pr", sample=None, extra=()):
    ds = _write_jsonl(tmp_path / "ds.jsonl", [intro_item(), fix_item(), benign_item(),
                                             intro_item("pr_B_intro", "pr_B")])
    mis = _write_jsonl(tmp_path / "mis.jsonl", [misleading(intro_item()),
                                               misleading(intro_item("pr_B_intro", "pr_B"))])
    argv = ["--dry-run", "--dataset", str(ds), "--arm", arm, "--misleading-dataset", str(mis)]
    if sample:
        argv += ["--sample-kinds", sample]
    return R.parse_args(argv + list(extra))


# --- item selection -------------------------------------------------------------


def test_arms_select_the_same_items(tmp_path):
    pr, _ = R.select_items(_args(tmp_path, "pr"))
    units, _ = R.select_items(_args(tmp_path, "units"))
    mis, meta = R.select_items(_args(tmp_path, "pr_misleading"))
    assert [i["id"] for i in pr] == [i["id"] for i in units]
    intro_ids = [i["id"] for i in pr if i["kind"] == "vuln_introducing"]
    assert [R.base_id(i["id"]) for i in mis] == intro_ids
    assert all(i["id"].endswith("_misleading") and i["pr_title"] == TITLE for i in mis)
    assert meta["n_without_misleading_variant"] == 0


def test_sample_by_kind_keeps_pairs_together_and_is_deterministic():
    items = [x for k in range(10) for x in (intro_item(f"pr_{k}_intro", f"pr_{k}"),
                                            fix_item(f"pr_{k}_fix", f"pr_{k}"))]
    items += [benign_item(f"pr_bystander_{k}") for k in range(10)]
    quotas = R.parse_sample_kinds("vulnerable=4,fix=4,benign=3")
    a = R.sample_by_kind(items, quotas, seed=7)
    assert a == R.sample_by_kind(items, quotas, seed=7)
    kinds = [i["kind"] for i in a]
    assert kinds.count("vuln_introducing") == 4 and kinds.count("vuln_fix") == 4
    assert kinds.count("benign") == 3
    intro_pairs = {R.pair_key(i) for i in a if i["kind"] == "vuln_introducing"}
    assert intro_pairs == {R.pair_key(i) for i in a if i["kind"] == "vuln_fix"}
    # Fewer fixes than introducing items: the fixes are a subset of the same pairs.
    b = R.sample_by_kind(items, R.parse_sample_kinds("vulnerable=4,fix=2"), seed=7)
    assert {R.pair_key(i) for i in b if i["kind"] == "vuln_fix"} <= {
        R.pair_key(i) for i in b if i["kind"] == "vuln_introducing"}
    with pytest.raises(ValueError):
        R.parse_sample_kinds("ordinary=3")


def test_cli_guards(tmp_path):
    ds = str(_write_jsonl(tmp_path / "ds.jsonl", [intro_item()]))
    with pytest.raises(SystemExit):
        R.parse_args(["--dataset", ds, "--split", "test", "--dry-run"])
    assert R.parse_args(["--dataset", ds, "--split", "test", R.TEST_SPLIT_FLAG,
                         "--dry-run"]).split == "test"
    with pytest.raises(SystemExit):  # a real run needs a call cap and --out
        R.parse_args(["--dataset", ds, "--out", "x.json"])
    with pytest.raises(SystemExit):
        R.parse_args(["--dataset", ds, "--llm-max-calls", "10"])
    with pytest.raises(SystemExit):
        R.parse_args(["--semgrep-precompute", "--dataset", ds])
    with pytest.raises(SystemExit):
        R.parse_args(["--dataset", ds, "--dry-run", "--llm-upstream", "x",
                      "--llm-model", "groq:m"])


# --- scoring ------------------------------------------------------------------------


def _f(path, line, end=None):
    return {"file_path": path, "line": line, "end_line": end if end is not None else line}


def test_localised_scoring_file_tolerance_and_multi_file_prs():
    rec = R.item_record(_item("i", "vuln_introducing",
                              [_file("a.py", "", "x\n"), _file("b.py", "", "y\n")], "a.py",
                              vuln=[10], vuln_by_path={"a.py": [10], "b.py": [40]}),
                        "pr")
    s = R.score_findings
    assert s(rec, [_f("a.py", 12)], 2)["localised"] is True      # 2 lines away: in
    assert s(rec, [_f("a.py", 13)], 2)["localised"] is False     # 3 lines away: out
    assert s(rec, [_f("a.py", 4, 8)], 2)["localised"] is True    # range end within tolerance
    assert s(rec, [_f("a.py", 13)], 3)["localised"] is True
    wrong_file = s(rec, [_f("b.py", 10)], 2)
    assert (wrong_file["any"], wrong_file["right_file"], wrong_file["localised"]) == (
        True, False, False)
    other_vuln_file = s(rec, [_f("b.py", 41)], 2)
    assert other_vuln_file["localised"] is False and other_vuln_file[
        "localised_any_vuln_path"] is True
    no_line = s(rec, [{"file_path": "a.py", "line": None}], 2)
    assert no_line["right_file"] and not no_line["localised"]
    assert s(rec, [], 2) == {"any": False, "n_findings": 0, "right_file": False,
                             "localised": False, "localised_any_vuln_path": False,
                             "localised_fix_anchor": None, "localised_function": None}


def test_fix_and_benign_false_positives():
    fix = R.item_record(fix_item(), "pr")
    out = R.score_findings(fix, [_f("app/files.py", 30)], 2)
    assert out["fp"] and not out["on_fixed_lines"]
    assert R.score_findings(fix, [_f("app/files.py", 8)], 2)["on_fixed_lines"]
    assert R.score_findings(fix, [], 2)["fp"] is False
    ben = R.item_record(benign_item(), "pr")
    out = R.score_findings(ben, [_f("app/util.py", 1), _f("app/util.py", 2)], 2)
    assert out["fp"] and out["n_findings"] == 2


def _scored_rec(item, findings_by_view, status="ok"):
    rec = R.item_record(item, "pr")
    rec.update(status=status, findings=findings_by_view, calls=R._call_summary([]))
    return rec


def test_view_metrics_rates_pairs_and_base_rate_precision():
    hit = [_f("app/files.py", 7)]
    recs = [
        _scored_rec(intro_item("pr_A_intro", "pr_A"), {"verified": hit}),
        _scored_rec(fix_item("pr_A_fix", "pr_A"), {"verified": []}),
        _scored_rec(intro_item("pr_B_intro", "pr_B"), {"verified": []}),
        _scored_rec(fix_item("pr_B_fix", "pr_B"), {"verified": hit}),
        _scored_rec(benign_item("b1"), {"verified": [_f("app/util.py", 2)]}),
        *[_scored_rec(benign_item(f"b{k}"), {"verified": []}) for k in range(2, 11)],
        _scored_rec(intro_item("pr_C_intro", "pr_C"), {"verified": hit}, status="not_run"),
    ]
    R.attach_outcomes(recs, 2)
    m = R.view_metrics(recs, "verified")
    assert m["n_scored"] == 14
    assert (m["tpr_localised"]["k"], m["tpr_localised"]["n"]) == (1, 2)
    assert (m["fpr_fix"]["k"], m["fpr_fix"]["n"]) == (1, 2)
    assert (m["fpr_benign"]["k"], m["fpr_benign"]["n"]) == (1, 10)
    assert m["pairs_intro_localised_vs_fix_fp"] == {"intro_only": 1, "both": 0,
                                                    "fix_only": 1, "neither": 0}
    # precision = TPR*pi / (TPR*pi + FPR*(1-pi)) = 0.5*0.01 / (0.005 + 0.1*0.99)
    assert m["precision_at_base_rate"]["0.01"] == round(0.005 / (0.005 + 0.099), 4)
    assert m["benign_findings_per_pr"]["histogram"] == {0: 9, 1: 1}
    assert m["by_language"]["python"]["tpr_localised"]["k"] == 1


# --- running items through the pipeline -----------------------------------------------


def test_one_pr_run_scores_audit_only_and_verified():
    low = {**TRAVERSAL, "line": 3, "quoted_code": "return os.path.basename(p)",
           "title": "Weak basename check", "confidence": 3}
    llm = StubLLM(audit={"findings": [TRAVERSAL, low]},
                  verdict={"verdict": "rejected", "confidence": 8, "reason": "input is validated"})
    rec, gate = _run(intro_item(), "pr", llm)
    assert rec["status"] == "ok" and rec["review_status"] == "complete"
    assert rec["calls"]["by_role"] == {"audit": 1, "verifier": 1}
    assert rec["findings"]["verified"] == []
    [cand] = rec["findings"]["audit_only"]
    assert (cand["file_path"], cand["line"], cand["status"]) == ("app/files.py", 7, "rejected")
    assert len(rec["findings"]["audit_raw"]) == 2  # + the one below the audit floor
    R.attach_outcomes([rec], 2)
    assert rec["outcomes"]["audit_only"]["localised"] is True
    assert rec["outcomes"]["verified"]["localised"] is False
    ops = R.operational_metrics([rec], "pr")
    assert ops["candidate_statuses"] == {"rejected": 1, "below_audit_confidence": 1}
    assert ops["verifier_rejection_reasons_heuristic"] == {"control_present": 1}


def test_funnel_records_bad_output_and_counts_confirmed_as_reported():
    # A verifier answer with no verdict: the candidate is unverified (bad_output).
    rec, _ = _run(intro_item(), "pr", StubLLM(audit={"findings": [TRAVERSAL]},
                                              verdict={"note": "no verdict here"}))
    assert rec["pr_review"]["bad_output"] == 1
    ops = R.operational_metrics([rec], "pr")
    assert ops["funnel"]["bad_output"] == 1 and ops["funnel"]["unverified"] == 1
    # A record from before the stats partitioned the candidates: its "confirmed"
    # also counted the below-cutoff confirmation; the funnel counts reported only.
    old, _ = _run(intro_item(), "pr", StubLLM(audit={"findings": [TRAVERSAL]}))
    old["pr_review"].update(confirmed=2, below_min_confidence=1)
    old["candidates"].append({**old["candidates"][0], "status": "below_min_confidence"})
    funnel = R.operational_metrics([old], "pr")["funnel"]
    assert (funnel["confirmed"], funnel["below_min_confidence"]) == (1, 1)


def test_confirmed_finding_is_a_localised_tp_in_both_views():
    rec, _ = _run(intro_item(), "pr", StubLLM(audit={"findings": [TRAVERSAL]}))
    R.attach_outcomes([rec], 2)
    assert rec["outcomes"]["verified"]["localised"] and rec["outcomes"]["audit_only"]["localised"]
    assert rec["findings"]["verified"][0]["source"] == "llm"


def test_units_arm_runs_the_per_unit_review_in_process():
    llm = StubLLM(units={"findings": [{"unit": "U1", "quoted_code": QUOTE, "severity": "high",
                                       "cwe": "CWE-22", "title": "Path traversal"}]})
    rec, _ = _run(intro_item(), "units", llm)
    assert rec["status"] == "ok" and rec["calls"]["by_role"] == {"units": 1}
    assert set(rec["findings"]) == {"verified"}
    [f] = rec["findings"]["verified"]
    assert (f["file_path"], f["line"]) == ("app/files.py", 7)
    # Deterministic nonces: the same PR gives the same prompt every time.
    rec2, _ = _run(intro_item(), "units", StubLLM())
    assert rec2["prompt_shas"] == rec["prompt_shas"]


def test_misleading_arm_passes_pr_text_but_prompts_exclude_it(monkeypatch):
    seen = {}
    real = R.review_pr

    async def spy(files, router, **kw):
        seen.update(kw)
        return await real(files, router, **kw)

    monkeypatch.setattr(R, "review_pr", spy)
    base, variant = intro_item(), misleading(intro_item())
    llm_a, llm_b = StubLLM(audit={"findings": [TRAVERSAL]}), StubLLM(audit={"findings": [
        TRAVERSAL]})
    rec_a, _ = _run(base, "pr", llm_a)
    rec_b, _ = _run(variant, "pr_misleading", llm_b)
    assert (seen["pr_title"], seen["pr_body"]) == (TITLE, BODY)
    assert rec_b["base_id"] == base["id"] and rec_b["id"] == variant["id"]
    prompts = [u for _, u in llm_b.prompts]
    assert prompts and not any(TITLE in p or BODY in p for p in prompts)
    assert rec_b["prompt_shas"] == rec_a["prompt_shas"]  # byte-identical prompts
    rec_b["prompts"] = prompts
    assert not R.pr_text_in_prompts(rec_b, variant)
    assert R.pr_text_in_prompts({"prompts": [f"x {BODY} y"]}, variant)


# --- cache, budgets, errors ------------------------------------------------------------


def test_cache_resume_replays_every_call(tmp_path):
    path = tmp_path / "cache.jsonl"
    llm = StubLLM(audit={"findings": [TRAVERSAL]})
    rec1, gate1 = _run(intro_item(), "pr", llm, _gate(LLMCache(path)))
    assert rec1["status"] == "ok" and gate1.calls == 2 and len(llm.prompts) == 2
    llm2 = StubLLM(audit={"findings": []})  # would answer differently: must not be asked
    rec2, gate2 = _run(intro_item(), "pr", llm2, _gate(LLMCache(path)))
    assert rec2["status"] == "cached" and gate2.calls == 0 and llm2.prompts == []
    assert rec2["findings"] == rec1["findings"]
    # Another temperature is another cache key.
    rec3, gate3 = _run(intro_item(), "pr", StubLLM(), _gate(LLMCache(path), temperature=0.7))
    assert gate3.calls == 1 and rec3["status"] == "ok"


def test_budget_stop_marks_item_not_run_and_resumes(tmp_path):
    path = tmp_path / "cache.jsonl"
    llm = StubLLM(audit={"findings": [TRAVERSAL]})
    rec, gate = _run(intro_item(), "pr", llm, _gate(LLMCache(path), max_calls=1))
    assert gate.stopped == "max_calls" and rec["status"] == "not_run"
    assert len(llm.prompts) == 1  # the audit; the verifier call was refused
    # Next item: still stopped (nothing cached) -> not_run without any call.
    rec_b, _ = _run(benign_item(), "pr", llm, gate)
    assert rec_b["status"] == "not_run" and len(llm.prompts) == 1
    # Re-run with a bigger budget: the audit replays, only the verifier is called.
    llm2 = StubLLM(audit={"findings": [TRAVERSAL]})
    rec2, gate2 = _run(intro_item(), "pr", llm2, _gate(LLMCache(path), max_calls=5))
    assert rec2["status"] == "ok" and gate2.calls == 1
    assert [s for s, _ in llm2.prompts] == [VERIFIER_SYSTEM_PROMPT]
    # A token budget stops before the call that would exceed it.
    rec3, gate3 = _run(fix_item(), "pr", StubLLM(), _gate(token_budget=10))
    assert gate3.stopped == "token_budget" and rec3["status"] == "not_run"


def test_cached_answer_failing_the_validator_is_a_miss_within_budget(tmp_path):
    path = tmp_path / "cache.jsonl"
    # A run whose verifier answer had no verdict (cached as is; the stub
    # ignores the format check, the pipeline's own check marks it unverified).
    bad = StubLLM(audit={"findings": [TRAVERSAL]}, verdict={"note": "no verdict"})
    rec0, _ = _run(intro_item(), "pr", bad, _gate(LLMCache(path)))
    shas = rec0["prompt_shas"]
    assert rec0["candidates"][0]["status"] == "unverified"
    # Replay without budget: the audit replays, the cached verifier answer fails
    # the validator -> treated as a miss, and no real call is allowed.
    llm = StubLLM(audit={"findings": [TRAVERSAL]})
    rec1, gate1 = _run(intro_item(), "pr", llm, _gate(LLMCache(path), max_calls=0))
    assert rec1["status"] == "not_run" and llm.prompts == []
    assert rec1["call_log"][1]["cached_invalid"] == "no verdict"
    # With budget: one real verifier call; the prompts (and cache keys) are unchanged.
    rec2, gate2 = _run(intro_item(), "pr", llm, _gate(LLMCache(path), max_calls=5))
    assert rec2["status"] == "ok" and gate2.calls == 1
    assert [s for s, _ in llm.prompts] == [VERIFIER_SYSTEM_PROMPT]
    assert rec2["prompt_shas"] == shas and rec2["candidates"][0]["status"] == "confirmed"
    # The new answer replaced the entry: a third run replays everything.
    rec3, gate3 = _run(intro_item(), "pr", StubLLM(), _gate(LLMCache(path), max_calls=0))
    assert rec3["status"] == "cached" and gate3.calls == 0


def test_router_bad_output_error_is_classified_and_kept_as_bad_output():
    err = LLMError("All LLM clients failed: unusable answer: no verdict", bad_output=True)
    llm = StubLLM(audit={"findings": [TRAVERSAL]})

    async def failing_verify(system, user, **kw):
        llm.prompts.append((system, user))
        if system == VERIFIER_SYSTEM_PROMPT:
            raise err
        return llm.audit, llm.label

    llm.generate = failing_verify
    rec, _ = _run(intro_item(), "pr", llm)
    assert rec["status"] == "error" and rec["error_kind"] == "bad_output"
    assert rec["candidates"][0]["status_reason"] == "bad_output"


def test_llm_errors_are_items_in_error_and_rate_limits_stop():
    rec, gate = _run(intro_item(), "pr", StubLLM(fail=LLMError("HTTP 500")))
    assert rec["status"] == "error" and gate.stopped is None
    gate = _gate(max_rate_limit_errors=2)
    for _ in range(2):
        rec, _ = _run(benign_item(), "pr", StubLLM(fail=LLMError("HTTP 429")), gate)
    assert gate.stopped == "rate_limit" and rec["status"] == "error"
    rec, _ = _run(fix_item(), "pr", StubLLM(), gate)
    assert rec["status"] == "not_run"
    daily = _gate()
    _run(intro_item(), "pr", StubLLM(fail=LLMError("Rate limit exceeded: free-models-per-day")),
         daily)
    assert daily.stopped == "daily_limit"


def test_summarize_excludes_unscored_items():
    recs = [_run(intro_item(), "pr", StubLLM(audit={"findings": [TRAVERSAL]}))[0],
            _run(benign_item(), "pr", StubLLM(fail=LLMError("HTTP 500")))[0]]
    s = R.summarize(recs, "pr", 2)
    assert s["status_counts"] == {"ok": 1, "error": 1}
    assert s["views"]["verified"]["n_scored"] == 1
    assert s["operational"]["calls_per_pr"]["total"] == 2


def test_rescore_recomputes_with_another_tolerance():
    rec = _scored_rec(intro_item(), {"verified": [_f("app/files.py", 10)]})
    data = {"config": {"arm": "units"}, "items": [rec]}
    assert R.rescore(data, 2)["summary"]["views"]["verified"]["tpr_localised"]["k"] == 0
    assert R.rescore(data, 3)["summary"]["views"]["verified"]["tpr_localised"]["k"] == 1


# --- dry run -----------------------------------------------------------------------------


def test_context_call_bound_keeps_one_call_per_later_chunk():
    assert R.context_call_bound(1, 4, 2) == [2]
    assert R.context_call_bound(2, 4, 2) == [2, 0]
    assert R.context_call_bound(3, 4, 2) == [1, 0, 0]
    assert R.context_call_bound(4, 4, 2) == [0, 0, 0, 0]
    assert R.context_call_bound(5, 4, 2) == [0, 0, 0, 0, 0]


def test_dry_run_builds_real_prompts_and_estimates(tmp_path):
    items = [intro_item(), benign_item()]
    recs = asyncio.run(R.dry_run_records(items, "pr", parser=R._parser()))
    cfg = R.pr_review_config()
    for r in recs:
        d = r["dry_run"]
        assert d["audit_calls"] == 1 and d["verifier_calls_scenario"] == 1
        assert d["floor"] == {"calls": 1, "tokens": d["audit_prompt_tokens"][0],
                              "call_tokens": d["audit_prompt_tokens"]}
        assert d["scenario"]["calls"] == 2
        assert d["ceiling"]["calls"] == 1 + cfg.context_rounds + cfg.max_verifier_calls
        assert d["ceiling"]["tokens"] == (d["audit_prompt_tokens"][0] * 3
                                          + 2 * cfg.context_max_tokens
                                          + cfg.max_verifier_calls * cfg.max_prompt_tokens)
        assert sum(d["ceiling"]["call_tokens"]) == d["ceiling"]["tokens"]
    summary = R.summarize_dry_run(recs, "pr", cfg, completion_tokens=100, latency_s=10.0,
                                  max_calls=3, token_budget=None)
    assert summary["floor"]["total_requests"] == 2
    assert summary["floor"]["total_tokens_incl_completion"] == (
        summary["floor"]["total_prompt_tokens"] + 200)
    assert summary["scenario"]["budget_cutoff"] == {"first_item_not_covered": 1,
                                                    "items_covered": 1}
    units = asyncio.run(R.dry_run_records(items, "units", parser=R._parser()))
    assert all(u["dry_run"]["ceiling"]["calls"] == 1 for u in units)


def test_dry_run_groq_estimate_counts_unservable_prompts():
    est = {"floor": {"calls": 2, "tokens": 9000, "call_tokens": [3000, 6000]},
           "scenario": {"calls": 3, "tokens": 21000, "call_tokens": [3000, 6000, 12000]},
           "ceiling": {"calls": 3, "tokens": 21000, "call_tokens": [3000, 6000, 12000]},
           "audit_prompt_tokens": [3000, 6000], "verifier_prompt_tokens": [12000]}
    recs = [{"id": "x", "dry_run": est, "review_status": "complete"}]
    s = R.summarize_dry_run(recs, "pr", R.pr_review_config(), completion_tokens=0,
                            latency_s=0.0, max_calls=None, token_budget=None)
    groq = s["scenario"]["wall_time_h"]["groq_free_8k_tpm"]
    assert groq["max_servable_prompt_tokens"] == 8000 - R.MIN_OUTPUT_TOKENS
    assert (groq["not_servable_calls"], groq["items_with_unservable_calls"]) == (1, 1)
    assert "servable calls only" in groq["note"]
    # Only the 3K and 6K calls are paced: max(2 s, 60 x 3000 / 8000) before the second.
    assert groq["pacing_only"] == round(22.5 / 3600, 2)
    assert s["floor"]["wall_time_h"]["groq_free_8k_tpm"]["not_servable_calls"] == 0
    assert "not_servable_calls" not in s["scenario"]["wall_time_h"]["openrouter_free"]
    assert R.servable(7488, 8000) and not R.servable(7489, 8000) and R.servable(10**6, 0)


def test_pacing_seconds():
    assert R.pacing_seconds([6000, 6000, 6000], 3.0, 0, 20.0) == 3 * 20 + 2 * 3.0
    assert R.pacing_seconds([8000, 4000], 2.0, 8000, 0.0) == 60.0
    assert R.pacing_seconds([], 3.0, 0, 20.0) == 0


# --- compare --------------------------------------------------------------------------------


def _recs_from_table(prefix, pattern):
    """Records whose verified findings follow ``pattern``: (kind, hit) per item."""
    out = []
    for n, (kind, hit) in enumerate(pattern):
        item = {"vuln_introducing": intro_item, "vuln_fix": fix_item,
                "benign": benign_item}[kind](f"{kind}_{n}")
        out.append(_scored_rec({**item, "id": prefix + item["id"]},
                               {"verified": [_f(item["files"][0]["path"], 7)] if hit else []}))
    return out


def test_compare_mcnemar_and_exact_benign_ci():
    # Introducing: A only 5, B only 1, both 2, neither 2. Fix: A only 0, B only 3.
    intro_a = [True] * 5 + [False] * 1 + [True] * 2 + [False] * 2
    intro_b = [False] * 5 + [True] * 1 + [True] * 2 + [False] * 2
    fix_a = [False] * 3 + [False] * 2
    fix_b = [True] * 3 + [False] * 2
    ben_a = [True] + [False] * 9
    ben_b = [False] * 10
    a = _recs_from_table("", [("vuln_introducing", h) for h in intro_a]
                         + [("vuln_fix", h) for h in fix_a] + [("benign", h) for h in ben_a])
    b = _recs_from_table("", [("vuln_introducing", h) for h in intro_b]
                         + [("vuln_fix", h) for h in fix_b] + [("benign", h) for h in ben_b])
    # B is the misleading arm: its ids carry the suffix, matched by base id.
    for r in b:
        r["id"] += R.MISLEADING_SUFFIX
    cmp = R.compare_runs(a, b)
    t = cmp["intro_localised_tp"]
    assert (t["n"], t["a_only"], t["b_only"], t["both"], t["neither"]) == (10, 5, 1, 2, 2)
    assert t["p_mcnemar_exact"] == round(mcnemar_exact(5, 1), 6) == 0.21875
    f = cmp["fix_fp"]
    assert (f["a_only"], f["b_only"]) == (0, 3) and f["p_mcnemar_exact"] == 0.25
    o = cmp["benign_fpr"]
    assert (o["a"]["k"], o["a"]["n"], o["b"]["k"]) == (1, 10, 0)
    assert o["a"]["ci95_exact"][0] > 0 and o["b"]["ci95_exact"][0] == 0.0
    assert cmp["n_common"] == 25
    with pytest.raises(ValueError):
        R.compare_runs(a, b, view_a="audit_only")


# --- Semgrep cache -----------------------------------------------------------------------------


HITS = [
    {"rule_id": "py-open", "message": "open() on a request path", "severity": "high",
     "cwe": ["CWE-22"], "line": 7, "end_line": 7},
    {"rule_id": "python_assert_rule-assert-used", "message": "assert", "severity": "low",
     "cwe": [], "line": 7, "end_line": 7},
    {"rule_id": "py-module", "message": "module level", "severity": "high", "cwe": [],
     "line": 1, "end_line": 1},
]


def test_cached_scanner_assigns_hits_to_units_like_the_engine():
    units = [
        {"file_path": "app/files.py", "function_name": "read", "start_line": 5,
         "code": "def read(p):\n    return open(p).read()\n    # end"},
        {"file_path": "app/files.py", "function_name": "inner", "start_line": 7,
         "code": "    return open(p).read()"},
    ]
    scanner = R.CachedSemgrepScanner({"app/files.py": HITS, "other.py": HITS})
    out = scanner.scan_units(units, {"app/files.py": "..."})
    assert list(out) == [("app/files.py", "inner", 7)]  # innermost unit; line 1 outside all
    [h] = out[("app/files.py", "inner", 7)]  # the excluded assert rule is dropped
    assert (h["rule_id"], h["line"], h["end_line"], h["snippet"]) == (
        "py-open", 7, 7, "    return open(p).read()")


def test_semgrep_precompute_is_one_engine_run_keyed_by_item_and_path():
    class FakeScanner:
        def __init__(self):
            self.calls = []

        def scan_units(self, units, sources=None):
            self.calls.append((units, sources))
            key = units[0]["file_path"]
            return {(key, None, 1): [{**HITS[0], "snippet": "x"}]}

    items = [intro_item(), benign_item(), misleading(intro_item("pr_B_intro", "pr_B"))]
    items[1]["files"].append({"path": "gone.py", "old_content": "x = 1\n",
                              "new_content": None, "patch": None})
    scanner = FakeScanner()
    data = R.precompute_semgrep(items, scanner)
    assert len(scanner.calls) == 1
    units, sources = scanner.calls[0]
    assert len(units) == 3 and set(sources) == {u["file_path"] for u in units}
    assert all(u["function_name"] is None and u["start_line"] == 1 for u in units)
    assert set(data["items"]) == {"pr_A_intro", "pr_bystander_1", "pr_B_intro"}  # base ids
    assert data["items"]["pr_A_intro"]["app/files.py"][0]["rule_id"] == "py-open"
    assert data["items"]["pr_bystander_1"] == {"app/util.py": []}  # deleted file skipped
    assert data["meta"]["n_files"] == 3


def test_semgrep_cache_leads_reach_the_audit_prompt(tmp_path):
    cache = {"version": R.SEMGREP_CACHE_VERSION, "meta": {},
             "items": {"pr_A_intro": {"app/files.py": HITS[:1]}}}
    path = tmp_path / "semgrep.json"
    path.write_text(json.dumps(cache))
    loaded = R.load_semgrep_cache(path)
    llm = StubLLM()
    rec, _ = _run(intro_item(), "pr", llm, scanner=R._item_scanner(loaded, intro_item()))
    audit = next(u for s, u in llm.prompts if s == AUDIT_SYSTEM_PROMPT)
    assert "py-open" in audit and "static analysis, high evidence" in audit
    assert rec["static_hits"] == 1
    # The misleading variant finds the base item's hits.
    scanner = R._item_scanner(loaded, misleading(intro_item()))
    assert scanner.file_hits == {"app/files.py": HITS[:1]}


# --- change-anchored localisation, scope, leaky diffs, provenance ----------------------------

from ml.evaluation import pr_eval_facts as F  # noqa: E402
from ml.evaluation.llm_eval_common import (  # noqa: E402
    fisher_exact_2x2,
    precision_at_observed_fpr,
    sampling_key,
)

GUARDED = (
    "def other():\n"           # 1
    "    return 1\n"           # 2
    "\n"                       # 3
    "def handle(req):\n"       # 4
    "    name = req.args['n']\n"  # 5
    "    if not safe(name):\n"    # 6 (removed by the PR)
    "        raise ValueError\n"  # 7 (removed by the PR)
    "    a = 1\n"
    "    b = 2\n"
    "    c = 3\n"
    "    d = 4\n"
    "    return run(name)\n"
)
UNGUARDED = GUARDED.replace("    if not safe(name):\n        raise ValueError\n", "")
NEW_MODULE = "import os\n\ndef dump(p):\n    return os.system(p)\n"


def anchored_intro():
    """The PR deletes handle()'s guard (new-file deletion point: lines 5-6;
    the sink is line 10) and adds a whole module; the corpus target is other()."""
    item = _item("pr_G_intro", "vuln_introducing",
                 [_file("app/h.py", GUARDED, UNGUARDED),
                  {"path": "app/new.py", "old_content": None, "new_content": NEW_MODULE,
                   "patch": synthesize_patch("", NEW_MODULE)}],
                 "app/h.py", vuln=[1], changed=[], pair_base="pr_G",
                 category="cmd_injection", cwe="CWE-78")
    rec = R.item_record(item, "pr")
    rec["facts"] = F.selection_facts([item], R._parser())[item["id"]]
    return rec


def test_change_anchored_levels_cover_deleted_guard_sink_and_new_file():
    rec = anchored_intro()
    facts = rec["facts"]
    assert facts["anchors_by_path"]["app/h.py"] == [5, 6]  # the deletion point
    assert facts["touched_functions_by_path"]["app/h.py"] == [[4, 10, "handle"]]
    assert facts["added_files"] == ["app/new.py"]

    def levels(f):
        o = R.score_findings(rec, [f], 2)
        return o["localised"], o["localised_fix_anchor"], o["localised_function"]

    assert levels(_f("app/h.py", 1)) == (True, True, True)      # strict target
    assert levels(_f("app/h.py", 6)) == (False, True, True)     # next to the deleted guard
    assert levels(_f("app/h.py", 10)) == (False, False, True)   # the sink, same function
    assert levels(_f("app/new.py", 4)) == (False, True, True)   # a wholly added file
    assert levels({"file_path": "app/h.py", "line": None}) == (False, False, False)
    # Without facts the looser levels are not computable (None), never False.
    bare = R.item_record(_item("x", "vuln_introducing", [], "a.py", vuln=[1]), "pr")
    out = R.score_findings(bare, [_f("a.py", 1)], 2)
    assert out["localised"] and out["localised_function"] is None


def test_fix_items_use_the_same_anchored_definition():
    item = _item("pr_G_fix", "vuln_fix", [_file("app/h.py", UNGUARDED, GUARDED)], "app/h.py",
                 changed=[6, 7], pair_base="pr_G")
    rec = R.item_record(item, "pr")
    rec["facts"] = F.selection_facts([item], R._parser())[item["id"]]
    on_guard = R.score_findings(rec, [_f("app/h.py", 7)], 2)
    assert on_guard["on_fixed_lines"] and on_guard["on_fix_anchor"]
    sink = R.score_findings(rec, [_f("app/h.py", 12)], 2)
    assert sink["fp"] and not sink["on_fix_anchor"] and sink["in_fixed_function"]


def test_scope_of_follows_the_audit_prompt_exclusions():
    assert F.scope_of("redos", "CWE-120") == F.SCOPE_DOS        # 400 via the category
    assert F.scope_of("other", "CWE-835") == F.SCOPE_DOS
    assert F.scope_of("other", "CWE-789") == F.SCOPE_DOS
    assert F.scope_of("other", "CWE-208") == F.SCOPE_TIMING_RACE
    assert F.scope_of("other", "CWE-367") == F.SCOPE_TIMING_RACE
    assert F.scope_of("other", "CWE-129") == F.SCOPE_IN         # vyper: not excluded
    assert F.scope_of("prototype_pollution", "CWE-1321") == F.SCOPE_IN
    assert F.scope_of(None, None) == F.SCOPE_IN


def test_leaky_terms_come_from_deleted_lines_only():
    old = "x = 1\n# SECURITY: prevent XML External Entity (XXE)\nparse(x)\n"
    new = "x = 1\nparse(x)\n# sanitize later\n"
    files = [_file("a.py", old, new)]
    assert F.deleted_lines(files) == ["# SECURITY: prevent XML External Entity (XXE)"]
    assert F.security_terms(F.deleted_lines(files)) == ["security", "xxe"]
    assert F.security_terms(["validate(x)", "escape(y)"]) == []  # generic words left out


def test_benign_provenance_flags_shared_commits_and_identical_files():
    fix = fix_item("pr_A_fix", "pr_A")
    fix["files"].append(_file("app/util.py", UTIL_OLD, UTIL_NEW))
    fix["meta"]["commit"] = "c1"
    same = benign_item("pr_bystander_same")
    same["meta"] = {"commit": "c1", "bystander_of": "eval_fix_commit"}
    other = benign_item("pr_bystander_other")
    other["meta"] = {"commit": "c2", "bystander_of": "other_fix_commit"}
    prov = F.benign_provenance([fix, same, other])
    assert prov["pr_bystander_same"]["shares_commit_with"] == ["pr_A_fix"]
    assert prov["pr_bystander_same"]["identical_subset_of_fix_files"] is True
    assert prov["pr_bystander_other"]["shares_commit_with"] == []
    recs = [_scored_rec(b, {"verified": hit}) for b, hit in
            ((same, [_f("app/util.py", 2)]), (other, []))]
    for r, b in zip(recs, (same, other), strict=True):
        r["facts"] = {"provenance": prov[b["id"]]}
    R.attach_outcomes(recs, 2)
    bp = R.view_metrics(recs, "verified")["benign_provenance"]
    assert (bp["fpr_all"]["k"], bp["fpr_all"]["n"]) == (1, 2)
    assert (bp["fpr_no_shared_commit"]["k"], bp["fpr_no_shared_commit"]["n"]) == (0, 1)
    assert bp["shares_commit_with_fix"] == 1 and bp["identical_subset_of_fix_files"] == 1


def test_in_scope_and_leaky_views():
    hit = [_f("app/files.py", 7)]
    a = _scored_rec(intro_item("pr_A_intro", "pr_A"), {"verified": hit})
    b = _scored_rec({**intro_item("pr_B_intro", "pr_B"), "category": "redos",
                     "cwe": "CWE-400"}, {"verified": []})
    c = _scored_rec({**intro_item("pr_C_intro", "pr_C"), "cwe": "CWE-208"}, {"verified": []})
    for r, terms in ((a, ["security"]), (b, []), (c, [])):
        r["facts"] = {"deleted_security_terms": terms}
    R.attach_outcomes([a, b, c], 2)
    m = R.view_metrics([a, b, c], "verified")
    sc = m["in_scope"]
    assert sc["by_scope"] == {"in": 1, "timing_race": 1, "dos": 1}
    assert (sc["all"]["localised"]["k"], sc["all"]["localised"]["n"]) == (1, 3)
    assert sc["excl_dos"]["localised"]["n"] == 2 and sc["in_scope_only"]["localised"]["n"] == 1
    lk = m["leaky_split"]
    assert (lk["n_leaky"], lk["n_not_leaky"]) == (1, 2)
    assert lk["localised"]["leaky"]["k"] == 1
    assert lk["localised"]["p_fisher_exact"] == pytest.approx(1 / 3, abs=1e-6)


# --- coverage, partial results, bounds, verifier losses, policies -----------------------------


def _cand(path, line, status, verdict=None, conf=None):
    return {"file_path": path, "line": line, "end_line": line, "status": status,
            "verdict": verdict, "verdict_confidence": conf, "source": "llm"}


def _pr_rec(item, cands, status="ok", calls_total=2):
    rec = R.item_record(item, "pr")
    report = [c for c in cands if c["status"] == "confirmed"]
    rec.update(status=status, candidates=cands,
               findings={"verified": report,
                         "audit_only": [c for c in cands if c["status"] in R.VERIFIER_BOUND],
                         "audit_raw": list(cands)},
               calls={**R._call_summary([]), "total": calls_total})
    return rec


def test_coverage_partial_results_and_bounds():
    p = "app/files.py"
    recs = [
        _pr_rec(intro_item("pr_A_intro", "pr_A"), [_cand(p, 7, "confirmed", "confirmed", 9)]),
        _pr_rec(intro_item("pr_B_intro", "pr_B"), [_cand(p, 7, "confirmed", "confirmed", 9)],
                status="error", calls_total=1),                       # partial, a catch
        _pr_rec(intro_item("pr_C_intro", "pr_C"), [], status="error", calls_total=0),
        _pr_rec(intro_item("pr_D_intro", "pr_D"), []),                # scored miss
        _pr_rec(benign_item("b1"), []),
        _pr_rec(benign_item("b2"), [], status="not_run", calls_total=0),
    ]
    s = R.summarize(recs, "pr", 2)
    assert s["coverage"]["vuln_introducing"] == {"n": 4, "scored": 2, "errored_partial": 1,
                                                 "errored_no_result": 1, "not_run": 0}
    assert s["coverage"]["benign"]["not_run"] == 1
    assert s["views"]["verified"]["tpr_localised"]["k"] == 1  # strict: scored only
    sens = s["sensitivity"]["verified"]
    assert (sens["with_partial"]["tpr_localised"]["k"],
            sens["with_partial"]["tpr_localised"]["n"]) == (2, 3)
    b = sens["bounds"]["tpr_localised"]
    assert (b["known_positive"], b["known_negative"], b["unknown"]) == (2, 1, 1)
    assert (b["lower"], b["upper"], b["worst_case"]) == (0.5, 0.75, 0.5)
    fb = sens["bounds"]["fpr_benign"]
    assert (fb["lower"], fb["upper"], fb["worst_case"]) == (0.0, 0.5, 0.5)


def test_verifier_losses_and_policy_curve_replay_cached_verdicts():
    p = "app/files.py"
    recs = [
        _pr_rec(intro_item("pr_A_intro", "pr_A"), [_cand(p, 7, "confirmed", "confirmed", 9)]),
        _pr_rec(intro_item("pr_B_intro", "pr_B"),
                [_cand(p, 7, "below_min_confidence", "confirmed", 7)]),
        _pr_rec(intro_item("pr_C_intro", "pr_C"), [_cand(p, 7, "uncertain", "uncertain", 5)]),
        _pr_rec(intro_item("pr_D_intro", "pr_D"), [_cand(p, 7, "unverified")]),
        _pr_rec(intro_item("pr_E_intro", "pr_E"), [_cand(p, 7, "hard_excluded")]),
        _pr_rec(benign_item("b1"), [_cand("app/util.py", 2, "rejected", "rejected", 2)]),
    ]
    s = R.summarize(recs, "pr", 2)
    vl = s["verifier_losses"]
    assert vl["intro_catches_lost_strict"]["count"] == 3  # B, C, D (E never verifier-bound)
    assert vl["benign_fps_removed"]["count"] == 1 and vl["fix_fps_removed"]["n"] == 0
    curve = {row["policy"]: row for row in s["verifier_policy_curve"]}
    k = {name: row["intro_localised"]["k"] for name, row in curve.items()}
    assert k["confirmed>=8"] == 1 and k["confirmed>=7"] == 2
    assert k["confirmed (any confidence)"] == 2 and k["confirmed or uncertain"] == 3
    assert k["no verifier (every verifier-bound candidate)"] == 4
    assert curve["no verifier (every verifier-bound candidate)"]["benign_flagged"]["k"] == 1
    assert curve["confirmed>=8"]["benign_flagged"]["k"] == 0
    # Two views of ONE run: losses, not a degenerate McNemar.
    cmp = R.compare_runs(recs, recs, "verified", "audit_only", same=True)
    assert cmp["nested_views"] and "p_mcnemar_exact" not in json.dumps(cmp)
    assert cmp["losses"]["intro_catches_lost_strict"]["count"] == 3


def test_precision_has_no_point_estimate_without_false_positives():
    recs = [_scored_rec(intro_item("pr_A_intro", "pr_A"), {"verified": [_f("app/files.py", 7)]}),
            *[_scored_rec(benign_item(f"b{k}"), {"verified": []}) for k in range(10)]]
    R.attach_outcomes(recs, 2)
    m = R.view_metrics(recs, "verified")
    assert set(m["precision_at_base_rate"].values()) == {None}
    assert m["precision_at_base_rate_note"].startswith("n/a: 0 of 10")
    upper = m["fpr_benign_upper95_exact"]
    assert upper == pytest.approx(1 - 0.025 ** 0.1, abs=1e-4)  # Clopper-Pearson, k = 0
    assert m["precision_at_base_rate_fpr_upper95"]["0.01"] == round(
        0.01 / (0.01 + upper * 0.99), 4)
    assert precision_at_observed_fpr(0.5, 1, 10, 0.01) == round(0.005 / (0.005 + 0.099), 4)
    assert fisher_exact_2x2(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact_2x2(0, 0, 0, 0) == 1.0


def test_rescore_recomputes_facts_and_refreshes_corrected_labels():
    item = intro_item()
    rec = _scored_rec({**item, "category": "redos", "cwe": "CWE-400"},
                      {"verified": [_f("app/files.py", 7)]})
    fixed = {**item, "category": "prototype_pollution", "cwe": "CWE-1321",
             "meta": {**item["meta"], "label_override": {"reason": "advisory text",
                                                         "from": {"cwe": "CWE-400",
                                                                  "category": "redos"}}}}
    data = {"config": {"arm": "units"}, "items": [rec]}
    out = R.rescore(data, 2, [fixed], parser=R._parser())
    [ch] = out["rescore"]["label_changes"]
    assert ch["to"] == {"category": "prototype_pollution", "cwe": "CWE-1321"}
    [r] = out["items"]
    assert r["category"] == "prototype_pollution" and r["facts"]["anchors_by_path"]
    assert out["summary"]["views"]["verified"]["in_scope"]["by_scope"]["in"] == 1
    assert out["summary"]["views"]["verified"]["tpr_localised_function"]["k"] == 1


# --- sampling / cache keys ------------------------------------------------------------------


class SamplingClient:
    def __init__(self, label="openrouter:qwen/qwen3.8-27b:free", temperature=None):
        self.label, self.provider, self.temperature = label, label.split(":")[0], temperature

    def sampling_params(self):
        if self.temperature is not None:
            return {"temperature": float(self.temperature)}
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


class SamplingLLM(StubLLM):
    def __init__(self, temperature=None, **kw):
        super().__init__(**kw)
        self.clients = [SamplingClient(temperature=temperature)]


def test_sampling_key_keeps_temperature_keys_and_separates_model_default():
    assert sampling_key({"temperature": 0.0}) == 0.0
    key = sampling_key({"top_k": 20, "temperature": 1.0, "top_p": 0.95})
    assert key == 'sampling:{"temperature":1.0,"top_k":20,"top_p":0.95}'
    assert R.parse_args(["--dry-run"]).llm_temperature is None  # model default
    assert R.parse_args(["--dry-run", "--llm-temperature", "0"]).llm_temperature == 0.0


def test_model_default_sampling_never_replays_temperature_zero_entries(tmp_path):
    path = tmp_path / "cache.jsonl"
    # A temperature-0 run (explicit override) fills the cache...
    rec0, g0 = _run(intro_item(), "pr", SamplingLLM(temperature=0.0, audit={"findings": [
        TRAVERSAL]}), _gate(LLMCache(path), temperature=None))
    assert g0.calls == 2
    assert {json.loads(x)["temperature"] for x in path.read_text().splitlines()} == {0.0}
    # ...which the model-default run does not replay (and records its sampling key).
    rec1, g1 = _run(intro_item(), "pr", SamplingLLM(), _gate(LLMCache(path), temperature=None))
    assert g1.calls == 1 and rec1["status"] == "ok"
    keys = [json.loads(x)["temperature"] for x in path.read_text().splitlines()]
    assert keys[-1].startswith("sampling:")
    # Each is replayed by its own setting; another repeat index is another call.
    _, g2 = _run(intro_item(), "pr", SamplingLLM(), _gate(LLMCache(path), temperature=None))
    _, g3 = _run(intro_item(), "pr", SamplingLLM(), _gate(LLMCache(path), temperature=0.0))
    assert g2.calls == 0 and g3.calls == 0
    _, g4 = _run(intro_item(), "pr", SamplingLLM(), _gate(LLMCache(path), temperature=None,
                                                         repeat=1))
    assert g4.calls == 1
    assert json.loads(path.read_text().splitlines()[-1])["repeat"] == 1


# --- the worth-a-look tier ------------------------------------------------------------------

SQL_OLD = ("def find(db, name):\n"
           "    return db.execute('SELECT * FROM users WHERE name = ?', (name,))\n")
SQL_NEW = ("def find(db, name):\n"
           "    return db.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n")


def sql_intro_item(item_id="pr_S_intro"):
    return _item(item_id, "vuln_introducing", [_file("app/db.py", SQL_OLD, SQL_NEW)],
                 "app/db.py", vuln=[2], changed=[2], pair_base="pr_S", category="sqli")


def test_run_records_review_suggestions_as_their_own_view():
    verdict = {"verdict": "uncertain", "confidence": 5, "reason": "caller not shown",
               "removed_control_quote": "    p = clean(p)"}
    rec, _ = _run(intro_item(), "pr", StubLLM(audit={"findings": [TRAVERSAL]}, verdict=verdict))
    assert rec["findings"]["verified"] == []
    [sug] = rec["findings"]["verified_plus_review"]
    assert (sug["file_path"], sug["line"], sug["source"]) == ("app/files.py", 7,
                                                              R.REVIEW_SOURCE)
    [cand] = rec["candidates"]
    assert cand["status"] == "review_suggested" and cand["review_evidence"] == ["verifier_quote"]
    assert rec["findings"]["audit_only"] == [cand]  # still verifier-bound
    assert rec["pr_review"]["review_suggested"] == 1 and rec["review_evidence_source"] == "run"
    s = R.summarize([rec], "pr", 2)
    assert s["views"]["verified"]["tpr_localised"]["k"] == 0
    assert s["views"]["verified_plus_review"]["tpr_localised"]["k"] == 1
    assert s["views"]["verified_plus_review"]["tpr_any_finding"]["k"] == 1
    assert s["review_tier_adds"]["intro_catches_lost_strict"]["count"] == 1  # = gained
    assert s["operational"]["funnel"]["review_suggested"] == 1
    curve = {row["policy"]: row for row in s["verifier_policy_curve"]}
    assert curve["confirmed>=7"]["intro_localised"]["k"] == 0
    assert curve["confirmed>=7 + worth a look (not blocking)"]["intro_localised"]["k"] == 1
    R.print_summary(s, "pr")  # renders the new view and rows


def _old_pr_rec(item, cands):
    """A record from a run before the tier: no verified_plus_review, no evidence."""
    rec = _pr_rec(item, cands)
    for c in rec["candidates"]:
        c.update(severity="high", audit_confidence=8)
    return rec


def test_rescore_replays_the_tier_from_guard_diff_only_at_the_runs_cutoff(capsys):
    p = "app/db.py"
    recs = [
        _old_pr_rec(sql_intro_item(), [_cand(p, 2, "uncertain", "uncertain", 5)]),
        _old_pr_rec(sql_intro_item("pr_T_intro"),
                    [_cand(p, 2, "below_min_confidence", "confirmed", 7)]),
        _old_pr_rec(sql_intro_item("pr_U_intro"), [_cand(p, 2, "rejected", "rejected", 6)]),
        # guard_diff sees nothing in this one (and no verifier quote in old runs)
        _old_pr_rec(intro_item(), [_cand("app/files.py", 7, "uncertain", "uncertain", 5)]),
    ]
    items = [sql_intro_item(), sql_intro_item("pr_T_intro"), sql_intro_item("pr_U_intro"),
             intro_item()]
    data = {"config": {"arm": "pr", "pr_review": {"min_confidence": 8}}, "items": recs}
    out = R.rescore(data, 2, items, parser=R._parser())
    meta = out["rescore"]["review_view"]
    assert meta["replayed"] == 4 and meta["suggestions"] == 2 and meta["min_confidence"] == 8
    assert "guard_diff only" in meta["evidence"]
    by_id = {r["base_id"]: r for r in out["items"]}
    assert len(by_id["pr_S_intro"]["findings"]["verified_plus_review"]) == 1
    assert len(by_id["pr_T_intro"]["findings"]["verified_plus_review"]) == 1  # 7 < 8
    assert by_id["pr_U_intro"]["findings"]["verified_plus_review"] == []  # rejected
    assert by_id["pr_A_intro"]["findings"]["verified_plus_review"] == []
    assert by_id["pr_S_intro"]["candidates"][0]["review_evidence"] == ["guard_diff"]
    assert data["items"][0]["candidates"][0].get("review_evidence") is None  # input untouched
    assert out["summary"]["views"]["verified_plus_review"]["tpr_localised"]["k"] == 2
    assert out["summary"]["review_evidence_sources"] == {R.REVIEW_EVIDENCE_RESCORED: 4}
    R.print_summary(out["summary"], "pr")
    assert "OLD verifier prompt, guard_diff evidence only" in capsys.readouterr().out
    # Without the dataset nothing is replayed: the view is reported as n/a.
    bare = R.rescore({"config": {"arm": "pr"}, "items": [_old_pr_rec(intro_item(), [])]}, 2)
    assert "verified_plus_review" not in bare["summary"]["views"]
    R.print_summary(bare["summary"], "pr")
    assert "n/a (run recorded before the tier" in capsys.readouterr().out
