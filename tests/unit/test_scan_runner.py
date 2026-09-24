"""Unit tests for the pure report-building logic in scan_runner."""
from backend.app.services.scan_runner import _build_report_findings

_ROW = {
    "finding_id": 1,
    "point_id": "pt-1",
    "source": "cve",
    "cve_id": "CVE-1",
    "team_pr_id": None,
    "file_path": "app.py",
    "start_line": 10,
    "function_name": "handle_login",
}
_UNITS = {"U1": {"uid": "U1", "file_path": "app.py", "function_name": "handle_login",
                 "start_line": 10}}


def _validated(**kw):
    return {"unit": "U1", "cve_id": None, "title": "SQLi", "explanation": "bad",
            "fix_snippet": "", "quoted_code": "q = 'x' + u", "line": 12, "end_line": 12,
            "cwe": "CWE-89", **kw}


def test_build_report_findings_anchors_to_unit_and_links_cited_row():
    [f] = _build_report_findings([_validated(cve_id="CVE-1")], _UNITS, [_ROW])
    assert (f["file_path"], f["function_name"], f["start_line"]) == ("app.py", "handle_login", 10)
    assert f["line"] == 12 and f["source"] == "llm" and f["deterministic"] is False
    assert (f["finding_id"], f["point_id"]) == (1, "pt-1")
    assert f["dedupe_key"].startswith("llm:")


def test_build_report_findings_without_citation_is_still_anchored():
    [f] = _build_report_findings([_validated()], _UNITS, [_ROW])
    assert f["file_path"] == "app.py" and f["function_name"] == "handle_login"
    assert f["finding_id"] is None and f["point_id"] is None
    # A cited CVE retrieved for a DIFFERENT function doesn't link to its row.
    other = {**_ROW, "function_name": "other"}
    [g] = _build_report_findings([_validated(cve_id="CVE-1")], _UNITS, [other])
    assert g["finding_id"] is None


def _unit_with_code():
    code = "def handle_login(u):\n    q = 'x' + u\n    return db.execute(q)\n"
    return {"U1": {"uid": "U1", "file_path": "app.py", "function_name": "handle_login",
                   "start_line": 10, "code": code, "prompt_code": code}}


def test_dedupe_key_ignores_llm_wording_and_quote_variation():
    units = _unit_with_code()
    a = _build_report_findings([_validated(quoted_code="q = 'x' + u", line=11)], units, [])[0]
    # Next run: different CWE and title, a partial / re-indented quote of the same line.
    b = _build_report_findings([_validated(quoted_code="'x'  +  u", line=11, cwe="CWE-564",
                                           title="Injection")], units, [])[0]
    c = _build_report_findings([_validated(quoted_code="return db.execute(q)", line=12)],
                               units, [])[0]
    assert a["dedupe_key"] == b["dedupe_key"] != c["dedupe_key"]
    # The key older servers used is still sent, for comment adoption.
    assert a["legacy_dedupe_keys"] and a["legacy_dedupe_keys"] != b["legacy_dedupe_keys"]


def test_two_findings_on_one_line_keep_distinct_keys():
    from backend.app.services.scan_runner import disambiguate_dedupe_keys

    units = _unit_with_code()
    found = _build_report_findings(
        [_validated(line=11, cwe="CWE-79", severity="medium", title="xss"),
         _validated(line=11, cwe="CWE-89", severity="high", title="sqli")], units, [])
    disambiguate_dedupe_keys(found)
    keys = {f["title"]: f["dedupe_key"] for f in found}
    assert keys["xss"] == keys["sqli"] + "#2"  # the more severe one keeps the plain key


def test_dedupe_key_stable_across_rewording_distinct_across_issues():
    a = _build_report_findings([_validated(title="SQL injection")], _UNITS, [])[0]
    b = _build_report_findings([_validated(title="Injectable SQL", line=40)], _UNITS, [])[0]
    c = _build_report_findings([_validated(quoted_code="os.system(c)", cwe="CWE-78")],
                               _UNITS, [])[0]
    assert a["dedupe_key"] == b["dedupe_key"] != c["dedupe_key"]


def test_finding_out_exposes_twin_scores_from_payload():
    import json

    from backend.app.db.models import Finding
    from backend.app.models.schemas import FindingOut
    from backend.app.services.scan_runner import _finding_out, _persist_findings

    class _Session:
        def add_all(self, rows):
            for i, r in enumerate(rows, start=1):
                r.id = i

        def flush(self):
            pass

    raw = {
        "ghost_hunter_findings": [
            {"cve_id": "CVE-T", "point_id": "p1", "similarity_score": 0.8,
             "sim_fixed": 0.7, "twin_margin": 0.1},
            {"cve_id": "CVE-H", "point_id": "p2", "similarity_score": 0.8,
             "sim_fixed": None, "twin_margin": None},
        ],
        "team_memory_findings": [{"pr_id": "9", "point_id": "p3", "similarity_score": 0.5}],
    }
    rows = _persist_findings(_Session(), "scan-1", raw)
    assert json.loads(rows[0].payload_json)["twin_margin"] == 0.1

    outs = [FindingOut(**_finding_out(r)) for r in rows]
    assert (outs[0].sim_fixed, outs[0].twin_margin) == (0.7, 0.1)
    assert (outs[1].sim_fixed, outs[1].twin_margin) == (None, None)
    assert (outs[2].sim_fixed, outs[2].twin_margin) == (None, None)  # team rows
    # Rows persisted before this change (no twin keys / no payload) still render.
    legacy = Finding(id=4, point_id="p4", source="cve", title="t", similarity_score=0.5,
                     rerank_prob=0.5, payload_json=None)
    assert FindingOut(**_finding_out(legacy)).twin_margin is None
