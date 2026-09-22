"""Unit tests for finding validation and deterministic Markdown rendering."""
from backend.app.core.markdown_renderer import (
    PROMPT_MAX_DIFF_LINES,
    build_user_prompt,
    fix_diff,
    render_markdown,
    severity_from_cvss,
    validate_findings,
)


def test_severity_from_cvss_bands():
    assert severity_from_cvss(9.8) == "critical"
    assert severity_from_cvss(7.5) == "high"
    assert severity_from_cvss(5.0) == "medium"
    assert severity_from_cvss(2.0) == "low"
    assert severity_from_cvss(None) is None


def test_validate_drops_hallucinated_cve():
    findings = [
        {"cve_id": "CVE-REAL", "title": "real"},
        {"cve_id": "CVE-FAKE", "title": "hallucinated"},
        {"team_pr_id": "pr_1", "title": "team-real"},
        {"team_pr_id": "pr_999", "title": "team-fake"},
        {"cve_id": None, "team_pr_id": None, "title": "generic"},
    ]
    out = validate_findings(findings, allowed_cves={"CVE-REAL"}, allowed_prs={"pr_1"})
    titles = {f["title"] for f in out}
    assert titles == {"real", "team-real", "generic"}


def test_render_clean_when_no_findings():
    md = render_markdown([], cve_count=0, team_count=0)
    assert "✅" in md
    assert "No known vulnerabilities" in md


def test_render_groups_cve_and_team():
    findings = [
        {"severity": "critical", "cve_id": "CVE-1", "title": "SQLi", "explanation": "bad",
         "fix_snippet": "use params"},
        {"severity": "medium", "team_pr_id": "pr_9", "title": "bare except", "explanation": "seen",
         "fix_snippet": ""},
    ]
    md = render_markdown(findings, cve_count=1, team_count=1)
    assert "The World Has Seen This Break Before" in md
    assert "CVE-1" in md
    assert "Your Team Has Seen This Break Before" in md
    assert "PR pr_9" in md
    assert "`CRITICAL`" in md


def test_render_orders_by_severity():
    findings = [
        {"severity": "low", "cve_id": "CVE-LOW", "title": "low"},
        {"severity": "critical", "cve_id": "CVE-CRIT", "title": "crit"},
    ]
    md = render_markdown(findings, cve_count=2, team_count=0)
    assert md.index("CVE-CRIT") < md.index("CVE-LOW")


_VULN = (
    'def get(db, i):\n    sql = "SELECT * FROM t WHERE id = \'%s\'" % i\n'
    "    return db.execute(sql)"
)
_FIXED = (
    'def get(db, i):\n    sql = "SELECT * FROM t WHERE id = %s"\n'
    "    return db.execute(sql, (i,))"
)


def test_fix_diff_is_compact_unified_diff():
    diff = fix_diff(_VULN, _FIXED)
    lines = diff.splitlines()
    assert lines[0].startswith("@@")  # ---/+++ file headers dropped
    assert "-    sql = \"SELECT * FROM t WHERE id = '%s'\" % i" in lines
    assert "+    return db.execute(sql, (i,))" in lines
    assert fix_diff(_VULN, _VULN) == ""


def test_fix_diff_truncates_long_diffs():
    vuln = "\n".join(f"a{i}" for i in range(100))
    fixed = "\n".join(f"b{i}" for i in range(100))
    lines = fix_diff(vuln, fixed).splitlines()
    assert len(lines) == PROMPT_MAX_DIFF_LINES + 1
    assert lines[-1].startswith("... (diff truncated,")


def test_prompt_includes_fix_diff_and_pre_post_instruction():
    cves = [
        {"cve_id": "CVE-TWIN", "severity": 9.0, "category": "sqli", "description": "d",
         "vulnerable_code": _VULN, "fixed_code": _FIXED},
        {"cve_id": "CVE-HAND", "severity": 7.0, "category": "sqli", "description": "d",
         "vulnerable_code": _VULN, "fixed_code": None},
    ]
    prompt = build_user_prompt("def x(): pass", cves, [])
    assert prompt.count("how this CVE was fixed") == 1  # only the entry with a twin
    twin_block = prompt.split("CVE-HAND")[0]
    assert "how this CVE was fixed" in twin_block
    assert "+    return db.execute(sql, (i,))" in twin_block
    assert "pre-fix" in prompt and "post-fix" in prompt


def test_prompt_without_twins_has_no_diff_section():
    cves = [{"cve_id": "CVE-HAND", "severity": 7.0, "category": "sqli", "description": "d",
             "vulnerable_code": _VULN}]
    prompt = build_user_prompt("def x(): pass", cves, [])
    assert "how this CVE was fixed" not in prompt
    assert "post-fix" not in prompt
    assert "vulnerable pattern:" in prompt


def test_prompt_caps_long_corpus_code():
    long_code = "\n".join(f"line_{i} = {i}" for i in range(500))
    cves = [{"cve_id": "CVE-LONG", "vulnerable_code": long_code, "fixed_code": long_code + "\nx"}]
    prompt = build_user_prompt("def x(): pass", cves, [])
    assert "line_39 = " in prompt and "line_40 = " not in prompt  # code capped at 40
    assert "+x" in prompt  # the diff still shows the change at the end
    assert "(code truncated, 460 more lines)" in prompt
    assert len(prompt.splitlines()) < 120
