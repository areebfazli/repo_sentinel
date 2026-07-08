"""Unit tests for finding validation and deterministic Markdown rendering."""
from backend.app.core.markdown_renderer import (
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
