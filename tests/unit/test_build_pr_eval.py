"""Tests for scripts/build_pr_eval.py (offline; no network, no models).

Patch generation / application (round trips on random edits and on real cached
GitHub samples in ``fixtures/pr_eval_cache_sample.json``), reversed vs real fix
direction, vuln_lines_new mapping (incl. insertion-only fixes), split mapping and
no dev/test straddling, no advisory text in PR text, multi-file assembly, caps,
the dev sample, and an end-to-end build from a tiny fake cache.
"""
import base64
import difflib
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.diff_utils import parse_patch_changed_lines  # noqa: E402
from scripts.build_pr_eval import (  # noqa: E402
    LABEL_OVERRIDES,
    LEAK_ID_RE,
    MISLEADING_TITLE,
    PatchError,
    apply_patch,
    build_all,
    build_bystander_item,
    build_vuln_items,
    file_entry,
    label_override,
    main,
    make_misleading,
    neutral_pr_text,
    patch_line_numbers,
    reconstruct,
    reverse_entry,
    sample_category,
    sample_dev,
    select_files,
    split_for_ids,
    split_lines,
    straddling,
    unified_diff,
    vuln_lines,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pr_eval_cache_sample.json"


def _round_trip(old, new, path="pkg/mod.py"):
    patch = unified_diff(old, new, path)
    assert apply_patch(old, patch) == (new or "")
    assert apply_patch(new, patch, reverse=True) == (old or "")
    return patch


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


def test_split_lines_only_on_newline():
    assert split_lines("a\nb") == ["a\n", "b"]
    assert split_lines("a\r\nb\n") == ["a\r\n", "b\n"]
    assert split_lines("x\x0cy z\rw\n") == ["x\x0cy z\rw\n"]
    assert split_lines("") == [] and split_lines(None) == []


def test_unified_diff_matches_difflib_format():
    old = "".join(f"line {i}\n" for i in range(20))
    new = old.replace("line 5\n", "line five\n").replace("line 15\n", "")
    ours = unified_diff(old, new, "src/x.py")
    ref = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       "a/src/x.py", "b/src/x.py", n=3))
    assert ours == ref
    assert ours.startswith("--- a/src/x.py\n+++ b/src/x.py\n@@ -3,7 +3,7 @@\n")


def test_round_trip_edge_cases():
    _round_trip("a\nb\nc", "a\nB\nc")  # no trailing newline on both sides
    p = _round_trip("a\nb\n", "a\nb")  # newline removed at EOF
    assert "\\ No newline at end of file" in p
    _round_trip("a\nb", "a\nb\nc\n")  # newline added at EOF
    _round_trip("x\r\ny\r\n", "x\r\nz\r\n")  # CRLF
    _round_trip("f\x0cg\nh i\n", "f\x0cg\nj i\n")  # form feed, U+2028
    added = _round_trip(None, "new\nfile\n")
    assert added.startswith("--- /dev/null\n+++ b/pkg/mod.py\n@@ -0,0 +1,2 @@\n")
    removed = _round_trip("old\n", None)
    assert removed.startswith("--- a/pkg/mod.py\n+++ /dev/null\n@@ -1 +0,0 @@\n")
    assert unified_diff("same\n", "same\n", "p") == ""
    assert file_entry("p", "same\n", "same\n") is None
    assert file_entry("p", None, "") is None  # empty file added: no change to show


def test_round_trip_random_edits():
    rng = random.Random(7)
    vocab = ["", "    return x\n", "}\n", "if (a) {\n", "pass\n", "x = 1\n", "# c\n"]
    for _ in range(300):
        old_lines = [rng.choice(vocab) or f"v{rng.randint(0, 9)}\n"
                     for _ in range(rng.randint(0, 40))]
        new_lines = list(old_lines)
        for _ in range(rng.randint(1, 6)):
            op = rng.random()
            pos = rng.randint(0, len(new_lines))
            if op < 0.4:
                new_lines.insert(pos, rng.choice(vocab) or "ins\n")
            elif op < 0.7 and new_lines:
                new_lines.pop(min(pos, len(new_lines) - 1))
            elif new_lines:
                new_lines[min(pos, len(new_lines) - 1)] = f"mod {rng.random()}\n"
        old, new = "".join(old_lines), "".join(new_lines)
        if rng.random() < 0.3:
            new = new.rstrip("\n")
        patch = _round_trip(old, new)
        added, removed = patch_line_numbers(patch) if patch else ([], 0)
        assert added == (parse_patch_changed_lines(patch) if patch else [])
        e = file_entry("pkg/mod.py", old, new)
        if e is not None:
            r = reverse_entry(e)
            assert apply_patch(r["old_content"], r["patch"]) == old
            assert r["_added"] == e["_removed"] and r["_removed"] == e["_added"]


def test_apply_patch_rejects_mismatch():
    patch = unified_diff("a\nb\nc\n", "a\nX\nc\n", "p")
    with pytest.raises(PatchError):
        apply_patch("a\nDIFFERENT\nc\n", patch)
    with pytest.raises(PatchError):
        apply_patch("a\nb\nc\n", "@@ -1,3 +1,3 @@\n a\n-b\n")  # truncated hunk


def test_real_cached_samples_round_trip():
    samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert {s["status"] for s in samples} == {"modified", "added"}
    assert any("\\ No newline at end of file" in s["github_patch"] for s in samples)
    for s in samples:
        gh = s["github_patch"]
        if s["status"] == "modified":
            assert apply_patch(s["pre"], gh) == s["post"]  # GitHub's own patch
            assert apply_patch(s["post"], gh, reverse=True) == s["pre"]
            old, new, how = reconstruct("modified", gh, None, s["post"])
            assert (old, new, how) == (s["pre"], s["post"], "pre_from_patch")
            assert reconstruct("modified", gh, s["pre"], s["post"])[2] == "cached"
            patch = _round_trip(s["pre"], s["post"], s["path"])
            assert patch_line_numbers(patch)[0] == parse_patch_changed_lines(patch)
            # difflib's diff need not equal git's hunk for hunk, but both are
            # exact and touch the same region.
            ours, theirs = patch_line_numbers(patch)[0], patch_line_numbers(gh)[0]
            assert ours and theirs and abs(min(ours) - min(theirs)) <= 3
        else:
            old, new, how = reconstruct("added", gh, None, None)
            assert old is None and how == "added"
            assert len(split_lines(new)) == len(patch_line_numbers(gh)[0])
            assert new.startswith("from bs4 import BeautifulSoup\n")
            _round_trip(None, new, s["path"])


def test_reconstruct_skips_and_mismatch():
    assert reconstruct("renamed", "@@ -1 +1 @@\n-a\n+b", None, None)[2] == "skip:renamed"
    assert reconstruct("added", None, None, None)[2] == "skip:added_not_cached"
    assert reconstruct("modified", None, None, None)[2] == "skip:modified_not_cached"
    gh = unified_diff("a\n", "b\n", "p")
    assert reconstruct("modified", gh, "a\n", "zzz\n")[2] == "cached_patch_mismatch"
    assert reconstruct("removed", "@@ -1 +0,0 @@\n-a", None, None) == ("a\n", None, "removed")
    assert reconstruct("modified", gh, None, "other\n")[2] == "skip:patch_does_not_apply"


# ---------------------------------------------------------------------------
# vuln_lines_new
# ---------------------------------------------------------------------------

FIXED = "def f(x):\n    check(x)\n    y = clean(x)\n    return run(y)\n\n\ndef g():\n    pass\n"
VULN = "def f(x):\n    y = x\n    return run(y)\n\n\ndef g():\n    pass\n"


def test_vuln_lines_deleted_or_modified():
    # Reversed fix: old = fixed, new = vulnerable. The fix deleted/modified
    # vulnerable line 2 ("y = x").
    lines, mode = vuln_lines(FIXED, VULN, [(1, 3)])
    assert (lines, mode) == ([2], "deleted_or_modified")
    assert set(lines) <= set(patch_line_numbers(unified_diff(FIXED, VULN, "p"))[0])


def test_vuln_lines_insertion_only_fix():
    fixed = "def f(x):\n    a = 1\n    b = 2\n    check(x)\n    c = 3\n    d = 4\n    return x\n"
    vuln = "def f(x):\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    return x\n"
    lines, mode = vuln_lines(fixed, vuln, [(1, 6)])
    assert mode == "insertion_point"
    assert lines == [1, 2, 3, 4, 5]  # the gap is after line 3: 3 +- 2
    # The reversed PR only deletes lines: nothing is "changed" in the new file.
    assert patch_line_numbers(unified_diff(fixed, vuln, "p"))[0] == []


def test_vuln_lines_restricted_to_function():
    fixed = FIXED.replace("    pass\n", "    return 1\n")
    lines, _ = vuln_lines(fixed, VULN, [(1, 3)])
    assert lines == [2]  # g's change (line 7) is outside the vulnerable function
    assert vuln_lines(fixed, VULN, None)[0] == [2, 7]
    assert vuln_lines(fixed, VULN, [(6, 6)])[1] == "none"  # nothing touched there


# ---------------------------------------------------------------------------
# Item assembly
# ---------------------------------------------------------------------------

REPO, SHA = "acme/app", "a" * 40


def _pair(base, vuln_code, adv="CVE-2024-0001", category="cmd_injection"):
    return {"base": base,
            "vuln": {"id": f"{base}_vuln", "code": vuln_code, "expected_cve_id": adv,
                     "category": category, "language": "python"},
            "safe": {"id": f"{base}_safe", "code": "x", "expected_cve_id": adv,
                     "category": category, "language": "python"}}


def _prov(path, fn="f", adv="GHSA-aaaa-bbbb-cccc"):
    return {"repo": REPO, "commit": SHA, "file_path": path, "function_name": fn,
            "advisory_id": adv, "cve_id": "CVE-2024-0001", "category": "cmd_injection",
            "cwe_id": "CWE-78", "language": "python"}


def _entries(extra: int = 0, extra_lines: int = 1):
    es = [file_entry("app/run.py", VULN, FIXED)]  # forward: pre-fix -> post-fix
    for i in range(extra):
        body = "".join(f"v{j}\n" for j in range(extra_lines))
        es.append(file_entry(f"app/other{i}.py", "old\n", body))
    return es


def _vuln_items(entries, split="dev", max_files=6, max_changed=2000):
    base = "CVE-2024-0001_f_12345678"
    stats = Counter()
    items = build_vuln_items((REPO, SHA), [base], {base: _pair(base, VULN.split("\n\n\n")[0])},
                             {base: [_prov("app/run.py")]}, entries, [], split, max_files,
                             max_changed, stats)
    return items, stats


def test_reversed_and_real_fix_direction():
    items, _ = _vuln_items(_entries(extra=2))
    intro, fix = items
    assert (intro["kind"], intro["source"]) == ("vuln_introducing", "reversed_fix")
    assert (fix["kind"], fix["source"]) == ("vuln_fix", "real_fix")
    ti = next(f for f in intro["files"] if f["path"] == "app/run.py")
    tf = next(f for f in fix["files"] if f["path"] == "app/run.py")
    assert (ti["old_content"], ti["new_content"]) == (FIXED, VULN)
    assert (tf["old_content"], tf["new_content"]) == (VULN, FIXED)
    for it in items:
        assert set(it) >= {"id", "kind", "language", "repo", "advisory_id", "cve_id",
                           "category", "cwe", "split", "pr_title", "pr_body", "files",
                           "target", "source", "notes"}
        for f in it["files"]:
            assert set(f) == {"path", "old_content", "new_content", "patch"}
            assert apply_patch(f["old_content"], f["patch"]) == (f["new_content"] or "")
            assert f["patch"].startswith(f"--- a/{f['path']}\n+++ b/{f['path']}\n@@ ")
    assert intro["target"] == {"path": "app/run.py", "vuln_lines_new": [2],
                               "changed_lines_new": [2]}
    assert fix["target"]["vuln_lines_new"] == []
    assert fix["target"]["changed_lines_new"] == [2, 3]
    assert intro["meta"]["vuln_lines_mode"] == "deleted_or_modified"
    assert intro["id"].endswith("_intro") and fix["id"].endswith("_fix")
    assert intro["meta"]["pair_base"] == fix["meta"]["pair_base"]
    assert (intro["advisory_id"], intro["cve_id"], intro["cwe"]) == (
        "CVE-2024-0001", "CVE-2024-0001", "CWE-78")
    assert [f["path"] for f in intro["files"]] == sorted(f["path"] for f in intro["files"])


def test_label_overrides_fix_contradicted_cwe_tags():
    # CVE-2020-7771 (prototype pollution tagged CWE-400) and CVE-2013-0270 (a
    # request-size DoS tagged CWE-119): the corrected label, category via the
    # corpus table.
    assert label_override(["GHSA-5pxj-mhwj-x5gv"])[1]["category"] == "prototype_pollution"
    osv_id, o = label_override(["PYSEC-2026-650", "GHSA-4ppj-4p4v-jf4p"])
    assert (osv_id, o["cwe"], o["category"]) == ("GHSA-4ppj-4p4v-jf4p", "CWE-400", "redos")
    assert label_override(["GHSA-aaaa-bbbb-cccc"]) is None
    assert all(o["reason"] for o in LABEL_OVERRIDES.values())
    base = "CVE-2024-0001_f_12345678"
    items = build_vuln_items((REPO, SHA), [base], {base: _pair(base, VULN.split("\n\n\n")[0])},
                             {base: [_prov("app/run.py", adv="GHSA-5pxj-mhwj-x5gv")]},
                             _entries(), [], "dev", 6, 2000, Counter())
    for it in items:  # intro and fix alike
        assert (it["cwe"], it["category"]) == ("CWE-1321", "prototype_pollution")
        assert it["meta"]["label_override"]["from"] == {"cwe": "CWE-78",
                                                        "category": "cmd_injection"}
        # The pre-registered sample keeps stratifying on the as-built category.
        assert sample_category(it) == "cmd_injection"
    plain, _ = _vuln_items(_entries())
    assert "label_override" not in plain[0]["meta"]
    assert sample_category(plain[0]) == plain[0]["category"] == "cmd_injection"


def test_pr_text_is_neutral_and_misleading_variant():
    items, _ = _vuln_items(_entries(extra=1))
    for it in items:
        text = it["pr_title"] + "\n" + it["pr_body"]
        assert not LEAK_ID_RE.search(text)
        assert "fix" not in text.lower() and "vulnerab" not in text.lower()
        assert it["pr_title"] == "Update app/other0.py and 1 other file"
    assert items[0]["pr_title"] == items[1]["pr_title"]  # kinds indistinguishable by text
    m = make_misleading(items[0])
    assert m["id"] == items[0]["id"] + "_misleading" and m["pr_title"] == MISLEADING_TITLE
    assert m["files"] == items[0]["files"] and m["target"] == items[0]["target"]
    assert not LEAK_ID_RE.search(m["pr_title"] + m["pr_body"])
    assert neutral_pr_text(["b.py"]) == ("Update b.py", "Changes:\n- b.py")
    assert LEAK_ID_RE.search("see GHSA-abcd-efgh-2345 and CVE-2021-1234")


def test_multi_file_caps_and_truncation_notes():
    items, _ = _vuln_items(_entries(extra=8))
    intro = items[0]
    assert len(intro["files"]) == 6
    assert "app/run.py" in [f["path"] for f in intro["files"]]  # vulnerable file first
    assert intro["meta"]["truncated"] and len(intro["meta"]["dropped_files"]) == 3
    assert "Truncated to 6 file(s)" in intro["notes"]
    # Line cap: the big bystander files are dropped, the vulnerable file kept.
    items, _ = _vuln_items(_entries(extra=3, extra_lines=50), max_changed=60)
    assert [f["path"] for f in items[0]["files"]] == ["app/other0.py", "app/run.py"]
    assert sum(len(patch_line_numbers(f["patch"])[0]) + patch_line_numbers(f["patch"])[1]
               for f in items[0]["files"]) <= 60
    # The vulnerable file alone over the cap: no item.
    items, stats = _vuln_items(_entries(), max_changed=2)
    assert items == [] and stats["skip_commit:vuln_file_over_caps"] == 1


def test_select_files_priority_and_order():
    es = [file_entry(p, "a\n", "b\n") for p in ("z.py", "a.py", "m.py")]
    kept, dropped = select_files(es, ["z.py"], max_files=2)
    assert [e["path"] for e in kept] == ["a.py", "z.py"] and dropped == ["m.py"]


def test_insertion_only_fix_and_unlocated_function():
    fixed = "def f(x):\n    a = 1\n    check(x)\n    return x\n"
    vuln = "def f(x):\n    a = 1\n    return x\n"
    base = "CVE-2024-0002_f_abcdef12"
    stats = Counter()
    items = build_vuln_items((REPO, SHA), [base], {base: _pair(base, vuln, "CVE-2024-0002")},
                             {base: [_prov("app/run.py")]}, [file_entry("app/run.py", vuln,
                                                                        fixed)],
                             [], "test", 6, 2000, stats)
    intro = items[0]
    assert intro["meta"]["vuln_lines_mode"] == "insertion_point"
    assert intro["target"]["vuln_lines_new"] == [1, 2, 3]
    assert intro["target"]["changed_lines_new"] == []
    # Function code not found verbatim: whole-file mapping, flagged.
    items = build_vuln_items((REPO, SHA), [base], {base: _pair(base, "def nope():\n")},
                             {base: [_prov("app/run.py")]}, [file_entry("app/run.py", vuln,
                                                                        fixed)],
                             [], "test", 6, 2000, stats)
    assert items[0]["meta"]["vuln_lines_mode"] == "insertion_point_function_not_located"


def test_bystander_item():
    es = [file_entry("app/run.py", VULN, FIXED), file_entry("app/util.py", "a\n", "a\nb\n")]
    it = build_bystander_item((REPO, SHA), es, {"app/run.py"}, "dev", "eval_fix_commit", 6,
                              2000, Counter())
    assert (it["kind"], it["source"], it["split"]) == ("benign", "bystander", "dev")
    assert [f["path"] for f in it["files"]] == ["app/util.py"]
    assert it["target"] == {"path": None, "vuln_lines_new": [], "changed_lines_new": []}
    assert it["advisory_id"] is None and "security-adjacent" in it["notes"]
    # No known vulnerable location, or only mined files: no bystander.
    assert build_bystander_item((REPO, SHA), es, set(), "dev", "x", 6, 2000, Counter()) is None
    assert build_bystander_item((REPO, SHA), es[:1], {"app/run.py"}, "dev", "x", 6, 2000,
                                Counter()) is None


# ---------------------------------------------------------------------------
# Splits and sampling
# ---------------------------------------------------------------------------


def test_split_for_ids():
    side = {"a_vuln": "dev", "a_safe": "dev", "b_vuln": "test"}
    assert split_for_ids(["a_vuln", "a_safe"], side) == "dev"
    assert split_for_ids(["c_vuln"], side) == "reserve"
    with pytest.raises(ValueError):
        split_for_ids(["a_vuln", "b_vuln"], side)


def test_straddling_detects_repo_and_advisory():
    def item(split, repo, adv=None):
        return {"split": split, "repo": repo, "advisory_id": adv, "cve_id": None,
                "meta": {"osv_ids": []}}
    ok = [item("dev", "o/a", "CVE-1"), item("test", "o/b", "CVE-2"), item("reserve", "o/a")]
    assert straddling(ok) == {"advisory": [], "repo": []}
    bad = ok + [item("test", "O/A")]
    assert straddling(bad)["repo"] == ["o/a"]
    assert straddling(ok + [item("test", "o/c", "CVE-1")])["advisory"] == ["CVE-1"]


def _fake_items():
    items = []
    for i in range(30):
        lang = "javascript" if i % 5 == 0 else "python"
        cat = ["xss", "sqli", "other"][i % 3]
        split = "dev" if i < 24 else "test"
        for kind in ("vuln_introducing", "vuln_fix"):
            items.append({"id": f"pr_{i}_{kind}", "kind": kind, "language": lang,
                          "category": cat, "split": split, "repo": f"o/r{i}",
                          "meta": {"pair_base": f"pr_{i}"}})
    for i in range(40):
        items.append({"id": f"pr_bystander_{i}", "kind": "benign",
                      "language": "javascript" if i % 4 == 0 else "python",
                      "category": None, "split": "dev" if i < 36 else "test",
                      "repo": f"o/b{i % 9}",
                      "meta": {"bystander_of": "eval_fix_commit" if i < 6 else "other"}})
    return items


def test_sample_dev_keeps_pairs_and_is_deterministic():
    items = _fake_items()
    s1 = sample_dev(items, 42, n_vuln=10, n_benign=12, benign_per_repo=2)
    s2 = sample_dev(list(reversed(items)), 42, n_vuln=10, n_benign=12, benign_per_repo=2)
    assert [i["id"] for i in s1] == [i["id"] for i in s2]
    kinds = Counter(i["kind"] for i in s1)
    assert kinds == {"vuln_introducing": 10, "vuln_fix": 10, "benign": 12}
    assert all(i["split"] == "dev" for i in s1)
    intro = {i["meta"]["pair_base"] for i in s1 if i["kind"] == "vuln_introducing"}
    fix = {i["meta"]["pair_base"] for i in s1 if i["kind"] == "vuln_fix"}
    assert intro == fix
    assert max(Counter(i["repo"] for i in s1 if i["kind"] == "benign").values()) <= 2
    assert sample_dev(items, 1, 10, 12, 2) != s1  # the seed matters


# ---------------------------------------------------------------------------
# End to end from a tiny fake cache (offline GithubClient: no network)
# ---------------------------------------------------------------------------


def _cache_put(cache: Path, url: str, data) -> None:
    (cache / "api").mkdir(parents=True, exist_ok=True)
    (cache / "api" / f"{hashlib.sha256(url.encode()).hexdigest()}.json").write_text(
        json.dumps(data), encoding="utf-8")


def _content(cache, repo, path, ref, text):
    _cache_put(cache, f"https://api.github.com/repos/{repo}/contents/{quote(path, safe='/')}"
                      f"?ref={ref}",
               {"content": base64.b64encode(text.encode()).decode(), "encoding": "base64"})


def test_build_all_end_to_end(tmp_path, monkeypatch):
    import requests

    def no_network(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(requests.Session, "get", no_network)

    cache = tmp_path / "cache"
    parent = "b" * 40
    helper_old, helper_new = "def h():\n    return 1\n", "def h():\n    return 2\n"
    fn_code = VULN.split("\n\n\n")[0]
    files = [
        {"filename": "app/run.py", "status": "modified",
         "patch": unified_diff(VULN, FIXED, "app/run.py").split("\n", 2)[2]},
        {"filename": "app/helper.py", "status": "modified",
         "patch": unified_diff(helper_old, helper_new, "x").split("\n", 2)[2]},
        {"filename": "app/new.py", "status": "added",
         "patch": "@@ -0,0 +1 @@\n+X = 1\n\\ No newline at end of file"},
        {"filename": "tests/test_run.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        {"filename": "README.md", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        {"filename": "app/moved.py", "status": "renamed"},
    ]
    _cache_put(cache, f"https://api.github.com/repos/{REPO}/commits/{SHA}",
               {"sha": SHA, "parents": [{"sha": parent}], "files": files})
    _content(cache, REPO, "app/run.py", parent, VULN)
    _content(cache, REPO, "app/run.py", SHA, FIXED)
    _content(cache, REPO, "app/helper.py", parent, helper_old)
    _content(cache, REPO, "app/helper.py", SHA, helper_new)
    pair = {"repo": REPO, "commit": SHA, "file_path": "app/run.py", "function_name": "f",
            "advisory_id": "GHSA-aaaa-bbbb-cccc", "cve_id": "CVE-2024-0001",
            "category": "cmd_injection", "cwe_id": "CWE-78", "language": "python",
            "vulnerable_code": fn_code, "fixed_code": "def f(x): fixed"}
    (cache / "processed_advisories.json").write_text(json.dumps(
        {"GHSA-aaaa-bbbb-cccc": {"pairs": [pair]}}), encoding="utf-8")
    from scripts.build_corpus_from_osv import pair_code_hash
    base = f"CVE-2024-0001_f_{pair_code_hash(fn_code, 'def f(x): fixed')}"
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text("".join(json.dumps(x) + "\n" for x in (
        {"id": f"{base}_vuln", "language": "python", "label": "vulnerable",
         "category": "cmd_injection", "expected_cve_id": "CVE-2024-0001", "code": fn_code},
        {"id": f"{base}_safe", "language": "python", "label": "safe",
         "category": "cmd_injection", "expected_cve_id": "CVE-2024-0001",
         "code": "def f(x): fixed"})), encoding="utf-8")
    split = tmp_path / "v1.json"
    split.write_text(json.dumps({"dev": {"ids": [f"{base}_vuln", f"{base}_safe"]},
                                 "test": {"ids": []}}), encoding="utf-8")

    items, stats = build_all(cache, [pairs_file], split, None)
    kinds = {it["kind"]: it for it in items}
    assert set(kinds) == {"vuln_introducing", "vuln_fix", "benign"}
    intro, fix, benign = kinds["vuln_introducing"], kinds["vuln_fix"], kinds["benign"]
    assert [f["path"] for f in fix["files"]] == ["app/helper.py", "app/new.py", "app/run.py"]
    new_file = next(f for f in fix["files"] if f["path"] == "app/new.py")
    assert (new_file["old_content"], new_file["new_content"]) == (None, "X = 1")
    assert "\\ No newline at end of file" in new_file["patch"]
    intro_new = next(f for f in intro["files"] if f["path"] == "app/new.py")
    assert intro_new["new_content"] is None  # added by the fix -> removed when reversed
    assert intro["split"] == fix["split"] == benign["split"] == "dev"
    assert intro["target"]["vuln_lines_new"] == [2]
    assert intro["meta"]["omitted_files"] == ["app/moved.py"]
    assert [f["path"] for f in benign["files"]] == ["app/helper.py", "app/new.py"]
    assert stats["file_skip:pure_rename"] == 1 and stats["file:added"] == 1

    out = tmp_path / "out"
    rc = main(["--cache-dir", str(cache), "--pairs", str(pairs_file), "--split", str(split),
               "--ordinary", str(tmp_path / "none.jsonl"), "--out-dir", str(out)])
    assert rc == 0
    written = [json.loads(x) for x in (out / "pr_eval_v1.jsonl").read_text().splitlines()]
    assert [w["id"] for w in written] == [it["id"] for it in items]
    mis = [json.loads(x) for x in (out / "pr_eval_v1_misleading.jsonl").read_text()
           .splitlines()]
    assert [m["id"] for m in mis] == [intro["id"] + "_misleading"]
    manifest = json.loads((out / "pr_eval_v1_manifest.json").read_text())
    assert manifest["counts"]["by_kind"] == {"benign": 1, "vuln_fix": 1,
                                             "vuln_introducing": 1}
    digest = hashlib.sha256((out / "pr_eval_v1.jsonl").read_bytes()).hexdigest()
    assert manifest["files"]["pr_eval_v1.jsonl"]["sha256"] == digest
