"""Unit tests for the unified-diff changed-line parser."""
from backend.app.core.diff_utils import parse_patch_changed_lines


def test_single_hunk_added_lines():
    patch = (
        "@@ -1,3 +1,4 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    x = 1\n"
        "+    return x\n"
        " # end\n"
    )
    # New file: line1 context(1), then +x=1 at 2, +return x at 3, context at 4.
    assert parse_patch_changed_lines(patch) == [2, 3]


def test_multiple_hunks():
    patch = (
        "@@ -1,2 +1,2 @@\n"
        " a\n"
        "+b\n"
        "@@ -10,2 +11,3 @@\n"
        " j\n"
        "+k\n"
        "+l\n"
    )
    assert parse_patch_changed_lines(patch) == [2, 12, 13]


def test_ignores_headers_before_hunk():
    patch = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1 +1,2 @@\n"
        " keep\n"
        "+added\n"
    )
    # The +++ header must NOT be counted as a change.
    assert parse_patch_changed_lines(patch) == [2]


def test_empty_patch():
    assert parse_patch_changed_lines("") == []


def test_blank_context_line_without_leading_space():
    # A blank context line may lose its leading space; it must still advance the
    # new-file counter so following additions get the right line number.
    patch = "@@ -1,3 +1,3 @@\n a\n\n+b\n"
    assert parse_patch_changed_lines(patch) == [3]


def test_removed_lines_do_not_advance():
    patch = "@@ -1,3 +1,2 @@\n keep\n-gone\n+added\n"
    # new file: keep(1), added(2); the removed line doesn't advance.
    assert parse_patch_changed_lines(patch) == [2]
