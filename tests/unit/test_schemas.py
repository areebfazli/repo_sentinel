"""Unit tests for AnalyzeRequest mode validation."""
import pytest
from pydantic import ValidationError

from backend.app.models.schemas import AnalyzeRequest, FileInput


def test_empty_files_list_rejected():
    with pytest.raises(ValidationError):
        AnalyzeRequest(files=[])


def test_neither_mode_rejected():
    with pytest.raises(ValidationError):
        AnalyzeRequest()


def test_both_modes_rejected():
    with pytest.raises(ValidationError):
        AnalyzeRequest(code_snippet="x", files=[FileInput(path="a.py", content="y")])


def test_snippet_mode_defaults_to_no_language_filter():
    r = AnalyzeRequest(code_snippet="def x(): pass")
    assert r.language is None  # None => don't language-filter the CVE search


def test_files_mode_ok():
    r = AnalyzeRequest(files=[FileInput(path="a.py", content="def x(): pass")])
    assert r.code_snippet is None
