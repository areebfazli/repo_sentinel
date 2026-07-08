"""Unit tests for the BM25-style sparse encoder."""
from backend.app.core.sparse_encoder import encode, tokenize


def test_encode_shape_and_determinism():
    a = encode("subprocess.run(cmd, shell=True)")
    b = encode("subprocess.run(cmd, shell=True)")
    assert set(a["indices"]) == set(b["indices"])  # deterministic (cross-process safe)
    assert len(a["indices"]) == len(a["values"])
    assert all(v >= 1.0 for v in a["values"])


def test_snake_case_is_split():
    toks = tokenize("get_user_id")
    assert "get_user_id" in toks
    assert {"get", "user", "id"}.issubset(set(toks))


def test_distinctive_tokens_overlap():
    # Vulnerable command-injection shares shell/true/subprocess with a CVE pattern;
    # a safe list-args version does not.
    vuln = set(encode("subprocess.run(cmd, shell=True)")["indices"])
    cve = set(encode("result = subprocess.run(command, shell=True)")["indices"])
    safe = set(encode("subprocess.run(['gzip', path])")["indices"])
    assert len(vuln & cve) >= len(safe & cve)


def test_term_frequency_counts():
    vec = encode("a a a b")
    # 'a' appears 3x, 'b' once -> values include a 3.0 and a 1.0
    assert 3.0 in vec["values"]
    assert 1.0 in vec["values"]
