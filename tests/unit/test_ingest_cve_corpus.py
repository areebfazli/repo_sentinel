"""Unit tests for CVE corpus ingestion prep: point-id rule, payload/twin handling,
and the single-batch vuln+fixed embedding split. No models, no Qdrant."""
import uuid

from scripts.ingest_cve_corpus import embed_pairs, point_id_for, prepare_records


def _rec(cve_id="CVE-1", vuln="def f(x):\n    run(x)", **extra):
    return {"cve_id": cve_id, "vulnerable_code": vuln, "language": "python", **extra}


def test_handwritten_point_id_is_unchanged():
    # Feedback votes are keyed on point_id; handwritten ids must stay uuid5(cve_id).
    # (Literal = the id stored for this entry before the twin schema change.)
    assert point_id_for(_rec("CVE-2023-28450")) == "69eefcaf-780c-57ef-8587-5754daee7c66"
    assert point_id_for(_rec("CVE-2023-28450")) == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "CVE-2023-28450")
    )
    # Explicit nulls (the OSV schema) behave like absent fields.
    assert point_id_for(_rec("CVE-2023-28450", file_path=None, function_name=None)) == (
        point_id_for(_rec("CVE-2023-28450"))
    )


def test_located_entries_get_distinct_ids_per_function():
    a = _rec("GHSA-xxxx", file_path="pkg/a.py", function_name="load")
    b = _rec("GHSA-xxxx", file_path="pkg/a.py", function_name="save")
    assert point_id_for(a) == str(uuid.uuid5(uuid.NAMESPACE_URL, "GHSA-xxxx|pkg/a.py|load"))
    assert point_id_for(a) != point_id_for(b)
    assert point_id_for(a) != point_id_for(_rec("GHSA-xxxx"))


def test_prepare_records_payload_and_twin_normalisation():
    records = [
        _rec("CVE-A", fixed_code="def f(x):\n    run([x])", cwe_id="CWE-78", source="osv",
             repo="o/r", commit="abc", file_path="a.py", function_name="f", severity=None),
        _rec("CVE-B"),                                   # handwritten: no fixed_code key
        _rec("CVE-C", fixed_code="   "),                 # blank -> no twin
        _rec("CVE-D", fixed_code="def f(x):\n        run(x)"),  # whitespace-only change
        {"cve_id": "CVE-E"},                             # no vulnerable_code -> skipped
    ]
    ids, payloads, vuln, fixed = prepare_records(records)

    assert [p["cve_id"] for p in payloads] == ["CVE-A", "CVE-B", "CVE-C", "CVE-D"]
    assert len(ids) == len(vuln) == len(fixed) == 4
    assert fixed == ["def f(x):\n    run([x])", None, None, None]
    a, b = payloads[0], payloads[1]
    assert a["fixed_code"] == fixed[0]
    assert (a["cwe_id"], a["source"], a["repo"], a["commit"]) == ("CWE-78", "osv", "o/r", "abc")
    assert (a["file_path"], a["function_name"], a["severity"]) == ("a.py", "f", None)
    assert b["fixed_code"] is None and b["source"] == "handwritten"
    assert b["file_path"] is None and "embedding_model" in b


def test_prepare_records_collapses_duplicate_ids_last_wins():
    ids, payloads, _, _ = prepare_records(
        [_rec("CVE-A", description="old"), _rec("CVE-A", description="new")]
    )
    assert len(ids) == 1 and payloads[0]["description"] == "new"


class _RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


def test_embed_pairs_single_batch_and_alignment():
    emb = _RecordingEmbedder()
    vuln_vecs, fixed_vecs = embed_pairs(emb, ["a", "bb", "ccc"], ["xxxx", None, "yyyyy"])
    assert emb.calls == [["a", "bb", "ccc", "xxxx", "yyyyy"]]  # one cache-aware batch
    assert vuln_vecs == [[1.0], [2.0], [3.0]]
    assert fixed_vecs == [[4.0], None, [5.0]]
