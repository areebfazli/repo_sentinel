"""Lightweight BM25-style sparse encoder for hybrid retrieval.

Tokenizes code into identifier terms and hashes them to sparse indices with
term-frequency values; Qdrant applies IDF at query time (Modifier.IDF). This
captures distinctive lexical signals dense embeddings blur — e.g. `shell`+`True`,
`innerHTML`, `extractall`, `popen` — that often separate a vulnerable pattern from
its safe twin.

Deterministic hashing (crc32, not Python's salted hash) is required so ingest and
query, which run in different processes, map the same token to the same index.
"""
import re
import zlib

VOCAB_SIZE = 2**20
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text):
        tokens.append(tok.lower())
        if "_" in tok:  # snake_case -> also index subtokens for better overlap
            tokens.extend(part.lower() for part in tok.split("_") if part)
    return tokens


def encode(text: str) -> dict:
    """Return a term-frequency sparse vector: {"indices": [...], "values": [...]}."""
    counts: dict[int, int] = {}
    for token in tokenize(text):
        idx = zlib.crc32(token.encode("utf-8")) % VOCAB_SIZE
        counts[idx] = counts.get(idx, 0) + 1
    return {
        "indices": list(counts.keys()),
        "values": [float(v) for v in counts.values()],
    }
