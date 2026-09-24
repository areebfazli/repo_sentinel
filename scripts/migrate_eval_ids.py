"""Rewrite the OSV eval sets' item ids to the unique scheme, offline.

``build_eval_lines`` used to name a pair ``<advisory>_<function>``, which is not
unique: 30 ids repeated across ``detection_eval_osv_{pypi,npm}.jsonl`` (the same
function name in several files of one fix commit), so ``run_eval`` merged
different pairs into one and double-counted exact copies. The builder now
appends a short hash of the pair's code (``eval_pair_base_id``); this script
applies the same scheme to the existing files without re-running the corpus
build (no network, no parsing, no models): lines come in ``build_eval_lines``
order (``<prefix>_vuln`` then ``<prefix>_safe``), each pair's id becomes
``<prefix>_<hash8>_{vuln,safe}``, and exact duplicate pairs are dropped (first
kept). Idempotent: an already-migrated file is left unchanged.

    python scripts/migrate_eval_ids.py            # the default OSV eval files
    python scripts/migrate_eval_ids.py --check    # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_corpus_from_osv import pair_code_hash  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "ml" / "evaluation" / "datasets"
DEFAULT_FILES = [EVAL_DIR / "detection_eval_osv_pypi.jsonl",
                 EVAL_DIR / "detection_eval_osv_npm.jsonl"]


def migrate_lines(lines: list[dict]) -> tuple[list[dict], dict]:
    """New lines + stats {pairs, renamed, duplicates_dropped}. Raises
    ValueError if the lines aren't consecutive ``_vuln`` / ``_safe`` pairs."""
    if len(lines) % 2:
        raise ValueError("odd number of lines: not vuln/safe pairs")
    out: list[dict] = []
    seen: set[str] = set()
    renamed = dropped = 0
    for vuln, safe in zip(lines[0::2], lines[1::2], strict=True):
        vid, sid = vuln["id"], safe["id"]
        if not (vid.endswith("_vuln") and sid.endswith("_safe") and vid[:-5] == sid[:-5]):
            raise ValueError(f"not a vuln/safe pair: {vid!r}, {sid!r}")
        prefix = vid[:-5]
        digest = pair_code_hash(vuln["code"], safe["code"])
        base = prefix if prefix.endswith(f"_{digest}") else f"{prefix}_{digest}"
        if base in seen:
            dropped += 1
            continue
        seen.add(base)
        renamed += base != prefix
        out.append({**vuln, "id": f"{base}_vuln"})
        out.append({**safe, "id": f"{base}_safe"})
    return out, {"pairs": len(out) // 2, "renamed": renamed, "duplicates_dropped": dropped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)
    for path in args.files:
        raw = path.read_text(encoding="utf-8")
        lines = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        new, stats = migrate_lines(lines)
        ids = [ln["id"] for ln in new]
        assert len(ids) == len(set(ids)), "ids still not unique"
        print(f"{path.name}: {len(lines)} -> {len(new)} lines; {stats}")
        text = "".join(json.dumps(ln) + "\n" for ln in new)
        if not args.check and text != raw:
            path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
