"""Unified-diff helpers (pure)."""
import re

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_patch_changed_lines(patch: str) -> list[int]:
    """Return the new-file line numbers added/modified by a unified-diff patch.

    Walks each hunk starting from the new-file offset in its ``@@`` header:
    context lines advance the counter, ``+`` lines are recorded (and advance),
    ``-`` lines don't advance. Header lines outside hunks are ignored.
    """
    changed: list[int] = []
    new_line = 0
    in_hunk = False

    for line in patch.splitlines():
        header = _HUNK_HEADER.match(line)
        if header:
            new_line = int(header.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue

        # A blank context line may arrive as "" if a tool stripped its leading
        # space; treat it as context so it still advances the new-file counter.
        tag = line[0] if line else " "
        if tag == "+":
            changed.append(new_line)
            new_line += 1
        elif tag == "-" or tag == "\\":
            continue  # removals / no-newline markers don't advance the new file
        else:
            new_line += 1  # context line (leading space, or stripped blank)

    return changed
