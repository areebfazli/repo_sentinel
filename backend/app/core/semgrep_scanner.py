"""Static-analysis evidence for analysis units (ROADMAP 1f, step 5).

Runs a Semgrep-compatible engine (Semgrep CE, or Opengrep: same CLI) with the
vendored, permissively licensed rule set in ``backend/app/rules/semgrep`` over a
batch of per-function units and returns the hits per unit. The hits are meant
as evidence the LLM prompt can cite (rule id, CWE, line), not as a gate.

Rules are NOT selected by the retrieved CVE category: retrieval picks the wrong
class most of the time on named categories, so the whole language rule set runs.

How a unit becomes a scan target: each unit's code is written to its own temp
file with the right extension, padded with blank lines so that line N of the
temp file is line N of the real file — reported lines need no remapping. Units
that don't parse standalone (a JS/TS or Java method without its class, a Go
function without a package clause) get a one-line wrapper placed on the padding
line just above the unit (or prefixed onto its first line when it starts at
line 1), so line numbers are still preserved. All units go to ONE engine
process: startup + rule compilation dominate (seconds), per-unit cost is small.

Snippet mode loses context: many rules key on imports / ``require()`` or a
class, which a lone function lacks (on the upstream rule fixtures a function
scanned alone recovers only ~30% of the in-function hits a whole-file scan finds
for Python/JS/Java, ~80% for Go). When the caller has the full file text (files
mode always does), pass ``sources={file_path: content}``: each such file is
scanned once, whole, and every hit is assigned to the innermost unit whose line
range contains it — full context, same output shape.

Never breaks a scan: engine missing, timeout, crash or unreadable output logs a
warning and yields ``{}``.

Settings are constructor arguments with module-level defaults; wiring them into
``config.Settings`` is left to the integration step. ``scan_units`` is blocking
(subprocess) — call it via ``asyncio.to_thread`` from async code.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[1] / "rules" / "semgrep"
DEFAULT_TIMEOUT_S = 120.0  # whole engine invocation
DEFAULT_RULE_TIMEOUT_S = 5  # engine --timeout: per rule per file
DEFAULT_JOBS = 2
DEFAULT_MAX_MEMORY_MB = 1024  # engine --max-memory, per target file
# Rules whose hits are dropped. Bandit's B101 (``assert`` used) is a code-quality
# rule, not a vulnerability, and was the single largest source of hits on
# ordinary code AND on fixed twins in the eval (ROADMAP 1f step 5).
DEFAULT_EXCLUDED_RULES = frozenset({"python_assert_rule-assert-used"})
# Vendored layout: file extension -> rule subdirectories. Passing only the
# subdirectories a batch needs halves engine startup (rule compilation is most
# of the fixed cost). Used only when rules_dir is the vendored default.
DEFAULT_LANGUAGE_DIRS: dict[str, tuple[str, ...]] = {
    ".py": ("python",),
    **{ext: ("javascript", "javascript-lgpl3")
       for ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")},
    ".go": ("go",),
    ".java": ("java",),
}
MAX_MESSAGE_CHARS = 200
MAX_SNIPPET_LINES = 3
MAX_SNIPPET_LINE_CHARS = 200

UnitKey = tuple[str, str | None, int]
Hit = dict[str, Any]

# Unit language -> temp-file extension. The unit's own file suffix wins when it is
# one of the known ones (so .ts/.tsx/.jsx keep their parser).
_LANGUAGE_EXT = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "java": ".java",
}
_KNOWN_EXTS = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".java"}
_JS_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}

_WRAPPER_CLASS = "__RepoSentinelUnit__"
# A JS/TS class-method definition (``foo(a) {``, ``async *gen() {``, ``get x() {``,
# ``static #p() {``) — not a ``function`` declaration or an arrow function, which
# parse standalone.
_JS_METHOD_RE = re.compile(
    r"^\s*(?:(?:static|async|get|set|public|private|protected|override|readonly)\s+)*"
    r"\*?\s*(?:#?[A-Za-z_$][\w$]*|\[[^\]]*\]|'[^']*'|\"[^\"]*\")\s*(?:<[^>]*>)?\s*\("
)
_JS_NOT_METHOD_RE = re.compile(r"^\s*(?:async\s+)?function\b|^\s*async\s*\(")
_GO_PACKAGE_RE = re.compile(r"^\s*package\s+\w+", re.MULTILINE)
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
_RULE_ID_RE = re.compile(r"""^\s*-\s*id:\s*["']?([^"'\s#]+)""", re.MULTILINE)
_SEVERITY_MAP = {"ERROR": "high", "WARNING": "medium", "INFO": "low"}

# Environment passed to the engine: enough to run, nothing else (no API keys,
# no SEMGREP_APP_TOKEN that would switch on logged-in / networked behaviour).
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME")


def unit_key(unit: dict[str, Any]) -> UnitKey:
    """Key a unit the way results are returned: (file_path, function_name, start_line)."""
    return (unit["file_path"], unit.get("function_name"), int(unit.get("start_line") or 1))


def _find_engine() -> str | None:
    """semgrep on PATH, else next to this interpreter (a venv run without
    activation has no venv/bin on PATH), else opengrep."""
    found = shutil.which("semgrep")
    if found:
        return found
    beside = Path(sys.executable).parent / "semgrep"
    if beside.is_file() and os.access(beside, os.X_OK):
        return str(beside)
    return shutil.which("opengrep")


def _unit_ext(unit: dict[str, Any]) -> str | None:
    suffix = Path(unit.get("file_path") or "").suffix.lower()
    if suffix in _KNOWN_EXTS:
        return suffix
    return _LANGUAGE_EXT.get((unit.get("language") or "").lower()) or (suffix or None)


def _wrapper(unit: dict[str, Any], ext: str) -> tuple[str, str]:
    """(prefix, suffix) that make a function unit parse standalone. Whole-file
    units (function_name None) are never wrapped."""
    if unit.get("function_name") is None:
        return "", ""
    code = unit.get("code") or ""
    if ext == ".java":
        return f"class {_WRAPPER_CLASS} {{", "}"
    if ext == ".go" and not _GO_PACKAGE_RE.search(code):
        return "package main;", ""
    is_arrow = "=>" in code.split("{", 1)[0]
    if (
        ext in _JS_EXTS
        and not is_arrow
        and not _JS_NOT_METHOD_RE.match(code)
        and _JS_METHOD_RE.match(code)
    ):
        return f"class {_WRAPPER_CLASS} {{", "}"
    return "", ""


def build_target_source(unit: dict[str, Any], ext: str) -> str:
    """The temp-file text for a unit: its code at its real line numbers."""
    code_lines = (unit.get("code") or "").splitlines() or [""]
    start = max(int(unit.get("start_line") or 1), 1)
    prefix, suffix = _wrapper(unit, ext)
    if start >= 2:
        # Blank lines 1..start-2, the wrapper (or a blank) on line start-1.
        lines = [""] * (start - 2) + [prefix] + code_lines
    else:
        lines = ([f"{prefix} {code_lines[0]}"] if prefix else [code_lines[0]]) + code_lines[1:]
    if suffix:
        lines.append(suffix)
    return "\n".join(lines) + "\n"


def _normalize_cwe(raw: Any) -> list[str]:
    items = raw if isinstance(raw, list) else [raw] if raw else []
    out: list[str] = []
    for item in items:
        for match in _CWE_RE.findall(str(item)):
            cwe = match.upper()
            if cwe not in out:
                out.append(cwe)
    return out


def _normalize_severity(extra: dict[str, Any]) -> str:
    meta = extra.get("metadata") or {}
    sec = meta.get("security-severity")
    if isinstance(sec, str) and sec.strip():
        return sec.strip().lower()
    return _SEVERITY_MAP.get(str(extra.get("severity", "")).upper(), "info")


def _trim_message(message: str) -> str:
    text = " ".join((message or "").split())
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[: MAX_MESSAGE_CHARS - 1].rstrip() + "…"


def load_rule_ids(rules_dir: Path) -> set[str]:
    """Rule ids declared in a rules directory (regex over the YAML — cheap, and
    only used to strip the engine's path-derived prefix from check_id)."""
    ids: set[str] = set()
    for path in rules_dir.rglob("*.y*ml"):
        try:
            ids.update(_RULE_ID_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return ids


def short_rule_id(check_id: str, known_ids: set[str]) -> str:
    """Semgrep prefixes a local rule's id with its config path, dotted
    (``home.me.repo.rules.python.exec.python_exec_rule-x``). Return the longest
    dotted suffix that is a declared id (ids may contain dots), else the last
    component."""
    parts = check_id.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in known_ids:
            return candidate
    return parts[-1]


def _unit_span(unit: dict[str, Any]) -> tuple[int, int]:
    """(first, last) real-file line of a unit, from its code (end_line may be
    stale or missing; the code is what was scanned)."""
    start = max(int(unit.get("start_line") or 1), 1)
    return start, start + len((unit.get("code") or "").splitlines() or [""]) - 1


def _owner(units: list[dict[str, Any]], line: int) -> dict[str, Any] | None:
    """Innermost unit containing ``line`` (nested functions: the inner one)."""
    best, best_len = None, None
    for unit in units:
        first, last = _unit_span(unit)
        if first <= line <= last and (best_len is None or last - first < best_len):
            best, best_len = unit, last - first
    return best


def parse_results(
    payload: dict[str, Any],
    targets: dict[str, dict[str, Any] | list[dict[str, Any]]],
    known_ids: set[str] | None = None,
) -> dict[UnitKey, list[Hit]]:
    """Turn engine JSON into hits per unit. ``targets`` maps the temp file name
    (as passed to the engine) to its unit, or to the list of units of a file
    scanned whole. A hit goes to the innermost unit containing its line; hits
    outside every unit (wrapper lines, unchanged code) are dropped, and
    duplicates (same unit, rule and line) collapse."""
    known_ids = known_ids or set()
    out: dict[UnitKey, list[Hit]] = {}
    seen: set[tuple[UnitKey, str, int]] = set()
    for result in payload.get("results") or []:
        try:
            owners = targets.get(Path(result["path"]).name)
            if owners is None:
                continue
            line = int(result["start"]["line"])
            end_line = int((result.get("end") or {}).get("line", line))
            extra = result.get("extra") or {}
        except (KeyError, TypeError, ValueError):
            continue
        unit = _owner(owners if isinstance(owners, list) else [owners], line)
        if unit is None:
            continue
        code_lines = (unit.get("code") or "").splitlines() or [""]
        start, last = _unit_span(unit)
        end_line = min(max(end_line, line), last)
        rule_id = short_rule_id(str(result.get("check_id", "")), known_ids)
        key = unit_key(unit)
        if (key, rule_id, line) in seen:
            continue
        seen.add((key, rule_id, line))
        snippet_lines = code_lines[line - start : min(end_line, line + MAX_SNIPPET_LINES - 1)
                                   - start + 1]
        out.setdefault(key, []).append(
            {
                "rule_id": rule_id,
                "message": _trim_message(extra.get("message", "")),
                "severity": _normalize_severity(extra),
                "cwe": _normalize_cwe((extra.get("metadata") or {}).get("cwe")),
                "line": line,
                "end_line": end_line,
                "snippet": "\n".join(s.rstrip()[:MAX_SNIPPET_LINE_CHARS] for s in snippet_lines),
            }
        )
    for hits in out.values():
        hits.sort(key=lambda h: (h["line"], h["rule_id"]))
    return out


class SemgrepScanner:
    def __init__(
        self,
        rules_dir: str | Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        enabled: bool = True,
        engine_path: str | None = None,
        rule_timeout_s: int = DEFAULT_RULE_TIMEOUT_S,
        jobs: int = DEFAULT_JOBS,
        max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
        exclude_rules: frozenset[str] | set[str] = DEFAULT_EXCLUDED_RULES,
        language_dirs: dict[str, tuple[str, ...]] | None = None,
    ):
        """``language_dirs`` (ext -> rule subdirs) scopes each run to the
        subdirectories its targets need; it defaults to the vendored layout for
        the default rules_dir and to None (whole directory) for a custom one."""
        self.rules_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
        self.exclude_rules = frozenset(exclude_rules)
        self.language_dirs = (
            language_dirs if language_dirs is not None
            else DEFAULT_LANGUAGE_DIRS if rules_dir is None else None
        )
        self.timeout_s = timeout_s
        self.enabled = enabled
        self.rule_timeout_s = rule_timeout_s
        self.jobs = jobs
        self.max_memory_mb = max_memory_mb
        self._engine_path = engine_path
        self._engine_resolved = engine_path is not None
        self._known_ids: set[str] | None = None

    @property
    def engine(self) -> str | None:
        if not self._engine_resolved:
            self._engine_path = _find_engine()
            self._engine_resolved = True
        return self._engine_path

    def available(self) -> bool:
        """True when the engine binary exists and the rules directory is present."""
        engine = self.engine
        if not engine:
            return False
        path = Path(engine)
        if path.is_absolute() and not (path.is_file() and os.access(path, os.X_OK)):
            return False
        if not path.is_absolute() and shutil.which(engine) is None:
            return False
        return self.rules_dir.is_dir()

    def _configs(self, files: list[str]) -> list[str]:
        """Rule paths for these targets: the whole rules_dir, or with
        language_dirs only the subdirectories their extensions map to."""
        if self.language_dirs is None:
            return [str(self.rules_dir)]
        subdirs: list[str] = []
        for name in files:
            for sub in self.language_dirs.get(Path(name).suffix.lower(), ()):
                if sub not in subdirs and (self.rules_dir / sub).is_dir():
                    subdirs.append(sub)
        return [str(self.rules_dir / sub) for sub in subdirs]

    def _command(self, files: list[str], configs: list[str] | None = None) -> list[str]:
        cmd = [self.engine or "semgrep", "scan"]
        for config in configs if configs is not None else self._configs(files):
            cmd += ["--config", config]
        cmd += [
            "--json", "--quiet",
            "--no-git-ignore",
            "--timeout", str(self.rule_timeout_s),
            "--timeout-threshold", "3",
            "--max-memory", str(self.max_memory_mb),
            "--jobs", str(self.jobs),
        ]
        if "opengrep" not in Path(self.engine or "").name:
            # Semgrep CE only: no telemetry, no update check (no network at all
            # with a local --config). Opengrep has neither.
            cmd += ["--metrics=off", "--disable-version-check"]
        return cmd + files

    def _env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        env["SEMGREP_ENABLE_VERSION_CHECK"] = "0"
        env["SEMGREP_SEND_METRICS"] = "off"
        return env

    def _run(self, cmd: list[str], cwd: str) -> str | None:
        """Run the engine; stdout on success, None (after a warning) otherwise.
        Runs in its own process group so a timeout also kills semgrep-core."""
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=self._env(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True,
            )
        except OSError as exc:
            logger.warning(f"semgrep: could not start engine ({exc}); no static evidence")
            return None
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            logger.warning(f"semgrep: timed out after {self.timeout_s}s; no static evidence")
            return None
        # 0 = ok, 1 = findings with --error; anything else is a failure, but the
        # JSON may still hold results + per-file errors, so let the caller parse.
        if proc.returncode not in (0, 1) and not (stdout or "").lstrip().startswith("{"):
            logger.warning(
                f"semgrep: engine exited {proc.returncode}: {(stderr or '').strip()[-300:]}"
            )
            return None
        return stdout

    @staticmethod
    def _write_targets(
        tmp: str, units: list[dict[str, Any]], sources: dict[str, str]
    ) -> dict[str, dict[str, Any] | list[dict[str, Any]]]:
        """Write one temp file per whole source file (for units whose file text
        was supplied) and one per remaining unit; return name -> unit(s)."""
        targets: dict[str, dict[str, Any] | list[dict[str, Any]]] = {}
        by_file: dict[str, list[dict[str, Any]]] = {}
        for unit in units:
            if unit.get("file_path") in sources and _unit_ext(unit):
                by_file.setdefault(unit["file_path"], []).append(unit)
        for idx, (file_path, file_units) in enumerate(by_file.items()):
            name = f"f{idx:05d}{_unit_ext(file_units[0])}"
            Path(tmp, name).write_text(sources[file_path], encoding="utf-8")
            targets[name] = file_units
        for idx, unit in enumerate(units):
            ext = _unit_ext(unit)
            if unit.get("file_path") in by_file or not ext or not unit.get("code"):
                continue
            name = f"u{idx:05d}{ext}"
            Path(tmp, name).write_text(build_target_source(unit, ext), encoding="utf-8")
            targets[name] = unit
        return targets

    def scan_units(
        self, units: list[dict[str, Any]], sources: dict[str, str] | None = None
    ) -> dict[UnitKey, list[Hit]]:
        """Scan all units in one engine run. Returns {unit_key: [Hit, ...]} for
        units with at least one hit; {} when disabled, unavailable or failed.

        ``sources`` (optional) maps file_path -> full file text: those files are
        scanned whole (imports and enclosing class in context) and hits are
        assigned to their units by line; other units are scanned as snippets.
        """
        if not self.enabled or not units:
            return {}
        if not self.available():
            logger.warning(
                f"semgrep: engine or rules unavailable (engine={self.engine}, "
                f"rules={self.rules_dir}); no static evidence"
            )
            return {}
        if self._known_ids is None:
            self._known_ids = load_rule_ids(self.rules_dir)

        with tempfile.TemporaryDirectory(prefix="reposentinel_semgrep_") as tmp:
            try:
                targets = self._write_targets(tmp, units, sources or {})
            except (OSError, ValueError) as exc:
                logger.warning(f"semgrep: could not write scan targets ({exc})")
                return {}
            if not targets:
                return {}

            if self.language_dirs is not None:
                # No rules for a target's language -> don't hand it to the engine.
                targets = {n: u for n, u in targets.items()
                           if Path(n).suffix.lower() in self.language_dirs}
            files = sorted(targets)
            configs = self._configs(files)
            if not files or not configs:
                return {}
            stdout = self._run(self._command(files, configs), cwd=tmp)
            if stdout is None:
                return {}
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                logger.warning(f"semgrep: unreadable JSON output ({exc}); no static evidence")
                return {}
            if not isinstance(payload, dict):
                logger.warning("semgrep: unexpected JSON output; no static evidence")
                return {}

        errors = payload.get("errors") or []
        if errors:
            logger.debug(f"semgrep: {len(errors)} engine error(s), e.g. {str(errors[0])[:300]}")
        hits = parse_results(payload, targets, self._known_ids)
        if self.exclude_rules:
            hits = {k: kept for k, v in hits.items()
                    if (kept := [h for h in v if h["rule_id"] not in self.exclude_rules])}
        return hits
