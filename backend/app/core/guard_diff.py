"""PR diff-direction check: did this change REMOVE a security guard? (ROADMAP step 6)

Retrieval and LLM judging of the *final* function cannot tell vulnerable code from
its fixed twin, but a reviewer always has the diff, and the *direction* of a change
is informative: a PR that removes an escaping call, an auth check, a bounds check or
a ``raise``-guard, swaps ``yaml.safe_load`` for ``yaml.load``, flips ``shell=True``
or ``verify=False``, or turns a parameterised query into an f-string is suspicious;
a PR that adds one is fix-like.

Design (deterministic, cheap, no model):

1. Each side (old / new code of one function) is parsed with tree-sitter
   (Python, JavaScript; TypeScript goes through the JS grammar like
   ``CodeParser``) and reduced to a multiset of *features*:

   - ``call``   — calls to guard functions (``GUARD_CALLS``: sanitisers, auth /
     permission checks, path-containment helpers, validators, prototype guards);
   - ``block``  — guard blocks: ``if cond: raise/return/throw/abort/continue``
     (either branch) and asserts, whose condition references a function parameter
     or a taint-ish name, or matches a guard-kind vocabulary (``BLOCK_KINDS``);
   - ``cond``   — length/bounds comparisons in other ``if`` conditions;
   - ``hazard`` — unsafe APIs (``HAZARDS``, e.g. ``yaml.load``, ``pickle.loads``,
     ``eval``, ``innerHTML =``) with their safe twins (``yaml.safe_load`` ...);
   - ``flag``   — security-relevant flag values (``FLAGS``: ``shell=True``,
     ``verify=False``, ``rejectUnauthorized: false``, ``autoescape`` ...);
   - ``sql``    — SQL string literals that are interpolated (f-string, ``%``,
     ``+``, ``.format``, template literal) vs parameterised (placeholders).

2. Features are compared as *counts per (family, kind)*, so moved code, reordered
   statements, reformatting, comments and local renames are invisible: keys use
   callee / attribute names and an identifier-anonymised condition shape, never
   local variable names or whitespace. A protective feature whose count drops is
   ``removed`` (``weakened`` for a flag); a hazard whose count rises is
   ``weakened`` — merged with a falling safe twin into one high-confidence API
   swap. The mirror images are ``added`` / ``strengthened``, so swapping old and
   new flips every direction (removed<->added, weakened<->strengthened).

3. ``risk`` nets the evidence: ``neg`` = sum of confidences of removed/weakened
   changes, ``pos`` = added/strengthened; ``guard_removed`` when ``neg >= RISK_MIN``
   and ``neg - pos >= RISK_MARGIN`` (``guard_added`` symmetrically), else ``none``.

Vocabularies are module-level tables, each entry with a short rationale; tune them
with ``python -m ml.evaluation.eval_guard_diff``. The corpus builder's rename /
format detector (``scripts/build_corpus_from_osv.py``) inspired the
count-and-anonymise approach; nothing is imported from it.

Production input: files mode already sends each changed file's post-change
``content`` plus GitHub's unified ``patch`` (``github_action/scan_pr.py``). The
patch covers every change in the file, so reverse-applying it to ``content``
reconstructs the whole pre-change file (``reverse_apply_patch``), and the old
version of each unit is the same-named function in it (``old_code_for_units``).
No signal is possible when the patch is missing (GitHub omits it for very large
diffs), does not apply, or the function is new; an optional ``base_content``
attribute on the file (not yet in ``FileInput``) is used when present.
``plan_units`` selects functions by *added* lines only, so a function whose only
change is a deleted guard is never a unit: integration must plan with
``patch_touched_lines(patch)`` (deletion points included) as ``changed_lines``.
"""
from __future__ import annotations

import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Node, Parser

Direction = Literal["removed", "added", "weakened", "strengthened"]
Risk = Literal["guard_removed", "guard_added", "none"]

# Decision thresholds on the netted confidence sums (see module docstring).
RISK_MIN = 0.5
RISK_MARGIN = 0.25
# ``GuardDiffResult.alert`` tier: see its docstring.
ALERT_MIN_CONFIDENCE = 0.7
# Protective changes explained by an extract/inline-method refactor are scaled by this.
HELPER_DISCOUNT = 0.4
# Callees too common to count as "a new helper the guard moved into".
COMMON_CALLEES = frozenset(
    "print len str int float bool list dict set tuple isinstance issubclass getattr setattr "
    "hasattr type format join get append extend items keys values split strip lower upper "
    "replace startswith endswith encode decode open read write debug info warning warn error "
    "exception log push map filter foreach then catch resolve tostring parse stringify "
    "includes indexof slice substring trim sort reverse update pop add remove copy range "
    "enumerate zip sorted min max sum any all super next iter exists isfile isdir sub match "
    "search compile findall fullmatch".split())

_NEGATIVE: frozenset[str] = frozenset({"removed", "weakened"})
_FLIP: dict[str, str] = {
    "removed": "added",
    "added": "removed",
    "weakened": "strengthened",
    "strengthened": "weakened",
}


@dataclass(frozen=True)
class GuardChange:
    direction: Direction
    kind: str  # sanitiser, auth_check, bounds_check, safe_api, tls_verify, shell, ...
    line: int | None  # 1-based; in the OLD code for removed/strengthened, NEW otherwise
    old_text: str
    new_text: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class GuardDiffResult:
    risk: Risk
    changes: tuple[GuardChange, ...] = ()
    removed_score: float = 0.0  # sum of confidences of removed + weakened changes
    added_score: float = 0.0  # sum of confidences of added + strengthened changes
    note: str | None = None  # why there is no / only partial signal

    @property
    def has_signal(self) -> bool:
        return self.risk != "none"

    @property
    def alert(self) -> bool:
        """High-precision tier: guard_removed backed by an unsafe API swap, a
        security-flag flip or SQL interpolation (a ``weakened`` change with
        confidence >= ``ALERT_MIN_CONFIDENCE``), not only by removed checks."""
        return self.risk == "guard_removed" and any(
            c.direction == "weakened" and c.confidence >= ALERT_MIN_CONFIDENCE
            for c in self.changes)


# ---------------------------------------------------------------------------
# Vocabularies (tunable). Regexes are case-insensitive full matches on the last
# segment of the callee (``os.path.realpath`` -> ``realpath``) unless noted.
# First matching kind wins, so order matters.
# ---------------------------------------------------------------------------

# (kind, regex on callee last segment, weight). Weight = confidence of one
# removal/addition of such a call.
GUARD_CALLS: list[tuple[str, str, float]] = [
    # Auth / permission / authenticity checks. Django/DRF/Flask-Login idioms,
    # constant-time comparisons (compare_digest, timingSafeEqual), JWT/signature
    # verification, CSRF helpers.
    ("auth_check",
     r"has_?perms?|has_?permissions?|has_object_permission|check_?perms?\w*"
     r"|check_?permissions?\w*|check_?access\w*|check_?auth\w*|checkAuth\w*"
     r"|require_?(login|auth|perm|admin|role|scope|user|owner)\w*|requireAuth\w*"
     r"|\w*login_required|\w*permission_required|user_passes_test|staff_member_required"
     r"|require_\w*(login|auth)\w*"
     r"|is_?authenticated|is_?authori[sz]ed|authori[sz]e\w*|isLoggedIn|ensureLoggedIn"
     r"|ensure_?(authenticated|logged_in|admin|owner|permission)\w*"
     r"|can_?(access|edit|view|read|write|delete|modify|manage|admin|update)\w*"
     r"|is_?admin|is_?owner|is_?superuser|is_?staff"
     r"|compare_digest|timingSafeEqual|constant_time_compare|safe_str_cmp"
     r"|verify\w*|csrf_protect|validate_csrf\w*|check_csrf\w*|protect_from_forgery",
     0.6),
    # Path containment / traversal defences: canonicalisation used for a
    # containment test, safe joins, filename sanitisers.
    ("path_containment",
     r"realpath|commonpath|commonprefix|is_relative_to|relative_to|safe_?join\w*"
     r"|safeJoin\w*|safe_?path\w*"
     r"|secure_filename|sanitize_?filename|sanitizeFilename|basename|resolve_?safe\w*",
     0.55),
    # Output encoding / sanitisers (XSS, header/CRLF, shell, SQL escaping, regex).
    ("sanitiser",
     r"(?!\w*unescape)\w*escape\w*|htmlspecialchars|conditional_escape"
     r"|format_html|escape_?js|escapejs|escape_?string|escape_?like|escapeId|escapeRegExp"
     r"|escape_?regex|escape_?shell\w*|escapeshellarg|quote|quote_plus|shlex_quote"
     r"|\w*saniti[sz]e\w*|\w*Saniti[sz]e\w*|clean_?html|strip_?tags|striptags|purify|xss"
     r"|filterXSS"
     r"|encodeURIComponent|encodeURI|encode_?uri\w*|urlencode|safe_?header\w*",
     0.55),
    # Prototype-pollution guards (JS): own-property checks, null-prototype objects.
    ("proto_guard", r"\w*OwnProperty\w*|hasOwn|freeze|isPrototypeOf", 0.5),
    # Generic validators. check_output/check_call are subprocess APIs, not checks.
    ("validation",
     r"_?validat\w*|\w+_validat\w*|is_?valid\w*|isValid\w*|check_(?!output\b|call\b)\w+"
     r"|ensure_?(safe|valid|allowed|within)\w*|is_?safe\w*|isSafe\w*|is_?allowed\w*"
     r"|isAllowed\w*|security_?checks?\w*|url_has_allowed_host_and_scheme|is_safe_url"
     r"|sanity_?check\w*|assert_\w+",
     0.45),
    # "safe_*" wrappers (safe_get, safeJoinModulesDir, safe_load): the name states
    # intent. mark_safe is a hazard (HAZARDS), not matched here.
    # safe_load is the yaml hazard's twin (HAZARDS) - not counted twice.
    ("safe_api", r"(?!safe_?load)safe_\w+|(?!safeLoad)safe[A-Z]\w*", 0.45),
]
# Attribute reads that are auth checks without a call (``user.is_authenticated``
# as a property); counted like the call so a () -> property change nets out.
AUTH_ATTRIBUTES = re.compile(r"is_authenticated|is_staff|is_superuser|is_admin|isAdmin"
                             r"|isAuthenticated")

# A ``replace``/``sub`` whose arguments mention traversal or markup characters is
# a hand-rolled sanitiser (``s.replace('..', '')``, ``.replace(/</g, '&lt;')``).
REPLACE_SANITISER_CALLEE = r"replace|replaceAll|sub|gsub|translate|strip"
REPLACE_SANITISER_ARGS = re.compile(
    r"\.\.|<|>|&(lt|gt|amp|quot|#)|\\x00|\\0|%00|\\r|\\n|javascript:|\\u0000|[\"']/[\"']")
REPLACE_SANITISER_WEIGHT = 0.4

# Guard-block kind vocabulary, matched against the condition text + the first
# exit statement (order matters; first hit wins). (kind, regex, bonus weight).
BLOCK_KINDS: list[tuple[str, str, float]] = [
    ("proto_guard", r"__proto__|\bconstructor\b|\bprototype\b|hasOwnProperty|hasOwn\b", 0.2),
    ("attr_guard", r"startswith\(\s*[\"']_|__(class|globals|builtins|subclasses|mro|base)__",
     0.2),
    # startswith only counts next to a path-ish name (``p.startswith(root)``);
    # prefix tests on ordinary strings are everywhere.
    ("path_containment",
     r"\.\.|realpath|commonpath|commonprefix|is_relative_to|isabs|abspath|normpath"
     r"|path\.sep|os\.sep|\bresolve\("
     r"|(?i:(path|dir|root|base|folder|file)\w*\.(startswith|startsWith)"
     r"|(startswith|startsWith)\(\s*\w*(path|dir|root|base|folder))", 0.2),
    ("auth_check",
     r"perm|auth|login|logged|\brole|admin|owner|staff|superuser|csrf|is_active|access"
     r"|forbidden|unauthori[sz]ed|\b40[13]\b|PermissionDenied|Forbidden|session|signature"
     r"|hmac|digest|verify|token", 0.2),
    ("bounds_check",
     r"(\blen\(|\.length\b|\bsize\b|\bcount\b|MAX|max_|_max\b|limit|\bdepth\b)", 0.1),
    ("validation",
     r"isinstance|typeof|instanceof|\bre\.(match|search|fullmatch)|\.test\(|\.match\("
     r"|scheme|valid|safe|allow|whitelist|allowlist|blacklist|denylist|\bin\s+[A-Z_]{3,}"
     r"|includes\(|indexOf|\btype\(", 0.1),
]
# Names that make an otherwise-unclassified guard condition count ("input_check").
TAINT_NAMES = re.compile(
    r"^(req|request|params?|query|body|input|user_?input|data|payload|path|file_?name"
    r"|filename|file_?path|url|uri|href|link|host|hostname|domain|redirect\w*|next"
    r"|target|dest\w*|cmd|command|args?|argv|key|name|value|headers?|cookies?|token"
    r"|sql|template|expr\w*|src|source|content|html|text|user\w*|email|obj|options?|opts)$",
    re.I)
# Base weight by how the guard exits; kind bonus is added, capped at BLOCK_WEIGHT_CAP.
EXIT_WEIGHTS: dict[str, float] = {"raise": 0.55, "abort": 0.55, "return": 0.4,
                                  "continue": 0.3, "assert": 0.45}
BLOCK_WEIGHT_CAP = 0.8
# Calls that end a request / fail hard when they are a guard's body.
ABORT_CALLS = re.compile(
    r"abort|exit|_exit|fail|deny|forbid|reject|send_error|sendStatus|throw\w*"
    r"|HttpResponseForbidden|HttpResponseBadRequest|HttpResponseNotFound|PermissionDenied",
    re.I)
MAX_GUARD_BODY_STATEMENTS = 4  # longer bodies are logic, not a guard
# Statements allowed before the exit of a guard body (logging / flashing).
LOG_CALLEE = re.compile(
    r"(^|\.)(log|logger|logging|_logger|LOG|console|warnings|messages)\.\w+$"
    r"|(^|\.)(print|warn|warning|error|info|debug|exception|log|flash|notify)$", re.I)
# A ``return`` counts as a guard exit only when it returns nothing, a literal or an
# error-ish value (``return None``, ``return res.status(403)``, ``return next(err)``);
# ``return computed_value`` is ordinary control flow.
LITERAL_RETURN = re.compile(
    r"None|False|True|null|undefined|false|true|-?\d+|''|\"\"|\[\]|\{\}|\(\)|NotImplemented")
ERRORISH = re.compile(
    r"err|fail|forbid|denied|deny|invalid|unauthori|status|abort|reject|\b4\d\d\b|bad"
    r"|block|not_?found|refuse|disallow|illegal|unsafe|insecure|redirect", re.I)
HTTP_4XX = re.compile(r"\b4(0[0-9]|1[0-9]|29)\b")
COND_BOUNDS = re.compile(
    r"\blen\(|\.length\b|\bsize\b|\bMAX|\bmax_|_max\b|limit|\bdepth\b|_LIMIT|_SIZE", re.I)
COND_WEIGHT = 0.25
_ORDER_OPS = frozenset({"<", ">", "<=", ">="})

# Unsafe APIs and their safe twins. (group, kind, languages, hazard regex, safe-twin
# regex or None, "unless" regex on the call text making it the safe twin instead,
# weight alone, weight as a swap). Regexes run on masked code (``_mask``), anchored
# as a call/assignment. Weight 0 = only meaningful as a swap (ordinary code uses it).
HAZARDS: list[tuple[str, str, frozenset[str], str, str | None, str | None, float, float]] = [
    # PyYAML: load/full_load construct arbitrary objects; safe_load does not.
    ("yaml_load", "safe_api", frozenset({"python", "javascript"}),
     r"\byaml\.(load|load_all|unsafe_load|unsafe_load_all|full_load)\s*\(",
     r"\byaml\.(safe_load|safe_load_all|safeLoad|safeLoadAll)\s*\(",
     r"C?SafeLoader|JSON_SCHEMA|FAILSAFE_SCHEMA", 0.5, 0.9),
    # pickle-family deserialisers execute code on load.
    ("pickle", "safe_api", frozenset({"python"}),
     r"\b(c?[Pp]ickle|dill|cloudpickle|marshal|shelve|jsonpickle)\.(loads?|open|decode"
     r"|Unpickler)\s*\(", r"\bjson\.loads?\s*\(", None, 0.5, 0.9),
    # eval/exec vs literal parsing.
    ("py_eval", "safe_api", frozenset({"python"}),
     r"(?<![\w.])(eval|exec)\s*\(", r"\b(ast\.)?literal_eval\s*\(|\bjson\.loads?\s*\(",
     None, 0.5, 0.9),
    ("js_eval", "safe_api", frozenset({"javascript"}),
     r"(?<![\w.])eval\s*\(|\bnew\s+Function\s*\(|\bvm\.(runInNewContext|runInThisContext"
     r"|runInContext|compileFunction)\s*\(", r"\bJSON\.parse\s*\(", None, 0.5, 0.9),
    # Shell-string execution vs argv execution.
    ("os_system", "shell", frozenset({"python"}),
     r"\bos\.(system|popen)\s*\(|\b(commands|subprocess)\.(getoutput|getstatusoutput)\s*\(",
     r"\bsubprocess\.(run|call|check_call|check_output|Popen)\s*\(", None, 0.45, 0.85),
    ("child_exec", "shell", frozenset({"javascript"}),
     r"(?<![\w$.])(exec|execSync)\s*\(|\b(child_process|childProcess|cp|shell|shelljs)\."
     r"(exec|execSync)\s*\(",
     r"\b(execFile|execFileSync|spawn|spawnSync)\s*\(", None, 0.45, 0.85),
    # DOM sinks vs text sinks.
    ("inner_html", "sanitiser", frozenset({"javascript"}),
     r"\.(innerHTML|outerHTML)\s*(\+?=)(?!=)|\binsertAdjacentHTML\s*\(|\bdocument\.write(ln)?"
     r"\s*\(|\bdangerouslySetInnerHTML\b",
     r"\.(textContent|innerText)\s*=(?!=)|\bcreateTextNode\s*\(", None, 0.45, 0.85),
    # Django/Jinja "trust this HTML" markers vs escaping.
    ("mark_safe", "sanitiser", frozenset({"python"}),
     r"(?<![\w.])(mark_safe|Markup|SafeString|SafeText)\s*\((?!\s*[\"'][^\"'{}%]*[\"']\s*\))",
     r"(?<![\w.])(escape|conditional_escape|format_html)\s*\(", None, 0.4, 0.85),
    # Server-side template injection.
    ("template_string", "safe_api", frozenset({"python"}),
     r"\brender_template_string\s*\(|\.from_string\s*\(", r"\brender_template\s*\(",
     None, 0.35, 0.8),
    # Temp-file races.
    ("mktemp", "safe_api", frozenset({"python"}),
     r"(?<![\w])(tempfile\.)?mktemp\s*\(",
     r"\b(tempfile\.)?(mkstemp|mkdtemp|NamedTemporaryFile|TemporaryFile|TemporaryDirectory)"
     r"\s*\(", None, 0.3, 0.7),
    # Weak hashes (ordinary code uses md5 for cache keys -> low weight alone).
    ("weak_hash", "weak_crypto", frozenset({"python", "javascript"}),
     r"\bhashlib\.(md5|sha1)\s*\(|\bcreateHash\s*\(\s*[\"'](md5|sha1)[\"']",
     r"\bhashlib\.(sha256|sha384|sha512|sha3_\w+|blake2[bs]|pbkdf2_hmac|scrypt)\s*\("
     r"|\bcreateHash\s*\(\s*[\"']sha(256|384|512)[\"']|\b(bcrypt|argon2|scrypt)\b",
     None, 0.2, 0.7),
    # Non-crypto randomness: only a swap is informative.
    ("weak_random", "weak_crypto", frozenset({"python", "javascript"}),
     r"\brandom\.(random|randint|choice|choices|randrange|getrandbits)\s*\(|\bMath\.random"
     r"\s*\(", r"\bsecrets\.\w+\s*\(|\bos\.urandom\s*\(|\bSystemRandom\b|\bcrypto\."
     r"(randomBytes|randomUUID|getRandomValues|randomInt)\s*\(", None, 0.0, 0.6),
    # Path handling: abspath does not resolve symlinks; send_file vs the
    # directory-confined send_from_directory. Swap-only.
    ("abspath", "path_containment", frozenset({"python"}),
     r"\b(os\.path\.)?abspath\s*\(", r"\b(os\.path\.)?realpath\s*\(", None, 0.0, 0.7),
    ("send_file", "path_containment", frozenset({"python"}),
     r"(?<![\w.])(flask\.)?send_file\s*\(", r"\bsend_from_directory\s*\(", None, 0.0, 0.7),
    # XML parsers without defusedxml (swap to defusedxml is usually an import
    # change, so this mostly fires alone -> low weight).
    ("xml_parse", "safe_api", frozenset({"python"}),
     r"\b(xml\.etree\.\w+|minidom|pulldom|xml\.sax|expatbuilder)\.(parse|fromstring|XML"
     r"|parseString|iterparse)\s*\(", r"\bdefusedxml\.\w+", None, 0.25, 0.8),
    ("unserialize", "safe_api", frozenset({"javascript"}),
     r"(?<![\w$])(\w+\.)?unserialize\s*\(", r"\bJSON\.parse\s*\(", None, 0.5, 0.9),
    # Django CSRF opt-out.
    ("csrf_exempt", "auth_check", frozenset({"python"}), r"\bcsrf_exempt\b", None, None,
     0.5, 0.5),
]

# Security flags. (name regex, unsafe-value regex, safe-value regex or None,
# default_is_safe, kind, weight). When the library default is safe, only the
# unsafe literal is tracked; otherwise dropping an explicit safe setting also
# counts as weakening. Matched as ``name = value`` / ``name: value`` /
# ``'name': value`` on masked code (never ``==``).
FLAGS: list[tuple[str, str, str | None, bool, str, float]] = [
    ("shell", r"True|true|1", None, True, "shell", 0.8),
    ("verify|ssl_verify|verify_ssl|verify_certs?", r"False|false|0", None, True,
     "tls_verify", 0.8),
    ("check_hostname", r"False|false", None, True, "tls_verify", 0.8),
    ("rejectUnauthorized|strictSSL", r"false|0", None, True, "tls_verify", 0.8),
    ("insecure|insecureSkipVerify|InsecureSkipVerify", r"true|True", None, True,
     "tls_verify", 0.6),
    ("NODE_TLS_REJECT_UNAUTHORIZED", r"[\"']?0[\"']?", None, True, "tls_verify", 0.8),
    ("verify_signature|verify_exp|verify_aud|verify_iss", r"False|false", None, True,
     "auth_check", 0.7),
    ("ignoreExpiration|ignoreNotBefore", r"true", None, True, "auth_check", 0.6),
    ("autoescape|autoEscape", r"False|false", r"True|true|select_autoescape", False,
     "sanitiser", 0.7),
    ("noEscape", r"true", None, True, "sanitiser", 0.6),
    ("sanitize", r"false", None, True, "sanitiser", 0.5),
    ("resolve_entities|noent", r"True|true", r"False|false", False, "xxe", 0.6),
    ("load_dtd|huge_tree|dtd_validation|no_network", r"True|true", None, True, "xxe", 0.4),
    ("allow_pickle", r"True", None, True, "safe_api", 0.6),
    ("Loader", r"(yaml\.)?(Loader|UnsafeLoader|FullLoader)\b", r"(yaml\.)?C?SafeLoader",
     False, "safe_api", 0.7),
    ("allowProtoPropertiesByDefault|allowProtoMethodsByDefault|allowPrototypes",
     r"true", None, True, "proto_guard", 0.6),
    ("nodeIntegration", r"true", None, True, "safe_api", 0.6),
    ("contextIsolation|webSecurity|sandbox", r"false", None, True, "safe_api", 0.6),
    ("httponly|httpOnly|HttpOnly|secure", r"False|false", r"True|true", False,
     "cookie_flags", 0.35),
    ("allow_redirects|follow_redirects|followRedirects", r"True|true", r"False|false",
     False, "ssrf", 0.35),
]

# SQL: a string literal counts as SQL when it has two SQL keywords (at least one
# written upper-case, or the string is an argument of an execute-like call).
SQL_KEYWORDS = re.compile(
    r"\b(select|insert\s+into|update|delete\s+from|where|from|values|order\s+by"
    r"|group\s+by|join|set|limit|drop\s+table|create\s+table)\b", re.I)
SQL_UPPER = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|WHERE|FROM|VALUES|JOIN)\b")
SQL_EXEC_CALLEE = re.compile(
    r"execute|executemany|executescript|exec_driver_sql|query|raw|rawQuery|text|prepare"
    r"|all|run|get", re.I)
SQL_PLACEHOLDER = re.compile(r"%s|%\(\w+\)s|\?|(?<!:):\w+|\$\d+")
SQL_SWAP_WEIGHT = 0.85
SQL_ALONE_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LANG_ALIASES = {
    "python": "python", "py": "python", ".py": "python",
    "javascript": "javascript", "js": "javascript", ".js": "javascript",
    "typescript": "javascript", "ts": "javascript", ".ts": "javascript",
    "jsx": "javascript", ".jsx": "javascript", "tsx": "javascript", ".tsx": "javascript",
    "mjs": "javascript", ".mjs": "javascript", "cjs": "javascript", ".cjs": "javascript",
}


def normalize_language(language: str | None) -> str | None:
    """Map a language / extension to "python" / "javascript", else None."""
    if not language:
        return None
    return _LANG_ALIASES.get(language.strip().lower())


@lru_cache(maxsize=2)
def _language(lang: str) -> Language:
    mod = tree_sitter_python if lang == "python" else tree_sitter_javascript
    return Language(mod.language())


def _parse(code: str, lang: str) -> tuple[Node, bytes, int]:
    """Parse ``code``; returns (root, source bytes, line offset to subtract).

    A bare JS method (``foo(a) { ... }``, how ``CodeParser`` extracts
    ``method_definition``s) is not a valid program; wrap it in a class so the
    parameters and body parse properly.
    """
    parser = Parser(_language(lang))
    src = code.encode("utf-8", "replace")
    tree = parser.parse(src)
    root = tree.root_node
    if lang == "javascript" and root.has_error:
        wrapped = ("class __G__ {\n" + code + "\n}").encode("utf-8", "replace")
        wtree = parser.parse(wrapped)
        if not wtree.root_node.has_error:
            return wtree.root_node, wrapped, 1
    return root, src, 0


def _text(node: Node | None, src: bytes) -> str:
    if node is None:
        return ""
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _walk(node: Node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _first_line(text: str, limit: int = 160) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Feature:
    family: str  # call | block | cond | hazard | safe | flag | flag_safe | sql | sql_safe
    kind: str
    group: str  # netting key (with family/kind)
    key: str  # rename/format-insensitive identity, used to pick representatives
    polarity: int  # +1 protective (presence good), -1 hazard (presence bad)
    weight: float  # confidence of one unmatched add/remove (0 = swap-only)
    swap_weight: float
    line: int
    text: str
    idents: frozenset[str] = frozenset()  # identifiers the guard tests / the call takes


def _mask(root: Node, src: bytes) -> str:
    """Source for the regex detectors: comments blanked, and the text of "prose"
    strings (docstrings, messages: containing whitespace or > 40 bytes) blanked
    except quote characters, so ``shell=True`` in a docstring doesn't count while
    ``'verify_signature': False`` and ``createHash('md5')`` still do. Newlines
    are kept, so offsets and line numbers hold."""
    buf = bytearray(src)
    for n in _walk(root):
        prose = n.type in ("string", "template_string") and (
            n.end_byte - n.start_byte > 40 or re.search(rb"\s", src[n.start_byte:n.end_byte]))
        if n.type == "comment" or prose:
            for i in range(n.start_byte, n.end_byte):
                if buf[i] not in (10, 13, 34, 39, 96):
                    buf[i] = 32
    return buf.decode("utf-8", "replace")


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _callee(node: Node, src: bytes, lang: str) -> str:
    if lang == "javascript" and node.type == "new_expression":
        fn = node.child_by_field_name("constructor")
    else:
        fn = node.child_by_field_name("function")
    return re.sub(r"\s+", "", _text(fn, src))


def _last_segment(callee: str) -> str:
    """``os.path.realpath`` -> ``realpath``; ``Object.prototype.hasOwnProperty.call``
    -> ``hasOwnProperty`` (``call``/``apply``/``bind`` look through to the method)."""
    callee = re.sub(r"\?\.", ".", callee)
    parts = [p for p in re.split(r"[.\[\]]", callee) if p] if callee else []
    if len(parts) > 1 and parts[-1] in ("call", "apply", "bind"):
        return parts[-2]
    return parts[-1] if parts else ""


@lru_cache(maxsize=1)
def _guard_call_res() -> list[tuple[str, re.Pattern, float]]:
    return [(k, re.compile(rf"(?:{rx})", re.I), w) for k, rx, w in GUARD_CALLS]


def _classify_call(last: str) -> tuple[str, float] | None:
    for kind, rx, weight in _guard_call_res():
        if rx.fullmatch(last):
            return kind, weight
    return None


_KEEP_ID_PARENT_FIELDS = (("attribute", "attribute"), ("member_expression", "property"),
                          ("call", "function"), ("call_expression", "function"))


def _shape(node: Node, src: bytes) -> str:
    """Identifier-anonymised token shape of ``node`` (rename/format immune):
    plain identifiers -> ``ID``; callee names, attribute/property names, keywords,
    operators and numbers kept; strings keep their content (quote style dropped)."""
    out: list[str] = []

    def visit(n: Node) -> None:
        if n.type in ("string", "template_string"):
            body = re.sub(r"^[rRbBuUfF]*([\"'`]{1,3})|([\"'`]{1,3})$", "", _text(n, src))
            out.append("S:" + re.sub(r"\s+", " ", body)[:40])
            return
        if n.type == "comment":
            return
        if n.child_count:
            for c in n.children:
                visit(c)
            return
        t = _text(n, src)
        if n.type == "identifier":
            parent = n.parent
            keep = parent is not None and any(
                parent.type == ptype and parent.child_by_field_name(field) == n
                for ptype, field in _KEEP_ID_PARENT_FIELDS)
            out.append(t if keep else "ID")
        else:
            out.append(t)

    visit(node)
    return " ".join(out)


def _in_string(n: Node) -> bool:
    p = n.parent
    while p is not None:
        if p.type in ("string", "template_string"):
            return True
        p = p.parent
    return False


def _identifiers(node: Node, src: bytes) -> set[str]:
    return {_text(n, src) for n in _walk(node) if n.type == "identifier" and not _in_string(n)}


def _param_names(root: Node, src: bytes, lang: str) -> set[str]:
    wanted = ("parameters", "lambda_parameters") if lang == "python" else ("formal_parameters",)
    for n in _walk(root):
        if n.type in wanted:
            return _identifiers(n, src)
    return set()


def _statements(block: Node) -> list[Node]:
    if block.type in ("block", "statement_block"):
        return [c for c in block.named_children if c.type != "comment"]
    return [block]


def _exit_kind(block: Node | None, src: bytes, lang: str) -> tuple[str, Node] | None:
    """If ``block`` is a short guard body ending the flow, return (exit, stmt)."""
    if block is None:
        return None
    if block.type == "else_clause":
        body = block.child_by_field_name("body")
        if body is None:
            named = [c for c in block.named_children if c.type != "comment"]
            body = named[0] if named else None
        if body is None:
            return None
        block = body
    stmts = _statements(block)
    if not stmts or len(stmts) > MAX_GUARD_BODY_STATEMENTS:
        return None
    # The exit must come first, or after logging only: ``if x: y = f(); return y``
    # is logic, not a guard.
    for s in stmts:
        kind = _stmt_exit(s, src, lang)
        if kind:
            return kind, s
        if not _is_log_stmt(s, src, lang):
            return None
    return None


def _first_call(stmt: Node) -> Node | None:
    for c in _walk(stmt):
        if c.type in ("call", "call_expression"):
            return c
    return None


def _is_log_stmt(stmt: Node, src: bytes, lang: str) -> bool:
    if stmt.type != "expression_statement":
        return False
    call = _first_call(stmt)
    return call is not None and bool(LOG_CALLEE.search(_callee(call, src, lang)))


def _stmt_exit(s: Node, src: bytes, lang: str) -> str | None:
    t = s.type
    if t in ("raise_statement", "throw_statement"):
        return "raise"
    if t == "continue_statement":
        return "continue"
    if t == "return_statement":
        value = " ".join(_text(c, src) for c in s.named_children if c.type != "comment")
        value = value.strip()
        if not value or LITERAL_RETURN.fullmatch(value) or ERRORISH.search(value):
            return "return"
        return None
    if t == "expression_statement":
        call = _first_call(s)
        if call is None:
            return None
        if ABORT_CALLS.fullmatch(_last_segment(_callee(call, src, lang)) or ""):
            return "abort"
        args = call.child_by_field_name("arguments")
        if args is not None and HTTP_4XX.search(_text(args, src)):
            return "abort"  # res.send(401), res.status(403)
    return None


def _block_kind(text: str, idents: set[str], params: set[str]) -> tuple[str, float] | None:
    for kind, rx, bonus in BLOCK_KINDS:
        if re.search(rx, text):
            return kind, bonus
    if idents & params or any(TAINT_NAMES.match(i) for i in idents):
        return "input_check", 0.0
    return None


def _guard_block_features(root: Node, src: bytes, lang: str, params: set[str],
                          line_off: int) -> tuple[list[_Feature], set[int]]:
    feats: list[_Feature] = []
    guard_conds: set[int] = set()  # ids of condition nodes already used by a guard
    for n in _walk(root):
        cond = body = None
        branches: list[Node | None] = []
        if n.type in ("if_statement", "elif_clause"):
            cond = n.child_by_field_name("condition")
            body = n.child_by_field_name("consequence")
            branches = [body]
            alt = n.child_by_field_name("alternative")
            if lang == "javascript" and alt is not None and alt.type == "else_clause":
                named = [c for c in alt.named_children if c.type != "comment"]
                if named and named[0].type != "if_statement":
                    branches.append(alt)
            elif lang == "python":
                for c in n.children:
                    if c.type == "else_clause":
                        branches.append(c)
        elif n.type == "assert_statement":
            cond = n.named_children[0] if n.named_children else None
            branches = []
        elif n.type == "call_expression":
            last = _last_segment(_callee(n, src, lang))
            if last in ("assert", "ok", "invariant") and _callee(n, src, lang) in (
                    "assert", "assert.ok", "invariant", "console.assert"):
                args = n.child_by_field_name("arguments")
                cond = args.named_children[0] if args is not None and args.named_children \
                    else None
        if cond is None:
            continue
        exit_info = None
        if n.type in ("assert_statement", "call_expression"):
            exit_info = ("assert", n)
        else:
            for br in branches:
                exit_info = _exit_kind(br, src, lang)
                if exit_info:
                    break
        if exit_info is None:
            continue
        exit_name, stmt = exit_info
        cond_text = _text(cond, src)
        classify_text = cond_text + "\n" + _text(stmt, src)
        kb = _block_kind(classify_text, _identifiers(cond, src), params)
        # Unclassified guards are kept with weight 0: they never count on their
        # own, but let a renamed guard (whose kind changed with the name) cancel.
        kind, bonus = kb if kb is not None else ("unclassified", 0.0)
        weight = min(EXIT_WEIGHTS[exit_name] + bonus, BLOCK_WEIGHT_CAP) if kb else 0.0
        if kb is not None:
            guard_conds.add(cond.id)
        line = n.start_point[0] + 1 - line_off
        feats.append(_Feature("block", kind, kind, f"{exit_name}|{_shape(cond, src)}", 1,
                              weight, weight, line, _first_line(_text(n, src)),
                              frozenset(_identifiers(cond, src))))
    return feats, guard_conds


def _cond_features(root: Node, src: bytes, lang: str, guard_conds: set[int],
                   line_off: int) -> list[_Feature]:
    feats: list[_Feature] = []
    for n in _walk(root):
        if n.type not in ("if_statement", "elif_clause"):
            continue
        cond = n.child_by_field_name("condition")
        if cond is None or cond.id in guard_conds:
            continue
        for c in _walk(cond):
            if c.type == "comparison_operator":
                is_cmp = any(_text(ch, src) in _ORDER_OPS for ch in c.children)
            elif c.type == "binary_expression":
                op = c.child_by_field_name("operator")
                is_cmp = op is not None and _text(op, src) in _ORDER_OPS
            else:
                is_cmp = False
            if is_cmp and COND_BOUNDS.search(_text(c, src)):
                feats.append(_Feature("cond", "bounds_check", "bounds_check", _shape(c, src), 1,
                                      COND_WEIGHT, COND_WEIGHT,
                                      c.start_point[0] + 1 - line_off,
                                      _first_line(_text(c, src)),
                                      frozenset(_identifiers(c, src))))
    return feats


def _call_features(root: Node, src: bytes, lang: str, line_off: int) -> list[_Feature]:
    feats: list[_Feature] = []
    call_types = ("call",) if lang == "python" else ("call_expression", "new_expression")
    for n in _walk(root):
        if n.type in call_types:
            callee = _callee(n, src, lang)
            last = _last_segment(callee)
            if not last:
                continue
            hit = _classify_call(last)
            if hit is None and re.fullmatch(REPLACE_SANITISER_CALLEE, last):
                args = n.child_by_field_name("arguments")
                if args is not None and REPLACE_SANITISER_ARGS.search(_text(args, src)):
                    hit = ("sanitiser", REPLACE_SANITISER_WEIGHT)
            if hit is None:
                continue
            kind, weight = hit
        elif n.type in ("attribute", "member_expression"):
            name_node = n.child_by_field_name("attribute") or n.child_by_field_name("property")
            last = _text(name_node, src)
            parent = n.parent
            if not AUTH_ATTRIBUTES.fullmatch(last) or (
                    parent is not None and parent.type in call_types
                    and parent.child_by_field_name("function") == n):
                continue
            kind, weight = "auth_check", GUARD_CALLS[0][2]
        elif n.type == "decorator" and lang == "python":
            name = re.sub(r"\(.*", "", _text(n, src).lstrip("@").strip(), flags=re.S)
            last = _last_segment(name)
            hit = _classify_call(last)
            if hit is None:
                continue
            kind, weight = hit
        else:
            continue
        args = n.child_by_field_name("arguments")
        idents = frozenset(_identifiers(args, src)) if args is not None else frozenset()
        feats.append(_Feature("call", kind, kind, last.lower(), 1, weight, weight,
                              n.start_point[0] + 1 - line_off, _first_line(_text(n, src)),
                              idents))
    return feats


def _call_context(root: Node, src: bytes, lang: str) -> dict[str, frozenset[str]]:
    """callee last segment (lower-cased) -> identifiers passed to it, over all
    calls; used to spot a guard moved into a newly introduced helper."""
    ctx: dict[str, set[str]] = defaultdict(set)
    call_types = ("call",) if lang == "python" else ("call_expression", "new_expression")
    for n in _walk(root):
        if n.type in call_types:
            last = _last_segment(_callee(n, src, lang)).lower()
            if not last:
                continue
            args = n.child_by_field_name("arguments")
            ctx[last] |= _identifiers(args, src) if args is not None else set()
    return {k: frozenset(v) for k, v in ctx.items()}


def _call_span(code: str, start: int) -> str:
    """Text from ``start`` through the matching close paren of the first ``(``
    (bounded), for "unless" checks on a call's arguments."""
    i = code.find("(", start)
    if i < 0 or i - start > 80:
        return code[start:start + 200]
    depth = 0
    for j in range(i, min(len(code), i + 2000)):
        ch = code[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return code[start:j + 1]
    return code[start:start + 2000]


@lru_cache(maxsize=1)
def _hazard_res():
    out = []
    for group, kind, langs, hz, safe, unless, w, sw in HAZARDS:
        out.append((group, kind, langs, re.compile(hz), re.compile(safe) if safe else None,
                    re.compile(unless) if unless else None, w, sw))
    return out


def _hazard_features(code: str, lang: str, line_off: int) -> list[_Feature]:
    feats: list[_Feature] = []
    for group, kind, langs, hz, safe, unless, w, sw in _hazard_res():
        if lang not in langs:
            continue
        for m in hz.finditer(code):
            span = _call_span(code, m.start())
            line = _line_of(code, m.start()) - line_off
            if unless is not None and unless.search(span):
                feats.append(_Feature("safe", kind, group, group, 1, 0.0, sw, line,
                                      _first_line(span)))
                continue
            feats.append(_Feature("hazard", kind, group, group, -1, w, sw, line,
                                  _first_line(span)))
        if safe is not None:
            for m in safe.finditer(code):
                feats.append(_Feature("safe", kind, group, group, 1, 0.0, sw,
                                      _line_of(code, m.start()) - line_off,
                                      _first_line(_call_span(code, m.start()))))
    return feats


@lru_cache(maxsize=1)
def _flag_res():
    out = []
    for names, unsafe, safe, default_safe, kind, w in FLAGS:
        head = rf"(?<![\w$])[\"']?(?:{names})[\"']?\s*(?:(?<![=!<>])=(?!=)|:)\s*"
        out.append((names, re.compile(head + rf"(?:{unsafe})(?![\w])"),
                    re.compile(head + rf"(?:{safe})(?![\w])") if safe else None,
                    default_safe, kind, w))
    return out


def _flag_features(code: str, line_off: int) -> list[_Feature]:
    feats: list[_Feature] = []
    for names, unsafe_re, safe_re, default_safe, kind, w in _flag_res():
        group = "flag:" + names
        for m in unsafe_re.finditer(code):
            feats.append(_Feature("flag", kind, group, m.group(0), -1, w, w,
                                  _line_of(code, m.start()) - line_off, _first_line(m.group(0))))
        if safe_re is not None:
            alone = 0.0 if default_safe else w
            for m in safe_re.finditer(code):
                feats.append(_Feature("flag_safe", kind, group, m.group(0), 1, alone, w,
                                      _line_of(code, m.start()) - line_off,
                                      _first_line(m.group(0))))
    return feats


def _is_sql(text: str, in_exec_call: bool) -> bool:
    kws = {re.sub(r"\s+", " ", m.group(1).lower()) for m in SQL_KEYWORDS.finditer(text)}
    if len(kws) < 2:
        return False
    return bool(SQL_UPPER.search(text)) or in_exec_call


def _enclosing_call_callee(n: Node, src: bytes, lang: str) -> str:
    p = n.parent
    for _ in range(4):
        if p is None:
            return ""
        if p.type in ("call", "call_expression"):
            return _last_segment(_callee(p, src, lang))
        p = p.parent
    return ""


def _sql_features(root: Node, src: bytes, lang: str, line_off: int) -> list[_Feature]:
    feats: list[_Feature] = []
    for n in _walk(root):
        if n.type not in ("string", "template_string"):
            continue
        if n.parent is not None and n.parent.type in ("string", "template_string",
                                                      "concatenated_string"):
            continue
        text = _text(n, src)
        in_exec = bool(SQL_EXEC_CALLEE.fullmatch(_enclosing_call_callee(n, src, lang) or ""))
        if not _is_sql(text, in_exec):
            continue
        p = n.parent
        interpolated = False
        if lang == "python" and any(c.type == "interpolation" for c in n.children):
            interpolated = True
        if n.type == "template_string" and any(c.type == "template_substitution"
                                               for c in n.children):
            # A tagged template (sql`...${x}`) is parameterised by its tag.
            interpolated = not (p is not None and p.type == "call_expression")
        if p is not None and p.type in ("binary_operator", "binary_expression"):
            op = p.child_by_field_name("operator")
            if op is not None and _text(op, src) in ("%", "+"):
                interpolated = True
        if p is not None and p.type in ("attribute", "member_expression"):
            attr = p.child_by_field_name("attribute") or p.child_by_field_name("property")
            if _text(attr, src) in ("format", "format_map", "concat", "replace"):
                interpolated = True
        line = n.start_point[0] + 1 - line_off
        if interpolated:
            feats.append(_Feature("sql", "sql_param", "sql", "sql_interp", -1,
                                  SQL_ALONE_WEIGHT, SQL_SWAP_WEIGHT, line, _first_line(text)))
        elif SQL_PLACEHOLDER.search(text):
            feats.append(_Feature("sql_safe", "sql_param", "sql", "sql_placeholder", 1, 0.0,
                                  SQL_SWAP_WEIGHT, line, _first_line(text)))
    return feats


def extract_features(code: str, lang: str
                     ) -> tuple[list[_Feature], str | None, dict[str, frozenset[str]]]:
    """All guard/hazard features of one function: (features, note, call context)."""
    code = textwrap.dedent(code)
    root, src, off = _parse(code, lang)
    note = None
    if root.has_error:
        err = sum(n.end_byte - n.start_byte for n in _walk(root) if n.type == "ERROR")
        note = "partial_parse" if err < 0.5 * max(len(src), 1) else "mostly_unparsable"
    masked = _mask(root, src)
    params = _param_names(root, src, lang)
    blocks, guard_conds = _guard_block_features(root, src, lang, params, off)
    feats = [
        *blocks,
        *_cond_features(root, src, lang, guard_conds, off),
        *_call_features(root, src, lang, off),
        *_hazard_features(masked, lang, off),
        *_flag_features(masked, off),
        *_sql_features(root, src, lang, off),
    ]
    return feats, note, _call_context(root, src, lang)


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def _representatives(olds: list[_Feature], news: list[_Feature], n: int,
                     from_old: bool) -> list[_Feature]:
    """``n`` features from the side that lost them, preferring keys whose count
    actually changed (so a moved guard isn't reported instead of the removed one)."""
    src_side, other = (olds, news) if from_old else (news, olds)
    diff = Counter(f.key for f in src_side) - Counter(f.key for f in other)
    picked: list[_Feature] = []
    for f in src_side:
        if len(picked) == n:
            break
        if diff[f.key] > 0:
            diff[f.key] -= 1
            picked.append(f)
    for f in reversed(src_side):  # counts net out across keys: fill up
        if len(picked) == n:
            break
        if f not in picked:
            picked.append(f)
    return picked


def _change(direction: str, f: _Feature, conf: float, rationale: str,
            other: _Feature | None = None, from_old: bool | None = None) -> GuardChange:
    """``f`` is the feature that changed; ``from_old`` says which side it came
    from (default: removed/strengthened = old side, i.e. a protective feature
    lost or a hazard dropped)."""
    old_side = direction in ("removed", "strengthened") if from_old is None else from_old
    old_text = f.text if old_side else (other.text if other else "")
    new_text = (other.text if other else "") if old_side else f.text
    return GuardChange(direction=direction, kind=f.kind, line=f.line if f.line > 0 else None,
                       old_text=old_text, new_text=new_text, confidence=round(conf, 3),
                       rationale=rationale)


_FAMILY_LABEL = {
    "call": "guard call", "block": "guard block", "cond": "bounds comparison",
    "hazard": "unsafe API", "flag": "unsafe flag value", "flag_safe": "explicit safe flag",
    "sql": "interpolated SQL string", "sql_safe": "parameterised SQL",
}


def _drop_common(feats: list[_Feature], common: Counter) -> list[_Feature]:
    budget = Counter(common)
    out = []
    for f in feats:
        k = (f.family, f.key)
        if budget[k] > 0:
            budget[k] -= 1
            continue
        out.append(f)
    return out


def _new_helpers(ctx_from: dict[str, frozenset[str]], ctx_to: dict[str, frozenset[str]]
                 ) -> frozenset[str]:
    """Identifiers passed to helpers that are called on the ``to`` side only."""
    out: set[str] = set()
    for callee, idents in ctx_to.items():
        if callee not in ctx_from and callee not in COMMON_CALLEES:
            out |= idents
    return frozenset(out - {"self", "cls", "this"})


def diff_features(old_feats: list[_Feature], new_feats: list[_Feature],
                  old_ctx: dict[str, frozenset[str]] | None = None,
                  new_ctx: dict[str, frozenset[str]] | None = None) -> list[GuardChange]:
    """Net the two feature multisets into changes (see module docstring). With call
    contexts, a protective feature that disappears while a *new* helper is called
    on the same identifiers (extract-method refactor: the guard probably moved
    into it) is discounted by ``HELPER_DISCOUNT``; mirrored for additions."""
    old_ctx, new_ctx = old_ctx or {}, new_ctx or {}
    into_new_helper = _new_helpers(old_ctx, new_ctx)  # guard removed -> moved into it?
    from_old_helper = _new_helpers(new_ctx, old_ctx)  # guard added <- inlined from it?
    # Identical (family, key) features on both sides are the same guard, even if
    # a rename changed how it was classified: cancel them before netting by kind.
    common = (Counter((f.family, f.key) for f in old_feats)
              & Counter((f.family, f.key) for f in new_feats))
    old_feats = _drop_common(old_feats, common)
    new_feats = _drop_common(new_feats, common)
    groups: dict[tuple, tuple[list[_Feature], list[_Feature]]] = defaultdict(lambda: ([], []))
    for f in old_feats:
        groups[(f.family, f.kind, f.group)][0].append(f)
    for f in new_feats:
        groups[(f.family, f.kind, f.group)][1].append(f)

    # Pair hazard families with their safe twins for swaps.
    twin_family = {"hazard": "safe", "flag": "flag_safe", "sql": "sql_safe"}
    changes: list[GuardChange] = []
    consumed: set[tuple] = set()
    for (family, kind, group), (olds, news) in sorted(groups.items()):
        if family not in twin_family:
            continue
        twin_key = (twin_family[family], kind, group)
        t_olds, t_news = groups.get(twin_key, ([], []))
        d_haz = len(news) - len(olds)
        d_safe = len(t_news) - len(t_olds)
        consumed.add(twin_key)
        swaps = 0
        if d_haz > 0 and d_safe < 0:
            swaps = min(d_haz, -d_safe)
            haz = _representatives(olds, news, swaps, from_old=False)
            safe = _representatives(t_olds, t_news, swaps, from_old=True)
            for h, s in zip(haz, safe, strict=True):
                changes.append(_change("weakened", h, h.swap_weight,
                                       f"safe form replaced by {_FAMILY_LABEL[family]}"
                                       f" ({group})", other=s))
        elif d_haz < 0 and d_safe > 0:
            swaps = min(-d_haz, d_safe)
            haz = _representatives(olds, news, swaps, from_old=True)
            safe = _representatives(t_olds, t_news, swaps, from_old=False)
            for h, s in zip(haz, safe, strict=True):
                changes.append(_change("strengthened", h, h.swap_weight,
                                       f"unsafe {_FAMILY_LABEL[family]} replaced by safe form"
                                       f" ({group})", other=s))
        rest_haz = abs(d_haz) - swaps
        if rest_haz and (olds or news):
            reps = _representatives(olds, news, rest_haz, from_old=d_haz < 0)
            direction = "weakened" if d_haz > 0 else "strengthened"
            for h in reps:
                if h.weight > 0:
                    verb = "introduced" if d_haz > 0 else "dropped"
                    changes.append(_change(direction, h, h.weight,
                                           f"{_FAMILY_LABEL[family]} {verb} ({group})"))
        rest_safe = abs(d_safe) - swaps
        if rest_safe:
            reps = _representatives(t_olds, t_news, rest_safe, from_old=d_safe < 0)
            direction = "weakened" if d_safe < 0 else "strengthened"
            for s in reps:
                if s.weight > 0:
                    verb = "dropped" if d_safe < 0 else "added"
                    changes.append(_change(direction, s, s.weight,
                                           f"{_FAMILY_LABEL[twin_family[family]]} {verb}"
                                           f" ({group})", from_old=d_safe < 0))

    for gkey, (olds, news) in sorted(groups.items()):
        family = gkey[0]
        if family in twin_family or gkey in consumed:
            continue
        if family in twin_family.values():
            # Safe twin without any hazard occurrence on either side.
            delta = len(news) - len(olds)
            if delta == 0:
                continue
            reps = _representatives(olds, news, abs(delta), from_old=delta < 0)
            direction = "weakened" if delta < 0 else "strengthened"
            for s in reps:
                if s.weight > 0:
                    verb = "dropped" if delta < 0 else "added"
                    changes.append(_change(direction, s, s.weight,
                                           f"{_FAMILY_LABEL[family]} {verb} ({gkey[2]})",
                                           from_old=delta < 0))
            continue
        delta = len(news) - len(olds)
        if delta == 0:
            continue
        reps = _representatives(olds, news, abs(delta), from_old=delta < 0)
        direction = "removed" if delta < 0 else "added"
        helper_idents = into_new_helper if delta < 0 else from_old_helper
        for f in reps:
            if f.weight <= 0:
                continue
            why = f"{_FAMILY_LABEL[family]} {direction}: {f.kind}"
            conf = f.weight
            if family in ("block", "cond") and f.idents & helper_idents:
                conf *= HELPER_DISCOUNT
                why += (" (discounted: a helper called only on the other side takes the"
                        " same values, so the check may have moved there)")
            changes.append(_change(direction, f, conf, why))
    changes.sort(key=lambda c: (c.direction not in _NEGATIVE, -c.confidence, c.line or 0))
    return changes


def _decide(changes: list[GuardChange]) -> tuple[Risk, float, float]:
    neg = round(sum(c.confidence for c in changes if c.direction in _NEGATIVE), 3)
    pos = round(sum(c.confidence for c in changes if c.direction not in _NEGATIVE), 3)
    if neg >= RISK_MIN and neg - pos >= RISK_MARGIN:
        return "guard_removed", neg, pos
    if pos >= RISK_MIN and pos - neg >= RISK_MARGIN:
        return "guard_added", neg, pos
    return "none", neg, pos


def guard_diff(old_code: str | None, new_code: str | None, language: str | None) -> GuardDiffResult:
    """Compare one unit's old and new code. Never raises on bad input: an
    unsupported language, missing side or parse failure yields ``risk="none"``
    with a ``note``."""
    lang = normalize_language(language)
    if lang is None:
        return GuardDiffResult("none", note=f"unsupported_language:{language}")
    if old_code is None:
        return GuardDiffResult("none", note="no_old_code")
    if new_code is None:
        return GuardDiffResult("none", note="no_new_code")
    try:
        old_feats, old_note, old_ctx = extract_features(old_code, lang)
        new_feats, new_note, new_ctx = extract_features(new_code, lang)
    except Exception as exc:  # tree-sitter/regex edge cases must never break a scan
        return GuardDiffResult("none", note=f"analysis_failed:{type(exc).__name__}")
    changes = diff_features(old_feats, new_feats, old_ctx, new_ctx)
    risk, neg, pos = _decide(changes)
    notes = sorted({n for n in (old_note, new_note) if n})
    return GuardDiffResult(risk, tuple(changes), neg, pos, ",".join(notes) or None)


def flip_direction(direction: str) -> str:
    return _FLIP[direction]


# ---------------------------------------------------------------------------
# Files mode: reconstruct each unit's old code from content + patch
# ---------------------------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchMismatch(ValueError):
    """The patch does not apply (in reverse) to the given content."""


def _parse_hunks(patch: str) -> list[tuple[int, int, int, int, list[str]]]:
    hunks: list[tuple[int, int, int, int, list[str]]] = []
    cur: list[str] | None = None
    remaining_old = remaining_new = 0
    for line in patch.splitlines():
        m = _HUNK.match(line)
        if m:
            a, b, c, d = (int(m.group(1)), int(m.group(2) or 1), int(m.group(3)),
                          int(m.group(4) or 1))
            cur = []
            hunks.append((a, b, c, d, cur))
            remaining_old, remaining_new = b, d
            continue
        if cur is None:
            continue  # diff --git / index / ---/+++ headers
        if remaining_old <= 0 and remaining_new <= 0:
            if line.startswith("\\"):
                continue
            cur = None  # trailing junk after a complete hunk
            continue
        tag = line[0] if line else " "
        if tag == "\\":
            continue
        cur.append(line if line else " ")
        if tag == "-":
            remaining_old -= 1
        elif tag == "+":
            remaining_new -= 1
        else:
            remaining_old -= 1
            remaining_new -= 1
    return hunks


def reverse_apply_patch(new_content: str, patch: str) -> str:
    """Pre-change file content from the post-change ``new_content`` and its unified
    ``patch`` (GitHub's per-file ``patch`` field; ``---``/``+++`` headers optional).
    Raises ``PatchMismatch`` if the patch doesn't match the content."""
    hunks = _parse_hunks(patch)
    if not hunks:
        raise PatchMismatch("no hunks")
    new_lines = new_content.split("\n")
    old: list[str] = []
    pos = 0
    for _a, b, c, d, lines in hunks:
        start = c - 1 if d > 0 else c
        if start < pos or start > len(new_lines):
            raise PatchMismatch(f"hunk at +{c} out of order or past end")
        old.extend(new_lines[pos:start])
        pos = start
        n_old = n_new = 0
        for line in lines:
            tag, text = line[0], line[1:]
            if tag == "-":
                old.append(text)
                n_old += 1
                continue
            if pos >= len(new_lines) or new_lines[pos].rstrip("\r") != text.rstrip("\r"):
                raise PatchMismatch(f"line {pos + 1} does not match the patch")
            if tag == "+":
                n_new += 1
            else:
                old.append(new_lines[pos])
                n_old += 1
                n_new += 1
            pos += 1
        if n_old != b or n_new != d:
            raise PatchMismatch(f"hunk at +{c} has wrong line counts (truncated patch?)")
    old.extend(new_lines[pos:])
    return "\n".join(old)


def patch_touched_lines(patch: str) -> list[int]:
    """New-file lines a patch touches, *including deletion points*: added lines
    plus, for each run of removed lines, the new-file lines just before and
    after it. ``diff_utils.parse_patch_changed_lines`` returns added lines only,
    so ``plan_units`` never selects a function whose only change is a deletion
    (e.g. a removed guard); pass these as ``changed_lines`` for this check."""
    touched: set[int] = set()
    for _a, _b, c, d, lines in _parse_hunks(patch):
        new_line = c if d > 0 else c + 1
        prev_minus = False
        for line in lines:
            tag = line[0]
            if tag == "-":
                if not prev_minus and new_line > 1:
                    touched.add(new_line - 1)
                touched.add(new_line)
                prev_minus = True
                continue
            prev_minus = False
            if tag == "+":
                touched.add(new_line)
            new_line += 1
    return sorted(touched)


def _old_line_for(patch: str, new_line: int) -> int:
    """Approximate pre-change line number of post-change ``new_line``."""
    delta = 0
    for a, _b, c, _d, lines in _parse_hunks(patch):
        if c > new_line:
            break
        # Offset at the hunk start, then walk it.
        delta = a - c
        o, nl = a, c
        for line in lines:
            if nl > new_line:
                break
            tag = line[0]
            if tag == "-":
                o += 1
            elif tag == "+":
                nl += 1
            else:
                o += 1
                nl += 1
            delta = o - nl
    return new_line + delta


def old_content_for_file(file) -> tuple[str | None, str | None]:
    """(old_content, note) for a files-mode ``FileInput``-like object. Uses an
    explicit ``base_content`` attribute when present, else reverse-applies
    ``patch`` to ``content``."""
    base = getattr(file, "base_content", None)
    if base is not None:
        return base, None
    patch = getattr(file, "patch", None)
    if not patch:
        return None, "no_patch"
    try:
        return reverse_apply_patch(file.content, patch), None
    except PatchMismatch as exc:
        return None, f"patch_mismatch:{exc}"


def old_code_for_units(file, units: list[dict], parser) -> list[tuple[str | None, str | None]]:
    """For each unit (``plan_units`` dicts of this file), its pre-change code and
    a note. Functions are matched by name; among same-named functions (e.g.
    ``<anonymous>`` arrows) the one nearest the patch-mapped start line wins. A
    whole-file unit (``function_name`` None) maps to the whole old file."""
    old_content, note = old_content_for_file(file)
    if old_content is None:
        return [(None, note) for _ in units]
    ext = "." + file.path.rsplit(".", 1)[-1].lower() if "." in file.path else ""
    try:
        old_funcs = parser.extract_functions(old_content, ext) if parser.supports(ext) else []
    except Exception as exc:
        return [(None, f"old_parse_failed:{type(exc).__name__}") for _ in units]
    by_name: dict[str, list[dict]] = defaultdict(list)
    for f in old_funcs:
        by_name[f["name"]].append(f)
    patch = getattr(file, "patch", None) or ""
    out: list[tuple[str | None, str | None]] = []
    for unit in units:
        if unit.get("function_name") is None:
            out.append((old_content, None))
            continue
        cands = by_name.get(unit["function_name"], [])
        if not cands:
            out.append((None, "new_function"))
            continue
        target = _old_line_for(patch, unit["start_line"]) if patch else unit["start_line"]
        best = min(cands, key=lambda f: abs(f["start_line"] - target))
        out.append((best["code"], None))
    return out


def guard_diff_for_file(file, units: list[dict], parser) -> list[GuardDiffResult]:
    """``guard_diff`` for each unit of one files-mode file (same order as ``units``)."""
    results = []
    for unit, (old_code, note) in zip(units, old_code_for_units(file, units, parser),
                                      strict=True):
        if old_code is None:
            results.append(GuardDiffResult("none", note=note))
            continue
        results.append(guard_diff(old_code, unit["code"], unit.get("language")))
    return results
