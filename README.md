# RepoSentinel

**An AI security reviewer that reads every pull request and flags changes that open a security hole.**

## What it is

A *pull request* (PR) is a proposed change to a codebase that teammates review before it is
merged. Reviewers are good at spotting broken features, but security mistakes are easy to miss:
a deleted permission check, or user input passed straight into a shell command or a database
query. RepoSentinel reviews every PR on GitHub for these mistakes and leaves a comment on the
exact line when it finds one.

How a review works:

1. **Read the change.** It collects the changed files, the diff, and each changed function
   before and after the change.
2. **Gather evidence.** Cheap, deterministic checks point at suspicious spots:
   - [Semgrep](https://semgrep.dev) rules (a static-analysis tool that matches known risky
     code patterns);
   - a "did this PR remove a guard?" check that compares old and new code for deleted
     sanitisers, auth checks or safe-API calls;
   - optional reference material from a vector database (a search index that finds similar
     code): known vulnerable code from public CVEs (published security bugs) and the team's
     own past review comments.
3. **Audit the whole PR with an LLM** (a large language model). It asks "what does this change
   newly make exploitable?" It may ask for more code from the PR before it answers (the
   option exists, but the model never used it in the 160-PR eval run).
4. **Double-check every finding.** Each candidate goes to a separate verifier call. Only
   findings confirmed with high confidence are reported. A candidate the verifier can't
   confirm, at a spot where the change deleted code that looks like a security check
   (deleted lines, recognised as a control, not found elsewhere in the new code), is listed
   separately as "worth a look"; it never fails the check.
5. **Report.** It posts inline comments on the PR and can fail the check when a finding is
   severe enough.

Two rules keep it honest:

- **It ignores the PR title and description.** A description that says "harmless refactor"
  can't talk it out of a finding.
- **It never fakes a clean result.** If part of the review failed or ran out of budget, the
  result says `partial` or `failed`, not "no issues found".

It ships as a **GitHub Action** (the main way to use it), a **FastAPI** backend with an async
job API, and a small **web dashboard** for pasting code by hand.

## Key numbers

| What was measured | Result | Sample |
|---|---|---|
| Similarity search alone | Can't tell a bug from its fix: precision 0.5 (a coin flip) at every threshold. A function and its fixed version embed at cosine 0.96-0.998. | 1,054 eval functions |
| Old design: LLM reviews each changed function on its own | Caught **1 of 8** real bugs on the right lines (95% CI 2-47%) | 8 bug/fix pairs + 13 ordinary functions; a different, much smaller test than the PR-level row, so not directly comparable |
| Static checks only (Semgrep + guard-removal check), best case | Reach **23%** of bugs at a **4%** false-alarm rate on benign changes | 502 held-out bug/fix pairs |
| Guard-removal "alert" tier alone | Fires on **3%** of bugs, **0** false alarms measured | 506 pairs + 994 benign edits |
| **Current design: PR-level review** (preliminary; measured with the previous configuration, see below) | Reported findings (after the verifier): catches **12/55** bug-introducing PRs on the exact lines and **16/55** within the changed function; **16/41** within the function when denial-of-service and timing bugs (which the prompt doesn't target) are left out. Wrongly flags **2/57** fix PRs and **0/48** benign PRs. Audit candidates before the verifier: **35/55** caught within the function, 4/57 fix and 2/48 benign PRs flagged. | 160 of 200 dev PRs scored: 55 of 60 bug-introducing, 57 of 60 fix, only 48 of 80 benign |

What this means:

- Finding "similar" code is not enough to call something vulnerable, so retrieval is only
  context. The LLM review with verification makes the decision.
- Static checks are precise but see less than a quarter of the bugs. They work as leads for
  the LLM, not as the verdict.
- The PR-level numbers come from a partial run on a dev sample built from real CVE fix commits
  (each fix reversed gives a bug-introducing PR), using the free Qwen 3.8 27B model through
  OpenRouter, with temperature 0, the old verifier prompt and a confidence cutoff of 8. The
  current defaults differ (the model is now Nemotron 3 Super 120B, free on OpenRouter, at
  temperature 0.2; a rewritten verifier prompt, cutoff 7, 12K-token prompts). **A rerun with the current defaults is
  pending**, and so is the held-out test split.
- The benign PRs are files that changed alongside security fixes in the same commits, not
  ordinary PRs. Only 48 of the 80 were scored: if all 32 unknown ones were flagged, the
  false-alarm rate would be 32/80. The 0/48 also rests on the cutoff of 8: one benign PR
  (pypiserver) was confirmed at confidence 7, so the same verdicts at today's cutoff of 7
  would give 1/48.
- The verifier trades recall for quiet: within the changed function it keeps 16 of the 35
  catches the audit made, and removes both benign false alarms and 2 of the 4 fix-PR alarms.
  Tuning that trade-off is the next step.

Other facts worth knowing: the CVE corpus holds 2,890 vulnerable/fixed code pairs (25
handwritten, the rest mined from real fix commits of PyPI and npm advisories). Code is
embedded with `jina-embeddings-v2-base-code` on CPU. A typical PR review costs about 1-6 LLM
calls (roughly 5K-40K tokens), at most 12.

Method, per-stage results and caveats: [docs/DETAILS.md](docs/DETAILS.md) and
[ROADMAP.md](ROADMAP.md).

## Run it on your machine

### Requirements

- Python 3.12 or newer (developed on 3.14; the Action runs on 3.12).
- A few GB of free RAM. The embedding model loads into memory at start (CPU only, no GPU
  needed); the first run downloads it.
- An [OpenRouter](https://openrouter.ai) API key (the default model is the free
  `nvidia/nemotron-3-super-120b-a12b:free`, with no fallback). Or run without any key in mock
  mode (no real review).
- Semgrep is installed by `requirements.txt`. If the engine is missing, scans still run and
  only log a warning.

### 1. Install and configure

Run every command from the project root. The code uses absolute `backend.app.*` imports and is
not an installed package.

```bash
pip install -r requirements.txt
cp .env.example .env                 # fill in keys; defaults to ENVIRONMENT=development
```

`.env` only needs secrets. Provider and model choice are code defaults in
`backend/app/config.py` (any setting can still be overridden in `.env`). There is no LLM
fallback by default, so an OpenRouter outage or rate limit on the free model fails the review
(`failed`; the Action's coverage gate turns red). To add one, set `OPENROUTER_FALLBACK_MODEL`
and/or `LLM_FALLBACK_PROVIDER` (e.g. `groq` plus `GROQ_API_KEY`).

| Key | Needed? | Used for |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes (primary provider; startup fails without it) | The LLM review. Free models need "allow free endpoints that may train on inputs" in OpenRouter's privacy settings. |
| `GROQ_API_KEY` | Optional (not used by default) | Only with `LLM_PROVIDER=groq` or `LLM_FALLBACK_PROVIDER=groq`. Groq's free tier only fits prompts up to ~7.5K tokens; for a Groq-only setup set `PR_REVIEW_MAX_PROMPT_TOKENS=6000`. |
| `GITHUB_TOKEN` | Optional | Crawling a real repo's past review comments into Team Memory. |
| `REPOSENTINEL_API_KEY` | Optional in development, required in production | Shared secret clients send as `X-RepoSentinel-Key`. |

`ENVIRONMENT=development` (the default) stores vectors in `./qdrant_data/` and data in SQLite,
so no Docker is needed. `production` switches to networked Qdrant and Postgres.

### 2. Seed both vector collections

The API returns useful results only after **both** collections are seeded. Local Qdrant is
single-process, so **stop the API before running any ingest script**.

```bash
# CVE corpus (reads data/cve_corpus/*.json; stores each fix too, so prompts can show the fix diff)
python scripts/ingest_cve_corpus.py --recreate

# Team review history: demo data (offline)
python scripts/ingest_team_history.py --mock --recreate

# ...or real closed-PR review comments (needs GITHUB_TOKEN)
python scripts/ingest_team_history.py --repo owner/name
```

Re-run both with `--recreate` whenever the embedding model changes; mixed-model vectors
silently ruin search quality.

### 3. Start the API

```bash
python -m backend.app.main                     # serves 0.0.0.0:8000, loads models on boot
LLM_PROVIDER=mock python -m backend.app.main   # no LLM key: pipeline runs, no real review
```

Clients `POST /api/v1/analyze/`, get `202` and a `job_id`, then poll
`GET /api/v1/analyze/{job_id}`.

### 4. Open the dashboard (optional)

```bash
cd frontend && python -m http.server 8080      # then open http://localhost:8080
```

Opening `index.html` via `file://` won't work: CORS only allows `CORS_ORIGINS` (default
`localhost:8080`). The dashboard talks to `http://127.0.0.1:8000`.

### 5. Review PRs with the GitHub Action

1. Host the API where GitHub's runners can reach it.
2. Copy `.github/workflows/repo_sentinel.yml` and `github_action/scan_pr.py` into the
   repository you want reviewed.
3. Add two repository secrets: `REPOSENTINEL_URL` (where the API is reachable) and
   `REPOSENTINEL_API_KEY` (the same value as the server's `REPOSENTINEL_API_KEY`).
4. Adjust the gates in the workflow's `env` if needed:

| Input | Workflow value | Effect |
|---|---|---|
| `INPUT_FAIL_ON_SEVERITY` | `high` | Fail the check on a finding at or above this severity (`none`, `low`, `medium`, `high`, `critical`). |
| `INPUT_FAIL_ON_PARTIAL` | `true` | Fail when the review was `partial` or `failed` instead of passing as clean. |
| `INPUT_GATE_ON_DETERMINISTIC` | `false` | Let a guard-removal alert fail the check on its own. Off: the LLM review or Semgrep must flag the same spot. |

The Action posts inline comments on the quoted line, updates them in place on later pushes
instead of duplicating them, and adds a summary comment.

### 6. Tests and lint

```bash
python -m pytest -m "not slow"        # fast suite (no model loads, LLM mocked, fresh temp SQLite)
python -m pytest -m slow              # eval regression (needs a seeded cve_corpus + API stopped)
ruff check backend scripts tests ml github_action
```

## More detail

- [docs/DETAILS.md](docs/DETAILS.md): architecture, full configuration reference, the Semgrep
  and guard-removal checks, PR-level review internals, eval harnesses, corpus and dataset
  building, project layout, gotchas.
- [ROADMAP.md](ROADMAP.md): dated measurements, decisions and what comes next.
- The audit and verifier prompts are adapted from
  [anthropics/claude-code-security-review](https://github.com/anthropics/claude-code-security-review)
  (MIT) and [openai/codex-security](https://github.com/openai/codex-security) (Apache-2.0).
  Notices and licence texts:
  [backend/app/core/prompts/THIRD_PARTY_NOTICES.md](backend/app/core/prompts/THIRD_PARTY_NOTICES.md).
  Vendored Semgrep rules: [backend/app/rules/semgrep/README.md](backend/app/rules/semgrep/README.md).
