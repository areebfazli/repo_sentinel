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
   newly make exploitable?" It may ask for more code from the PR before it answers (it did
   in 25 of 214 PRs of the held-out test; code outside the PR can't be shown to it).
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

Measured once on a held-out test set never used for tuning, with the shipped default setup
(the free `nvidia/nemotron-3-super-120b-a12b:free` model on OpenRouter). Setup and metrics
were fixed before the run
([pre-registration](ml/evaluation/results/pr_eval_v2_test217_preregistration.md)); nothing was
tuned afterwards. The 217 test PRs come from real open-source Python and JavaScript projects:
60 **bug-introducing PRs** (real security fixes for published CVEs, played backwards, so the
PR puts the vulnerability back), the 60 real **fix PRs** (nothing to report), and 97
**everyday commits** from the same projects (commits mentioning security filtered out).

| Question | Result |
|---|---|
| Does it catch a bug-introducing PR, in the right function? | **34 of 57 (60%)**, 95% CI 47-71%. On the exact lines (±2): 27 of 57 (47%). |
| ...leaving out denial-of-service and timing bugs, which it isn't built to find | 29 of 47 (62%) |
| Does it wrongly flag a fix PR? | **1 of 60 (1.7%)** |
| Does it wrongly flag an everyday commit? | **4 of 97 (4.1%)**; with 95% confidence at most 10.2% |

3 bug-introducing PRs errored (see below); counting what they found before failing and the
rest as misses, 36 of 60 (60%) were caught. Larger PRs draw more false alarms:

| | Under 50 changed lines | 50 or more |
|---|---|---|
| Everyday commits flagged | 2 of 81 (2.5%) | 2 of 16 (12.5%) |
| Fix PRs flagged | 0 of 46 | 1 of 14 |
| Bug-introducing PRs caught (right function) | 23 of 45 | 11 of 12 |

What this means:

- **Use findings as review comments, not as a merge blocker.** Few real PRs introduce a
  vulnerability. If 2% do, about **1 in 5 warnings is a real bug** (1 in 10 at 1%, 2 in 5 at
  5%; worse if the false-alarm rate is at the top of its range). Keep the Action's severity
  gate at `high` and let a person judge each finding.
- **Expect more than 4% false alarms on your PRs.** About 1 in 8 larger everyday commits was
  flagged, and real PRs are often larger than this sample's.
- **The benchmark is narrow.** Its bugs are reversed fixes, so most of them delete a check;
  real vulnerabilities are often new code. Denial-of-service and timing bugs are out of scope.
  The reviewer sees only the PR's changed files, not the rest of the repository.
- **The default model is free, with no fallback.** An OpenRouter outage or rate limit fails
  the review (`failed`, a red check), and free models can be withdrawn (the previous default
  was). You can add a fallback model in `.env` (see below).
- **The model sometimes thinks too long.** 3 of 60 bug-introducing PRs (5%) failed because it
  spent its whole 16K-token output budget reasoning. Such a review says `partial` or `failed`,
  never clean.
- **Cost:** on average 1.6 LLM calls, 17K tokens and 69 s of model time per PR (1 in 10 PRs
  uses over 39K tokens; 1 in 10 takes over 190 s).

Earlier measurements explain the design: similarity search can't tell a bug from its fix
(precision 0.5 at every threshold, 1,054 functions), so CVE retrieval (2,890 vulnerable/fixed
pairs) is only context; static checks alone (Semgrep + guard-removal check) reach 23% of bugs
at a 4% false-alarm rate (502 held-out pairs), so they are leads for the LLM, not the verdict.
Full results (dev vs test, the verifier's effect, cost) and caveats:
[docs/DETAILS.md](docs/DETAILS.md#held-out-test-results-v2) and [ROADMAP.md](ROADMAP.md).

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
