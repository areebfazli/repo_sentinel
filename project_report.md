# RepoSentinel: Technical Project Report

**Version:** 1.0.0 MVP  
**Date:** June 2026  
**Architecture:** Python · FastAPI · Qdrant · CodeBERT · OpenRouter

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [System Architecture](#system-architecture)
5. [Core Components](#core-components)
   - [Phase 1: Foundation Layer](#phase-1-foundation-layer)
   - [Phase 2: Ghost Hunter (CVE Pipeline)](#phase-2-ghost-hunter-cve-pipeline)
   - [Phase 3: Team Memory Pipeline](#phase-3-team-memory-pipeline)
   - [Phase 4: The Unified API & RAG Merger](#phase-4-the-unified-api--rag-merger)
   - [Phase 5: Frontend Dashboard & GitHub Action](#phase-5-frontend-dashboard--github-action)
6. [Data Flow: End-to-End Request Lifecycle](#data-flow-end-to-end-request-lifecycle)
7. [Technology Stack](#technology-stack)
8. [Key Design Decisions](#key-design-decisions)
9. [Directory Structure](#directory-structure)
10. [Running the Project](#running-the-project)

---

## Overview

**RepoSentinel** is an AI-powered code security and institutional memory platform. It acts as a permanent, intelligent reviewer that sits on top of your team's Pull Request workflow. When a developer submits code, RepoSentinel simultaneously cross-references it against two separate intelligence databases:

1. **The Global CVE Threat Database (Ghost Hunter):** Has the world already seen this vulnerability?
2. **The Internal Team Memory Database:** Has *your team* complained about or fixed this exact pattern before?

The findings from both pipelines are merged and passed to a large language model (LLM) which generates a detailed, human-readable security report posted back to the developer — all before a single human reviewer has opened the PR.

---

## The Problem

Modern code review has two critical blind spots:

### 1. Known Vulnerability Blindness
Security vulnerabilities are well-catalogued in public databases (NVD, MITRE CVE). However, developers often unknowingly reintroduce patterns that match historically exploited code. There is no automated system that checks *new code* against *known vulnerability signatures* at a semantic level.

### 2. Institutional Memory Loss
When a senior developer reviews a PR and catches a bad pattern (e.g., "never use a bare `except:` here, it silently swallows authentication errors"), that knowledge lives in a comment thread buried in an old PR. The next junior developer makes the exact same mistake because there's no system to surface *what the team has already learned the hard way*.

---

## The Solution

RepoSentinel solves both problems using a **Retrieval-Augmented Generation (RAG)** architecture:

- Code is converted into mathematical vectors using a **code-specialized embedding model (CodeBERT)**.
- These vectors are stored in and retrieved from a **vector database (Qdrant)**, which finds *semantically similar* code — not just syntactically identical code.
- A **Cross-Encoder Reranker** acts as a second-pass precision filter to eliminate false positives.
- An **LLM (via OpenRouter)** synthesizes the raw technical findings into a clear, actionable report using natural language.

The result is a system that catches vulnerabilities that `grep` would miss (because they're structurally similar but not textually identical) and that brings forward institutional team knowledge at the exact moment it's needed.

---

## System Architecture

```
[ Developer Opens PR ]
         │
         ▼
[ GitHub Action Webhook Fires ]
         │  (sends code diff)
         ▼
┌─────────────────────────────────────┐
│         FastAPI Backend             │
│          POST /api/v1/analyze/      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         RAG Merger                  │
│    (runs two searches in parallel)  │
└─────────────────────────────────────┘
    │                         │
    ▼                         ▼
┌──────────────┐   ┌──────────────────────┐
│ CVERetriever │   │  TeamRetriever       │
│ Ghost Hunter │   │  Team Memory         │
└──────────────┘   └──────────────────────┘
    │                         │
    ▼                         ▼
┌──────────────────────────────────────────────────────┐
│         CodeBERT Embedder                            │
│   Converts code → 768-dimensional float vector      │
└──────────────────────────────────────────────────────┘
    │                         │
    ▼                         ▼
┌─────────────────┐  ┌────────────────────┐
│ Qdrant Collection│  │ Qdrant Collection  │
│  `cve_corpus`   │  │  `team_history`    │
└─────────────────┘  └────────────────────┘
    │                         │
    ▼                         ▼
┌──────────────────────────────────────────────────────┐
│         Cross-Encoder Reranker                       │
│   ms-marco-MiniLM-L-6-v2: kills false positives     │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│         Report Generator                             │
│   OpenRouter → openai/gpt-oss-120b:free              │
│   Renders structured Markdown security report        │
└──────────────────────────────────────────────────────┘
         │
         ▼
[ Markdown Report posted as PR Comment ]
```

---

## Core Components

### Phase 1: Foundation Layer

#### `backend/app/config.py`
A Pydantic-based settings manager. Reads configuration from environment variables and `.env` files. Defines the separation between `development` (local SQLite/in-memory Qdrant) and `production` (hosted services) environments. Key settings include `OPENROUTER_API_KEY`, `GITHUB_TOKEN`, and `QDRANT_LOCAL_PATH`.

#### `backend/app/core/github_crawler.py`
Wraps the PyGithub library to connect to GitHub's REST API and fetch closed Pull Requests from any repository. Extracts the PR title, description, all review comments (with author attribution), and the code diff for each PR. This is the data ingestion pipeline for Team Memory.

#### `backend/app/core/code_parser.py`
Uses the `tree-sitter` library (a fast, multi-language AST parser) to parse source files into their constituent function definitions. Supports Python, JavaScript, TypeScript, Go, and Java. By splitting code into discrete function-level chunks, we ensure vector embeddings are semantically meaningful at the right granularity.

#### `backend/app/core/embedder.py`
Wraps HuggingFace's `microsoft/codebert-base` model. CodeBERT is a transformer model pre-trained on both natural language and source code simultaneously, making it significantly better at capturing *code semantics* than general-purpose models like BERT.

Each code snippet is tokenized and passed through the model. The `[CLS]` token's hidden state (a 768-dimensional float vector) is extracted as the snippet's embedding. This vector is what gets stored in Qdrant and what gets compared during search.

#### `backend/app/core/vector_store.py`
Wraps the Qdrant client library. Manages two separate collections:
- **`cve_corpus`**: Stores CVE vulnerability entries with their code snippet embeddings.
- **`team_history`**: Stores past PR discussion embeddings.

In development mode, Qdrant runs locally by saving its data to a directory (`/qdrant_data`). In production, this would point to a hosted Qdrant Cloud instance. Uses `query_points` (the current Qdrant v1.18+ API) for similarity searches.

---

### Phase 2: Ghost Hunter (CVE Pipeline)

#### `scripts/ingest_cve_corpus.py`
A one-time ingestion script that seeds the CVE vector database. It takes a list of structured CVE entries (each containing a CVE ID, description, severity score, programming language, and a representative vulnerable code snippet), embeds each one using CodeBERT, and upserts them into the `cve_corpus` Qdrant collection.

The sample CVEs ingested cover: SQL Injection, Command Injection, Path Traversal, XML Injection, and LDAP Injection — a diverse set that tests the system's ability to distinguish between semantically similar vulnerability classes.

#### `backend/app/core/cve_retriever.py`
The query-time component of the Ghost Hunter. Takes a new code snippet, embeds it using CodeBERT, and issues an Approximate Nearest Neighbour (ANN) search against the `cve_corpus` collection to find the most similar vulnerability signatures.

Results are pre-filtered by a cosine similarity threshold before being passed to the Cross-Encoder reranker for precision scoring.

#### `backend/app/core/reranker.py`
One of the most critical components in the pipeline. The base CodeBERT ANN search finds *structurally similar* code, which is necessary but not sufficient — for example, SQL injection and Command injection have similar structural patterns (string interpolation into a function call) but are fundamentally different vulnerability classes.

The Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) solves this. Unlike the bi-encoder (CodeBERT), which embeds documents independently, a Cross-Encoder takes **both** the query and the candidate document as a single input and outputs a direct relevance score. This joint processing captures the nuanced relationship between the two texts.

The reranker acts as a second-pass filter: it takes the top-N broad matches from the ANN search and sorts them by true relevance, eliminating false positives before they reach the LLM.

---

### Phase 3: Team Memory Pipeline

#### `scripts/ingest_team_history.py`
Fetches closed Pull Requests from a specified GitHub repository (with an authenticated token) and processes them into Team Memory embeddings. For each PR, it constructs a rich text document combining the PR title, description, the specific review comment made by a senior developer, and the file name — then embeds it as a single semantic unit.

Falls back to a mock internal PR history dataset in development to avoid GitHub API rate limits during testing.

#### `backend/app/core/team_retriever.py`
Mirrors the `CVERetriever` architecture but queries the `team_history` Qdrant collection. When a developer submits code, the Team Retriever finds past PRs where the team discussed code with similar patterns. The `snippet_preview` from each matching PR (containing the original review comment and context) is surfaced to the report generator.

This creates the effect of a senior developer "remembering" a previous code review and citing it.

---

### Phase 4: The Unified API & RAG Merger

#### `backend/app/core/rag_merger.py`
The orchestration layer that ties the two pipelines together. Uses Python's `asyncio` to run the `CVERetriever` and `TeamRetriever` concurrently (in parallel), cutting the total query time roughly in half compared to running them sequentially.

Returns a merged dictionary containing:
```json
{
  "ghost_hunter_findings": [...],
  "team_memory_findings": [...],
  "is_vulnerable": true
}
```

#### `backend/app/core/report_generator.py`
Takes the raw JSON from the RAG Merger and constructs a structured prompt for the LLM. The prompt injects:
- The original code snippet under review
- The list of CVE matches with their IDs and descriptions
- The list of Team Memory matches with PR IDs and original review comments

It then calls the **OpenRouter API** (`openai/gpt-oss-120b:free`), requesting the LLM generate a response following a strict Markdown report structure with three sections:
- 🌐 **The World Has Seen This Break Before** (CVE findings)
- 🏠 **Your Team Has Seen This Break Before** (Team Memory findings)
- ✅ **Recommended Fix** (concrete corrected code)

Gracefully falls back to a mock report if no `OPENROUTER_API_KEY` is configured, allowing full end-to-end testing of the pipeline without an LLM.

#### `backend/app/api/routes/analyze.py`
A FastAPI router exposing the `POST /api/v1/analyze/` endpoint. Uses FastAPI's dependency injection (`Depends`) to ensure the `RagMerger` and `ReportGenerator` instances are singletons — loaded once at startup and reused across all requests. This prevents the expensive ML models from being reloaded on every request.

#### `backend/app/main.py`
The FastAPI application entry point. Uses FastAPI's `lifespan` context manager to pre-load all ML models into RAM during application startup, before the server begins accepting requests. This ensures zero cold-start latency on the first API call.

CORS is configured to allow cross-origin requests from the frontend dashboard during development.

---

### Phase 5: Frontend Dashboard & GitHub Action

#### `frontend/index.html` + `frontend/src/styles/main.css` + `frontend/src/app.js`
A premium, single-page web dashboard for manually testing the RepoSentinel pipeline. Features:

- **Dark glassmorphism design** with animated ambient gradient backgrounds
- **Code editor pane** (left) with a macOS-style toolbar, using JetBrains Mono font
- **Results pane** (right) that renders the LLM's Markdown report as rich HTML using `marked.js`
- **Micro-animations**: floating ghost icon idle state, shake animation on empty submit, slide-up animation on results
- **Loading states**: a spinner replaces the button text during the API call

The frontend communicates directly with the FastAPI backend at `http://127.0.0.1:8000/api/v1/analyze/` using the browser's native `fetch` API.

#### `.github/workflows/repo_sentinel.yml`
A GitHub Actions workflow that triggers on PR events (`opened`, `synchronize`, `reopened`). It:

1. Checks out the repository with full git history
2. Fetches the PR's unified diff using the GitHub API
3. Sends the diff to the hosted RepoSentinel FastAPI endpoint (configured via `REPOSENTINEL_URL` repository secret)
4. Captures the Markdown report response
5. Posts the report as a PR comment using `mshick/add-pr-comment@v2`

This workflow is what makes RepoSentinel a drop-in tool for any repository — a developer installs it by copying the workflow file and adding two repository secrets.

---

## Data Flow: End-to-End Request Lifecycle

```
1. Developer opens a PR
2. GitHub Action fires, fetches the diff
3. POST /api/v1/analyze/ receives {"code_snippet": "..."}
4. FastAPI resolves dependencies → RagMerger singleton
5. RagMerger fires two asyncio tasks concurrently:
   ├── CVERetriever.find_vulnerabilities(code)
   │     └── Embedder.embed_text(code) → 768-dim vector
   │     └── VectorStore.search_cve_corpus(vector, limit=5) → broad matches
   │     └── Reranker.rerank(code, matches) → precision-sorted results
   └── TeamRetriever.find_team_history(code)
         └── Embedder.embed_text(code) → 768-dim vector
         └── VectorStore.search_team_history(vector, limit=5) → broad matches
         └── Reranker.rerank(code, matches) → precision-sorted results
6. asyncio.gather() merges results from both tasks
7. ReportGenerator.generate_pr_comment(code, merged_findings)
   └── Constructs structured prompt
   └── POST https://openrouter.ai/api/v1/chat/completions
   └── Returns Markdown report string
8. AnalyzeResponse returned: {is_vulnerable, report_markdown, ghost_hunter_matches, team_memory_matches}
9. GitHub Action posts report_markdown as a PR comment
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Embedding Model** | `microsoft/codebert-base` | Converts code to semantic float vectors |
| **Reranker Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Eliminates false positives |
| **Vector Database** | Qdrant (local mode) | ANN similarity search |
| **Code Parsing** | tree-sitter | Multi-language AST function extraction |
| **ML Framework** | PyTorch + HuggingFace Transformers | Model inference |
| **Web Framework** | FastAPI + Uvicorn | HTTP API server |
| **Data Validation** | Pydantic v2 | Request/response schemas |
| **LLM Access** | OpenRouter API | LLM report generation |
| **LLM Model** | `openai/gpt-oss-120b:free` | Markdown report writer |
| **GitHub Integration** | PyGithub + GitHub Actions | PR crawling and webhook automation |
| **Frontend** | Vanilla HTML / CSS / JS | Interactive dashboard |
| **Settings** | python-dotenv + Pydantic Settings | Environment configuration |

---

## Key Design Decisions

### Why CodeBERT over general embedding models?
CodeBERT was pre-trained on a bimodal dataset of natural language documentation and corresponding source code. This gives it a shared semantic space between code and text. A general model like `text-embedding-ada-002` treats source code as raw text and misses structural code semantics like variable scope, call chains, and control flow.

### Why two-stage retrieval (ANN + Cross-Encoder)?
The ANN search is extremely fast (milliseconds) but imprecise — it finds *structurally similar* code, not *semantically equivalent* threats. The Cross-Encoder is slower but highly precise because it reasons about the *relationship* between the query and each candidate jointly. The two-stage architecture gets the best of both: speed from ANN, precision from the Cross-Encoder.

### Why local Qdrant over a cloud vector DB?
For development, running Qdrant as a local on-disk database eliminates the need for Docker, network connectivity, or API keys. The `qdrant_client` library supports seamless migration to hosted Qdrant Cloud in production by simply changing the `QdrantClient` initialization arguments in `config.py`.

### Why OpenRouter over direct Anthropic/OpenAI SDKs?
OpenRouter acts as a unified gateway to 200+ LLMs. By building against the OpenRouter API (which is OpenAI-compatible), the model can be swapped at any time by changing a single string — no SDK changes, no dependency changes. This is particularly valuable for a security tool where the optimal LLM may change as the field evolves.

### Why `asyncio.gather()` in the RAG Merger?
The CVE search and Team Memory search are completely independent operations that both hit different Qdrant collections. Running them sequentially would double the latency for no reason. `asyncio.gather()` fires both tasks at the same time and waits for both to complete, keeping response time tight even as the vector DBs grow.

---

## Directory Structure

```
repo_sentinel/
│
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes/
│       │       └── analyze.py          # POST /api/v1/analyze/ endpoint
│       ├── core/
│       │   ├── embedder.py             # CodeBERT wrapper
│       │   ├── code_parser.py          # tree-sitter AST function extractor
│       │   ├── vector_store.py         # Qdrant client (dual collection)
│       │   ├── reranker.py             # Cross-Encoder false positive filter
│       │   ├── cve_retriever.py        # Ghost Hunter: CVE search pipeline
│       │   ├── team_retriever.py       # Team Memory: PR history search pipeline
│       │   ├── rag_merger.py           # Async parallel orchestrator
│       │   ├── github_crawler.py       # GitHub PR data ingestion
│       │   └── report_generator.py     # OpenRouter LLM report writer
│       ├── models/
│       │   └── schemas.py              # Pydantic API schemas
│       ├── config.py                   # Environment-based settings
│       └── main.py                     # FastAPI app + lifespan manager
│
├── scripts/
│   ├── ingest_cve_corpus.py            # Seed the CVE vector database
│   └── ingest_team_history.py          # Seed the Team Memory vector database
│
├── frontend/
│   ├── index.html                      # Dashboard HTML structure
│   └── src/
│       ├── styles/main.css             # Glassmorphism design system
│       └── app.js                      # Fetch API + Markdown rendering
│
├── .github/
│   └── workflows/
│       └── repo_sentinel.yml           # GitHub Action: PR scan automation
│
├── qdrant_data/                        # Local Qdrant vector DB (dev only)
├── requirements.txt                    # Pinned Python dependencies
├── .env.example                        # Environment variable template
└── progress_tracker.md                 # Build phase log
```

---

## Running the Project

### Prerequisites
- Python 3.12+ with a virtual environment
- An OpenRouter API key (free tier works with `openai/gpt-oss-120b:free`)

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Install Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Seed the Databases (first time only)
```bash
# Ingest the CVE vulnerability corpus
PYTHONPATH=. python scripts/ingest_cve_corpus.py

# Ingest mock team PR history
PYTHONPATH=. python scripts/ingest_team_history.py
```

### 4. Start the API Server
```bash
PYTHONPATH=. python backend/app/main.py
# Server starts at http://0.0.0.0:8000
```

### 5. Open the Dashboard
Open `frontend/index.html` in your browser and paste code to scan.

### 6. Test with curl
```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"code_snippet": "query = \"SELECT * FROM users WHERE id = \" + user_id"}'
```

### 7. Install GitHub Action (optional)
Copy `.github/workflows/repo_sentinel.yml` into your target repository and add:
- `REPOSENTINEL_URL`: The public URL of your hosted FastAPI server
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
