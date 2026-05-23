# Job Listing Matcher

<img width="2318" height="1271" alt="image" src="https://github.com/user-attachments/assets/b9f3501b-72d4-4724-bc53-f65abd22ca3f" />


A chat-based job-matching assistant. The user describes their experience and what
they are looking for in plain language; the system turns that into a structured
profile, retrieves the best-matching vacancies from a vector database, re-ranks
them with a trained model, and presents them next to the conversation — which the
user can keep refining to request more results.

**Who is this for, and why?** Anyone job-hunting who would rather *describe* what
they want than wrestle with the dozen drop-down filters of a typical job board.
Free-text intent ("a senior backend role, remote, fintech, that lets me keep using
Go but pick up Rust") carries far more signal than checkboxes — and our pipeline is
built to exploit exactly that signal.

## What it does

1. **Collects preferences through conversation.** A friendly chat agent asks for
   the user's skills, experience, desired role, location, salary, work mode, and
   exclusions — one or two questions at a time, never a form.
2. **Extracts a structured profile.** Once enough is known, the free-text
   conversation is parsed into a rich, validated schema (skills, languages,
   desired stack, domains, locations to prefer/exclude, compensation, work mode).
3. **Retrieves and re-ranks vacancies.** The profile is matched against a vector
   database of vacancies using hybrid retrieval, then re-ranked by a trained model
   for precision.
4. **Presents results and keeps the loop open.** Ranked vacancies appear in a
   results panel; the user can continue chatting to refine or request more.

## Architecture

Cleanly separated components, each with a single responsibility:

```
            React chat UI  (SSE streaming, results panel)
                          │
                          ▼
   FastAPI backend  ── auth · chat · history · logs routes ─┐
                          │                                 │
   ┌──────────────────────┼─────────────────────────────┐   │
   │  Conversation LLM     Profile-extraction LLM       │   │  JSONL audit log
   │  (fast, streaming) →  (more capable, free-text →   │   │  of every LLM
   │  collects prefs       structured schema)           │   │  input/output
   └──────────────────────┬─────────────────────────────┘   │
                          ▼                                 │
      Hybrid retrieval  →  LambdaRank reranker  →  ranked vacancies
   (Chroma dense + BM25,
      fused with RRF)

   Background (scheduled):
     cron / Celery → LangGraph ReAct agent → Adzuna API scrape
      → field extraction → Chroma upsert (+TTL) + BM25 refresh
```

- **Frontend** (`frontend/`) — React single-page chat UI. Streams assistant
  replies token-by-token over Server-Sent Events and slides in a job-results panel
  when matches arrive. Basic loading/error states; usable with no instructions.
- **Backend** (`backend/`) — FastAPI service with cookie-authenticated routes for
  chat, conversation history, and log inspection. Streams the chat and orchestrates
  the ML pipeline.
- **Conversation + extraction** (`services/conversation.py`, `sara_candidate_poll/`)
  — two LLM roles by design: a fast model drives the live chat, and a more capable
  model does one-shot extraction of free text into the structured profile schema.
  Both are provider-swappable via LiteLLM.
- **Retrieval + reranking** (`services/sara_retrieve_rerank/`) — the ML core: a
  persistent Chroma dense index, an in-memory BM25 index over the same documents,
  reciprocal-rank-fusion hybrid retrieval, and a LightGBM LambdaRank reranker.
- **Live vacancy agent** (`services/adzuna/`) — a scheduled LangGraph **ReAct
  agent** that refreshes the vacancy store from the public Adzuna API, adapting its
  search (e.g. broadening geography) when results are thin, then re-indexing.

## Why this approach

The task is fundamentally a **retrieval-and-ranking** problem over noisy,
free-text descriptions on both sides, so the design leans on **RAG** (hybrid
dense + sparse retrieval) for recall and a **learned reranker** for precision —
a stronger fit than either pure semantic search or a single LLM judgement. An
**agent** is used in the background with the intent of adapting to different
data sources (only one is used in the demo, though). 

## Evaluation

Quality is measured on a labelled validation set (4,995 candidates with
ground-truth source vacancies; reranker evaluated on the top-20 hit groups).
Each stage is measured against a sensible baseline:

| Stage | Metric | Baseline | This system | Lift |
|-------|--------|---------:|------------:|-----:|
| Retrieval | Recall@20 | 0.671 (dense only) | **0.754** (hybrid) | +8.3 pp |
| Reranking | Recall@1 | 0.552 (cosine) | **0.859** (LambdaRank) | +30.7 pp |
| Reranking | NDCG@10 | 0.721 (cosine) | **0.936** (LambdaRank) | +0.216 |

Feature ablations (cosine-only vs. + schema features vs. + LLM "expert" scores)
isolate where the lift comes from — the structured schema features carry most of
it. A miss-analysis suite breaks failures down by region, specialization, work
format, and English level. Full numbers, plots, and reproduction commands live in
[the pipeline README](backend/services/sara_retrieve_rerank/README.md).

## Engineering quality

- **Tests** — 84 unit and integration tests (`tests/`) covering preprocessing,
  retrieval (dense/BM25/hybrid), reranking, schema features, evaluation, the
  Adzuna agent.
- **Logging** — every LLM-based component writes a structured JSONL audit log of
  its inputs and outputs (`llm_logging.py`, `services/log_sink.py`), exposed
  through a logs route in the API.

## Project layout

```text
.
├── frontend/                       # React + Vite chat UI
├── backend/
│   ├── main.py                     # FastAPI app entry point
│   ├── routes/                     # chat + jobs endpoints
│   └── services/
│       ├── conversation.py         # Claude streaming + search-trigger detection
│       ├── recommender.py          # ML adapter (plugs in sara_retrieve_rerank)
│       ├── sara_retrieve_rerank/   # retrieval + reranking pipeline
│       ├── sara_candidate_poll/    # candidate profile extraction
│       └── adzuna/                 # live vacancy scraper
├── scripts/                        # CLI scripts for the ML pipeline
├── tests/                          # pytest suite
├── notebooks/                      # experiment notebooks
├── data/
│   ├── raw/                        # input JSONL (gitignored)
│   ├── processed/                  # generated feature rows (gitignored)
│   └── chroma/                     # persistent dense index
├── outputs/                        # plots, trained models
├── config.yaml                     # path / model / sampling overrides
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -e .[server,rerank,dev]
```

Copy `.env.example` to `.env` and fill in your keys:

```
MISTRAL_API_KEY=...
ADZUNA_APP_ID=
ADZUNA_APP_KEY=
```

## Running

**Backend** (from the repo root):

```bash
cd backend
uvicorn main:app --reload
```

API at `http://localhost:8000`. Interactive docs at `/docs`.

**Frontend** (in a second terminal):

```bash
cd frontend
npm install
npm run dev
```

UI at `http://localhost:5173`. All `/api/*` requests are proxied to the backend.

## Sub-module READMEs

| Component | README |
|-----------|--------|
| FastAPI backend | [backend/README.md](backend/README.md) |
| Retrieval + reranking pipeline | [backend/services/sara_retrieve_rerank/README.md](backend/services/sara_retrieve_rerank/README.md) |
| Frontend UI | [frontend/README.md](frontend/README.md) |
| Adzuna RAG search | [scripts/ADZUNA_RAG_README.md](scripts/ADZUNA_RAG_README.md) |
