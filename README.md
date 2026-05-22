# Job Listing Matcher

AI training course assignment. A chat-based job matching application: a React frontend collects job preferences from the user, a FastAPI backend streams responses and drives a retrieval + reranking ML pipeline, and the sara_retrieve_rerank service surfaces the best-matching vacancies.

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
ANTHROPIC_API_KEY=sk-ant-...
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
