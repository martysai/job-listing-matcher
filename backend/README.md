# Backend — Job Recommendation API

FastAPI service that powers the job-matching chat interface. It streams conversation responses via Server-Sent Events from an LLM, runs profile extraction, and drives the `sara_retrieve_rerank` retrieval + reranking pipeline — returning ranked vacancies inline in the chat stream.

## Requirements

- Python 3.11+
- An LLM API token (read from `MISTRAL_API_KEY`)

## Setup

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e .[server,rerank]
```

Create a `.env` file (or copy the existing one) and fill in your LLM API token plus the vacancy corpus the recommender loads at startup:

```
MISTRAL_API_KEY=...                                       # LLM API token (chat + extraction)
VACANCIES_PATH=.jsonl
CHROMA_DIR=data/chroma                                    # optional, defaults to data/chroma
LAMBDARANK_MODEL_PATH=.pkl                               # optional; absent = retrieval only, no reranking
```

## Running

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Interactive docs at `/docs`.

## API Endpoints

Every `/api/*` route except `auth/login` and `auth/logout` requires the session cookie set at login.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/auth/login` | Log in with an HTTP Basic header; sets the session cookie |
| `POST` | `/api/auth/logout` | Clear the session cookie |
| `GET` | `/api/auth/me` | Verify the current session |
| `POST` | `/api/chat/stream` | Stream the chat reply as SSE; runs the job search inline |
| `GET` | `/api/chat/history/{session_id}` | Full message history for a session |
| `POST` | `/api/chat/reset/{session_id}` | Reset a session |
| `GET` | `/api/logs` | Query the JSONL audit log of LLM calls |

### `POST /api/chat/stream`

Accepts the conversation history and streams SSE events:

```json
{ "messages": [{"role": "user", "content": "..."}], "session_id": "..." }
```

The chat model collects preferences turn by turn. Once it has enough, it emits a `searching` event, extracts a structured profile, retrieves + re-ranks vacancies, and streams them back as a `jobs` event — all within the same request.

| Type | Payload | Meaning |
|------|---------|---------|
| `text` | `{"content": "..."}` | Streamed text chunk |
| `searching` | — | Enough info collected; profile extraction + retrieval started |
| `jobs` | `{"jobs": [Job, ...]}` | Ranked vacancies for the collected profile |
| `done` | — | Stream complete |

Each `Job` object has `id`, `title`, `company`, `location`, `salary`, `tags`, `summary`, `url`, and `match_score`.

### `GET /api/logs`

Returns recent audit-log rows (one per LLM call / pipeline event). Optional query params: `session_id`, `event`, `component`, `limit` (≤ 500), `offset`.

## Project Structure

```
backend/
├── main.py                  # FastAPI app, CORS + auth middleware, router registration
├── .env                     # secrets + corpus paths (not committed)
├── routes/
│   ├── auth.py              # login / logout / me (cookie session)
│   ├── chat.py              # POST /api/chat/stream
│   ├── history.py           # chat history + reset
│   └── logs.py              # GET /api/logs (audit-log query)
└── services/
    ├── conversation.py      # LLM streaming + <SEARCH> detection; runs the recommender inline
    ├── recommender.py       # adapter onto the sara_retrieve_rerank pipeline
    ├── database.py          # SQLite store (sessions + messages)
    ├── log_sink.py          # JSONL audit-log writer + query helper
    ├── sara_retrieve_rerank/  # retrieval + reranking pipeline
    ├── sara_candidate_poll/   # candidate profile extraction (free text → schema)
    └── adzuna/                # live vacancy scraper + LangGraph refresh agent
```
