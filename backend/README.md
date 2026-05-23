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
| `GET` | `/health` | Health check |
| `POST` | `/api/auth/login` | Log in with an HTTP Basic header; sets the session cookie |
| `POST` | `/api/auth/logout` | Clear the session cookie |
| `GET` | `/api/auth/me` | Verify the current session |
| `POST` | `/api/chat/stream` | Stream the chat reply as SSE; runs the job search inline |
| `GET` | `/api/chat/history/{session_id}` | Full message history for a session |
| `POST` | `/api/chat/reset/{session_id}` | Reset a session |
| `GET` | `/api/logs` | Query the JSONL audit log of LLM calls |
| `POST` | `/api/admin/vacancy-refresh` | Manually trigger one Adzuna refresh cycle |

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

## Vacancy refresh loop

On startup, the lifespan schedules `VacancyRefreshService` (in
`services/vacancy_refresh.py`), a background asyncio task that periodically
runs the LangGraph ReAct Adzuna agent (`services/adzuna/adzuna_agent.py`)
and hot-merges any new vacancies into the live Chroma + BM25 index without
restarting the API.

Each cycle is fully isolated: every layer (env validation, Adzuna HTTP,
LLM ReAct loop, Chroma upsert, BM25 rebuild) is wrapped in try/except, so
a refresh failure preserves the previous in-memory state and the API keeps
serving from the existing index.

The default plan covers three locations × three roles (London / Berlin /
Amsterdam × Data Scientist / ML Engineer / Software Engineer) every six
hours. Override via env (see `.env.example`):

```
ADZUNA_APP_ID, ADZUNA_APP_KEY
MISTRAL_API_KEY  (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
VACANCY_REFRESH_ENABLED          (true / false)
VACANCY_REFRESH_LOCATIONS        country:city,country:city,...
VACANCY_REFRESH_ROLES            role,role,role
VACANCY_REFRESH_INTERVAL_SECONDS
VACANCY_REFRESH_INITIAL_DELAY_SECONDS
ADZUNA_MAX_VACANCIES             (cap per cycle; throttle)
VACANCY_EXTRACTOR_DELAY          (seconds between extractor LLM calls)
```

If `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` or an LLM key is missing, the loop
auto-disables and logs the reason — the API still boots and serves
recommendations from the existing index.

To run a cycle on demand: `POST /api/admin/vacancy-refresh` (auth required).

To run refresh **independently of the API** (so cron survives API restarts),
use the standalone daemon and disable the in-API loop:

```
# Terminal 1 — daemon
python scripts/run_refresh_daemon.py

# Terminal 2 — API server, reader-only
VACANCY_REFRESH_ENABLED=false python scripts/run_backend.py --port 8001
```



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
