# Backend — Job Recommendation API

FastAPI service that powers the job-matching chat interface. It streams conversation responses via Server-Sent Events using the (stub: Anthropic Claude API), and exposes a job recommendation endpoint backed by a pluggable ML retriever/reranker.

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e .[server,rerank]
```

Create a `.env` file (or copy the existing one) and fill in your key:

```
ANTHROPIC_API_KEY=sk-ant-...
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

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat/stream` | Stream chat response as SSE |
| `POST` | `/api/jobs/recommend` | Return ranked job listings |

### `POST /api/chat/stream`

Accepts a conversation history and streams SSE events:

```json
{ "messages": [{"role": "user", "content": "..."}], "session_id": "..." }
```

Event types emitted:

| Type | Payload | Meaning |
|------|---------|---------|
| `text` | `{"content": "..."}` | Streamed text token |
| `ready_to_search` | `{"profile": {...}}` | Bot has collected enough info; trigger job search |
| `done` | — | Stream complete |

### `POST /api/jobs/recommend`

```json
{
  "profile": {
    "title": "Senior Python Engineer",
    "skills": ["Python", "FastAPI"],
    "location": "Remote",
    "experience_years": 5,
    "job_type": "full-time",
    "salary_min": 90000
  },
  "top_k": 10
}
```

Returns a list of `Job` objects with `id`, `title`, `company`, `location`, `salary`, `tags`, `summary`, `url`, and `match_score`.

## Connecting your ML pipeline

`services/recommender.py` contains a stub implementation that returns placeholder jobs. Replace it with your real retriever and reranker:

```python
# services/recommender.py
from your_ml_module import VectorStore, Retriever, Reranker

class RecommenderService:
    def __init__(self):
        self.store     = VectorStore(path="...")
        self.retriever = Retriever(self.store)
        self.reranker  = Reranker(model="...")

    async def search(self, profile: dict, top_k: int = 10) -> list[dict]:
        query      = _build_query(profile)
        candidates = await asyncio.to_thread(self.retriever.retrieve, query, k=top_k * 3)
        ranked     = await asyncio.to_thread(self.reranker.rerank, query, candidates, k=top_k)
        return [_format_job(j) for j in ranked]
```

The `_build_query` and `_format_job` helpers in that file are ready to use as-is.

## Project Structure

```
backend/
├── main.py                  # FastAPI app, CORS, router registration
├── .env                     # ANTHROPIC_API_KEY (not committed)
├── routes/
│   ├── chat.py              # POST /api/chat/stream
│   └── jobs.py              # POST /api/jobs/recommend
└── services/
    ├── conversation.py      # Claude streaming + <SEARCH_READY> detection
    ├── recommender.py       # ML adapter (stub — plug in your code here)
    ├── sara_retrieve_rerank/  # retrieval + reranking pipeline
    ├── sara_candidate_poll/   # candidate profile extraction
    └── adzuna/               # vacancy scraper + refresh agent
```
