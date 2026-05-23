# Adzuna Vacancy Refresh Agent

The `backend.services.adzuna` subpackage is a scheduled agent that keeps the Chroma vacancy index up to date by scraping fresh job listings from the [Adzuna API](https://developer.adzuna.com/docs/). It runs independently of the live candidate request path and has no impact on it.

---

## Role in the system

```
Candidate input
      ↓
 LLM Parser  →  Structured profile
                      ↓
                 Retriever  ←  Chroma (vacancies)  ←─────────────────┐
                      ↓                                               │
                 Reranker                              Adzuna Agent   │
                      ↓                               (scheduled)    │
                 Top-N results → UI                        │          │
                                                     scrape → extract │
                                                     → vectorise      │
                                                     → upsert ────────┘
```

The agent does not block user requests — it runs on a schedule (twice a day) and writes new vacancies into the same Chroma index used by the retriever and reranker.

---

## Agent architecture

The agent is implemented as a **LangGraph ReAct agent**: a tool-calling LLM decides which tool to invoke and in what order, guided by a system prompt and the text responses returned by each tool.

```
ProfileCounter → QueryBuilder → list of AdzunaQuery
                                        ↓
                               LangGraph ReAct Agent
                               (LLM + 4 @tool functions)
                                        │
                   ┌────────────────────┼──────────────────────┐
                   ↓                    ↓                       ↓
          scrape_vacancies      extract_fields      index_to_vector_store
          (Adzuna HTTP API)   (LLM SGR extraction)  (Chroma upsert + BM25)
                   │
          check_scrape_quality
          (context diagnostics)
```

### Tools (`tools.py`)

| Tool | Purpose | Arguments |
|---|---|---|
| `scrape_vacancies` | HTTP scraping of Adzuna; stores results in `RefreshContext` | `broaden: bool = False` |
| `check_scrape_quality` | Reports statistics on what is currently stored in context | — |
| `extract_fields` | LLM SGR extraction of structured fields from title + description | — |
| `index_to_vector_store` | Vectorisation + Chroma upsert + TTL cleanup + BM25 append | — |

The agent calls `scrape_vacancies(broaden=True)` if the first scrape returns fewer than 50 vacancies.

### Adzuna query strategy (`counter.py`)

`ProfileCounter` accumulates frequency counts over candidate profile fields (`preferred_domains`, `desired_positions`, `preferred_locations`) in a 7-day sliding window. `QueryBuilder` allocates the daily API budget proportionally to these counts (60 % on domains, 40 % on positions).

During **cold start** (fewer than 10 events in the last 7 days) a fixed set of `FALLBACK_QUERIES` is used, prioritising the categories identified as hard miss-buckets in the retriever miss analysis.

---

## File structure

```
backend/services/adzuna/
├── __init__.py       Package public API
├── config.py         Constants, env vars, AdzunaQuery dataclass, mapping tables
├── counter.py        ProfileCounter, QueryBuilder, ADZUNA_CATEGORIES, mappings
├── scraper.py        Adzuna HTTP scraping, raw API → vacancy dict mapper
├── extractor.py      VacancyExtracted (Pydantic SGR), LLM batch extraction
├── indexer.py        Vectorisation (vacancy_to_text), Chroma upsert, BM25 append
├── tools.py          RefreshContext, @tool functions
└── adzuna_agent.py   build_llm, build_agent, run_scheduled_job
```

---

## Installation

The package is installed together with the main project:

```bash
cd /path/to/adzuna
python -m pip install -e .[server,rerank,dev]
```

Additional dependencies for the agent:

```bash
pip install langgraph langchain-core langchain-community
pip install langchain-huggingface chromadb litellm aiohttp
```

---

## Environment variables

Add to `.env` (loaded automatically by Celery and the API layer):

```env
# ── Adzuna API ────────────────────────────────────────────────────────────────
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
ADZUNA_COUNTRY=gb                        # two-letter country code

# ── Adzuna rate limits (free tier: 250 req/day, 25 req/min) ──────────────────
ADZUNA_DAILY_REQUEST_BUDGET=230          # 250 minus a small retry reserve
ADZUNA_SCRAPE_DELAY=2.5                  # seconds between requests (25 req/min)
ADZUNA_VACANCY_TTL_DAYS=14              # days before a vacancy expires in Chroma

# ── LLM for the orchestrating agent ──────────────────────────────────────────
VACANCY_AGENT_LLM_MODEL=mistral/mistral-large-latest
MISTRAL_API_KEY=your_key                 # or OPENAI_API_KEY / ANTHROPIC_API_KEY
LITELLM_RETRY_AFTER_WAIT_TIME=2          # base pause between retries on 429 (sec)

# ── LLM for SGR vacancy field extraction ─────────────────────────────────────
VACANCY_EXTRACTOR_LLM_MODEL=mistral/mistral-small-latest
VACANCY_EXTRACTOR_LLM_LOG_PATH=data/logs/vacancy_extractor.jsonl

# ── Infrastructure ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_PATH=data/chroma
ADZUNA_VACANCIES_JSONL=data/raw/adzuna_vacancies.jsonl
ADZUNA_COUNTER_PATH=data/processed/profile_counter.json
```

---

## Running the agent

### One-off run (debugging, testing)

```bash
python -c "
from backend.services.adzuna import run_scheduled_job
result = run_scheduled_job()
print(result)
"
```

Or from a script:

```python
from backend.services.adzuna import run_scheduled_job

result = run_scheduled_job()
print(result["status"])   # "ok" or "error"
print(result["summary"])  # final text produced by the agent
```

### Scheduled via Celery

**1. Register the task** — uncomment at the bottom of `adzuna_agent.py`:

```python
@celery_app.task(name="vacancy_refresh")
def vacancy_refresh_task() -> dict:
    return run_scheduled_job()
```

**2. Add the schedule** to `celeryconfig.py`:

```python
from celery.schedules import crontab

beat_schedule = {
    "vacancy-refresh-morning": {
        "task":     "vacancy_refresh",
        "schedule": crontab(hour=2, minute=0),    # 02:00 UTC
    },
    "vacancy-refresh-afternoon": {
        "task":     "vacancy_refresh",
        "schedule": crontab(hour=14, minute=0),   # 14:00 UTC
    },
}
```

**3. Start the worker and beat:**

```bash
celery -A your_celery_app worker --loglevel=info &
celery -A your_celery_app beat   --loglevel=info
```

### Scheduled via cron

```cron
0  2  * * *  /path/to/.venv/bin/python -c "from backend.services.adzuna import run_scheduled_job; run_scheduled_job()"
0 14  * * *  /path/to/.venv/bin/python -c "from backend.services.adzuna import run_scheduled_job; run_scheduled_job()"
```

---

## Integration into the live request path

The only change required in the candidate submission handler is a single line added after parsing the structured profile:

```python
from backend.services.adzuna import ProfileCounter

counter = ProfileCounter()   # singleton — create once at application startup

# In the request handler (FastAPI, Flask, etc.):
async def handle_candidate(candidate_text: str, background_tasks):
    profile = await parse_candidate(candidate_text)                    # existing
    background_tasks.add_task(counter.record_profile, profile)         # ← new
    results = await pipeline.match({"text": candidate_text})           # existing
    return results
```

`record_profile` runs in the background and does not add latency to the user response.

---

## BM25 compatibility

Adzuna vacancies are written to two locations:

| Store | Path | Purpose |
|---|---|---|
| Chroma | `data/chroma/` | Dense retriever |
| JSONL | `data/raw/adzuna_vacancies.jsonl` | BM25 sparse retriever |

When starting `RecommendationPipeline`, merge both JSONL sources when building the in-memory BM25 index:

```python
import json
from pathlib import Path

original = list(map(json.loads, Path("data/raw/vacancies_safe_ml_dataset_nozip.jsonl").read_text().splitlines()))
adzuna   = list(map(json.loads, Path("data/raw/adzuna_vacancies.jsonl").read_text().splitlines()))

# Deduplicate by dataset_id; Adzuna entries override originals on collision
all_vacancies = {v["dataset_id"]: v for v in original + adzuna}.values()
```

---

## Monitoring

Every LLM call made by the field extractor (Tool 1.5) is written to a JSONL audit log:

```bash
tail -f data/logs/vacancy_extractor.jsonl | python -m json.tool
```

Each line contains: `timestamp`, `vacancy_id`, `prompt`, `response`, `latency_ms`, and `error` (when present).

The agent logs via the standard Python `logging` module under the `sara.agent` logger:

```bash
export PYTHONUNBUFFERED=1
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from backend.services.adzuna import run_scheduled_job
run_scheduled_job()
"
```

---

## Known limitations

| Limitation | Details |
|---|---|
| Adzuna free tier limits | 250 req/day, 25 req/min. With two runs per day: ~115 requests × 50 results ≈ 5,750 vacancies per cycle |
| Incomplete fields | Adzuna does not provide `grades`, `english_level`, or `company_type`. Tool 1.5 extracts them via LLM; quality depends on the vacancy description |
| BM25 deduplication | `adzuna_vacancies.jsonl` grows with each run; deduplication by `dataset_id` is the responsibility of the code that reads the file |
| `vacancy_score` | Adzuna vacancies receive `vacancy_score = 0.0` — a neutral default that does not negatively affect reranker scoring |
