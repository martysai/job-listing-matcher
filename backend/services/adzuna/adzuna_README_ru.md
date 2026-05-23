# Adzuna Vacancy Refresh Agent

Субпакет `backend.services.adzuna` — периодический агент, который пополняет базу вакансий Chroma свежими объявлениями из [Adzuna API](https://developer.adzuna.com/docs/). Работает независимо от живого пути обработки запросов кандидата и не влияет на него.

---

## Место в системе

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

Агент не блокирует пользовательский запрос — он запускается по расписанию (дважды в сутки) и пишет новые вакансии в тот же Chroma-индекс, с которым работают ретривер и реранкер.

---

## Архитектура агента

Используется **LangGraph ReAct агент**: LLM с поддержкой tool calling самостоятельно решает, какой инструмент вызвать и в каком порядке, опираясь на системный промпт и текстовые ответы каждого инструмента.

```
ProfileCounter → QueryBuilder → список AdzunaQuery
                                        ↓
                               LangGraph ReAct Agent
                               (LLM + 4 @tool функции)
                                        │
                   ┌────────────────────┼──────────────────────┐
                   ↓                    ↓                       ↓
          scrape_vacancies      extract_fields      index_to_vector_store
          (Adzuna HTTP API)   (LLM SGR экстракция)  (Chroma upsert + BM25)
                   │
          check_scrape_quality
          (диагностика контекста)
```

### Инструменты (`tools.py`)

| Tool | Назначение | Аргументы |
|---|---|---|
| `scrape_vacancies` | HTTP-скрэпинг Adzuna, сохраняет в `RefreshContext` | `broaden: bool = False` |
| `check_scrape_quality` | Статистика по текущему содержимому контекста | — |
| `extract_fields` | LLM SGR-экстракция структурных полей из title + description | — |
| `index_to_vector_store` | Векторизация + Chroma upsert + TTL-чистка + BM25 append | — |

Агент вызывает `scrape_vacancies(broaden=True)`, если первый скрэп вернул < 50 вакансий.

### Стратегия запросов к Adzuna (`counter.py`)

`ProfileCounter` накапливает частоты по полям профилей кандидатов (`preferred_domains`, `desired_positions`, `preferred_locations`) в скользящем окне 7 дней. `QueryBuilder` распределяет дневной бюджет API пропорционально этим частотам (60 % — домены, 40 % — позиции).

При **cold start** (< 10 событий за 7 дней) используется фиксированный набор `FALLBACK_QUERIES`, покрывающий наиболее проблемные категории по данным miss-анализа.

---

## Структура файлов

```
backend/services/adzuna/
├── __init__.py       Публичный API пакета
├── config.py         Константы, env-переменные, AdzunaQuery, таблицы маппинга
├── counter.py        ProfileCounter, QueryBuilder, ADZUNA_CATEGORIES, маппинги
├── scraper.py        HTTP-скрэпинг Adzuna, маппинг raw API → vacancy dict
├── extractor.py      VacancyExtracted (Pydantic SGR), LLM-батч экстракция
├── indexer.py        Векторизация (vacancy_to_text), Chroma upsert, BM25 append
├── tools.py          RefreshContext, @tool функции
└── adzuna_agent.py   build_llm, build_agent, run_scheduled_job
```

---

## Установка

Пакет устанавливается вместе с основным проектом:

```bash
cd /path/to/services
python -m pip install -e .[server,rerank,dev]
```

Дополнительные зависимости для агента:

```bash
pip install langgraph langchain-core langchain-community
pip install langchain-huggingface chromadb litellm aiohttp
```

---

## Переменные окружения

Добавить в `.env` (файл читается Celery и API-слоем автоматически):

```env
# ── Adzuna API ────────────────────────────────────────────────────────────────
ADZUNA_APP_ID=ваш_app_id
ADZUNA_APP_KEY=ваш_app_key
ADZUNA_COUNTRY=gb                        # двухбуквенный код страны

# ── Лимиты Adzuna (free tier: 250 req/day, 25 req/min) ───────────────────────
ADZUNA_DAILY_REQUEST_BUDGET=230          # 250 минус резерв на ошибки
ADZUNA_SCRAPE_DELAY=2.5                  # секунды между запросами (25 req/min)
ADZUNA_VACANCY_TTL_DAYS=14              # срок хранения вакансии в Chroma

# ── LLM для оркестрирующего агента ───────────────────────────────────────────
VACANCY_AGENT_LLM_MODEL=mistral/mistral-large-latest
MISTRAL_API_KEY=ваш_ключ                 # или OPENAI_API_KEY / ANTHROPIC_API_KEY
LITELLM_RETRY_AFTER_WAIT_TIME=2          # базовая пауза между retry на 429 (сек)

# ── LLM для SGR-экстракции полей вакансий ────────────────────────────────────
VACANCY_EXTRACTOR_LLM_MODEL=mistral/mistral-small-latest
VACANCY_EXTRACTOR_LLM_LOG_PATH=data/logs/vacancy_extractor.jsonl

# ── Инфраструктура ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_PATH=data/chroma
ADZUNA_VACANCIES_JSONL=data/raw/adzuna_vacancies.jsonl
ADZUNA_COUNTER_PATH=data/processed/profile_counter.json
```

---

## Запуск

### Разовый запуск (отладка, тест)

```bash
python -c "
from backend.services.adzuna import run_scheduled_job
result = run_scheduled_job()
print(result)
"
```

Или из скрипта:

```python
from backend.services.adzuna import run_scheduled_job

result = run_scheduled_job()
print(result["status"])   # "ok" или "error"
print(result["summary"])  # итоговый текст агента
```

### По расписанию через Celery

**1. Зарегистрировать задачу** — раскомментировать в конце `adzuna_agent.py`:

```python
@celery_app.task(name="vacancy_refresh")
def vacancy_refresh_task() -> dict:
    return run_scheduled_job()
```

**2. Добавить расписание** в `celeryconfig.py`:

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

**3. Запустить воркер и beat:**

```bash
celery -A your_celery_app worker --loglevel=info &
celery -A your_celery_app beat   --loglevel=info
```

### По расписанию через cron

```cron
0  2  * * *  /path/to/.venv/bin/python -c "from backend.services.adzuna import run_scheduled_job; run_scheduled_job()"
0 14  * * *  /path/to/.venv/bin/python -c "from backend.services.adzuna import run_scheduled_job; run_scheduled_job()"
```

---

## Интеграция в живой путь

Единственное изменение в обработчике запроса кандидата — одна строка после парсинга профиля:

```python
from backend.services.adzuna import ProfileCounter

counter = ProfileCounter()   # синглтон — создать один раз при старте

# В обработчике запроса (FastAPI, Flask и т.д.):
async def handle_candidate(candidate_text: str, background_tasks):
    profile = await parse_candidate(candidate_text)       # уже есть
    background_tasks.add_task(counter.record_profile, profile)  # ← новое
    results = await pipeline.match({"text": candidate_text})    # уже есть
    return results
```

`record_profile` выполняется в фоне и не блокирует ответ пользователю.

---

## BM25 и совместимость с ретривером

Вакансии из Adzuna сохраняются в два места:

| Хранилище | Файл | Назначение |
|---|---|---|
| Chroma | `data/chroma/` | Плотный (dense) ретривер |
| JSONL | `data/raw/adzuna_vacancies.jsonl` | BM25 sparse ретривер |

При запуске `RecommendationPipeline` нужно смёржить оба JSONL-источника при построении BM25-индекса:

```python
import json
from pathlib import Path

adzuna   = list(map(json.loads, Path("data/raw/adzuna_vacancies.jsonl").read_text().splitlines()))

# Дедупликация по dataset_id
all_vacancies = {v["dataset_id"]: v for v in original + adzuna}.values()
```

---

## Мониторинг

Каждый LLM-вызов в Tool 1.5 (экстракция полей) логируется в JSONL:

```bash
tail -f data/logs/vacancy_extractor.jsonl | python -m json.tool
```

Каждая строка содержит: `timestamp`, `vacancy_id`, `prompt`, `response`, `latency_ms`, `error` (при ошибке).

Агент логирует через стандартный Python logging (`sara.agent`):

```bash
# Настроить уровень в своём logging config или через переменную:
export PYTHONUNBUFFERED=1
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from backend.services.adzuna import run_scheduled_job
run_scheduled_job()
"
```

---

## Известные ограничения

| Ограничение | Описание |
|---|---|
| Лимиты Adzuna free tier | 250 req/day, 25 req/min. При двух запусках в сутки: ~115 запросов × 50 результатов = ~5 750 вакансий за цикл |
| Неполные поля | Adzuna не предоставляет `grades`, `english_level`, `company_type`. Tool 1.5 извлекает их через LLM, качество зависит от описания вакансии |
| BM25 дедупликация | `adzuna_vacancies.jsonl` накапливается при каждом запуске; дедупликацию по `dataset_id` выполняет код, читающий файл |
| `vacancy_score` | Adzuna-вакансии получают `vacancy_score = 0.0` — нейтральное значение, не влияющее отрицательно на ранжирование |
