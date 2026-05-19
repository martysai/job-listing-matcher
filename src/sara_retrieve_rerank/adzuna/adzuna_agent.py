"""Vacancy refresh agent — LangGraph ReAct agent.

Назначение
----------
Периодически обновляет базу вакансий в Chroma, скрэпя свежие объявления
из Adzuna API.  Запускается по расписанию (Celery beat / cron / Airflow)
один-два раза в сутки, не затрагивая живой путь обработки запросов
кандидата.

Архитектура
-----------
Вместо детерминированного пайплайна используется LangGraph ReAct агент:
LLM с поддержкой tool calling самостоятельно решает, какой инструмент
вызвать и в каком порядке, опираясь на системный промпт и текстовые
ответы каждого инструмента.  Это позволяет агенту адаптироваться к
нестандартным ситуациям — например, расширить географию поиска, если
первый скрэпинг вернул мало результатов.

Основные компоненты
-------------------
build_llm(model_string)
    Фабрика LangChain-модели через ChatLiteLLM.  Принимает строку формата
    litellm (``"mistral/mistral-large-latest"``, ``"gpt-4o"`` и т.д.) —
    тот же формат, что уже используется в RERANK_LLM_MODEL.
    Поддерживает любой провайдер без отдельных provider-пакетов.

build_agent(ctx, llm)
    Собирает LangGraph ReAct агент:
      tools = build_tools(ctx)          # @tool-функции из tools.py
      agent = create_react_agent(llm, tools)
    Системный промпт передаётся первым сообщением в agent.invoke() —
    совместимо со всеми версиями LangGraph.
    Агент выполняет цикл: LLM выбирает tool → tool возвращает текстовый
    результат → LLM принимает следующее решение.

run_scheduled_job()
    Точка входа для планировщика.  Инициализирует инфраструктуру
    (HuggingFace embedder, Chroma, ProfileCounter), запускает агент
    и возвращает dict {"status": "ok"|"error", "summary"|"error": ...}.

Инструменты (определены в tools.py)
------------------------------------
scrape_vacancies        Скрэпит Adzuna, сохраняет вакансии в RefreshContext.
check_scrape_quality    Возвращает статистику по текущему содержимому контекста.
extract_fields          LLM-экстракция структурных полей (grades, tags и др.).
index_to_vector_store   Векторизация + Chroma upsert + TTL-чистка + BM25 append.

LLM-провайдеры
--------------
Задать через VACANCY_AGENT_LLM_MODEL + соответствующий API-ключ.
Используется ChatLiteLLM — API-ключи читаются так же, как в остальном
проекте (MISTRAL_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY).

    mistral/mistral-large-latest    MISTRAL_API_KEY
    gpt-4o / gpt-4o-mini            OPENAI_API_KEY
    claude-3-5-haiku-20241022       ANTHROPIC_API_KEY

Зависимости
-----------
    pip install langgraph langchain-core langchain-litellm
    pip install langchain-huggingface chromadb litellm aiohttp
"""

from __future__ import annotations

import logging
import time
from typing import Any

# ── LangChain / LangGraph ─────────────────────────────────────────────────────
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ── Проект ────────────────────────────────────────────────────────────────────
from sara_retrieve_rerank.adzuna.config import (
    AGENT_MODEL,
    AGENT_LLM_RPS,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    EXTRACTOR_MODEL,
)
from sara_retrieve_rerank.adzuna.counter import ProfileCounter, QueryBuilder
from sara_retrieve_rerank.adzuna.tools import RefreshContext, build_tools

import os  # noqa: E402 — after project imports for grouping clarity

log = logging.getLogger("sara.agent")


# ══════════════════════════════════════════════════════════════════════════════
# Rate-limit callback
# ══════════════════════════════════════════════════════════════════════════════

class _MinDelayCallback(BaseCallbackHandler):
    """Enforces a minimum wall-clock delay between consecutive LLM calls.

    Works with any LangChain model regardless of whether it supports the
    ``rate_limiter`` constructor parameter.  Hooks into ``on_chat_model_start``
    which fires before every model invocation inside the LangGraph ReAct loop.
    """

    def __init__(self, min_delay_seconds: float) -> None:
        self._delay = min_delay_seconds
        self._last: float = 0.0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:  # type: ignore[override]
        elapsed = time.monotonic() - self._last
        wait = self._delay - elapsed
        if wait > 0:
            log.debug("Rate limiter: waiting %.2f s before LLM call", wait)
            time.sleep(wait)

    def on_llm_end(self, response, **kwargs) -> None:  # type: ignore[override]
        self._last = time.monotonic()

    def on_llm_error(self, error, **kwargs) -> None:  # type: ignore[override]
        # Reset timer on error so next call isn't held back unnecessarily.
        self._last = time.monotonic()


# ══════════════════════════════════════════════════════════════════════════════
# Системный промпт
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = SystemMessage(content="""
You are a scheduled vacancy database refresh agent.
Your job is to populate the job vacancy vector store with fresh vacancies
from the Adzuna API by calling the available tools.

Available tools
---------------
scrape_vacancies(broaden: bool = False)
    Hit the Adzuna API and store raw results in shared context.
    Pass broaden=True to remove the location filter if the first scrape
    returned fewer than 50 vacancies.

check_scrape_quality()
    Inspect the vacancies currently stored in context.
    Use after scrape_vacancies to decide whether to proceed or broaden.

extract_fields()
    Run LLM-based structured extraction on the scraped vacancies.
    Must be called after scrape_vacancies.

index_to_vector_store()
    Vectorise processed vacancies and upsert them into Chroma.
    Also removes expired entries and appends to the BM25 JSONL file.
    Must be called after extract_fields.

Required workflow
-----------------
1. Call scrape_vacancies().
2. Call check_scrape_quality().
   - If fewer than 50 vacancies: call scrape_vacancies(broaden=True).
   - If still fewer than 10: stop and report.
3. Call extract_fields().
4. Call index_to_vector_store().
5. Report a concise summary: scraped / extracted / indexed counts.

Rules
-----
- Do not call extract_fields or index_to_vector_store before scrape_vacancies.
- If a tool returns a message starting with ERROR, stop and report it.
- Do not call the same tool more than twice per run.
""".strip())


# ══════════════════════════════════════════════════════════════════════════════
# LLM factory
# ══════════════════════════════════════════════════════════════════════════════

def build_llm(model_string: str) -> Any:
    """Вернуть LangChain chat-модель через ChatLiteLLM.

    Rate limiting реализован через _MinDelayCallback, а не через параметр
    rate_limiter конструктора — не все провайдеры в langchain_litellm
    поддерживают этот параметр.  Callback работает с любой моделью.

    Примеры model_string
    --------------------
    mistral/mistral-large-latest
    gpt-4o
    claude-3-5-haiku-20241022

    API-ключи читаются из окружения автоматически через litellm
    (MISTRAL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).
    """
    from langchain_litellm import ChatLiteLLM  # pip install langchain-litellm

    return ChatLiteLLM(
        model       = model_string,
        temperature = 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Agent builder
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(ctx: RefreshContext, llm: Any) -> Any:
    """Собрать LangGraph ReAct агент.

    Параметры
    ----------
    ctx : RefreshContext
        Разделяемое состояние одного запуска (вакансии, эмбеддер, Chroma).
    llm : BaseChatModel
        Модель с поддержкой tool calling. Получить через build_llm().

    Возвращает
    ----------
    Скомпилированный граф LangGraph (CompiledGraph).
    Вызывать: agent.invoke({"messages": [...]}).

    Как это работает
    ----------------
    tools     = список @tool функций из tools.py
    llm       = модель, которая умеет вызывать tools
    agent     = create_react_agent(llm, tools, ...)
                └── LangGraph ReAct цикл:
                    LLM → выбирает tool → tool выполняется → LLM видит результат
                    → повторяет, пока не решит завершить
    """
    tools = build_tools(ctx)

    log.info(
        "Создаём агент: модель=%s  инструменты=%s",
        type(llm).__name__,
        [t.name for t in tools],
    )

    # Системный промпт НЕ передаём в create_react_agent, чтобы избежать
    # несовместимости между версиями LangGraph:
    #   < 0.2.27 — параметра нет совсем
    #   0.2.x    — state_modifier
    #   0.3+     — prompt
    # Вместо этого SYSTEM_PROMPT передаётся первым сообщением в agent.invoke().
    agent = create_react_agent(
        model = llm,
        tools = tools,
    )
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_scheduled_job() -> dict:
    """Запустить один цикл обновления базы вакансий.

    Вызывается Celery beat, cron или Airflow.
    Все настройки берутся из переменных окружения.

    Переменные окружения
    --------------------
    VACANCY_AGENT_LLM_MODEL   строка модели (default: mistral/mistral-large-latest)
    MISTRAL_API_KEY           ключ Mistral (или OPENAI_API_KEY / ANTHROPIC_API_KEY)
    VACANCY_AGENT_LLM_RPS     темп LLM-вызовов агента, req/s (default: 0.3 ≈ 1/3сек)
    EMBEDDING_MODEL_NAME      HuggingFace-модель для эмбеддингов вакансий
    CHROMA_DB_PATH            путь к Chroma (default: data/chroma)
    + все переменные из config.py (ADZUNA_APP_ID, ADZUNA_APP_KEY, ...)

    Возвращает
    ----------
    {"status": "ok",    "summary": "<текст агента>"}
    {"status": "error", "error":   "<сообщение об ошибке>"}

    Celery
    ------
    Раскомментировать декоратор ниже и добавить расписание в celeryconfig.py:

        beat_schedule = {
            "vacancy-refresh-morning":
                {"task": "vacancy_refresh", "schedule": crontab(hour=2,  minute=0)},
            "vacancy-refresh-afternoon":
                {"task": "vacancy_refresh", "schedule": crontab(hour=14, minute=0)},
        }
    """
    import chromadb
    from langchain_huggingface import HuggingFaceEmbeddings

    # ── Валидация Adzuna credentials ──────────────────────────────────────────
    # Пустые credentials дают HTTP 404 на каждый запрос к API.
    from sara_retrieve_rerank.adzuna.config import ADZUNA_APP_ID, ADZUNA_APP_KEY
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return {
            "status": "error",
            "error": (
                "ADZUNA_APP_ID и ADZUNA_APP_KEY не заданы. "
                "Зарегистрируйтесь на https://developer.adzuna.com/ "
                "и добавьте ключи в .env."
            ),
        }

    # ── Инфраструктура ────────────────────────────────────────────────────────
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    chroma_collection = (
        chromadb.PersistentClient(path=CHROMA_DB_PATH)
        .get_or_create_collection(CHROMA_COLLECTION_NAME)
    )

    # ── Контекст одного запуска ───────────────────────────────────────────────
    counter = ProfileCounter()
    ctx = RefreshContext(
        counter           = counter,
        query_builder     = QueryBuilder(counter),
        embedder          = embedder,
        chroma_collection = chroma_collection,
        extractor_model   = os.environ.get("VACANCY_EXTRACTOR_LLM_MODEL", EXTRACTOR_MODEL),
    )

    # ── LLM + rate-limit callback ─────────────────────────────────────────────
    model_string = os.environ.get("VACANCY_AGENT_LLM_MODEL", AGENT_MODEL)
    min_delay    = 1.0 / float(os.environ.get("VACANCY_AGENT_LLM_RPS", str(AGENT_LLM_RPS)))
    log.info("LLM: %s  delay between calls: %.1f s", model_string, min_delay)

    try:
        llm = build_llm(model_string)
    except (ImportError, ValueError) as exc:
        log.error("Не удалось инициализировать LLM: %s", exc)
        return {"status": "error", "error": str(exc)}

    rate_cb = _MinDelayCallback(min_delay_seconds=min_delay)

    # ── Агент ─────────────────────────────────────────────────────────────────
    agent = build_agent(ctx, llm)

    log.info("Запуск цикла обновления вакансий.")
    try:
        result = agent.invoke(
            input={
                "messages": [
                    # Системный промпт передаётся первым сообщением —
                    # совместимо со всеми версиями LangGraph.
                    SYSTEM_PROMPT,
                    ("user", "Run the vacancy refresh cycle now."),
                ]
            },
            config={
                "recursion_limit": 20,
                # _MinDelayCallback перехватывает on_chat_model_start
                # и добавляет паузу перед каждым LLM-вызовом.
                "callbacks": [rate_cb],
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Агент завершился с ошибкой: %s", exc)
        return {"status": "error", "error": str(exc)}

    # Последнее сообщение от LLM — итоговое резюме того, что было сделано.
    final_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    summary = final_messages[-1].content if final_messages else "(no summary)"

    log.info("Цикл завершён: %s", summary)
    return {"status": "ok", "summary": summary}


# @celery_app.task(name="vacancy_refresh")
# def vacancy_refresh_task() -> dict:
#     return run_scheduled_job()
