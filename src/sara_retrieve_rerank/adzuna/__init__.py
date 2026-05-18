"""
sara_retrieve_rerank.adzuna
============================
Scheduled agent that keeps the Chroma vacancy index fresh by scraping
Adzuna API on a cron-like schedule.

Public surface
--------------
    from sara_retrieve_rerank.adzuna import VacancyRefreshAgent, ProfileCounter
    from sara_retrieve_rerank.adzuna.agent import run_scheduled_job

File layout
-----------
    config.py    — constants, env vars, AdzunaQuery dataclass
    counter.py   — ProfileCounter, QueryBuilder, query mapping tables
    scraper.py   — Tool 1: HTTP scraping + Adzuna → vacancy dict mapper
    extractor.py — Tool 1.5: LLM structured field extraction
    indexer.py   — Tool 2: vectorise  |  Tool 3: Chroma upsert + BM25 append
    agent.py     — VacancyRefreshAgent orchestrator + scheduler entry point
"""

from sara_retrieve_rerank.adzuna.adzuna_agent import run_scheduled_job
from sara_retrieve_rerank.adzuna.counter import ProfileCounter, QueryBuilder
from sara_retrieve_rerank.adzuna.config import AdzunaQuery

__all__ = [
    "run_scheduled_job",
    "ProfileCounter",
    "QueryBuilder",
    "AdzunaQuery",
]
