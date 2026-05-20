"""LangChain tools for the Adzuna vacancy refresh agent.

Each function is decorated with @tool so a LangChain-compatible LLM
(Mistral, GPT-4, etc.) can decide when and how to call it.

Tools communicate with each other through a shared RefreshContext object
rather than through the LLM prompt — vacancy dicts are too large to pass
as text.  The LLM receives only compact string summaries as return values
and uses them to decide the next action.

Tool call order (nominal)
-------------------------
1. scrape_vacancies        — hit Adzuna API, store raw dicts in ctx
2. extract_fields          — LLM-extract structured fields, store in ctx
3. index_to_vector_store   — vectorise + upsert to Chroma + append to BM25

The agent may deviate from this order, e.g.:
  - call scrape_vacancies(broaden=True) if the first scrape returned < 50 results
  - call check_scrape_quality at any point to inspect what is stored in ctx
  - stop early and report an error if an unrecoverable failure occurs
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool

from sara_retrieve_rerank.adzuna.config import EXTRACTOR_MODEL, MAX_VACANCIES_PER_RUN
from sara_retrieve_rerank.adzuna.counter import ProfileCounter, QueryBuilder
from sara_retrieve_rerank.adzuna.extractor import extract_vacancy_fields_batch
from sara_retrieve_rerank.adzuna.indexer import partial_reindex, vectorize_vacancies
from sara_retrieve_rerank.adzuna.scraper import (
    adzuna_to_vacancy_dict,
    merge_extracted,
    scrape_adzuna_batch,
)


# ── Shared context ────────────────────────────────────────────────────────────

@dataclass
class RefreshContext:
    """Mutable state shared across all tools within one refresh cycle.

    Vacancy dicts are stored here rather than passed through the LLM prompt
    because they are too large to fit in a context window.  Tools write to
    this object; the LLM only sees compact string summaries returned by each
    tool call.
    """

    counter:           ProfileCounter
    query_builder:     QueryBuilder
    embedder:          Any
    chroma_collection: Any
    extractor_model:   str = EXTRACTOR_MODEL

    # Populated by scrape_vacancies
    raw_vacancies:       list[dict] = field(default_factory=list)
    # Populated by extract_fields
    processed_vacancies: list[dict] = field(default_factory=list)


# ── Tool factory ──────────────────────────────────────────────────────────────

def build_tools(ctx: RefreshContext) -> list:
    """Return the list of @tool functions bound to *ctx*.

    Pass the returned list to create_react_agent() or AgentExecutor.
    All tools share the same RefreshContext instance so intermediate
    vacancy data persists across tool calls within one agent run.
    """

    log = logging.getLogger("sara.agent")

    @tool
    def scrape_vacancies(broaden: bool = False) -> str:
        """Scrape job vacancies from the Adzuna API and store them in context.

        Builds the query plan from the profile counter (or falls back to
        universal queries if the counter is cold).  When broaden=True the
        location filter is removed from every query so the search covers the
        whole country — use this if a previous scrape returned too few results.

        Returns a summary string with counts and mode.  Call this first.
        """
        queries = ctx.query_builder.build()

        if broaden:
            from sara_retrieve_rerank.adzuna.config import AdzunaQuery
            queries = [
                AdzunaQuery(q.what, "", q.category, q.results_per_page)
                for q in queries
            ]
            log.info("scrape_vacancies: broadened — location filter removed")

        raw = asyncio.run(scrape_adzuna_batch(queries, max_vacancies=MAX_VACANCIES_PER_RUN))
        ctx.raw_vacancies = raw

        mode = "FALLBACK" if ctx.counter.is_cold() else "COUNTER-DRIVEN"
        limit_msg = f"  (limit: {MAX_VACANCIES_PER_RUN})" if MAX_VACANCIES_PER_RUN else ""
        return (
            f"Scraped {len(raw)} unique vacancies {limit_msg}.  "
            f"Mode: {mode}.  Queries executed: {len(queries)}."
        )

    @tool
    def extract_fields() -> str:
        """Extract structured vacancy fields (grades, skills, work_format, etc.)
        from the vacancies stored by scrape_vacancies.

        Calls the LLM extractor on each vacancy's title and description.
        Merges extracted fields with the partial Adzuna data to produce
        fully populated vacancy dicts.  Results are stored in context.

        Returns a summary with success / failure counts.
        Requires scrape_vacancies to have been called first.
        """
        if not ctx.raw_vacancies:
            return "ERROR: no scraped vacancies in context. Call scrape_vacancies first."

        partials  = [adzuna_to_vacancy_dict(v) for v in ctx.raw_vacancies]
        extracted = extract_vacancy_fields_batch(partials, model=ctx.extractor_model)
        ctx.processed_vacancies = [
            merge_extracted(p, e) for p, e in zip(partials, extracted)
        ]

        n_ok = sum(
            1 for v in ctx.processed_vacancies
            if v.get("grades") or v.get("tags")
        )
        return (
            f"Extracted fields for {len(ctx.processed_vacancies)} vacancies.  "
            f"{n_ok} have grades or tags populated."
        )

    @tool
    def index_to_vector_store() -> str:
        """Vectorise processed vacancies and upsert them into Chroma.

        Also appends raw dicts to the BM25 JSONL file and removes expired
        Adzuna entries from the vector store (TTL-based cleanup).

        Returns indexing stats: upserted, expired_removed, bm25_appended.
        Requires extract_fields to have been called first.
        """
        if not ctx.processed_vacancies:
            return "ERROR: no processed vacancies in context. Call extract_fields first."

        vectorized = vectorize_vacancies(ctx.processed_vacancies, ctx.embedder)
        stats      = partial_reindex(vectorized, ctx.chroma_collection)

        return (
            f"Indexed {stats['upserted']} vacancies.  "
            f"Expired removed: {stats['expired_removed']}.  "
            f"BM25 appended: {stats['bm25_appended']}.  "
            f"Errors: {stats['errors']}."
        )

    @tool
    def check_scrape_quality() -> str:
        """Report on the quantity and coverage of vacancies currently in context.

        Use this after scrape_vacancies to decide whether the result is
        sufficient before proceeding to extract_fields, or to diagnose
        why a previous step produced unexpected results.

        Returns counts and a sample of categories present.
        """
        raw  = ctx.raw_vacancies
        proc = ctx.processed_vacancies

        if not raw:
            return "Context is empty — scrape_vacancies has not been called yet."

        categories: dict[str, int] = {}
        for v in raw:
            cat = (v.get("category") or {}).get("label", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        top_cats = sorted(categories.items(), key=lambda x: -x[1])[:5]
        top_str  = ", ".join(f"{c}({n})" for c, n in top_cats)

        return (
            f"Raw vacancies in context: {len(raw)}.  "
            f"Processed (post-extraction): {len(proc)}.  "
            f"Top categories: {top_str}."
        )

    return [scrape_vacancies, extract_fields, index_to_vector_store, check_scrape_quality]
