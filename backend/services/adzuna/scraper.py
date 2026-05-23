"""Tool 1 — Adzuna HTTP scraper and vacancy dict mapper.

scrape_adzuna_batch
    Async: executes a list of AdzunaQuery objects, deduplicates results by
    vacancy id, and returns raw Adzuna JSON dicts.

adzuna_to_vacancy_dict
    Maps one raw Adzuna dict to the field layout expected by:
      - documents.vacancy_to_text()       (embedding text)
      - documents.vacancy_to_metadata()   (Chroma metadata)
      - schema_features.vacancy_schema_features()  (reranker features)
    Fields unavailable from Adzuna are set to empty; extractor.py fills them.

merge_extracted
    Merges LLM-extracted fields (from extractor.py) back into the partial dict.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from adzuna.config import (
    ADZUNA_APP_ID,
    ADZUNA_APP_KEY,
    ADZUNA_BASE_URL,
    ADZUNA_CONTRACT_TO_WORK_FORMAT,
    ADZUNA_CONTRACT_TO_WORK_TYPE,
    ADZUNA_COUNTRY,
    AdzunaQuery,
    SCRAPE_DELAY_SECONDS,
)
from sara_retrieve_rerank.preprocessing import clean_html

if TYPE_CHECKING:
    from adzuna.extractor import VacancyExtracted


async def scrape_adzuna_batch(
    queries: list[AdzunaQuery],
    country: str = ADZUNA_COUNTRY,
    delay_seconds: float = SCRAPE_DELAY_SECONDS,
    max_vacancies: int | None = None,
    max_vacancies_per_query: int | None = None,
) -> list[dict]:
    """Execute all queries sequentially and return deduplicated raw vacancy dicts.

    Parameters
    ----------
    max_vacancies:
        Stop scraping after this many unique vacancies have been collected
        *in total*.  None (default) means no global cap — run all queries
        to completion.  Set via ADZUNA_MAX_VACANCIES env var.
    max_vacancies_per_query:
        Keep at most this many new (post-dedup) vacancies *per query*.  None
        (default) means no per-query cap.  Used to spread harvest across
        every (location × role) pair instead of letting the first few
        queries fill the global bucket.  Set via ADZUNA_MAX_PER_QUERY env
        var (see config.py).

    Skips queries that return 0 results immediately (saves API quota).
    Continues on HTTP errors and timeouts, logging a warning per failure.
    """
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("pip install aiohttp") from exc

    all_vacancies: list[dict] = []
    seen_ids: set[str] = set()
    log = logging.getLogger("sara.agent")

    n_ok = 0  # queries with ≥1 result
    n_empty = 0  # queries with 0 results
    n_error = 0  # HTTP errors / timeouts

    limit_msg_parts = []
    if max_vacancies:
        limit_msg_parts.append(f"total={max_vacancies}")
    if max_vacancies_per_query:
        limit_msg_parts.append(f"per_query={max_vacancies_per_query}")
    limit_msg = "  limits: " + ", ".join(limit_msg_parts) if limit_msg_parts else "  no limit"
    log.info(
        "Adzuna scrape started — %d queries planned  delay=%.1fs%s",
        len(queries), delay_seconds, limit_msg,
    )

    async with aiohttp.ClientSession() as session:
        for i, query in enumerate(queries):

            # Stop early if vacancy limit reached
            if max_vacancies and len(all_vacancies) >= max_vacancies:
                log.info(
                    "Vacancy limit reached (%d) — stopping scrape after %d/%d queries",
                    max_vacancies, i, len(queries),
                )
                break

            query_country = query.country or country
            url    = ADZUNA_BASE_URL.format(country=query_country, page=1)
            params = {**query.to_params(), "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY}

            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        # Log the actual URL (without api_key) and response body
                        # so the root cause of 404/401/403 can be diagnosed.
                        safe_url = str(resp.url).replace(ADZUNA_APP_KEY, "***")
                        body = await resp.text()
                        log.warning(
                            "Query %d HTTP %d — skip\n  url: %s\n  body: %s",
                            i, resp.status, safe_url, body[:300],
                        )
                        n_error += 1
                        continue

                    results = (await resp.json()).get("results", [])
                    if not results:
                        n_empty += 1
                        continue

                    new = [v for v in results if str(v["id"]) not in seen_ids]

                    # Per-query cap first — keeps geographic / role diversity
                    # by preventing the first few queries from filling the
                    # global bucket.
                    if max_vacancies_per_query:
                        new = new[:max_vacancies_per_query]

                    # Then trim to stay within the global limit.
                    if max_vacancies:
                        remaining = max_vacancies - len(all_vacancies)
                        new = new[:remaining]

                    seen_ids.update(str(v["id"]) for v in new)
                    all_vacancies.extend(new)
                    n_ok += 1
                    log.info(
                        "[%d/%d] +%3d new  total=%d  what=%r  where=%r  category=%r",
                        i + 1, len(queries), len(new), len(all_vacancies),
                        query.what, query.where, query.category,
                    )

            except asyncio.TimeoutError:
                log.warning("Query %d: timeout — skip", i)
                n_error += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Query %d: %s — skip", i, exc)
                n_error += 1

            await asyncio.sleep(delay_seconds)

    log.info(
        "Adzuna scrape finished — vacancies: %d  queries: %d ok / %d empty / %d error",
        len(all_vacancies), n_ok, n_empty, n_error,
    )
    return all_vacancies


def adzuna_to_vacancy_dict(raw: dict) -> dict:
    """Partially map a raw Adzuna API response to the project vacancy schema.

    The returned dict uses the same field names as vacancies_safe_ml_dataset
    so it can be passed directly to vacancy_to_text(), vacancy_to_metadata(),
    and vacancy_schema_features() without further adaptation.

    Fields that Adzuna does not expose (grades, english_level, company_type,
    tags, specializations, company_domains) are set to empty values.
    Call merge_extracted() after extractor.extract_vacancy_fields_batch()
    to fill them in.
    """
    contract = raw.get("contract_type", "")
    loc      = raw.get("location") or {}
    area_raw = loc.get("area") or []
    regions = [a for a in area_raw if isinstance(a, str)] if isinstance(area_raw, list) else []

    return {
        "dataset_id":      str(raw["id"]),
        "source":          "adzuna",
        # ── embedding text fields (vacancy_to_text) ───────────────────────────
        "title":           raw.get("title", ""),
        "original_title":  raw.get("title", ""),   # no separate original in Adzuna
        "work_format":     ADZUNA_CONTRACT_TO_WORK_FORMAT.get(contract, []),
        "work_type":       ADZUNA_CONTRACT_TO_WORK_TYPE.get(contract, ""),
        "english_level":   "",    # → extractor
        "employee_type":   [],    # → extractor
        "company_type":    "",    # → extractor
        "tags":            [],    # → extractor
        "grades":          [],    # → extractor
        "specializations": [],    # → extractor
        "company_domains": [],    # → extractor
        "title_aliases":   [],
        "cities":  [loc["display_name"]] if loc.get("display_name") else [],
        "regions": regions,
        "tldr_sanitized":  clean_html(raw.get("description", "")),
        # ── reranker salary features (schema_features._to_usd) ────────────────
        "salary": {
            "min":           raw.get("salary_min"),
            "max":           raw.get("salary_max"),
            "currency":      None,   # Adzuna does not reliably expose currency
            "salary_in_usd": None,
        },
        # Adzuna has no quality score; 0.0 is the neutral default.
        # There is no has_vacancy_score indicator in schema_features, so zeros
        # do not mislead the reranker.
        "vacancy_score": 0.0,
    }


def merge_extracted(partial: dict, extracted: "VacancyExtracted") -> dict:
    """Merge LLM-extracted structured fields into the partial vacancy dict.

    work_format from the LLM is applied only when the mapper could not
    infer it (empty list), to avoid an LLM hallucination overwriting a
    reliable contract_type signal.
    """
    merged = dict(partial)
    merged["grades"]          = extracted.grades
    merged["english_level"]   = extracted.english_level or ""
    merged["employee_type"]   = extracted.employee_type
    merged["company_type"]    = extracted.company_type or ""
    merged["tags"]            = extracted.tags
    merged["specializations"] = extracted.specializations
    merged["company_domains"] = extracted.company_domains
    if not merged["work_format"] and extracted.work_format:
        merged["work_format"] = extracted.work_format
    return merged
