"""Tiny Adzuna API client for the simple RAG-style vacancy search script.

The Adzuna free tier is rate-limited per month, so this module is intentionally
conservative:

- one HTTP call per (country, query, page) tuple,
- on-disk JSON cache so repeated runs do not burn quota,
- caller decides how many pages and countries to query.

Docs: https://developer.adzuna.com/docs/search
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests


ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
SUPPORTED_COUNTRIES = {
    "gb", "us", "at", "au", "be", "br", "ca", "ch", "de", "es",
    "fr", "in", "it", "mx", "nl", "nz", "pl", "sg", "za",
}


@dataclass(frozen=True)
class AdzunaJob:
    """A single Adzuna vacancy with the fields we actually use downstream."""

    id: str
    country: str
    title: str
    company: str
    location: str
    description: str
    salary_min: float | None
    salary_max: float | None
    contract_time: str | None
    contract_type: str | None
    category: str | None
    created: str | None
    redirect_url: str | None
    adzuna_relevance: float | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _cache_key(country: str, query: str, page: int, where: str | None, extra: dict) -> str:
    payload = json.dumps(
        {"country": country, "q": query, "page": page, "where": where, "extra": extra},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_search_params(
    *,
    app_id: str,
    app_key: str,
    query: str,
    results_per_page: int,
    where: str | None,
    max_days_old: int | None,
    strict_and: bool,
) -> dict:
    """Build the Adzuna /search query parameters.

    By default we send the candidate text as `what_or` (match any of these
    words) so long free-text queries still return a useful recall set; Adzuna
    then ranks those results by its own relevance. Set `strict_and=True` to use
    `what` (all words must match) for narrow, keyword-style queries.
    """
    params: dict = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "content-type": "application/json",
        "sort_by": "relevance",
    }
    if strict_and:
        params["what"] = query
    else:
        params["what_or"] = query
    if where:
        params["where"] = where
    if max_days_old is not None:
        params["max_days_old"] = max_days_old
    return params


def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")


def _parse_job(raw: dict, country: str) -> AdzunaJob:
    return AdzunaJob(
        id=str(raw.get("id", "")),
        country=country,
        title=str(raw.get("title", "")).strip(),
        company=str((raw.get("company") or {}).get("display_name", "")).strip(),
        location=str((raw.get("location") or {}).get("display_name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        salary_min=raw.get("salary_min"),
        salary_max=raw.get("salary_max"),
        contract_time=raw.get("contract_time"),
        contract_type=raw.get("contract_type"),
        category=(raw.get("category") or {}).get("label"),
        created=raw.get("created"),
        redirect_url=raw.get("redirect_url"),
        adzuna_relevance=raw.get("__relevance__"),
    )


def search_country(
    *,
    app_id: str,
    app_key: str,
    country: str,
    query: str,
    page: int = 1,
    results_per_page: int = 50,
    where: str | None = None,
    max_days_old: int | None = None,
    strict_and: bool = False,
    cache_dir: Path | None = None,
    timeout: float = 20.0,
    sleep_after_request: float = 0.0,
) -> tuple[list[AdzunaJob], dict]:
    """Search a single Adzuna country endpoint, with optional disk cache.

    Returns (jobs, raw_meta) where raw_meta contains Adzuna's count/mean fields.
    """
    country = country.lower().strip()
    if country not in SUPPORTED_COUNTRIES:
        raise ValueError(
            f"Unsupported Adzuna country '{country}'. Supported: {sorted(SUPPORTED_COUNTRIES)}"
        )

    params = _build_search_params(
        app_id=app_id,
        app_key=app_key,
        query=query,
        results_per_page=results_per_page,
        where=where,
        max_days_old=max_days_old,
        strict_and=strict_and,
    )

    cached_payload = None
    cache_path: Path | None = None
    if cache_dir is not None:
        key = _cache_key(
            country,
            query,
            page,
            where,
            {"max_days_old": max_days_old, "strict_and": strict_and},
        )
        cache_path = cache_dir / f"{country}_{key}.json"
        cached_payload = _load_cache(cache_path)

    if cached_payload is not None:
        payload = cached_payload
        from_cache = True
    else:
        url = f"{ADZUNA_BASE_URL}/{country}/search/{page}"
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Adzuna {country} page {page} returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        from_cache = False
        if cache_path is not None:
            _save_cache(cache_path, payload)
        if sleep_after_request > 0:
            time.sleep(sleep_after_request)

    raw_results = payload.get("results") or []
    jobs: list[AdzunaJob] = []
    for rank, raw in enumerate(raw_results):
        raw_with_score = dict(raw)
        raw_with_score["__relevance__"] = 1.0 - (rank / max(len(raw_results), 1))
        jobs.append(_parse_job(raw_with_score, country=country))

    meta = {
        "country": country,
        "page": page,
        "from_cache": from_cache,
        "count_returned": len(jobs),
        "count_total": payload.get("count"),
        "mean_salary": payload.get("mean"),
    }
    return jobs, meta


def search_many_countries(
    *,
    app_id: str,
    app_key: str,
    countries: Iterable[str],
    query: str,
    results_per_page: int = 50,
    where: str | None = None,
    max_days_old: int | None = None,
    strict_and: bool = False,
    cache_dir: Path | None = None,
    sleep_between_requests: float = 0.5,
) -> tuple[list[AdzunaJob], list[dict]]:
    """Query each country once (page=1) and concatenate the jobs."""
    all_jobs: list[AdzunaJob] = []
    metas: list[dict] = []
    for country in countries:
        jobs, meta = search_country(
            app_id=app_id,
            app_key=app_key,
            country=country,
            query=query,
            page=1,
            results_per_page=results_per_page,
            where=where,
            max_days_old=max_days_old,
            strict_and=strict_and,
            cache_dir=cache_dir,
            sleep_after_request=sleep_between_requests,
        )
        all_jobs.extend(jobs)
        metas.append(meta)
    return all_jobs, metas
