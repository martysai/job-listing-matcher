from __future__ import annotations

import json
from pathlib import Path

import pytest

from adzuna import (
    SUPPORTED_COUNTRIES,
    _cache_key,
    _parse_job,
    search_country,
)


def test_cache_key_is_stable_and_param_sensitive() -> None:
    a = _cache_key("gb", "data scientist", 1, None, {})
    b = _cache_key("gb", "data scientist", 1, None, {})
    c = _cache_key("gb", "data scientist", 2, None, {})
    d = _cache_key("de", "data scientist", 1, None, {})
    assert a == b
    assert a != c
    assert a != d


def test_parse_job_extracts_nested_company_and_location() -> None:
    raw = {
        "id": "12345",
        "title": "Senior NLP Engineer",
        "company": {"display_name": "AcmeAI"},
        "location": {"display_name": "London, UK"},
        "description": "Build RAG systems.",
        "salary_min": 60000,
        "salary_max": 90000,
        "contract_time": "full_time",
        "contract_type": "permanent",
        "category": {"label": "IT Jobs"},
        "created": "2026-05-01T00:00:00Z",
        "redirect_url": "https://example.com/job/12345",
        "__relevance__": 0.92,
    }

    job = _parse_job(raw, country="gb")

    assert job.id == "12345"
    assert job.country == "gb"
    assert job.title == "Senior NLP Engineer"
    assert job.company == "AcmeAI"
    assert job.location == "London, UK"
    assert job.salary_min == 60000
    assert job.salary_max == 90000
    assert job.category == "IT Jobs"
    assert job.adzuna_relevance == pytest.approx(0.92)


def test_search_country_uses_cache_and_skips_network(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_payload = {
        "count": 2,
        "mean": 50000,
        "results": [
            {
                "id": "1",
                "title": "Data Scientist",
                "company": {"display_name": "Co"},
                "location": {"display_name": "Berlin"},
                "description": "...",
                "redirect_url": "https://example.com/1",
            },
            {
                "id": "2",
                "title": "ML Engineer",
                "company": {"display_name": "Co2"},
                "location": {"display_name": "Munich"},
                "description": "...",
                "redirect_url": "https://example.com/2",
            },
        ],
    }
    key = _cache_key(
        "de",
        "machine learning",
        1,
        None,
        {"max_days_old": None, "strict_and": False},
    )
    (cache_dir / f"de_{key}.json").write_text(json.dumps(cached_payload), encoding="utf-8")

    jobs, meta = search_country(
        app_id="fake",
        app_key="fake",
        country="de",
        query="machine learning",
        cache_dir=cache_dir,
    )

    assert meta["from_cache"] is True
    assert meta["count_returned"] == 2
    assert meta["count_total"] == 2
    assert [j.title for j in jobs] == ["Data Scientist", "ML Engineer"]
    assert jobs[0].adzuna_relevance > jobs[1].adzuna_relevance


def test_search_country_rejects_unknown_country() -> None:
    with pytest.raises(ValueError):
        search_country(app_id="x", app_key="y", country="zz", query="anything")


def test_supported_countries_contains_gb_and_de() -> None:
    assert "gb" in SUPPORTED_COUNTRIES
    assert "de" in SUPPORTED_COUNTRIES
