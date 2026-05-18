"""Constants, environment-based settings, and the AdzunaQuery dataclass.

All tuneable values live here so no other module needs os.environ calls.
Override any value by setting the corresponding environment variable before
starting the process (or via .env loaded by Celery / the API layer).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ── Adzuna API ────────────────────────────────────────────────────────────────

ADZUNA_APP_ID:   str = os.environ.get("ADZUNA_APP_ID",  "")
ADZUNA_APP_KEY:  str = os.environ.get("ADZUNA_APP_KEY", "")
ADZUNA_BASE_URL: str = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
ADZUNA_COUNTRY:  str = os.environ.get("ADZUNA_COUNTRY", "eu")

# Free-tier rate limits. 25 hits/min, 250/day, 1000/week, 2500/month.
# 250 req × 50 results = 12 500 vacancies/day max.
# Reserve 20 requests for retries → budget 230.

DAILY_REQUEST_BUDGET: int = int(os.environ.get("ADZUNA_DAILY_REQUEST_BUDGET", "230"))
RESULTS_PER_REQUEST:  int = 50

# Seconds between HTTP requests to stay within Adzuna rate limits.
SCRAPE_DELAY_SECONDS: float = float(os.environ.get("ADZUNA_SCRAPE_DELAY", "2.5"))


# ── LLM extractor (Tool 1.5) ──────────────────────────────────────────────────

# litellm model string for structured field extraction.
EXTRACTOR_MODEL: str = os.environ.get(
    "VACANCY_EXTRACTOR_LLM_MODEL", "mistral/mistral-small-latest"
)
# Pause between LLM calls to avoid rate-limit errors.
EXTRACTOR_DELAY_SECONDS: float = float(
    os.environ.get("VACANCY_EXTRACTOR_DELAY", "0.3")
)


# ── Storage ───────────────────────────────────────────────────────────────────

from pathlib import Path

# Adzuna vacancies appended here for BM25 index rebuild at pipeline startup.
# Kept separate from the original dataset so provenance is clear.
ADZUNA_VACANCIES_JSONL: Path = Path(
    os.environ.get("ADZUNA_VACANCIES_JSONL", "data/raw/adzuna_vacancies.jsonl")
)

CHROMA_COLLECTION_NAME: str = os.environ.get("CHROMA_COLLECTION_NAME", "vacancies")
LLM_LOG_PATH: Path = Path(
    os.environ.get("VACANCY_EXTRACTOR_LLM_LOG_PATH", "data/logs/vacancy_extractor.jsonl")
)

# Adzuna vacancies are pruned from Chroma after this many days if not
# re-encountered by the scraper (i.e. the vacancy has likely been filled).
TTL_DAYS: int = int(os.environ.get("ADZUNA_VACANCY_TTL_DAYS", "14"))

# Counter state persisted to this path between agent runs.
COUNTER_STORAGE_PATH: Path = Path(
    os.environ.get("ADZUNA_COUNTER_PATH", "data/processed/profile_counter.json")
)


# ── Adzuna field mappings ─────────────────────────────────────────────────────
# Maps Adzuna's contract_type strings to the enum values used by
# schema_features.py (WORK_TYPES and WORK_FORMATS tuples).

ADZUNA_CONTRACT_TO_WORK_TYPE: dict[str, str] = {
    "permanent": "fulltime",
    "contract":  "contract",
    "part_time": "parttime",
    "temporary": "project",
}

# Initial work_format guess from contract type.
# Tool 1.5 (LLM extractor) overrides this when the description says remote/hybrid.
ADZUNA_CONTRACT_TO_WORK_FORMAT: dict[str, list[str]] = {
    "permanent": ["onsite"],
    "contract":  ["onsite"],
    "part_time": ["onsite"],
    "temporary": ["onsite"],
}


# ── AdzunaQuery ───────────────────────────────────────────────────────────────

@dataclass
class AdzunaQuery:
    """Parameters for one Adzuna search API call.

    ``to_params()`` returns the query-string dict for the HTTP request.
    Empty strings for what / where / category are omitted so the API
    applies the broadest filter for that dimension.
    """

    what:             str
    where:            str
    category:         str
    results_per_page: int = RESULTS_PER_REQUEST
    sort_by:          str = "date"

    def to_params(self) -> dict:
        p: dict = {
            "results_per_page": self.results_per_page,
            "sort_by":          self.sort_by,
        }
        if self.what:     p["what"]     = self.what
        if self.where:    p["where"]    = self.where
        if self.category: p["category"] = self.category
        return p
