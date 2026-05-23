"""ProfileCounter and QueryBuilder.

ProfileCounter
--------------
Records which job domains, positions, and locations appear most often in
parsed candidate profiles.  Must be called once per candidate submission
(in a background task) so the query plan reflects real user demand.

QueryBuilder
------------
Translates counter frequency data into a prioritised list of AdzunaQuery
objects within the daily API budget.  Falls back to FALLBACK_QUERIES when
the counter is cold (not enough recent events).

Category tags
-------------
All category tags are taken from the official Adzuna API response
(GET /api/jobs/{country}/categories).  Do not guess or invent slugs —
use only values from ADZUNA_CATEGORIES below.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from adzuna.config import (
    AdzunaQuery,
    COUNTER_STORAGE_PATH,
    DAILY_REQUEST_BUDGET,
    RESULTS_PER_REQUEST,
)


# ── Official Adzuna category tags ─────────────────────────────────────────────
# Source: GET https://api.adzuna.com/v1/api/jobs/gb/categories
# Update this dict if the category list changes for a new country.

ADZUNA_CATEGORIES: dict[str, str] = {
    "Accounting & Finance Jobs":        "accounting-finance-jobs",
    "Admin Jobs":                        "admin-jobs",
    "Charity & Voluntary Jobs":          "charity-voluntary-jobs",
    "Consultancy Jobs":                  "consultancy-jobs",
    "Creative & Design Jobs":            "creative-design-jobs",
    "Customer Services Jobs":            "customer-services-jobs",
    "Domestic help & Cleaning Jobs":     "domestic-help-cleaning-jobs",
    "Energy, Oil & Gas Jobs":            "energy-oil-gas-jobs",
    "Engineering Jobs":                  "engineering-jobs",
    "Graduate Jobs":                     "graduate-jobs",
    "HR & Recruitment Jobs":             "hr-jobs",
    "Healthcare & Nursing Jobs":         "healthcare-nursing-jobs",
    "Hospitality & Catering Jobs":       "hospitality-catering-jobs",
    "IT Jobs":                           "it-jobs",
    "Legal Jobs":                        "legal-jobs",
    "Logistics & Warehouse Jobs":        "logistics-warehouse-jobs",
    "Maintenance Jobs":                  "maintenance-jobs",
    "Manufacturing Jobs":                "manufacturing-jobs",
    "Other/General Jobs":                "other-general-jobs",
    "PR, Advertising & Marketing Jobs":  "pr-advertising-marketing-jobs",
    "Part time Jobs":                    "part-time-jobs",
    "Property Jobs":                     "property-jobs",
    "Retail Jobs":                       "retail-jobs",
    "Sales Jobs":                        "sales-jobs",
    "Scientific & QA Jobs":              "scientific-qa-jobs",
    "Social work Jobs":                  "social-work-jobs",
    "Teaching Jobs":                     "teaching-jobs",
    "Trade & Construction Jobs":         "trade-construction-jobs",
    "Travel Jobs":                       "travel-jobs",
}


# ── Two-layer domain mapping ──────────────────────────────────────────────────
# Layer 1: profile domain term → Adzuna category tag (always set).
# Layer 2: profile domain term → keyword for `what` param (optional).
#
# Keywords are used only for broad categories (it-jobs) where the category
# alone returns too wide a result set.  Narrow categories (hr-jobs,
# healthcare-nursing-jobs, etc.) need no keyword — the category itself is
# already specific enough.

DOMAIN_TO_CATEGORY: dict[str, str] = {
    # IT specialisations → "IT Jobs"
    "machine learning":  "it-jobs",
    "deep learning":     "it-jobs",
    "nlp":               "it-jobs",
    "computer vision":   "it-jobs",
    "data science":      "it-jobs",
    "devops":            "it-jobs",
    "backend":           "it-jobs",
    "frontend":          "it-jobs",
    "fullstack":         "it-jobs",
    "cloud":             "it-jobs",
    "security":          "it-jobs",
    "fintech":           "it-jobs",
    "b2b saas":          "it-jobs",
    "high-load":         "it-jobs",
    "product":           "it-jobs",
    # Non-IT domains → their own categories
    "accounting":        "accounting-finance-jobs",
    "finance":           "accounting-finance-jobs",
    "healthcare":        "healthcare-nursing-jobs",
    "marketing":         "pr-advertising-marketing-jobs",
    "hr":                "hr-jobs",
    "recruitment":       "hr-jobs",
    "qa":                "scientific-qa-jobs",
    "testing":           "scientific-qa-jobs",
    "engineering":       "engineering-jobs",
    "consulting":        "consultancy-jobs",
    "design":            "creative-design-jobs",
    "ux":                "creative-design-jobs",
    "sales":             "sales-jobs",
    "legal":             "legal-jobs",
    "logistics":         "logistics-warehouse-jobs",
    "energy":            "energy-oil-gas-jobs",
}

DOMAIN_TO_KEYWORD: dict[str, str] = {
    # Only IT domains need a keyword to narrow down within "IT Jobs".
    # All other categories are narrow enough on their own.
    "machine learning":  "machine learning",
    "deep learning":     "deep learning",
    "nlp":               "nlp",
    "computer vision":   "computer vision",
    "data science":      "data scientist",
    "devops":            "devops",
    "backend":           "backend engineer",
    "frontend":          "frontend developer",
    "fullstack":         "full stack developer",
    "cloud":             "cloud engineer",
    "security":          "cybersecurity",
    "fintech":           "fintech",
    "b2b saas":          "saas",
    "high-load":         "distributed systems",
    "product":           "product manager",
}


# ── Position → Adzuna (category, keyword) ────────────────────────────────────
# Job titles from desired_positions are specific enough to always warrant
# a keyword alongside the category.

POSITION_TO_QUERY: dict[str, tuple[str, str]] = {
    "data scientist":      ("it-jobs",                       "data scientist"),
    "ml engineer":         ("it-jobs",                       "machine learning engineer"),
    "backend engineer":    ("it-jobs",                       "backend engineer"),
    "frontend engineer":   ("it-jobs",                       "frontend developer"),
    "devops engineer":     ("it-jobs",                       "devops engineer"),
    "sre":                 ("it-jobs",                       "site reliability engineer"),
    "platform engineer":   ("it-jobs",                       "platform engineer"),
    "product manager":     ("it-jobs",                       "product manager"),
    "data engineer":       ("it-jobs",                       "data engineer"),
    "software engineer":   ("it-jobs",                       "software engineer"),
    "qa engineer":         ("scientific-qa-jobs",            "qa engineer"),
    "cloud architect":     ("it-jobs",                       "cloud architect"),
    "security engineer":   ("it-jobs",                       "security engineer"),
    "finance analyst":     ("accounting-finance-jobs",       "finance analyst"),
    "accountant":          ("accounting-finance-jobs",       "accountant"),
    "marketing manager":   ("pr-advertising-marketing-jobs", "marketing manager"),
    "hr manager":          ("hr-jobs",                       "hr manager"),
    "consultant":          ("consultancy-jobs",              "consultant"),
    "ux designer":         ("creative-design-jobs",          "ux designer"),
    "sales manager":       ("sales-jobs",                    "sales manager"),
}


# ── Fallback queries (cold-start) ────────────────────────────────────────────
# Used when the counter has fewer than MIN_EVENTS recent events.
# Covers the specialisations identified as hard miss-buckets in README
# (qa_testing, marketing, backend_dev, frontend_dev, product_design)
# plus high-volume IT categories.

FALLBACK_QUERIES: list[AdzunaQuery] = [
    # IT — broad sub-niches
    AdzunaQuery("software engineer",         "", "it-jobs"),
    AdzunaQuery("backend engineer python",   "", "it-jobs"),
    AdzunaQuery("frontend developer react",  "", "it-jobs"),
    AdzunaQuery("data scientist",            "", "it-jobs"),
    AdzunaQuery("machine learning engineer", "", "it-jobs"),
    AdzunaQuery("devops engineer",           "", "it-jobs"),
    AdzunaQuery("data engineer",             "", "it-jobs"),
    AdzunaQuery("product manager",           "", "it-jobs"),
    AdzunaQuery("cloud engineer",            "", "it-jobs"),
    # QA — known miss-bucket in README, has its own Adzuna category
    AdzunaQuery("qa engineer",               "", "scientific-qa-jobs"),
    AdzunaQuery("automation engineer",       "", "scientific-qa-jobs"),
    # Marketing — known miss-bucket
    AdzunaQuery("",                          "", "pr-advertising-marketing-jobs"),
    # Design — known miss-bucket (product_design)
    AdzunaQuery("",                          "", "creative-design-jobs"),
    # Other sectors
    AdzunaQuery("",                          "", "accounting-finance-jobs"),
    AdzunaQuery("",                          "", "engineering-jobs"),
    AdzunaQuery("",                          "", "hr-jobs"),
    AdzunaQuery("",                          "", "healthcare-nursing-jobs"),
    AdzunaQuery("",                          "", "consultancy-jobs"),
]


# ══════════════════════════════════════════════════════════════════════════════

class ProfileCounter:
    """Sliding-window frequency counter over parsed candidate profile fields.

    Storage: JSON file (replace with Redis for concurrent writers in prod).
    Each event is stored as an ISO timestamp; old events are pruned lazily.

    Call ``record_profile()`` in a background task on every candidate
    submission.  The counter drives ``QueryBuilder`` to allocate the daily
    Adzuna API budget toward the most in-demand job categories.
    """

    WINDOW_DAYS: int = 7
    # Below this many events in the window → cold start → fallback queries.
    MIN_EVENTS: int = 10

    def __init__(self, storage_path: Path = COUNTER_STORAGE_PATH):
        self.storage_path = storage_path
        self._store: dict[str, dict[str, list[str]]] = self._load()

    def record_profile(self, profile: dict) -> None:
        """Record frequency events from a freshly parsed candidate profile.

        ``profile`` is the dict produced by the LLM parser
        (JobAndCandidateDescription schema).
        """
        now = datetime.utcnow().isoformat()
        jd  = profile.get("job_description") or {}

        for val in jd.get("desired_positions")  or []:
            self._inc("positions",     val.lower().strip(), now)
        for val in jd.get("preferred_domains")   or []:
            self._inc("domains",       val.lower().strip(), now)
        for val in jd.get("preferred_locations") or []:
            self._inc("locations",     val.lower().strip(), now)
        for val in jd.get("preferred_companies") or []:
            self._inc("company_types", val.lower().strip(), now)

        self._save()

    def get_top(self, dimension: str, k: int = 10) -> dict[str, int]:
        """Top-K entries for *dimension* within the current time window."""
        cutoff = self._cutoff()
        counts: dict[str, int] = defaultdict(int)
        for key, events in self._store.get(dimension, {}).items():
            n = sum(1 for e in events if e >= cutoff)
            if n:
                counts[key] = n
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:k])

    def is_cold(self) -> bool:
        """True when too few recent events to drive counter-based queries."""
        cutoff = self._cutoff()
        total = sum(
            1
            for dim in self._store.values()
            for events in dim.values()
            for e in events
            if e >= cutoff
        )
        return total < self.MIN_EVENTS

    def _inc(self, dimension: str, key: str, ts: str) -> None:
        bucket = self._store.setdefault(dimension, {}).setdefault(key, [])
        bucket.append(ts)
        prune = (datetime.utcnow() - timedelta(days=self.WINDOW_DAYS * 2)).isoformat()
        self._store[dimension][key] = [e for e in bucket if e >= prune]

    def _cutoff(self) -> str:
        return (datetime.utcnow() - timedelta(days=self.WINDOW_DAYS)).isoformat()

    def _load(self) -> dict:
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self._store, ensure_ascii=False), encoding="utf-8"
        )


class QueryBuilder:
    """Builds a daily Adzuna query plan from ProfileCounter data.

    Budget split (counter-driven)
    ------------------------------
    60 % → top-6 domains  (DOMAIN_TO_CATEGORY + DOMAIN_TO_KEYWORD)
    40 % → top-6 positions (POSITION_TO_QUERY)
    Requests within each group are proportional to frequency with a
    guaranteed minimum of MIN_PAGES_PER_QUERY pages.

    Cold start
    ----------
    When ``counter.is_cold()`` returns True, distributes the budget evenly
    over FALLBACK_QUERIES instead.
    """

    MIN_PAGES_PER_QUERY: int = 1

    def __init__(self, counter: ProfileCounter):
        self.counter = counter

    def build(self) -> list[AdzunaQuery]:
        """Return ordered list of queries to execute within today's budget."""
        if self.counter.is_cold():
            logging.getLogger("sara.agent").info(
                "Counter cold — using fallback queries"
            )
            return self._fallback()
        return self._from_counters()

    def _from_counters(self) -> list[AdzunaQuery]:
        top_domains   = self.counter.get_top("domains",   k=6)
        top_positions = self.counter.get_top("positions", k=6)
        top_locations = self.counter.get_top("locations", k=3)

        locations = [
            "" if loc in ("remote", "worldwide", "any", "anywhere") else loc
            for loc in (list(top_locations.keys()) or ["london"])
        ]
        primary_loc   = locations[0]
        secondary_loc = locations[1] if len(locations) > 1 else ""

        domain_budget   = int(DAILY_REQUEST_BUDGET * 0.60)
        position_budget = int(DAILY_REQUEST_BUDGET * 0.40)
        weighted: list[tuple[int, AdzunaQuery]] = []

        # Domain-driven: category always set; keyword only for broad categories.
        total_d = sum(top_domains.values()) or 1
        for domain, count in top_domains.items():
            category = DOMAIN_TO_CATEGORY.get(domain)
            if not category:
                continue
            keyword = DOMAIN_TO_KEYWORD.get(domain, "")
            n = max(
                self.MIN_PAGES_PER_QUERY,
                int((count / total_d) * domain_budget // RESULTS_PER_REQUEST),
            )
            weighted.append((n, AdzunaQuery(keyword, primary_loc, category)))
            if secondary_loc:
                weighted.append((1, AdzunaQuery(keyword, secondary_loc, category)))

        # Position-driven: keyword is the job title itself.
        total_p = sum(top_positions.values()) or 1
        for position, count in top_positions.items():
            category, keyword = POSITION_TO_QUERY.get(position, ("it-jobs", position))
            n = max(
                self.MIN_PAGES_PER_QUERY,
                int((count / total_p) * position_budget // RESULTS_PER_REQUEST),
            )
            weighted.append((n, AdzunaQuery(keyword, primary_loc, category)))

        return self._paginate(weighted)

    def _fallback(self) -> list[AdzunaQuery]:
        pages_each = max(
            1,
            DAILY_REQUEST_BUDGET // (len(FALLBACK_QUERIES) * RESULTS_PER_REQUEST),
        )
        result = [
            AdzunaQuery(q.what, q.where, q.category, RESULTS_PER_REQUEST)
            for q in FALLBACK_QUERIES
            for _ in range(pages_each)
        ]
        return result[:DAILY_REQUEST_BUDGET]

    def _paginate(self, weighted: list[tuple[int, AdzunaQuery]]) -> list[AdzunaQuery]:
        result = [
            AdzunaQuery(q.what, q.where, q.category, RESULTS_PER_REQUEST)
            for n, q in weighted
            for _ in range(n)
        ]
        return result[:DAILY_REQUEST_BUDGET]


class FixedQueryBuilder:
    """Drop-in replacement for QueryBuilder that returns a static query list.

    Used by the cron-driven refresh path where we don't want demand-driven
    query planning — instead the operator specifies an explicit list of
    (country, location, role) tuples to scrape every cycle (typically the
    top 2-3 tech hubs × top 2-3 target roles).

    Matches the ``build() -> list[AdzunaQuery]`` interface so it slots into
    ``RefreshContext.query_builder`` without any agent-side changes.
    """

    def __init__(self, queries: list[AdzunaQuery]):
        self._queries = list(queries)

    def build(self) -> list[AdzunaQuery]:
        return list(self._queries)
