"""Tabular schema-based features for candidates and vacancies.

Two parsed-schema JSONL inputs feed this module:

* `data/raw/candidates_with_schema.jsonl` (LLM-extracted fields under
  ``filled.candidate_description`` and ``filled.job_description``).
* `data/raw/vacancies_safe_ml_dataset_nozip.jsonl` (existing categorical
  vacancy columns).

Each candidate-vacancy row consumed by the reranker is enriched with three
groups of numeric features:

* ``cand_*`` — candidate-only features (counts, one-hot education levels,
  preferred remote policy, etc.).
* ``vac_*`` — vacancy-only features (one-hot grades, english level, work
  format, company type, salary in USD).
* ``pair_*`` — interaction signals computed from both sides
  (format / employment-type / seniority / english / industry / salary).

All features are floats so they can be fed straight into LightGBM. Missing
information is encoded as a ``0`` paired with a ``cand_has_*`` /
``vac_has_*`` indicator wherever it matters, so the model can distinguish
"missing" from "negative".
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sara_retrieve_rerank.data import load_jsonl

EDUCATION_LEVELS: tuple[str, ...] = (
    "bachelor",
    "master",
    "mba",
    "phd",
    "vocational",
)
ENGLISH_LEVELS: tuple[str, ...] = ("a1", "a2", "b1", "b2", "c1", "c2")
WORK_FORMATS: tuple[str, ...] = ("remote", "hybrid", "onsite")
WORK_TYPES: tuple[str, ...] = ("fulltime", "parttime", "project", "contract")
EMPLOYEE_TYPES: tuple[str, ...] = ("employment", "b2b_contract")
COMPANY_TYPES: tuple[str, ...] = (
    "corporation",
    "product_company",
    "outsourcing_company",
    "startup",
)
GRADES: tuple[str, ...] = (
    "trainee",
    "junior",
    "middle",
    "senior",
    "lead",
    "head",
    "principal",
    "director",
    "c_level",
)

# Approximate FX rates used only to put salary numbers on a common scale so
# that pair-wise overlap features make sense. Missing currencies fall back
# to a 1.0 multiplier and the `has_currency` indicator marks them so the
# model can downweight unreliable rows.
CURRENCY_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "USDT": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "RUB": 0.011,
    "PLN": 0.25,
    "CAD": 0.73,
    "AUD": 0.66,
    "JPY": 0.0068,
    "CHF": 1.13,
    "SEK": 0.094,
    "DKK": 0.14,
    "INR": 0.012,
    "CNY": 0.14,
    "MXN": 0.058,
    "ZAR": 0.054,
    "BRL": 0.20,
    "SGD": 0.74,
    "AED": 0.27,
    "PHP": 0.018,
    "KZT": 0.0022,
    "THB": 0.028,
    "MYR": 0.21,
    "UZS": 0.00008,
    "RSD": 0.0093,
    "TWD": 0.031,
    "BYN": 0.31,
    "NGN": 0.0007,
    "TRY": 0.032,
}

# Map vacancy grade -> typical (min_years, max_years_or_None) range used for
# the pair seniority match feature. These are loose buckets; the reranker
# learns the actual cutoffs from the data.
GRADE_YEAR_RANGES: dict[str, tuple[float, float | None]] = {
    "trainee": (0.0, 1.0),
    "junior": (0.0, 3.0),
    "middle": (2.0, 6.0),
    "senior": (5.0, None),
    "lead": (6.0, None),
    "head": (7.0, None),
    "principal": (8.0, None),
    "director": (8.0, None),
    "c_level": (10.0, None),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_token(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _tokenize(value: str | None) -> set[str]:
    if not value:
        return set()
    return set(_TOKEN_RE.findall(str(value).lower()))


def _normalize_education_level(raw: str | None) -> str | None:
    if not raw:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if "mba" in text:
        return "mba"
    if "phd" in text or "ph.d" in text or "doctor" in text:
        return "phd"
    if "master" in text or "ll.m" in text or "graduate degree" in text:
        return "master"
    if "bachelor" in text or text in {"ba", "bs", "bsc", "b.sc"}:
        return "bachelor"
    if "vocational" in text or "hnd" in text or "associate" in text:
        return "vocational"
    return None


def _has_english_language(languages: Iterable[Any] | None) -> bool:
    if not languages:
        return False
    for entry in languages:
        if not entry:
            continue
        text = str(entry).strip().lower()
        if "english" in text or text in {"en", "eng"}:
            return True
    return False


def _normalize_remote_policy(value: str | None) -> str | None:
    token = _normalize_token(value)
    if not token:
        return None
    if token in {"remote", "fullremote", "fully_remote"}:
        return "remote"
    if token in {"hybrid"}:
        return "hybrid"
    if token in {"onsite", "office", "in_office", "on_site"}:
        return "onsite"
    return None


def _normalize_employment_type(value: str | None) -> str | None:
    token = _normalize_token(value)
    if not token:
        return None
    if token in {"full_time", "fulltime", "full"}:
        return "fulltime"
    if token in {"part_time", "parttime", "part"}:
        return "parttime"
    if token in {"contract", "contractor", "freelance", "b2b", "b2b_contract"}:
        return "contract"
    if token in {"project", "temporary"}:
        return "project"
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_usd(amount: Any, currency: Any) -> float | None:
    value = _safe_float(amount)
    if value is None or value <= 0:
        return None
    rate = CURRENCY_TO_USD.get(str(currency).upper()) if currency else None
    if rate is None:
        return value
    return value * rate


def _log1p_nonneg(value: float | None) -> float:
    if value is None or value <= 0:
        return 0.0
    return math.log1p(float(value))


def _candidate_filled(candidate: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not candidate:
        return {}
    filled = candidate.get("filled")
    return filled if isinstance(filled, Mapping) else {}


def candidate_schema_features(candidate: Mapping[str, Any] | None) -> dict[str, float]:
    """Return numeric features derived from a parsed candidate schema row."""
    features: dict[str, float] = {field: 0.0 for field in CANDIDATE_SCHEMA_FIELDS}

    filled = _candidate_filled(candidate)
    cd = filled.get("candidate_description") if isinstance(filled, Mapping) else None
    jd = filled.get("job_description") if isinstance(filled, Mapping) else None
    cd = cd if isinstance(cd, Mapping) else {}
    jd = jd if isinstance(jd, Mapping) else {}

    years = _safe_float(cd.get("years_experience"))
    if years is not None:
        features["cand_years_experience"] = float(min(max(years, 0.0), 40.0))
        features["cand_has_years_experience"] = 1.0

    languages = cd.get("languages")
    if isinstance(languages, list):
        features["cand_lang_count"] = float(len(languages))
        features["cand_has_english"] = 1.0 if _has_english_language(languages) else 0.0

    skills = cd.get("skills")
    if isinstance(skills, list):
        features["cand_skill_count"] = float(len(skills))

    education = _normalize_education_level(cd.get("education_level"))
    if education is not None:
        key = f"cand_edu_{education}"
        if key in features:
            features[key] = 1.0
            features["cand_has_education"] = 1.0
    elif cd.get("education_level"):
        features["cand_has_education"] = 1.0

    positions = jd.get("desired_positions")
    if isinstance(positions, list):
        features["cand_position_count"] = float(len(positions))

    stack = jd.get("desired_tech_stack")
    if isinstance(stack, list):
        features["cand_tech_stack_count"] = float(len(stack))

    domains = jd.get("preferred_domains")
    if isinstance(domains, list):
        features["cand_domain_count"] = float(len(domains))

    activities = jd.get("preferred_activities")
    if isinstance(activities, list):
        features["cand_activity_count"] = float(len(activities))

    companies = jd.get("preferred_companies")
    if isinstance(companies, list):
        features["cand_industry_count"] = float(len(companies))

    comp = jd.get("desired_compensation_monthly") if isinstance(jd, Mapping) else None
    if isinstance(comp, Mapping):
        salary_min = _safe_float(comp.get("salary_min"))
        salary_max = _safe_float(comp.get("salary_max"))
        currency = comp.get("currency")
        if salary_min is not None or salary_max is not None:
            features["cand_has_salary_expectation"] = 1.0
        if currency:
            features["cand_has_salary_currency"] = 1.0
        min_usd = _to_usd(salary_min, currency) if salary_min is not None else None
        max_usd = _to_usd(salary_max, currency) if salary_max is not None else None
        if min_usd is None and max_usd is not None:
            min_usd = max_usd
        if max_usd is None and min_usd is not None:
            max_usd = min_usd
        features["cand_salary_min_usd_log"] = _log1p_nonneg(min_usd)
        features["cand_salary_max_usd_log"] = _log1p_nonneg(max_usd)
        benefits = comp.get("benefits")
        if isinstance(benefits, list):
            features["cand_benefits_count"] = float(len(benefits))

    work_mode = jd.get("preferred_work_mode") if isinstance(jd, Mapping) else None
    if isinstance(work_mode, Mapping):
        features["cand_has_work_mode"] = 1.0
        for raw in work_mode.get("employment_type") or []:
            kind = _normalize_employment_type(raw)
            if kind is None:
                continue
            key = f"cand_emp_{kind}"
            if key in features:
                features[key] = 1.0
        for raw in work_mode.get("preferred_remote_policy") or []:
            kind = _normalize_remote_policy(raw)
            if kind is None:
                continue
            key = f"cand_pref_{kind}"
            if key in features:
                features[key] = 1.0
        for raw in work_mode.get("acceptable_remote_policy") or []:
            kind = _normalize_remote_policy(raw)
            if kind is None:
                continue
            key = f"cand_acc_{kind}"
            if key in features:
                features[key] = 1.0

    return features


def vacancy_schema_features(vacancy: Mapping[str, Any] | None) -> dict[str, float]:
    """Return numeric features derived from a raw vacancy record."""
    features: dict[str, float] = {field: 0.0 for field in VACANCY_SCHEMA_FIELDS}
    if not vacancy:
        return features

    for raw in vacancy.get("grades") or []:
        grade = _normalize_token(raw)
        key = f"vac_grade_{grade}"
        if key in features:
            features[key] = 1.0

    english = _normalize_token(vacancy.get("english_level"))
    key = f"vac_english_{english}"
    if key in features:
        features[key] = 1.0
        features["vac_has_english_level"] = 1.0

    for raw in vacancy.get("work_format") or []:
        fmt = _normalize_remote_policy(raw) or _normalize_token(raw)
        key = f"vac_format_{fmt}"
        if key in features:
            features[key] = 1.0

    work_type = _normalize_token(vacancy.get("work_type"))
    key = f"vac_wt_{work_type}"
    if key in features:
        features[key] = 1.0

    for raw in vacancy.get("employee_type") or []:
        et = _normalize_token(raw)
        key = f"vac_et_{et}"
        if key in features:
            features[key] = 1.0

    ct = _normalize_token(vacancy.get("company_type"))
    key = f"vac_ct_{ct}"
    if key in features:
        features[key] = 1.0

    salary = vacancy.get("salary")
    if isinstance(salary, Mapping):
        features["vac_has_salary"] = 1.0
        min_value = _safe_float(salary.get("min"))
        max_value = _safe_float(salary.get("max"))
        currency = salary.get("currency")
        salary_in_usd = _safe_float(salary.get("salary_in_usd"))
        min_usd = _to_usd(min_value, currency) if min_value is not None else None
        max_usd = _to_usd(max_value, currency) if max_value is not None else None
        if min_usd is None and salary_in_usd is not None:
            min_usd = salary_in_usd
        if max_usd is None and salary_in_usd is not None:
            max_usd = salary_in_usd
        features["vac_salary_min_usd_log"] = _log1p_nonneg(min_usd)
        features["vac_salary_max_usd_log"] = _log1p_nonneg(max_usd)

    score = _safe_float(vacancy.get("vacancy_score"))
    if score is not None:
        features["vac_score"] = float(score)

    return features


def pair_schema_features(
    candidate: Mapping[str, Any] | None,
    vacancy: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return interaction features between a parsed candidate and a vacancy."""
    features: dict[str, float] = {field: 0.0 for field in PAIR_SCHEMA_FIELDS}
    if not candidate or not vacancy:
        return features

    filled = _candidate_filled(candidate)
    cd = filled.get("candidate_description") if isinstance(filled, Mapping) else None
    jd = filled.get("job_description") if isinstance(filled, Mapping) else None
    cd = cd if isinstance(cd, Mapping) else {}
    jd = jd if isinstance(jd, Mapping) else {}

    # Work-format match
    cand_pref = {
        _normalize_remote_policy(x)
        for x in ((jd.get("preferred_work_mode") or {}).get("preferred_remote_policy") or [])
    }
    cand_acc = {
        _normalize_remote_policy(x)
        for x in ((jd.get("preferred_work_mode") or {}).get("acceptable_remote_policy") or [])
    }
    cand_pref.discard(None)
    cand_acc.discard(None)
    vac_format = {_normalize_remote_policy(x) for x in (vacancy.get("work_format") or [])}
    vac_format.discard(None)
    if cand_pref and vac_format:
        features["pair_format_match"] = 1.0 if cand_pref & vac_format else 0.0
        features["pair_format_compat"] = 1.0 if cand_pref & vac_format else 0.0
    if cand_acc and vac_format and not (cand_pref & vac_format):
        if cand_acc & vac_format:
            features["pair_format_compat"] = 1.0
            features["pair_format_acceptable_match"] = 1.0

    # Employment type match
    cand_emp = {
        _normalize_employment_type(x)
        for x in ((jd.get("preferred_work_mode") or {}).get("employment_type") or [])
    }
    cand_emp.discard(None)
    vac_wt = _normalize_token(vacancy.get("work_type"))
    if cand_emp and vac_wt:
        features["pair_employment_type_match"] = 1.0 if vac_wt in cand_emp else 0.0

    # Seniority match
    years = _safe_float(cd.get("years_experience"))
    if years is not None and (vacancy.get("grades") or []):
        any_match = False
        for raw in vacancy.get("grades") or []:
            grade = _normalize_token(raw)
            bucket = GRADE_YEAR_RANGES.get(grade)
            if bucket is None:
                continue
            low, high = bucket
            if years >= low and (high is None or years <= high):
                any_match = True
                break
        features["pair_seniority_match"] = 1.0 if any_match else 0.0
        features["pair_has_seniority_info"] = 1.0

    # English match — vacancy requires English ≥ B1 and candidate has English
    english = _normalize_token(vacancy.get("english_level"))
    if english in {"b1", "b2", "c1", "c2"}:
        features["pair_english_match"] = (
            1.0 if _has_english_language(cd.get("languages")) else 0.0
        )
        features["pair_has_english_requirement"] = 1.0

    # Industry / domain overlap
    cand_industry_tokens: set[str] = set()
    for entry in jd.get("preferred_companies") or []:
        if isinstance(entry, Mapping):
            cand_industry_tokens |= _tokenize(entry.get("industry"))
    vac_industry_tokens: set[str] = set()
    for raw in vacancy.get("company_domains") or []:
        vac_industry_tokens |= _tokenize(raw)
    for raw in vacancy.get("specializations") or []:
        vac_industry_tokens |= _tokenize(raw)
    if cand_industry_tokens and vac_industry_tokens:
        overlap = cand_industry_tokens & vac_industry_tokens
        features["pair_industry_overlap"] = float(len(overlap))
        features["pair_industry_match"] = 1.0 if overlap else 0.0

    # Skill / tech-stack overlap
    cand_skill_tokens: set[str] = set()
    for raw in (cd.get("skills") or []) + (jd.get("desired_tech_stack") or []):
        cand_skill_tokens |= _tokenize(raw)
    vac_skill_tokens: set[str] = set()
    for raw in vacancy.get("tags") or []:
        vac_skill_tokens |= _tokenize(raw)
    if cand_skill_tokens and vac_skill_tokens:
        overlap = cand_skill_tokens & vac_skill_tokens
        features["pair_skill_overlap"] = float(len(overlap))
        features["pair_skill_match"] = 1.0 if overlap else 0.0

    # Position match — does candidate's desired_position substring appear in vacancy title?
    cand_position_tokens: set[str] = set()
    for raw in jd.get("desired_positions") or []:
        cand_position_tokens |= _tokenize(raw)
    title_tokens = _tokenize(vacancy.get("title"))
    if cand_position_tokens and title_tokens:
        overlap = cand_position_tokens & title_tokens
        features["pair_position_overlap"] = float(len(overlap))
        features["pair_position_match"] = 1.0 if overlap else 0.0

    # Salary overlap in USD
    comp = jd.get("desired_compensation_monthly") if isinstance(jd, Mapping) else None
    salary = vacancy.get("salary")
    if isinstance(comp, Mapping) and isinstance(salary, Mapping):
        cand_min_usd = _to_usd(comp.get("salary_min"), comp.get("currency"))
        cand_max_usd = _to_usd(comp.get("salary_max"), comp.get("currency"))
        if cand_min_usd is None and cand_max_usd is not None:
            cand_min_usd = cand_max_usd
        if cand_max_usd is None and cand_min_usd is not None:
            cand_max_usd = cand_min_usd
        vac_min_usd = _safe_float(salary.get("salary_in_usd"))
        vac_max_usd = vac_min_usd
        raw_min = _to_usd(salary.get("min"), salary.get("currency"))
        raw_max = _to_usd(salary.get("max"), salary.get("currency"))
        if raw_min is not None:
            vac_min_usd = raw_min
        if raw_max is not None:
            vac_max_usd = raw_max
        if vac_min_usd is None and vac_max_usd is not None:
            vac_min_usd = vac_max_usd
        if vac_max_usd is None and vac_min_usd is not None:
            vac_max_usd = vac_min_usd
        if (
            cand_min_usd is not None
            and cand_max_usd is not None
            and vac_min_usd is not None
            and vac_max_usd is not None
        ):
            features["pair_has_salary_info"] = 1.0
            overlap_low = max(cand_min_usd, vac_min_usd)
            overlap_high = min(cand_max_usd, vac_max_usd)
            if overlap_high >= overlap_low:
                features["pair_salary_overlap"] = 1.0
            denom = max(cand_max_usd, vac_max_usd, 1.0)
            features["pair_salary_gap_log"] = math.log1p(
                max(0.0, cand_min_usd - vac_max_usd) / denom
            )

    return features


CANDIDATE_SCHEMA_FIELDS: tuple[str, ...] = (
    "cand_years_experience",
    "cand_has_years_experience",
    "cand_lang_count",
    "cand_has_english",
    "cand_skill_count",
    "cand_has_education",
    *(f"cand_edu_{level}" for level in EDUCATION_LEVELS),
    "cand_position_count",
    "cand_tech_stack_count",
    "cand_domain_count",
    "cand_activity_count",
    "cand_industry_count",
    "cand_has_salary_expectation",
    "cand_has_salary_currency",
    "cand_salary_min_usd_log",
    "cand_salary_max_usd_log",
    "cand_benefits_count",
    "cand_has_work_mode",
    *(f"cand_emp_{kind}" for kind in WORK_TYPES),
    *(f"cand_pref_{kind}" for kind in WORK_FORMATS),
    *(f"cand_acc_{kind}" for kind in WORK_FORMATS),
)

VACANCY_SCHEMA_FIELDS: tuple[str, ...] = (
    *(f"vac_grade_{grade}" for grade in GRADES),
    "vac_has_english_level",
    *(f"vac_english_{level}" for level in ENGLISH_LEVELS),
    *(f"vac_format_{kind}" for kind in WORK_FORMATS),
    *(f"vac_wt_{kind}" for kind in WORK_TYPES),
    *(f"vac_et_{kind}" for kind in EMPLOYEE_TYPES),
    *(f"vac_ct_{kind}" for kind in COMPANY_TYPES),
    "vac_has_salary",
    "vac_salary_min_usd_log",
    "vac_salary_max_usd_log",
    "vac_score",
)

PAIR_SCHEMA_FIELDS: tuple[str, ...] = (
    "pair_format_match",
    "pair_format_acceptable_match",
    "pair_format_compat",
    "pair_employment_type_match",
    "pair_seniority_match",
    "pair_has_seniority_info",
    "pair_english_match",
    "pair_has_english_requirement",
    "pair_industry_match",
    "pair_industry_overlap",
    "pair_skill_match",
    "pair_skill_overlap",
    "pair_position_match",
    "pair_position_overlap",
    "pair_salary_overlap",
    "pair_has_salary_info",
    "pair_salary_gap_log",
)

SCHEMA_FEATURE_FIELDS: tuple[str, ...] = (
    *CANDIDATE_SCHEMA_FIELDS,
    *VACANCY_SCHEMA_FIELDS,
    *PAIR_SCHEMA_FIELDS,
)


def load_candidate_schema_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load `candidates_with_schema.jsonl` into a `{candidate_id: row}` map.

    Keys are normalized to strings so they can be looked up with ids of any
    type (int, str). Returns an empty map when the file is missing so callers
    can degrade gracefully.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}
    rows = load_jsonl(file_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("candidate_id")
        if key is None:
            key = row.get("id")
        if key is None:
            continue
        by_id[str(key)] = row
    return by_id


def enrich_rows_with_schema_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_schema_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    vacancies_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return new rows with schema feature columns appended.

    Missing schema or vacancy lookups produce zero-filled features so the
    output schema is uniform across rows. The original rows are not mutated.
    """
    candidate_schema_by_id = candidate_schema_by_id or {}
    vacancies_by_id = vacancies_by_id or {}

    enriched: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = row.get("candidate_id")
        vacancy_id = row.get("vacancy_id")
        candidate_schema = (
            candidate_schema_by_id.get(str(candidate_id)) if candidate_id is not None else None
        )
        vacancy = (
            vacancies_by_id.get(str(vacancy_id)) if vacancy_id is not None else None
        )

        merged = dict(row)
        merged.update(candidate_schema_features(candidate_schema))
        merged.update(vacancy_schema_features(vacancy))
        merged.update(pair_schema_features(candidate_schema, vacancy))
        enriched.append(merged)
    return enriched


def rows_have_schema_features(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return True iff rows already contain the schema feature columns."""
    if not rows:
        return False
    sample = rows[0]
    return all(field in sample for field in SCHEMA_FEATURE_FIELDS[:3])
