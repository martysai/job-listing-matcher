"""Tests for the schema-feature extraction module."""

from __future__ import annotations

import json

from sara_retrieve_rerank.schema_features import (
    CANDIDATE_SCHEMA_FIELDS,
    PAIR_SCHEMA_FIELDS,
    SCHEMA_FEATURE_FIELDS,
    VACANCY_SCHEMA_FIELDS,
    candidate_schema_features,
    enrich_rows_with_schema_features,
    load_candidate_schema_map,
    pair_schema_features,
    rows_have_schema_features,
    vacancy_schema_features,
)


CANDIDATE_SCHEMA_ROW = {
    "candidate_id": 1,
    "source_vacancy_id": "vacancy_001526",
    "candidate_type": "cv",
    "status": "ok",
    "filled": {
        "candidate_description": {
            "years_experience": 8,
            "languages": ["English"],
            "education_level": "master's",
            "skills": ["Python", "SQL", "Kafka", "Airflow", "Snowflake", "DBT"],
        },
        "job_description": {
            "desired_positions": ["Senior Data Pipeline Engineer"],
            "desired_tech_stack": ["Python", "Airflow"],
            "preferred_domains": [],
            "preferred_activities": [],
            "preferred_companies": [
                {"industry": "cybersecurity", "location": None},
            ],
            "desired_compensation_monthly": {
                "salary_min": 12500,
                "salary_max": 15000,
                "currency": "USD",
                "is_gross": None,
                "benefits": ["health"],
            },
            "preferred_work_mode": {
                "employment_type": ["full_time"],
                "preferred_remote_policy": ["remote"],
                "acceptable_remote_policy": ["hybrid"],
            },
        },
    },
}


VACANCY_ROW = {
    "dataset_id": "vacancy_001526",
    "title": "Senior Data Pipeline Engineer",
    "grades": ["senior"],
    "english_level": "c1",
    "work_format": ["hybrid"],
    "work_type": "fulltime",
    "employee_type": ["employment"],
    "company_type": "corporation",
    "company_domains": ["cybersecurity"],
    "specializations": ["data_engineering"],
    "tags": ["python", "airflow", "kafka"],
    "salary": {"min": 150000, "max": 180000, "currency": "USD", "salary_in_usd": 165000},
    "vacancy_score": 7,
}


def test_candidate_schema_features_emits_all_fields_with_expected_values():
    features = candidate_schema_features(CANDIDATE_SCHEMA_ROW)

    assert set(features) == set(CANDIDATE_SCHEMA_FIELDS)
    assert features["cand_years_experience"] == 8.0
    assert features["cand_has_years_experience"] == 1.0
    assert features["cand_lang_count"] == 1.0
    assert features["cand_has_english"] == 1.0
    assert features["cand_skill_count"] == 6.0
    assert features["cand_edu_master"] == 1.0
    assert features["cand_edu_phd"] == 0.0
    assert features["cand_emp_fulltime"] == 1.0
    assert features["cand_pref_remote"] == 1.0
    assert features["cand_acc_hybrid"] == 1.0
    assert features["cand_has_salary_expectation"] == 1.0
    assert features["cand_salary_min_usd_log"] > 0


def test_candidate_schema_features_handles_missing_filled_block():
    features = candidate_schema_features({})
    assert set(features) == set(CANDIDATE_SCHEMA_FIELDS)
    assert all(value == 0.0 for value in features.values())


def test_vacancy_schema_features_one_hots_categorical_columns():
    features = vacancy_schema_features(VACANCY_ROW)

    assert set(features) == set(VACANCY_SCHEMA_FIELDS)
    assert features["vac_grade_senior"] == 1.0
    assert features["vac_grade_middle"] == 0.0
    assert features["vac_english_c1"] == 1.0
    assert features["vac_format_hybrid"] == 1.0
    assert features["vac_wt_fulltime"] == 1.0
    assert features["vac_et_employment"] == 1.0
    assert features["vac_ct_corporation"] == 1.0
    assert features["vac_has_salary"] == 1.0
    assert features["vac_salary_min_usd_log"] > 0
    assert features["vac_score"] == 7.0


def test_pair_schema_features_matches_format_and_english():
    features = pair_schema_features(CANDIDATE_SCHEMA_ROW, VACANCY_ROW)

    assert set(features) == set(PAIR_SCHEMA_FIELDS)
    # Candidate prefers remote, vacancy is hybrid only -> pair_format_match is 0
    # but the acceptable policy includes hybrid -> acceptable match is 1.
    assert features["pair_format_match"] == 0.0
    assert features["pair_format_acceptable_match"] == 1.0
    assert features["pair_format_compat"] == 1.0
    assert features["pair_employment_type_match"] == 1.0
    assert features["pair_seniority_match"] == 1.0
    assert features["pair_english_match"] == 1.0
    assert features["pair_industry_match"] == 1.0
    assert features["pair_skill_match"] == 1.0
    assert features["pair_position_match"] == 1.0


def test_pair_schema_features_salary_overlap_when_ranges_intersect():
    candidate = {
        "filled": {
            "candidate_description": {},
            "job_description": {
                "desired_compensation_monthly": {
                    "salary_min": 100000,
                    "salary_max": 200000,
                    "currency": "USD",
                },
            },
        }
    }
    vacancy = {
        "salary": {"min": 150000, "max": 180000, "currency": "USD", "salary_in_usd": 165000},
    }
    features = pair_schema_features(candidate, vacancy)
    assert features["pair_salary_overlap"] == 1.0
    assert features["pair_has_salary_info"] == 1.0


def test_pair_schema_features_returns_zeros_for_missing_inputs():
    features = pair_schema_features(None, None)
    assert all(value == 0.0 for value in features.values())


def test_schema_feature_fields_are_unique_and_well_ordered():
    assert len(SCHEMA_FEATURE_FIELDS) == len(set(SCHEMA_FEATURE_FIELDS))
    # The combined list must be exactly the concatenation of the three groups.
    assert SCHEMA_FEATURE_FIELDS == (
        *CANDIDATE_SCHEMA_FIELDS,
        *VACANCY_SCHEMA_FIELDS,
        *PAIR_SCHEMA_FIELDS,
    )


def test_enrich_rows_with_schema_features_adds_columns_without_mutating_inputs():
    rows = [{"candidate_id": 1, "vacancy_id": "vacancy_001526", "cosine_similarity": 0.9}]
    enriched = enrich_rows_with_schema_features(
        rows,
        candidate_schema_by_id={"1": CANDIDATE_SCHEMA_ROW},
        vacancies_by_id={"vacancy_001526": VACANCY_ROW},
    )
    assert len(enriched) == 1
    assert "cand_years_experience" in enriched[0]
    assert "vac_grade_senior" in enriched[0]
    assert "pair_english_match" in enriched[0]
    assert enriched[0]["pair_english_match"] == 1.0
    # original rows untouched
    assert "cand_years_experience" not in rows[0]


def test_enrich_rows_zero_fills_when_lookup_misses():
    rows = [{"candidate_id": "nonexistent", "vacancy_id": "missing"}]
    enriched = enrich_rows_with_schema_features(
        rows,
        candidate_schema_by_id={},
        vacancies_by_id={},
    )
    assert all(enriched[0][field] == 0.0 for field in SCHEMA_FEATURE_FIELDS)


def test_rows_have_schema_features_detects_enriched_rows():
    assert rows_have_schema_features([{"cand_years_experience": 0.0, "cand_has_years_experience": 0.0, "cand_lang_count": 0.0}]) is True
    assert rows_have_schema_features([{"cosine_similarity": 0.9}]) is False
    assert rows_have_schema_features([]) is False


def test_load_candidate_schema_map_indexes_by_string_id(tmp_path):
    path = tmp_path / "candidates_with_schema.jsonl"
    rows = [
        {"candidate_id": 1, "filled": {"candidate_description": {}, "job_description": {}}},
        {"candidate_id": 2, "filled": {"candidate_description": {}, "job_description": {}}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    by_id = load_candidate_schema_map(path)
    assert set(by_id) == {"1", "2"}
    assert by_id["1"]["candidate_id"] == 1


def test_load_candidate_schema_map_returns_empty_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.jsonl"
    assert load_candidate_schema_map(path) == {}
