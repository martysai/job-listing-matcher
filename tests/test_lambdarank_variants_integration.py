"""Integration test for the three-variant LambdaRank training flow.

Builds synthetic feature rows that carry every required column (cosine,
bm25, LLM expert, schema features) and verifies that:

- All three variants train successfully end-to-end.
- The main `lambdarank_score` is at least competitive with cosine on a
  small but separable synthetic dataset.
- Saved validation rows carry all variant score columns so the output
  table is self-contained.

Skipped if LightGBM is not installed (the heavy training dependency is
optional in this repo).
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("lightgbm")

from sara_retrieve_rerank.reranking import (
    add_weighted_score,
    evaluate_ranking_rows,
    score_rows_with_model,
    train_lambdarank,
)
from sara_retrieve_rerank.schema_features import SCHEMA_FEATURE_FIELDS


def _make_synthetic_rows(num_groups: int = 30, group_size: int = 8, seed: int = 0):
    """Build labeled rows where ground-truth signals correlate with features.

    The "positive" row in each group has higher cosine, BM25 and schema
    signals than the negatives, so a properly fitted ranker should place it
    on top.
    """
    rng = random.Random(seed)
    rows = []
    for group_id in range(num_groups):
        positive_index = rng.randrange(group_size)
        for index in range(group_size):
            label = 1 if index == positive_index else 0
            base_score = 0.7 + rng.uniform(-0.1, 0.1) if label else 0.3 + rng.uniform(-0.1, 0.1)
            row = {
                "candidate_id": f"c{group_id}",
                "vacancy_id": f"v{group_id}_{index}",
                "label": label,
                "retrieval_rank": index + 1,
                "cosine_similarity": base_score,
                "bm25_score": base_score * 2,
                "general_score": 0.95 if label else 0.2,
                "location_score": 0.9 if label else 0.3,
                "seniority_score": 0.8 if label else 0.4,
                "salary_score": 0.9 if label else 0.3,
                "experience_score": 0.85 if label else 0.25,
            }
            for field in SCHEMA_FEATURE_FIELDS:
                # Make a few schema columns informative; rest are zero.
                if field == "pair_skill_match":
                    row[field] = 1.0 if label else 0.0
                elif field == "pair_seniority_match":
                    row[field] = 1.0 if label else 0.0
                elif field == "cand_years_experience":
                    row[field] = 5.0 if label else rng.uniform(0.0, 2.0)
                else:
                    row[field] = 0.0
            rows.append(row)
    return rows


def test_three_variants_train_and_score_validation_rows():
    rows = _make_synthetic_rows()
    weighted_fields = (
        "general_score",
        "location_score",
        "seniority_score",
        "salary_score",
        "experience_score",
    )
    rows = add_weighted_score(
        rows,
        {field: 1.0 for field in weighted_fields},
        output_key="weighted_llm_score",
    )

    main_features = (
        "cosine_similarity",
        "bm25_score",
        *weighted_fields,
        *SCHEMA_FEATURE_FIELDS,
    )
    no_llm_features = ("cosine_similarity", "bm25_score", *SCHEMA_FEATURE_FIELDS)
    no_schema_features = ("cosine_similarity", "bm25_score", *weighted_fields)

    main_model, _train_rows, validation_rows = train_lambdarank(
        rows,
        feature_fields=main_features,
        validation_fraction=0.34,
        seed=42,
    )
    scored = score_rows_with_model(
        validation_rows,
        main_model,
        feature_fields=main_features,
        output_key="lambdarank_score",
    )

    for variant_name, features in (
        ("lambdarank_no_llm_expert", no_llm_features),
        ("lambdarank_no_schema", no_schema_features),
    ):
        ablation_model, _ab_train, _ab_val = train_lambdarank(
            rows,
            feature_fields=features,
            validation_fraction=0.34,
            seed=42,
        )
        scored = score_rows_with_model(
            scored,
            ablation_model,
            feature_fields=features,
            output_key=variant_name,
        )

    # All variant scores must be present on every validation row.
    for row in scored:
        for key in ("lambdarank_score", "lambdarank_no_llm_expert", "lambdarank_no_schema"):
            assert key in row

    main_metrics = evaluate_ranking_rows(scored, score_key="lambdarank_score", ks=(1,))
    cosine_metrics = evaluate_ranking_rows(scored, score_key="cosine_similarity", ks=(1,))
    # On separable synthetic data the main model should at least match cosine.
    assert main_metrics["recall@1"] >= cosine_metrics["recall@1"]
