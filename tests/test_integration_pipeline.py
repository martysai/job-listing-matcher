"""End-to-end integration test using the BM25 retriever.

Covers the recommendation flow without external services:
    candidates --> retrieve_top_vacancies_bm25 --> build_pair_feature_rows
        --> score with deterministic fake LLM invoke
        --> train_lambdarank (skipped if LightGBM is not installed)
        --> rerank_all_matches
        --> assert ground-truth vacancy ranks at top-1.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from sara_retrieve_rerank.bm25_retrieval import BM25Index, retrieve_top_vacancies_bm25
from sara_retrieve_rerank.reranking import (
    DEFAULT_FEATURE_FIELDS_WITH_BM25,
    LLMExpert,
    add_weighted_score,
    build_pair_feature_rows,
    evaluate_ranking_rows,
    rerank_all_matches,
    score_feature_rows_with_experts,
)


VACANCIES = [
    {
        "dataset_id": "ml-1",
        "title": "Senior Machine Learning Engineer",
        "tldr_sanitized": "Build ranking pipelines using PyTorch and Python on GCP.",
        "specializations": ["Machine Learning"],
    },
    {
        "dataset_id": "fe-1",
        "title": "Frontend Engineer",
        "tldr_sanitized": "React TypeScript design systems and CSS animations.",
        "specializations": ["Frontend"],
    },
    {
        "dataset_id": "de-1",
        "title": "Data Engineer",
        "tldr_sanitized": "Spark Airflow SQL data pipelines on AWS.",
        "specializations": ["Data Engineering"],
    },
    {
        "dataset_id": "ml-2",
        "title": "Junior ML Researcher",
        "tldr_sanitized": "Run experiments with Jupyter and small Pytorch models.",
        "specializations": ["Machine Learning"],
    },
]


CANDIDATES = [
    {
        "id": "alice",
        "text": "Senior ML engineer with PyTorch ranking experience and Python.",
        "source_vacancy_id": "ml-1",
    },
    {
        "id": "bob",
        "text": "Frontend developer specialising in React TypeScript and design systems.",
        "source_vacancy_id": "fe-1",
    },
    {
        "id": "carol",
        "text": "Data engineer with Spark Airflow and SQL pipeline experience.",
        "source_vacancy_id": "de-1",
    },
]


def _vacancy_documents() -> list[Document]:
    from sara_retrieve_rerank.documents import create_vacancy_documents

    return create_vacancy_documents(VACANCIES)


def _fake_invoke(_llm, prompt: str) -> str:
    """Deterministic LLM stand-in that returns a high `general_score` only when
    the candidate text shares meaningful tokens with the vacancy text."""
    lower = prompt.lower()
    ground_truth_pairs = [
        ("senior ml engineer", "senior machine learning engineer"),
        ("frontend developer", "frontend engineer"),
        ("data engineer", "data engineer"),
    ]
    for candidate_marker, vacancy_marker in ground_truth_pairs:
        if candidate_marker in lower and vacancy_marker in lower:
            return '{"general_score": 0.95}'
    return '{"general_score": 0.1}'


def test_bm25_retrieves_ground_truth_in_top1_for_all_candidates():
    index = BM25Index(_vacancy_documents())

    for candidate in CANDIDATES:
        matches = retrieve_top_vacancies_bm25(candidate, index, k=4)
        assert matches[0]["vacancy_id"] == candidate["source_vacancy_id"], (
            f"BM25 top-1 should be ground truth for {candidate['id']}, got {matches[0]['vacancy_id']}"
        )


def test_end_to_end_retrieve_score_rerank_places_ground_truth_first():
    index = BM25Index(_vacancy_documents())

    all_matches = []
    for candidate in CANDIDATES:
        all_matches.extend(retrieve_top_vacancies_bm25(candidate, index, k=4))

    rows = build_pair_feature_rows(CANDIDATES, all_matches)
    assert len(rows) == len(CANDIDATES) * 4
    assert sum(row["label"] for row in rows) == len(CANDIDATES)

    scored_rows = score_feature_rows_with_experts(
        rows,
        candidates=CANDIDATES,
        vacancies=VACANCIES,
        llm=object(),
        experts=(LLMExpert(name="general", prompt="{candidate_text}\n{vacancy_text}"),),
        invoke=_fake_invoke,
    )

    fused_rows = add_weighted_score(
        scored_rows,
        {"cosine_similarity": 0.0, "bm25_score": 0.3, "general_score": 0.7},
        output_key="weighted_llm_score",
    )
    reranked = rerank_all_matches(fused_rows, score_key="weighted_llm_score")

    metrics = evaluate_ranking_rows(
        reranked,
        score_key="weighted_llm_score",
        ks=(1, 3),
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@3"] == 1.0


@pytest.mark.skipif(
    pytest.importorskip("lightgbm", reason="lightgbm not installed") is None,
    reason="lightgbm not installed",
)
def test_lambdarank_training_uses_bm25_feature_field_when_present():
    from sara_retrieve_rerank.reranking import (
        score_rows_with_model,
        train_lambdarank,
    )

    index = BM25Index(_vacancy_documents())
    all_matches = []
    for candidate in CANDIDATES:
        all_matches.extend(retrieve_top_vacancies_bm25(candidate, index, k=4))
    rows = build_pair_feature_rows(CANDIDATES, all_matches)

    # Inject the LLM score column so the LambdaRank model has enough features.
    for row in rows:
        row.setdefault("cosine_similarity", 0.0)
        row.setdefault("general_score", 1.0 if row["label"] == 1 else 0.05)

    model, _train_rows, _val_rows = train_lambdarank(
        rows,
        feature_fields=("cosine_similarity", "bm25_score", "general_score"),
        validation_fraction=0.34,
    )
    scored = score_rows_with_model(
        rows,
        model,
        feature_fields=("cosine_similarity", "bm25_score", "general_score"),
        output_key="lambdarank_score",
    )
    metrics = evaluate_ranking_rows(scored, score_key="lambdarank_score", ks=(1,))

    assert metrics["recall@1"] >= 0.66
    # Ensure the BM25 feature is actually consumable through the standard set.
    assert "bm25_score" in DEFAULT_FEATURE_FIELDS_WITH_BM25
