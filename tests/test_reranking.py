import math

from sara_retrieve_rerank.reranking import (
    LLMExpert,
    add_weighted_score,
    build_pair_feature_rows,
    evaluate_ranking_rows,
    filter_groups_with_positive,
    filter_rows_with_positive_groups,
    make_rate_limited_invoker,
    parse_llm_score,
    prepare_rows_for_llm_scoring,
    score_feature_rows_with_experts,
    rerank_candidate_matches,
    score_pair_with_experts,
    subsample_training_negatives,
    to_lambdarank_arrays,
)


def test_build_pair_feature_rows_adds_labels_and_retrieval_rank():
    candidates = [
        {"id": "candidate-1", "source_vacancy_id": "vacancy-2"},
        {"id": "candidate-2", "source_vacancy_id": "vacancy-9"},
    ]
    matches = [
        {"candidate_id": "candidate-1", "vacancy_id": "vacancy-1", "rank": 1},
        {"candidate_id": "candidate-1", "vacancy_id": "vacancy-2", "rank": 2},
        {"candidate_id": "unknown", "vacancy_id": "vacancy-3", "rank": 1},
    ]

    rows = build_pair_feature_rows(candidates, matches)

    assert len(rows) == 2
    assert rows[0]["label"] == 0
    assert rows[1]["label"] == 1
    assert rows[1]["retrieval_rank"] == 2
    assert rows[1]["source_vacancy_id"] == "vacancy-2"


def test_add_weighted_score_and_rerank_candidate_matches():
    rows = [
        {"vacancy_id": "a", "rank": 1, "cosine_similarity": 0.9, "general_score": 0.2},
        {"vacancy_id": "b", "rank": 2, "cosine_similarity": 0.5, "general_score": 1.0},
    ]

    scored = add_weighted_score(
        rows,
        {"cosine_similarity": 0.25, "general_score": 0.75},
        output_key="weighted_llm_score",
    )
    reranked = rerank_candidate_matches(scored, score_key="weighted_llm_score")

    assert reranked[0]["vacancy_id"] == "b"
    assert reranked[0]["rank"] == 1
    assert reranked[0]["retrieval_rank"] == 2
    assert math.isclose(reranked[0]["rerank_score"], 0.875)


def test_evaluate_ranking_rows_reports_recall_and_ndcg():
    rows = [
        {"candidate_id": "c1", "vacancy_id": "v1", "label": 0, "rank": 1, "score": 0.9},
        {"candidate_id": "c1", "vacancy_id": "v2", "label": 1, "rank": 2, "score": 0.8},
        {"candidate_id": "c2", "vacancy_id": "v3", "label": 1, "rank": 1, "score": 0.7},
        {"candidate_id": "c2", "vacancy_id": "v4", "label": 0, "rank": 2, "score": 0.6},
    ]

    metrics = evaluate_ranking_rows(rows, score_key="score", ks=(1, 2))

    assert metrics["num_groups"] == 2
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["ndcg@1"] == 0.5
    assert math.isclose(metrics["ndcg@2"], (1 / math.log2(3) + 1) / 2)


def test_filter_groups_with_positive_keeps_trainable_groups():
    rows = [
        {"candidate_id": "c1", "label": 0},
        {"candidate_id": "c1", "label": 1},
        {"candidate_id": "c2", "label": 0},
    ]

    filtered = filter_groups_with_positive(rows)

    assert [row["candidate_id"] for row in filtered] == ["c1", "c1"]


def test_filter_rows_with_positive_groups_keeps_only_positive_groups():
    rows = [
        {"candidate_id": "c1", "label": 0},
        {"candidate_id": "c1", "label": 1},
        {"candidate_id": "c2", "label": 0},
    ]

    filtered = filter_rows_with_positive_groups(rows)

    assert [row["candidate_id"] for row in filtered] == ["c1", "c1"]


def test_to_lambdarank_arrays_preserves_group_sizes():
    rows = [
        {"candidate_id": "c1", "label": 0, "cosine_similarity": 0.4, "general_score": 0.8},
        {"candidate_id": "c1", "label": 1, "cosine_similarity": 0.7, "general_score": 0.2},
        {"candidate_id": "c2", "label": 1, "cosine_similarity": 0.9, "general_score": 0.5},
    ]

    matrix, labels, groups = to_lambdarank_arrays(
        rows,
        feature_fields=("cosine_similarity", "general_score"),
    )

    assert matrix == [[0.4, 0.8], [0.7, 0.2], [0.9, 0.5]]
    assert labels == [0, 1, 1]
    assert groups == [2, 1]


def test_parse_llm_score_clamps_and_reads_message_content():
    class Message:
        content = "Score: 0.82"

    assert parse_llm_score(Message()) == 0.82
    assert parse_llm_score("1.5") == 1.0
    assert parse_llm_score("-0.2") == 0.0


def test_score_pair_with_experts_uses_injected_llm_invoker():
    expert = LLMExpert(
        name="general",
        prompt="Candidate: {candidate_text}\nVacancy: {vacancy_text}",
    )
    calls = []

    def invoke(_llm, prompt):
        calls.append(prompt)
        assert "Candidate text" in prompt
        assert "Title: ML Engineer" in prompt
        return '{"general_score": 0.73}'

    scores = score_pair_with_experts(
        {"text": "Candidate text"},
        {"title": "ML Engineer"},
        llm=object(),
        experts=(expert,),
        invoke=invoke,
    )

    assert scores == {"general_score": 0.73}
    assert len(calls) == 1


def test_score_feature_rows_with_experts_enriches_matching_pairs_only():
    expert = LLMExpert(name="general", prompt="{candidate_text}\n{vacancy_text}")
    rows = [
        {"candidate_id": "c1", "vacancy_id": "v1", "cosine_similarity": 0.9},
        {"candidate_id": "missing", "vacancy_id": "v1", "cosine_similarity": 0.1},
    ]

    scored = score_feature_rows_with_experts(
        rows,
        candidates=[{"id": "c1", "text": "Candidate text"}],
        vacancies=[{"dataset_id": "v1", "title": "ML Engineer"}],
        llm=object(),
        experts=(expert,),
        invoke=lambda _llm, _prompt: '{"general_score": 0.6}',
    )

    assert len(scored) == 1
    assert scored[0]["candidate_id"] == "c1"
    assert scored[0]["general_score"] == 0.6


def test_score_feature_rows_with_experts_calls_progress_callback():
    expert = LLMExpert(name="general", prompt="{candidate_text}\n{vacancy_text}")
    progress = []

    scored = score_feature_rows_with_experts(
        rows=[{"candidate_id": "c1", "vacancy_id": "v1"}],
        candidates=[{"id": "c1", "text": "Candidate text"}],
        vacancies=[{"dataset_id": "v1", "title": "ML Engineer"}],
        llm=object(),
        experts=(expert,),
        invoke=lambda _llm, _prompt: '{"general_score": 0.6}',
        progress_callback=lambda index, total, _row: progress.append((index, total)),
    )

    assert len(scored) == 1
    assert progress == [(1, 1)]


def test_score_feature_rows_with_experts_parallel_workers():
    expert = LLMExpert(name="general", prompt="{candidate_text}\n{vacancy_text}")
    rows = [
        {"candidate_id": "c1", "vacancy_id": "v1"},
        {"candidate_id": "c2", "vacancy_id": "v2"},
    ]
    candidates = [
        {"id": "c1", "text": "Candidate one"},
        {"id": "c2", "text": "Candidate two"},
    ]
    vacancies = [
        {"dataset_id": "v1", "title": "Role one"},
        {"dataset_id": "v2", "title": "Role two"},
    ]

    scored = score_feature_rows_with_experts(
        rows=rows,
        candidates=candidates,
        vacancies=vacancies,
        llm=object(),
        experts=(expert,),
        invoke=lambda _llm, _prompt: '{"general_score": 0.6}',
        max_workers=2,
    )

    assert len(scored) == 2
    assert [row["candidate_id"] for row in scored] == ["c1", "c2"]
    assert all(row["general_score"] == 0.6 for row in scored)


def test_score_feature_rows_with_experts_micro_batch_uses_single_llm_call():
    expert = LLMExpert(name="general", prompt="{candidate_text}\n{vacancy_text}")
    rows = [
        {"candidate_id": "c1", "vacancy_id": "v1"},
        {"candidate_id": "c2", "vacancy_id": "v2"},
    ]
    candidates = [
        {"id": "c1", "text": "Candidate one"},
        {"id": "c2", "text": "Candidate two"},
    ]
    vacancies = [
        {"dataset_id": "v1", "title": "Role one"},
        {"dataset_id": "v2", "title": "Role two"},
    ]

    call_count = {"value": 0}

    def invoke(_llm, _prompt):
        call_count["value"] += 1
        return (
            '{"results": ['
            '{"item_id":"0","general_score":0.6},'
            '{"item_id":"1","general_score":0.7}'
            "]}"
        )

    scored = score_feature_rows_with_experts(
        rows=rows,
        candidates=candidates,
        vacancies=vacancies,
        llm=object(),
        experts=(expert,),
        invoke=invoke,
        micro_batch_size=2,
    )

    assert len(scored) == 2
    assert call_count["value"] == 1
    assert scored[0]["general_score"] == 0.6
    assert scored[1]["general_score"] == 0.7


def test_subsample_training_negatives_limits_each_candidate_group():
    rows = [
        {"candidate_id": "c1", "label": 1, "retrieval_rank": 1},
        *[
            {"candidate_id": "c1", "label": 0, "retrieval_rank": rank}
            for rank in range(2, 12)
        ],
        {"candidate_id": "c2", "label": 1, "retrieval_rank": 1},
        {"candidate_id": "c2", "label": 0, "retrieval_rank": 2},
        {"candidate_id": "c2", "label": 0, "retrieval_rank": 3},
    ]

    sampled = subsample_training_negatives(
        rows,
        max_negatives_per_candidate=2,
        seed=123,
    )

    c1_rows = [row for row in sampled if row["candidate_id"] == "c1"]
    c2_rows = [row for row in sampled if row["candidate_id"] == "c2"]
    assert sum(1 for row in c1_rows if row["label"] == 1) == 1
    assert sum(1 for row in c1_rows if row["label"] == 0) == 2
    assert sum(1 for row in c2_rows if row["label"] == 1) == 1
    assert sum(1 for row in c2_rows if row["label"] == 0) == 2


def test_prepare_rows_for_llm_scoring_subsamples_train_and_validation():
    rows = []
    for candidate_idx in range(10):
        candidate_id = f"c{candidate_idx}"
        rows.append({"candidate_id": candidate_id, "label": 1, "retrieval_rank": 1})
        for rank in range(2, 21):
            rows.append({"candidate_id": candidate_id, "label": 0, "retrieval_rank": rank})

    prepared_rows, stats = prepare_rows_for_llm_scoring(
        rows,
        validation_fraction=0.2,
        seed=42,
        train_max_negatives_per_candidate=2,
        validation_max_negatives_per_candidate=2,
    )

    assert stats["input_rows"] == 200
    assert stats["positive_rows"] == 200
    assert stats["train_rows_for_scoring"] == 8 * 3
    assert stats["validation_rows_for_scoring"] == 2 * 3
    assert stats["total_rows_for_scoring"] == 30

    train_groups = {}
    validation_groups = {}
    for row in prepared_rows:
        target = train_groups if row["dataset_split"] == "train" else validation_groups
        target.setdefault(row["candidate_id"], []).append(row)

    assert len(train_groups) == 8
    assert len(validation_groups) == 2
    for group_rows in train_groups.values():
        assert sum(1 for row in group_rows if row["label"] == 1) == 1
        assert sum(1 for row in group_rows if row["label"] == 0) == 2
    for group_rows in validation_groups.values():
        assert sum(1 for row in group_rows if row["label"] == 1) == 1
        assert sum(1 for row in group_rows if row["label"] == 0) == 2


def test_prepare_rows_for_llm_scoring_can_keep_validation_full():
    rows = []
    for candidate_idx in range(10):
        candidate_id = f"c{candidate_idx}"
        rows.append({"candidate_id": candidate_id, "label": 1, "retrieval_rank": 1})
        for rank in range(2, 21):
            rows.append({"candidate_id": candidate_id, "label": 0, "retrieval_rank": rank})

    prepared_rows, stats = prepare_rows_for_llm_scoring(
        rows,
        validation_fraction=0.2,
        seed=42,
        train_max_negatives_per_candidate=2,
        validation_max_negatives_per_candidate=None,
    )

    assert stats["train_rows_for_scoring"] == 8 * 3
    assert stats["validation_rows_for_scoring"] == 2 * 20
    assert stats["total_rows_for_scoring"] == 64


def test_make_rate_limited_invoker_retries_with_backoff_and_delay():
    class RateLimitError(Exception):
        pass

    class DummyMessage:
        content = "0.42"

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            if self.calls < 3:
                raise RateLimitError("429 Too Many Requests")
            return DummyMessage()

    sleep_calls = []
    retry_events = []
    llm = DummyLLM()
    invoke = make_rate_limited_invoker(
        request_delay_seconds=2.0,
        max_retries=3,
        backoff_base_seconds=1.0,
        backoff_max_seconds=10.0,
        sleep_fn=sleep_calls.append,
        on_retry=lambda attempt, wait_seconds, _exc: retry_events.append((attempt, wait_seconds)),
    )

    result = invoke(llm, "prompt")

    assert result == "0.42"
    assert retry_events == [(1, 1.0), (2, 2.0)]
    assert sleep_calls == [1.0, 2.0, 2.0]
