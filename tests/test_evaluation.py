from sara_retrieve_rerank.evaluation import evaluate_matches_retriever


def test_evaluate_matches_retriever_reports_recall_and_counts():
    candidates = [
        {"id": "c1", "source_vacancy_id": "v2"},
        {"id": "c2", "source_vacancy_id": "v4"},
        {"id": "c3", "source_vacancy_id": "v9"},
    ]
    matches = [
        {"candidate_id": "c1", "vacancy_id": "v1", "rank": 1},
        {"candidate_id": "c1", "vacancy_id": "v2", "rank": 2},
        {"candidate_id": "c2", "vacancy_id": "v4", "rank": 1},
    ]

    metrics = evaluate_matches_retriever(candidates, matches, ks=(1, 2, 3))

    assert metrics["recall@1"] == 1 / 3
    assert metrics["recall@2"] == 2 / 3
    assert metrics["recall@3"] == 2 / 3
    assert metrics["num_candidates"] == 3
    assert metrics["num_hits@max_k"] == 2
    assert metrics["num_misses@max_k"] == 1


def test_evaluate_matches_retriever_ignores_candidates_without_ground_truth():
    candidates = [
        {"id": "c1", "source_vacancy_id": "v1"},
        {"id": "c2", "source_vacancy_id": None},
    ]
    matches = [{"candidate_id": "c1", "vacancy_id": "v1", "rank": 1}]

    metrics = evaluate_matches_retriever(candidates, matches, ks=(1,))

    assert metrics["recall@1"] == 1.0
    assert metrics["num_candidates"] == 1
    assert metrics["num_hits@max_k"] == 1
    assert metrics["num_misses@max_k"] == 0


def test_evaluate_matches_retriever_handles_empty_k_list():
    candidates = [{"id": "c1", "source_vacancy_id": "v1"}]
    matches = [{"candidate_id": "c1", "vacancy_id": "v1", "rank": 1}]

    metrics = evaluate_matches_retriever(candidates, matches, ks=())

    assert metrics == {
        "num_candidates": 1,
        "num_hits@max_k": 0,
        "num_misses@max_k": 1,
    }
