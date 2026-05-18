from sara_retrieve_rerank.hybrid_retrieval import reciprocal_rank_fusion


def test_rrf_orders_by_summed_reciprocal_rank():
    dense = [
        {"vacancy_id": "v1", "rank": 1, "cosine_similarity": 0.9},
        {"vacancy_id": "v2", "rank": 2, "cosine_similarity": 0.8},
        {"vacancy_id": "v3", "rank": 3, "cosine_similarity": 0.7},
    ]
    bm25 = [
        {"vacancy_id": "v2", "rank": 1, "bm25_score": 5.0},
        {"vacancy_id": "v3", "rank": 2, "bm25_score": 3.0},
        {"vacancy_id": "v4", "rank": 3, "bm25_score": 1.0},
    ]

    fused = reciprocal_rank_fusion([dense, bm25], rrf_k=1)

    # v2 appears at rank 2 in dense and rank 1 in bm25 => 1/3 + 1/2 = 0.833
    # v1 appears only in dense at rank 1                 => 1/2 = 0.5
    # v3 appears at rank 3 in dense and rank 2 in bm25  => 1/4 + 1/3 = 0.583
    # v4 appears only in bm25 at rank 3                  => 1/4 = 0.25
    ids = [row["vacancy_id"] for row in fused]
    assert ids == ["v2", "v3", "v1", "v4"]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_preserves_metadata_from_both_sources():
    dense = [{"vacancy_id": "v1", "rank": 1, "cosine_similarity": 0.91, "title": "Engineer"}]
    bm25 = [{"vacancy_id": "v1", "rank": 1, "bm25_score": 8.0}]

    fused = reciprocal_rank_fusion([dense, bm25])

    assert fused[0]["cosine_similarity"] == 0.91
    assert fused[0]["bm25_score"] == 8.0
    assert fused[0]["title"] == "Engineer"
    assert fused[0]["source_0_rank"] == 1
    assert fused[0]["source_1_rank"] == 1


def test_rrf_weights_can_bias_one_source():
    dense = [{"vacancy_id": "v_dense_only", "rank": 1}]
    bm25 = [{"vacancy_id": "v_bm25_only", "rank": 1}]

    biased_dense = reciprocal_rank_fusion([dense, bm25], weights=[10.0, 1.0])
    biased_bm25 = reciprocal_rank_fusion([dense, bm25], weights=[1.0, 10.0])

    assert biased_dense[0]["vacancy_id"] == "v_dense_only"
    assert biased_bm25[0]["vacancy_id"] == "v_bm25_only"


def test_rrf_skips_rows_without_key_or_rank():
    dense = [
        {"vacancy_id": None, "rank": 1},
        {"vacancy_id": "v1", "rank": 0},
        {"vacancy_id": "v1", "rank": 1},
    ]
    fused = reciprocal_rank_fusion([dense])
    assert [row["vacancy_id"] for row in fused] == ["v1"]
