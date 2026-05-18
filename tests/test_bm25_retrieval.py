from langchain_core.documents import Document

from sara_retrieve_rerank.bm25_retrieval import (
    BM25Index,
    retrieve_all_matches_bm25,
    retrieve_top_vacancies_bm25,
    tokenize,
)


def _make_documents() -> list[Document]:
    return [
        Document(
            page_content="Title: Senior ML Engineer\nDescription: Build ranking systems using Python and PyTorch.",
            metadata={"dataset_id": "v1", "title": "Senior ML Engineer", "regions": "EU"},
        ),
        Document(
            page_content="Title: Frontend Developer\nDescription: React, TypeScript, CSS, design systems.",
            metadata={"dataset_id": "v2", "title": "Frontend Developer", "regions": "US"},
        ),
        Document(
            page_content="Title: Data Engineer\nDescription: Spark, Airflow, SQL pipelines.",
            metadata={"dataset_id": "v3", "title": "Data Engineer", "regions": "EU"},
        ),
    ]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Senior ML-Engineer (PyTorch)!") == ["senior", "ml", "engineer", "pytorch"]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_bm25_index_ranks_relevant_vacancy_first():
    index = BM25Index(_make_documents())

    results = index.top_k("python pytorch ranking", k=3)

    assert [doc.metadata["dataset_id"] for doc, _ in results][0] == "v1"
    assert all(score >= 0 for _, score in results)


def test_retrieve_top_vacancies_bm25_returns_match_schema():
    index = BM25Index(_make_documents())
    candidate = {"id": "c1", "text": "Python engineer with PyTorch ranking experience"}

    matches = retrieve_top_vacancies_bm25(candidate, index, k=2)

    assert len(matches) == 2
    assert matches[0]["candidate_id"] == "c1"
    assert matches[0]["rank"] == 1
    assert matches[0]["vacancy_id"] == "v1"
    assert "bm25_score" in matches[0]
    assert matches[0]["bm25_score"] >= matches[1]["bm25_score"]


def test_retrieve_all_matches_bm25_handles_multiple_candidates():
    index = BM25Index(_make_documents())
    candidates = [
        {"id": "c1", "text": "machine learning ranking"},
        {"id": "c2", "text": "react typescript frontend"},
    ]

    matches = retrieve_all_matches_bm25(candidates, index, k=1)

    assert len(matches) == 2
    by_candidate = {row["candidate_id"]: row["vacancy_id"] for row in matches}
    assert by_candidate["c1"] == "v1"
    assert by_candidate["c2"] == "v2"


def test_bm25_index_handles_empty_documents():
    docs = [Document(page_content="", metadata={"dataset_id": "v1", "title": "Empty"})]
    index = BM25Index(docs)

    matches = retrieve_top_vacancies_bm25({"id": "c1", "text": "anything"}, index, k=5)

    assert len(matches) == 1
    assert matches[0]["vacancy_id"] == "v1"


def test_bm25_index_save_and_load_round_trip(tmp_path):
    index = BM25Index(_make_documents())
    path = tmp_path / "bm25.pkl"

    index.save(path)
    restored = BM25Index.load(path)

    candidate = {"id": "c1", "text": "spark airflow"}
    assert (
        retrieve_top_vacancies_bm25(candidate, index, k=1)[0]["vacancy_id"]
        == retrieve_top_vacancies_bm25(candidate, restored, k=1)[0]["vacancy_id"]
    )


def test_inverted_index_scoring_matches_rank_bm25():
    """The precomputed-inverted-index path must produce the same scores as
    `BM25Okapi.get_scores`, otherwise downstream metrics would silently drift."""
    import math

    index = BM25Index(_make_documents())
    for query in [
        "python pytorch ranking",
        "react typescript design",
        "spark airflow sql",
        "completely unrelated tokens",
        "python python ranking ranking ranking",  # duplicate query terms
    ]:
        expected = index.model.get_scores(tokenize(query))
        actual = index.get_scores(query)
        assert len(expected) == len(actual)
        for got, want in zip(actual, expected, strict=True):
            assert math.isclose(got, float(want), rel_tol=1e-5, abs_tol=1e-5), (
                f"score mismatch for query {query!r}: {got} vs {want}"
            )
