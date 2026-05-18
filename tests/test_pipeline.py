"""Pipeline wrapper test using only the BM25 retriever path.

The dense retriever path requires `langchain-chroma` and sentence-transformers,
which are heavy and incompatible with offline CI. We exercise the wrapper's
BM25 mode here, plus the rerank short-circuit when no model is loaded, and
fall back to dense-mode coverage in the integration test.
"""

from __future__ import annotations

import types

from sara_retrieve_rerank.bm25_retrieval import BM25Index
from sara_retrieve_rerank.documents import create_vacancy_documents


def _make_pipeline_with_stub_chroma():
    from sara_retrieve_rerank import pipeline as pipeline_module

    vacancies = [
        {"dataset_id": "v1", "title": "ML Engineer", "tldr_sanitized": "Python PyTorch ranking."},
        {"dataset_id": "v2", "title": "Frontend Engineer", "tldr_sanitized": "React TypeScript."},
        {"dataset_id": "v3", "title": "Data Engineer", "tldr_sanitized": "Spark Airflow SQL."},
    ]
    documents = create_vacancy_documents(vacancies)

    class StubPipeline(pipeline_module.RecommendationPipeline):
        def __init__(self):
            self.vacancies = vacancies
            self.documents = documents
            self.bm25 = BM25Index(documents)
            self.vectorstore = types.SimpleNamespace()
            self.reranker_model = None

    return StubPipeline()


def test_pipeline_match_bm25_returns_match_rows():
    pipeline = _make_pipeline_with_stub_chroma()
    candidate = {"id": "c1", "text": "python pytorch ranking engineer"}

    matches = pipeline.match(candidate, k=1, retriever="bm25")

    assert len(matches) == 1
    assert matches[0]["vacancy_id"] == "v1"
    assert matches[0]["candidate_id"] == "c1"


def test_pipeline_match_skips_rerank_when_no_model():
    pipeline = _make_pipeline_with_stub_chroma()
    candidate = {"id": "c1", "text": "react typescript frontend developer"}

    matches = pipeline.match(candidate, k=2, retriever="bm25", rerank=True)

    assert matches[0]["vacancy_id"] == "v2"
    # No `lambdarank_score` is written because no model is loaded.
    assert "lambdarank_score" not in matches[0]


def test_pipeline_match_raises_for_unknown_retriever():
    pipeline = _make_pipeline_with_stub_chroma()

    try:
        pipeline.match({"id": "c1", "text": "x"}, retriever="nope")
    except ValueError as exc:
        assert "Unknown retriever" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
