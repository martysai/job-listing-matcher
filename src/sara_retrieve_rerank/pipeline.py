"""High-level recommendation pipeline.

Wraps loading vacancies, the dense (Chroma) index, the BM25 index, and an
optional LambdaRank reranker model behind a single `match(candidate)` entry
point. The Chroma index is reused from `persist_directory` when it already
contains the documents, so the second start of the service is fast.

This module is the integration seam the UI / API layer is meant to import.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

from sara_retrieve_rerank.bm25_retrieval import BM25Index, retrieve_top_vacancies_bm25
from sara_retrieve_rerank.config import (
    BATCH_SIZE,
    DEFAULT_CHROMA_DIR,
    EMBEDDING_MODEL,
    TOP_K,
)
from sara_retrieve_rerank.data import load_jsonl
from sara_retrieve_rerank.documents import create_vacancy_documents
from sara_retrieve_rerank.hybrid_retrieval import (
    DEFAULT_RRF_K,
    retrieve_top_vacancies_hybrid,
)
from sara_retrieve_rerank.reranking import (
    DEFAULT_FEATURE_FIELDS,
    DEFAULT_FEATURE_FIELDS_WITH_BM25,
    rerank_candidate_matches,
    score_rows_with_model,
)
from sara_retrieve_rerank.retrieval import retrieve_top_vacancies
from sara_retrieve_rerank.vector_store import (
    create_vectorstore,
    index_documents,
    vectorstore_is_empty,
)


class RecommendationPipeline:
    """End-to-end recommendation entry point.

    Loads (and optionally builds) a persistent Chroma index and an in-memory
    BM25 index from the same set of vacancies, then exposes `match(candidate)`
    returning ranked vacancy match rows. If a LambdaRank model is supplied,
    matches are reordered by its predicted score before being returned.
    """

    def __init__(
        self,
        *,
        vacancies: Sequence[dict[str, Any]] | None = None,
        vacancies_path: str | Path | None = None,
        persist_directory: str | Path | None = DEFAULT_CHROMA_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE,
        reranker_model_path: str | Path | None = None,
        rebuild_index: bool = False,
    ):
        if vacancies is None:
            if vacancies_path is None:
                raise ValueError("Provide either `vacancies` or `vacancies_path`")
            vacancies = load_jsonl(vacancies_path)

        self.vacancies: list[dict[str, Any]] = list(vacancies)
        self.documents = create_vacancy_documents(self.vacancies)

        self.vectorstore = create_vectorstore(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            reset=rebuild_index,
        )
        if rebuild_index or vectorstore_is_empty(self.vectorstore):
            index_documents(self.vectorstore, self.documents, batch_size=batch_size)

        self.bm25 = BM25Index(self.documents)

        self.reranker_model: Any | None = None
        if reranker_model_path is not None:
            with Path(reranker_model_path).open("rb") as model_file:
                self.reranker_model = pickle.load(model_file)

    def match(
        self,
        candidate: dict[str, Any],
        *,
        k: int = TOP_K,
        retriever: str = "hybrid",
        candidate_pool_k: int | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        weight_dense: float = 1.0,
        weight_bm25: float = 1.0,
        rerank: bool = True,
        rerank_feature_fields: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked match rows for one candidate.

        `retriever` is one of `"dense"`, `"bm25"`, or `"hybrid"`. The reranker
        is applied only when `rerank=True` and a model is loaded; otherwise
        the raw retrieval order is preserved.
        """
        matches = self._retrieve(
            candidate,
            k=k,
            retriever=retriever,
            candidate_pool_k=candidate_pool_k,
            rrf_k=rrf_k,
            weight_dense=weight_dense,
            weight_bm25=weight_bm25,
        )
        if not rerank or self.reranker_model is None or not matches:
            return matches

        feature_fields = rerank_feature_fields
        if feature_fields is None:
            has_bm25 = any("bm25_score" in row for row in matches)
            feature_fields = (
                DEFAULT_FEATURE_FIELDS_WITH_BM25 if has_bm25 else DEFAULT_FEATURE_FIELDS
            )

        scored = score_rows_with_model(
            matches,
            self.reranker_model,
            feature_fields=feature_fields,
            output_key="lambdarank_score",
        )
        return rerank_candidate_matches(scored, score_key="lambdarank_score")

    def _retrieve(
        self,
        candidate: dict[str, Any],
        *,
        k: int,
        retriever: str,
        candidate_pool_k: int | None,
        rrf_k: int,
        weight_dense: float,
        weight_bm25: float,
    ) -> list[dict[str, Any]]:
        if retriever == "dense":
            return retrieve_top_vacancies(candidate, self.vectorstore, k=k)
        if retriever == "bm25":
            return retrieve_top_vacancies_bm25(candidate, self.bm25, k=k)
        if retriever == "hybrid":
            return retrieve_top_vacancies_hybrid(
                candidate,
                self.vectorstore,
                self.bm25,
                k=k,
                candidate_pool_k=candidate_pool_k,
                rrf_k=rrf_k,
                weight_dense=weight_dense,
                weight_bm25=weight_bm25,
            )
        raise ValueError(
            f"Unknown retriever {retriever!r}; expected one of dense, bm25, hybrid"
        )
