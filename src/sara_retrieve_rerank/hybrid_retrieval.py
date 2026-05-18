"""Hybrid (dense + BM25) retrieval via reciprocal rank fusion.

Reciprocal rank fusion (Cormack et al., 2009) merges multiple ranked lists
without requiring score normalization. For a vacancy at rank `r` in a list,
its contribution is `weight / (rrf_k + r)`. RRF is robust because it depends
on *rank* not on raw score scale, so it can fuse Chroma cosine similarities
with raw BM25 scores cleanly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_chroma import Chroma

from sara_retrieve_rerank.bm25_retrieval import (
    BM25Index,
    _wrap_progress,
    retrieve_top_vacancies_bm25,
)
from sara_retrieve_rerank.config import TOP_K
from sara_retrieve_rerank.retrieval import retrieve_top_vacancies

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    key: str = "vacancy_id",
    weights: Sequence[float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists of dict rows by reciprocal rank.

    Each row must carry a `rank` field (1-indexed). Rows are grouped by `key`
    and merged: the resulting row keeps the union of fields (later lists win
    on overlap, except for the `rank` field which is replaced by the fused
    rank), plus an `rrf_score` and per-source rank fields like `<source>_rank`.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match number of ranked lists")

    fused: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []

    for list_index, (rows, weight) in enumerate(zip(ranked_lists, weights, strict=True)):
        for row in rows:
            row_key = row.get(key)
            if row_key is None:
                continue
            row_key = str(row_key)
            rank = int(row.get("rank") or 0)
            if rank <= 0:
                continue
            contribution = float(weight) / (rrf_k + rank)
            if row_key not in fused:
                merged = dict(row)
                merged["rrf_score"] = contribution
                merged[f"source_{list_index}_rank"] = rank
                fused[row_key] = merged
                order.append(row_key)
            else:
                existing = fused[row_key]
                for field, value in row.items():
                    # Preserve the original `rank` of the first list under
                    # `source_<i>_rank` so callers can audit each retriever,
                    # but allow non-conflicting fields to fill in.
                    if field == "rank":
                        continue
                    if field not in existing or existing[field] in (None, "", 0, 0.0):
                        existing[field] = value
                existing["rrf_score"] = float(existing.get("rrf_score", 0.0)) + contribution
                existing[f"source_{list_index}_rank"] = rank

    ranked = sorted(fused.values(), key=lambda row: row.get("rrf_score", 0.0), reverse=True)
    return ranked


def retrieve_top_vacancies_hybrid(
    candidate: dict[str, Any],
    vectorstore: Chroma,
    bm25: BM25Index,
    *,
    k: int = TOP_K,
    candidate_pool_k: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    weight_dense: float = 1.0,
    weight_bm25: float = 1.0,
) -> list[dict[str, Any]]:
    """Hybrid retrieval: dense top-N + BM25 top-N fused by RRF, returning top-K.

    `candidate_pool_k` controls how many candidates each base retriever
    proposes before fusion. A larger pool (default `2 * k`) widens recall;
    `k` is the final fused result size.
    """
    pool_k = candidate_pool_k if candidate_pool_k is not None else max(2 * k, k + 10)

    dense_matches = retrieve_top_vacancies(candidate, vectorstore, k=pool_k)
    bm25_matches = retrieve_top_vacancies_bm25(candidate, bm25, k=pool_k)

    fused = reciprocal_rank_fusion(
        [dense_matches, bm25_matches],
        key="vacancy_id",
        weights=[weight_dense, weight_bm25],
        rrf_k=rrf_k,
    )[:k]

    result: list[dict[str, Any]] = []
    for new_rank, row in enumerate(fused, start=1):
        out = dict(row)
        out["candidate_id"] = candidate.get("id")
        out["rank"] = new_rank
        out["dense_rank"] = out.pop("source_0_rank", None)
        out["bm25_rank"] = out.pop("source_1_rank", None)
        out.setdefault("bm25_score", 0.0)
        out.setdefault("cosine_similarity", 0.0)
        result.append(out)
    return result


def retrieve_all_matches_hybrid(
    candidates: Sequence[dict[str, Any]],
    vectorstore: Chroma,
    bm25: BM25Index,
    *,
    k: int = TOP_K,
    candidate_pool_k: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    weight_dense: float = 1.0,
    weight_bm25: float = 1.0,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    iterable = _wrap_progress(candidates, label="Hybrid retrieval", enabled=show_progress)
    all_matches: list[dict[str, Any]] = []
    for candidate in iterable:
        all_matches.extend(
            retrieve_top_vacancies_hybrid(
                candidate,
                vectorstore,
                bm25,
                k=k,
                candidate_pool_k=candidate_pool_k,
                rrf_k=rrf_k,
                weight_dense=weight_dense,
                weight_bm25=weight_bm25,
            )
        )
    return all_matches
