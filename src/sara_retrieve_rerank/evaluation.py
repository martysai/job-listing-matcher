from __future__ import annotations

from collections.abc import Sequence

from langchain_chroma import Chroma

from sara_retrieve_rerank.config import DEFAULT_EVAL_KS, TOP_K
from sara_retrieve_rerank.retrieval import retrieve_top_vacancy_ids


def candidates_with_ground_truth(candidates: Sequence[dict]) -> list[dict]:
    """Keep only candidates that have a source vacancy ID."""
    return [candidate for candidate in candidates if candidate.get("source_vacancy_id")]


def is_hit_at_k(candidate: dict, vectorstore: Chroma, k: int = TOP_K) -> bool:
    """Return True if the known source vacancy is retrieved within top-K."""
    true_vacancy_id = candidate.get("source_vacancy_id")
    if not true_vacancy_id:
        return False

    retrieved_ids = retrieve_top_vacancy_ids(candidate, vectorstore, k=k)
    return true_vacancy_id in retrieved_ids


def calculate_recall_at_k(candidates: Sequence[dict], vectorstore: Chroma, k: int = TOP_K) -> float:
    """Calculate Recall@K over candidates with ground truth labels."""
    evaluated_candidates = candidates_with_ground_truth(candidates)
    if not evaluated_candidates:
        return 0.0

    hits = sum(1 for candidate in evaluated_candidates if is_hit_at_k(candidate, vectorstore, k=k))
    return hits / len(evaluated_candidates)


def evaluate_retriever(
    candidates: Sequence[dict],
    vectorstore: Chroma,
    ks: Sequence[int] = DEFAULT_EVAL_KS,
) -> dict:
    """Evaluate Recall@K for multiple K values."""
    evaluated_candidates = candidates_with_ground_truth(candidates)
    if not evaluated_candidates:
        return {f"recall@{k}": 0.0 for k in ks} | {
            "num_candidates": 0,
            "num_hits@max_k": 0,
            "num_misses@max_k": 0,
        }

    max_k = max(ks)
    hits_by_k = {k: 0 for k in ks}

    for candidate in evaluated_candidates:
        true_vacancy_id = candidate["source_vacancy_id"]
        retrieved_ids = retrieve_top_vacancy_ids(candidate, vectorstore, k=max_k)

        if true_vacancy_id in retrieved_ids:
            rank = retrieved_ids.index(true_vacancy_id) + 1
            for k in ks:
                if rank <= k:
                    hits_by_k[k] += 1

    n = len(evaluated_candidates)
    metrics = {f"recall@{k}": hits_by_k[k] / n for k in ks}
    metrics["num_candidates"] = n
    metrics["num_hits@max_k"] = hits_by_k[max_k]
    metrics["num_misses@max_k"] = n - hits_by_k[max_k]
    return metrics


def missed_candidate_ids_at_k(
    candidates: Sequence[dict],
    vectorstore: Chroma,
    k: int = TOP_K,
) -> list:
    """Return candidate IDs whose source vacancy was not retrieved within top-K."""
    missed_ids = []

    for candidate in candidates:
        true_id = candidate.get("source_vacancy_id")
        if not true_id:
            continue

        retrieved_ids = retrieve_top_vacancy_ids(candidate, vectorstore, k=k)
        if true_id not in retrieved_ids:
            missed_ids.append(candidate.get("id"))

    return missed_ids
