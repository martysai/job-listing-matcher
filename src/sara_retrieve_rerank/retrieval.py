from __future__ import annotations

from langchain_chroma import Chroma

from sara_retrieve_rerank.config import TOP_K
from sara_retrieve_rerank.preprocessing import candidate_to_query


def retrieve_top_vacancies(candidate: dict, vectorstore: Chroma, k: int = TOP_K) -> list[dict]:
    """Retrieve top matching vacancies for one candidate."""
    query = candidate_to_query(candidate)
    results = vectorstore.similarity_search_with_score(query, k=k)

    matches: list[dict] = []
    for rank, (doc, distance) in enumerate(results, start=1):
        matches.append(
            {
                "candidate_id": candidate.get("id"),
                "rank": rank,
                "vacancy_id": doc.metadata.get("dataset_id"),
                "title": doc.metadata.get("title"),
                "cosine_distance": float(distance),
                "cosine_similarity": float(1 - distance),
                "work_type": doc.metadata.get("work_type"),
                "work_format": doc.metadata.get("work_format"),
                "english_level": doc.metadata.get("english_level"),
                "regions": doc.metadata.get("regions"),
                "cities": doc.metadata.get("cities"),
                "specializations": doc.metadata.get("specializations"),
                "tags": doc.metadata.get("tags"),
            }
        )

    return matches


def retrieve_top_vacancy_ids(candidate: dict, vectorstore: Chroma, k: int = TOP_K) -> list[str]:
    """Retrieve only vacancy IDs for evaluation."""
    query = candidate_to_query(candidate)
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [doc.metadata["dataset_id"] for doc, _score in results]


def retrieve_all_matches(candidates: list[dict], vectorstore: Chroma, k: int = TOP_K) -> list[dict]:
    """Retrieve top-K matches for every candidate."""
    all_matches: list[dict] = []
    for candidate in candidates:
        all_matches.extend(retrieve_top_vacancies(candidate, vectorstore, k=k))
    return all_matches
