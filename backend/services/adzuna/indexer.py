"""Tool 2 — vectorise vacancies  |  Tool 3 — Chroma upsert + BM25 append.

vectorize_vacancies (Tool 2)
    Calls documents.vacancy_to_text() — the same function used to build the
    original Chroma index — so Adzuna vacancies land in the same region of
    the embedding space as the existing corpus.

partial_reindex (Tool 3)
    Upserts embeddings into Chroma incrementally (no full rebuild).
    Manages TTL: each Adzuna vacancy gets an expire_at timestamp and is
    pruned after TTL_DAYS days if not re-encountered.
    Appends raw vacancy dicts to a JSONL file for BM25 index rebuilds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from adzuna.config import ADZUNA_VACANCIES_JSONL, TTL_DAYS
from sara_retrieve_rerank.documents import vacancy_to_metadata, vacancy_to_text


def vectorize_vacancies(
    vacancy_dicts: list[dict],
    embedder: Any,
    batch_size: int = 64,
) -> list[dict]:
    """Tool 2 — embed vacancy dicts using the project's standard text function.

    Uses vacancy_to_text() from documents.py — identical to how the original
    dataset was indexed — so both corpora share the same embedding space and
    the Chroma retriever treats them uniformly.

    Pass the same embedder instance that is already loaded in
    RecommendationPipeline to avoid loading a second model copy.

    Returns a list of dicts with keys: id, text, embedding, metadata, raw.
    """
    texts     = [vacancy_to_text(v) for v in vacancy_dicts]
    metadatas = [vacancy_to_metadata(v) for v in vacancy_dicts]

    try:
        embeddings = embedder.embed_documents(texts, batch_size=batch_size)
    except TypeError:
        # Some LangChain embedders do not accept batch_size.
        embeddings = embedder.embed_documents(texts)

    return [
        {
            "id":        v["dataset_id"],
            "text":      text,
            "embedding": emb,
            "metadata":  meta,
            "raw":       v,
        }
        for v, text, emb, meta in zip(vacancy_dicts, texts, embeddings, metadatas)
    ]


def partial_reindex(
    vectorized:        list[dict],
    chroma_collection: Any,
    *,
    ttl_days:           int  = TTL_DAYS,
    adzuna_jsonl_path:  Path = ADZUNA_VACANCIES_JSONL,
) -> dict:
    """Tool 3 — upsert to Chroma and persist raw dicts for BM25.

    Chroma (incremental)
    --------------------
    collection.upsert() inserts new vacancies and refreshes the TTL of
    existing ones without rebuilding the index.  Expired vacancies
    (expire_at < now, source == adzuna) are deleted at the start of each run.

    BM25
    ----
    Raw vacancy dicts are appended to adzuna_jsonl_path.  At pipeline startup
    RecommendationPipeline merges this file with the original vacancy JSONL
    when building the in-memory BM25 index.  The two files stay separate so
    the original dataset is never modified.  Duplicate ids across runs are
    acceptable; the BM25 builder should deduplicate by dataset_id on load.

    Returns a stats dict: upserted, expired_removed, bm25_appended, errors.
    """
    now_iso    = datetime.utcnow().isoformat()
    expire_iso = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
    stats      = {"upserted": 0, "expired_removed": 0, "bm25_appended": 0, "errors": 0}
    log        = logging.getLogger("sara.agent")

    # 1. Prune stale Adzuna vacancies before inserting new ones.
    try:
        deleted = chroma_collection.delete(
            where={"$and": [
                {"source":    {"$eq": "adzuna"}},
                {"expire_at": {"$lt": now_iso}},
            ]}
        )
        stats["expired_removed"] = len(deleted) if isinstance(deleted, list) else 0
    except Exception as exc:  # noqa: BLE001
        log.warning("TTL cleanup failed: %s", exc)
        stats["errors"] += 1

    # 2. Upsert new / refreshed vacancies with updated expire_at.
    try:
        chroma_collection.upsert(
            ids        = [item["id"] for item in vectorized],
            embeddings = [item["embedding"] for item in vectorized],
            documents  = [item["text"] for item in vectorized],
            metadatas  = [
                {**item["metadata"], "expire_at": expire_iso, "source": "adzuna"}
                for item in vectorized
            ],
        )
        stats["upserted"] = len(vectorized)
    except Exception as exc:  # noqa: BLE001
        log.error("Chroma upsert failed: %s", exc)
        stats["errors"] += 1

    # 3. Append raw dicts so the BM25 index can include them on next startup.
    try:
        adzuna_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with adzuna_jsonl_path.open("a", encoding="utf-8") as fh:
            for item in vectorized:
                fh.write(json.dumps(item["raw"], ensure_ascii=False) + "\n")
        stats["bm25_appended"] = len(vectorized)
    except Exception as exc:  # noqa: BLE001
        log.warning("JSONL append failed: %s", exc)
        stats["errors"] += 1

    return stats
