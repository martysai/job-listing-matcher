"""Tests for backend/services/adzuna/indexer.py — TTL contract.

Regression cover for the bug discovered during PR #17 bring-up:
``expire_at`` was being written as an ISO-8601 string, but Chroma's ``$lt``
comparator only accepts numeric operands, so TTL cleanup raised
``Expected operand value to be an int or a float for operator $lt`` on
every cycle and stale Adzuna vacancies were never evicted.

These tests pin the metadata schema to POSIX float timestamps so the
cleanup query and the upserted metadata stay in sync.
"""
from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import pytest

from adzuna.indexer import partial_reindex


class _RecordingCollection:
    """Minimal stand-in for chromadb.Collection that records calls."""

    def __init__(self) -> None:
        self.delete_calls: list[dict] = []
        self.upsert_calls: list[dict] = []

    def delete(self, *, where: dict) -> list:
        self.delete_calls.append({"where": where})
        return []

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls.append(kwargs)


def _vectorized_item(item_id: str = "v1") -> dict:
    return {
        "id":        item_id,
        "text":      "Some vacancy text",
        "embedding": [0.0, 0.1, 0.2],
        "metadata":  {"title": "DS", "city": "London"},
        "raw":       {"dataset_id": item_id, "title": "DS"},
    }


def test_ttl_cleanup_filter_uses_numeric_timestamp(tmp_path: Path) -> None:
    collection = _RecordingCollection()

    stats = partial_reindex(
        vectorized=[_vectorized_item()],
        chroma_collection=collection,
        ttl_days=7,
        adzuna_jsonl_path=tmp_path / "adzuna.jsonl",
    )

    assert stats["errors"] == 0, "TTL cleanup must not raise / be skipped"
    assert len(collection.delete_calls) == 1

    where = collection.delete_calls[0]["where"]
    clauses = where["$and"]
    expire_clause = next(c for c in clauses if "expire_at" in c)
    lt_operand = expire_clause["expire_at"]["$lt"]

    # Chroma rejects strings here — must be int/float epoch.
    assert isinstance(lt_operand, numbers.Real)
    assert not isinstance(lt_operand, bool)


def test_upserted_expire_at_is_numeric_and_in_the_future(tmp_path: Path) -> None:
    collection = _RecordingCollection()

    partial_reindex(
        vectorized=[_vectorized_item("v-future")],
        chroma_collection=collection,
        ttl_days=14,
        adzuna_jsonl_path=tmp_path / "adzuna.jsonl",
    )

    assert len(collection.upsert_calls) == 1
    metadatas = collection.upsert_calls[0]["metadatas"]
    assert len(metadatas) == 1
    md = metadatas[0]

    assert md["source"] == "adzuna"
    assert isinstance(md["expire_at"], numbers.Real)
    assert not isinstance(md["expire_at"], bool)

    # And the upsert value must agree with the cleanup filter contract:
    # both are POSIX timestamps, and the cleanup `$lt now` must be < the
    # freshly-written expire_at for any item just upserted.
    cleanup_now = collection.delete_calls[0]["where"]["$and"][1]["expire_at"]["$lt"]
    assert md["expire_at"] > cleanup_now


def test_jsonl_audit_log_is_appended(tmp_path: Path) -> None:
    jsonl = tmp_path / "adzuna.jsonl"
    collection = _RecordingCollection()

    stats = partial_reindex(
        vectorized=[_vectorized_item("v-a"), _vectorized_item("v-b")],
        chroma_collection=collection,
        ttl_days=1,
        adzuna_jsonl_path=jsonl,
    )

    assert stats["bm25_appended"] == 2
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
