"""BM25 sparse retrieval over vacancy documents.

Mirrors the dict schema produced by `retrieval.retrieve_top_vacancies` so the
same downstream reranker, evaluation, and fusion code can consume both
sparse (BM25) and dense (Chroma) match rows.

`BM25Index` precomputes an inverted index of (term -> [(doc_id, weight)])
where `weight` is the per-(term, doc) BM25 contribution, so scoring at
query time only touches documents that actually contain at least one query
term. On a 10k-vacancy corpus this is roughly two orders of magnitude
faster than calling `rank_bm25.BM25Okapi.get_scores` directly, which is
necessary to make per-candidate BM25 retrieval over thousands of CVs
practical.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from sara_retrieve_rerank.preprocessing import candidate_to_query

# Word-character tokenizer kept on purpose: BM25 is built on a sparse vocabulary,
# so light normalization (lowercase, alphanumeric) outperforms aggressive stemming
# for short multi-language vacancy text.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenization used by the BM25 index."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over vacancy `Document` objects.

    Wraps `rank_bm25.BM25Okapi` to get its IDF / length-normalization
    parameters, then materializes the per-(term, doc) BM25 weight as an
    inverted index. Query-time scoring is a sparse scatter-add of those
    precomputed weights, which avoids the O(corpus_size * query_tokens)
    inner loop that `BM25Okapi.get_scores` runs for every query.
    """

    def __init__(self, documents: Sequence[Document]):
        self.documents: list[Document] = list(documents)
        self._tokens: list[list[str]] = [tokenize(doc.page_content) for doc in self.documents]
        # `BM25Okapi` does not accept empty token lists and crashes on them;
        # replace truly empty tokenizations with a sentinel so missing
        # vacancy text never breaks the index.
        safe_tokens = [tokens or [""] for tokens in self._tokens]
        self.model: BM25Okapi | None = BM25Okapi(safe_tokens) if safe_tokens else None
        # term -> (doc_ids[np.int32], weights[np.float32])
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.model is not None:
            self._build_postings()

    def __len__(self) -> int:
        return len(self.documents)

    def _build_postings(self) -> None:
        """Precompute the per-(term, doc) BM25 weight as an inverted index."""
        assert self.model is not None
        k1 = self.model.k1
        b = self.model.b
        avgdl = self.model.avgdl or 1.0

        # Bucket (doc_id, weight) lists per term.
        per_term_docs: dict[str, list[int]] = {}
        per_term_weights: dict[str, list[float]] = {}
        for doc_id, freqs in enumerate(self.model.doc_freqs):
            dl = self.model.doc_len[doc_id]
            denom_base = k1 * (1 - b + b * dl / avgdl)
            for term, tf in freqs.items():
                idf = self.model.idf.get(term)
                if idf is None:
                    continue
                weight = idf * (tf * (k1 + 1)) / (tf + denom_base)
                bucket_docs = per_term_docs.get(term)
                if bucket_docs is None:
                    per_term_docs[term] = [doc_id]
                    per_term_weights[term] = [weight]
                else:
                    bucket_docs.append(doc_id)
                    per_term_weights[term].append(weight)

        self._postings = {
            term: (
                np.asarray(per_term_docs[term], dtype=np.int32),
                np.asarray(per_term_weights[term], dtype=np.float32),
            )
            for term in per_term_docs
        }

    def get_scores(self, query: str) -> list[float]:
        if self.model is None or not self.documents:
            return []
        scores = self._score_array(query)
        return scores.tolist()

    def _score_array(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not self._postings:
            return scores
        # Counter preserves rank_bm25 semantics: duplicate query tokens
        # contribute multiple times. Dedup via Counter is purely a speed win.
        for term, count in Counter(tokenize(query)).items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, weights = posting
            if count == 1:
                np.add.at(scores, doc_ids, weights)
            else:
                np.add.at(scores, doc_ids, count * weights)
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self.documents or k <= 0:
            return []
        scores = self._score_array(query)
        n = scores.shape[0]
        k = min(k, n)
        if k == n:
            ordered = np.argsort(-scores, kind="stable")
        else:
            # Use argpartition for the top-K then sort just the K winners.
            partition = np.argpartition(-scores, k - 1)[:k]
            ordered = partition[np.argsort(-scores[partition], kind="stable")]
        return [(self.documents[int(idx)], float(scores[int(idx)])) for idx in ordered]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(
                {
                    "documents": [
                        {"page_content": doc.page_content, "metadata": doc.metadata}
                        for doc in self.documents
                    ]
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload["documents"]
        ]
        return cls(documents)


def bm25_match_row(
    candidate: dict[str, Any],
    document: Document,
    *,
    rank: int,
    score: float,
) -> dict[str, Any]:
    """Format one BM25 result as a candidate-vacancy match row."""
    metadata = document.metadata or {}
    return {
        "candidate_id": candidate.get("id"),
        "rank": rank,
        "vacancy_id": metadata.get("dataset_id"),
        "title": metadata.get("title"),
        "bm25_score": float(score),
        "work_type": metadata.get("work_type"),
        "work_format": metadata.get("work_format"),
        "english_level": metadata.get("english_level"),
        "regions": metadata.get("regions"),
        "cities": metadata.get("cities"),
        "specializations": metadata.get("specializations"),
        "tags": metadata.get("tags"),
    }


def retrieve_top_vacancies_bm25(
    candidate: dict[str, Any],
    bm25: BM25Index,
    k: int = 20,
) -> list[dict[str, Any]]:
    """Return BM25 top-K matches in the same shape as `retrieve_top_vacancies`."""
    query = candidate_to_query(candidate)
    results = bm25.top_k(query, k=k)
    return [
        bm25_match_row(candidate, doc, rank=rank, score=score)
        for rank, (doc, score) in enumerate(results, start=1)
    ]


def retrieve_all_matches_bm25(
    candidates: Sequence[dict[str, Any]],
    bm25: BM25Index,
    k: int = 20,
    *,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    iterable = _wrap_progress(candidates, label="BM25 retrieval", enabled=show_progress)
    all_matches: list[dict[str, Any]] = []
    for candidate in iterable:
        all_matches.extend(retrieve_top_vacancies_bm25(candidate, bm25, k=k))
    return all_matches


def _wrap_progress(
    iterable: Sequence[Any],
    *,
    label: str,
    enabled: bool,
):
    """Return `iterable` with a tqdm progress bar if available; else a plain print fallback."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm(iterable, desc=label, total=len(iterable))
    except ImportError:
        return _PrintProgress(iterable, label=label)


class _PrintProgress:
    """Minimal progress reporter used when tqdm is not installed."""

    def __init__(self, iterable: Sequence[Any], *, label: str, every_pct: int = 5):
        self._iterable = iterable
        self._label = label
        self._every_pct = max(1, every_pct)

    def __iter__(self):
        total = len(self._iterable)
        if total == 0:
            return
        step = max(1, total * self._every_pct // 100)
        for index, item in enumerate(self._iterable, start=1):
            yield item
            if index == total or index % step == 0:
                print(f"[{self._label}] {index}/{total} ({index / total:.0%})", flush=True)
