from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sara_retrieve_rerank.config import (
    BATCH_SIZE,
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_EVAL_KS,
    DEFAULT_VACANCIES_PATH,
    EMBEDDING_MODEL,
)
from sara_retrieve_rerank.data import load_jsonl
from sara_retrieve_rerank.documents import create_vacancy_documents
from sara_retrieve_rerank.evaluation import evaluate_retriever, missed_candidate_ids_at_k
from sara_retrieve_rerank.vector_store import create_vectorstore, index_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-to-vacancy retrieval.")
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_EVAL_KS))
    parser.add_argument("--misses-k", type=int, default=100)
    parser.add_argument("--persist-directory", default=None, help="Set a directory to persist Chroma. Defaults to in-memory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidates = load_jsonl(args.candidates_path)
    vacancies = load_jsonl(args.vacancies_path)
    print(f"Loaded {len(candidates)} candidates")
    print(f"Loaded {len(vacancies)} vacancies")

    vacancy_docs = create_vacancy_documents(vacancies)
    vectorstore = create_vectorstore(
        embedding_model=args.embedding_model,
        persist_directory=args.persist_directory,
        reset=True,
    )
    index_documents(vectorstore, vacancy_docs, batch_size=args.batch_size)

    metrics = evaluate_retriever(candidates, vectorstore, ks=args.ks)
    for metric_name, value in metrics.items():
        print(metric_name, value)

    missed_ids = missed_candidate_ids_at_k(candidates, vectorstore, k=args.misses_k)
    print(f"Missed candidates @{args.misses_k}: {len(missed_ids)}")


if __name__ == "__main__":
    main()
