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
    DEFAULT_CHROMA_DIR,
    DEFAULT_VACANCIES_PATH,
    EMBEDDING_MODEL,
)
from sara_retrieve_rerank.data import load_jsonl
from sara_retrieve_rerank.documents import create_vacancy_documents
from sara_retrieve_rerank.vector_store import create_vectorstore, index_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the vacancy Chroma index.")
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument("--persist-directory", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-reset", action="store_true", help="Do not reset the existing Chroma collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vacancies = load_jsonl(args.vacancies_path)
    print(f"Loaded {len(vacancies)} vacancies")

    vacancy_docs = create_vacancy_documents(vacancies)
    print(f"Created {len(vacancy_docs)} vacancy documents")

    vectorstore = create_vectorstore(
        embedding_model=args.embedding_model,
        persist_directory=args.persist_directory,
        reset=not args.no_reset,
    )
    index_documents(vectorstore, vacancy_docs, batch_size=args.batch_size)
    print("Vector store ready")


if __name__ == "__main__":
    main()
