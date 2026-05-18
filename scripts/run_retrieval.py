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
    DEFAULT_CANDIDATES_SCHEMA_PATH,
    DEFAULT_MATCHES_OUTPUT_PATH,
    DEFAULT_RERANK_FEATURES_OUTPUT_PATH,
    DEFAULT_VACANCIES_PATH,
    EMBEDDING_MODEL,
    TOP_K,
)
from sara_retrieve_rerank.bm25_retrieval import BM25Index, retrieve_all_matches_bm25
from sara_retrieve_rerank.data import load_jsonl, write_jsonl
from sara_retrieve_rerank.documents import create_vacancy_documents
from sara_retrieve_rerank.hybrid_retrieval import (
    DEFAULT_RRF_K,
    retrieve_all_matches_hybrid,
)
from sara_retrieve_rerank.retrieval import retrieve_all_matches
from sara_retrieve_rerank.reranking import build_pair_feature_rows
from sara_retrieve_rerank.schema_features import (
    enrich_rows_with_schema_features,
    load_candidate_schema_map,
)
from sara_retrieve_rerank.vector_store import create_vectorstore, index_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-to-vacancy retrieval.")
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument(
        "--candidates-schema-path",
        default=str(DEFAULT_CANDIDATES_SCHEMA_PATH),
        help=(
            "Path to candidates_with_schema.jsonl (LLM-parsed candidate fields). "
            "When --feature-output-path is used, the saved feature rows are "
            "enriched with schema-based tabular features for the reranker. "
            "Missing file is tolerated."
        ),
    )
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_MATCHES_OUTPUT_PATH))
    parser.add_argument(
        "--feature-output-path",
        default=None,
        help=f"Optionally save labeled reranker feature rows. Example: {DEFAULT_RERANK_FEATURES_OUTPUT_PATH}",
    )
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--persist-directory", default=None, help="Set a directory to persist Chroma. Defaults to in-memory.")
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
        help="Which retriever to run. 'hybrid' fuses dense + BM25 via reciprocal rank fusion.",
    )
    parser.add_argument(
        "--hybrid-pool-k",
        type=int,
        default=None,
        help="Number of candidates each base retriever returns before fusion (default: 2 * top_k).",
    )
    parser.add_argument(
        "--hybrid-rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant. Smaller values favor top-ranked candidates more aggressively.",
    )
    parser.add_argument(
        "--hybrid-weight-dense",
        type=float,
        default=1.0,
        help="Weight on the dense retriever during RRF fusion.",
    )
    parser.add_argument(
        "--hybrid-weight-bm25",
        type=float,
        default=1.0,
        help="Weight on the BM25 retriever during RRF fusion.",
    )
    parser.add_argument(
        "--report-recall-ks",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optionally print Recall@K from the produced matches. "
            "For meaningful values, keep each K <= --top-k."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidates = load_jsonl(args.candidates_path)
    vacancies = load_jsonl(args.vacancies_path)
    print(f"Loaded {len(candidates)} candidates")
    print(f"Loaded {len(vacancies)} vacancies")

    vacancy_docs = create_vacancy_documents(vacancies)

    vectorstore = None
    bm25_index = None
    if args.retriever in {"dense", "hybrid"}:
        vectorstore = create_vectorstore(
            embedding_model=args.embedding_model,
            persist_directory=args.persist_directory,
            reset=True,
        )
        index_documents(vectorstore, vacancy_docs, batch_size=args.batch_size)
    if args.retriever in {"bm25", "hybrid"}:
        bm25_index = BM25Index(vacancy_docs)
        print(f"Built BM25 index over {len(bm25_index)} vacancies")

    if args.retriever == "dense":
        assert vectorstore is not None
        matches = retrieve_all_matches(
            candidates, vectorstore, k=args.top_k, show_progress=True
        )
    elif args.retriever == "bm25":
        assert bm25_index is not None
        matches = retrieve_all_matches_bm25(
            candidates, bm25_index, k=args.top_k, show_progress=True
        )
    else:
        assert vectorstore is not None and bm25_index is not None
        matches = retrieve_all_matches_hybrid(
            candidates,
            vectorstore,
            bm25_index,
            k=args.top_k,
            candidate_pool_k=args.hybrid_pool_k,
            rrf_k=args.hybrid_rrf_k,
            weight_dense=args.hybrid_weight_dense,
            weight_bm25=args.hybrid_weight_bm25,
            show_progress=True,
        )

    write_jsonl(matches, args.output_path)
    print(f"Saved {len(matches)} matches ({args.retriever}) to {args.output_path}")

    if args.report_recall_ks:
        from sara_retrieve_rerank.evaluation import evaluate_matches_retriever

        ks = sorted(set(args.report_recall_ks))
        if any(k <= 0 for k in ks):
            raise ValueError("--report-recall-ks values must be positive integers")
        if any(k > args.top_k for k in ks):
            print(
                "Warning: some requested Recall@K values are greater than --top-k; "
                "those metrics are capped by available retrieved matches."
            )
        metrics = evaluate_matches_retriever(candidates, matches, ks=ks)
        print("[retrieval.recall_from_matches]")
        for k in ks:
            print(f"recall@{k}", metrics[f"recall@{k}"])
        print("num_candidates", metrics["num_candidates"])
        max_k = max(ks)
        print(f"num_hits@{max_k}", metrics["num_hits@max_k"])
        print(f"num_misses@{max_k}", metrics["num_misses@max_k"])

    if args.feature_output_path:
        feature_rows = build_pair_feature_rows(candidates, matches)
        candidate_schema_by_id = load_candidate_schema_map(args.candidates_schema_path)
        if candidate_schema_by_id:
            vacancies_by_id = {
                str(vacancy.get("dataset_id") or vacancy.get("id")): vacancy
                for vacancy in vacancies
            }
            feature_rows = enrich_rows_with_schema_features(
                feature_rows,
                candidate_schema_by_id=candidate_schema_by_id,
                vacancies_by_id=vacancies_by_id,
            )
            print(
                f"Enriched feature rows with schema features from {args.candidates_schema_path}"
            )
        else:
            print(
                f"No candidate schema found at {args.candidates_schema_path}; "
                "skipping schema feature enrichment."
            )
        write_jsonl(feature_rows, args.feature_output_path)
        print(f"Saved {len(feature_rows)} reranker feature rows to {args.feature_output_path}")


if __name__ == "__main__":
    main()
