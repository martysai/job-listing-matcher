from __future__ import annotations

import pickle
import sys
from collections.abc import Mapping, Sequence
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
    DEFAULT_LAMBDARANK_MODEL_OUTPUT_PATH,
    DEFAULT_LAMBDARANK_TRAIN_ROWS_OUTPUT_PATH,
    DEFAULT_LAMBDARANK_VALIDATION_ROWS_OUTPUT_PATH,
    DEFAULT_LAMBDARANK_VALIDATION_SCORED_OUTPUT_PATH,
    DEFAULT_RERANK_FEATURES_OUTPUT_PATH,
    DEFAULT_VACANCIES_PATH,
    EMBEDDING_MODEL,
)
from sara_retrieve_rerank.data import load_jsonl, write_jsonl
from sara_retrieve_rerank.reranking import (
    DEFAULT_LLM_SCORE_FIELDS,
    DEFAULT_RERANK_KS,
    add_weighted_score,
    evaluate_ranking_rows,
    score_rows_with_model,
    train_lambdarank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-to-vacancy retrieval.")
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_EVAL_KS))
    parser.add_argument("--misses-k", type=int, default=100)
    parser.add_argument("--persist-directory", default=None, help="Set a directory to persist Chroma. Defaults to in-memory.")
    parser.add_argument(
        "--rerank-features-path",
        default=None,
        help=f"Evaluate cached reranker feature rows instead of rebuilding Chroma. Example: {DEFAULT_RERANK_FEATURES_OUTPUT_PATH}",
    )
    parser.add_argument("--rerank-ks", nargs="+", type=int, default=list(DEFAULT_RERANK_KS))
    parser.add_argument("--weighted-score-fields", nargs="+", default=list(DEFAULT_LLM_SCORE_FIELDS))
    parser.add_argument("--train-lambdarank", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--lambdarank-train-output-path",
        default=str(DEFAULT_LAMBDARANK_TRAIN_ROWS_OUTPUT_PATH),
    )
    parser.add_argument(
        "--lambdarank-validation-output-path",
        default=str(DEFAULT_LAMBDARANK_VALIDATION_ROWS_OUTPUT_PATH),
    )
    parser.add_argument(
        "--lambdarank-scored-validation-output-path",
        default=str(DEFAULT_LAMBDARANK_VALIDATION_SCORED_OUTPUT_PATH),
    )
    parser.add_argument(
        "--lambdarank-model-output-path",
        default=str(DEFAULT_LAMBDARANK_MODEL_OUTPUT_PATH),
    )
    parser.add_argument(
        "--comparison-plot-path",
        default="outputs/rerank_method_comparison.png",
        help="Base path for comparison charts. For train mode, _all_rows and _validation suffixes are added.",
    )
    parser.add_argument(
        "--disable-comparison-plot",
        action="store_true",
        help="Skip writing comparison PNG charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rerank_features_path:
        evaluate_rerank_feature_rows(args)
        return

    from sara_retrieve_rerank.documents import create_vacancy_documents
    from sara_retrieve_rerank.evaluation import evaluate_retriever, missed_candidate_ids_at_k
    from sara_retrieve_rerank.vector_store import create_vectorstore, index_documents

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


def evaluate_rerank_feature_rows(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rerank_features_path)
    print(f"Loaded {len(rows)} reranker feature rows")

    all_rows_method_metrics: dict[str, dict[str, float | int]] = {}
    all_rows_method_metrics["cosine_similarity"] = evaluate_ranking_rows(
        rows,
        score_key="cosine_similarity",
        ks=args.rerank_ks,
    )
    _print_metrics("cosine_similarity.all_rows", all_rows_method_metrics["cosine_similarity"])
    weighted_fields = _available_fields(rows, args.weighted_score_fields)
    weighted_rows_all = rows
    if weighted_fields:
        weighted_rows_all = add_weighted_score(
            rows,
            {field: 1.0 for field in weighted_fields},
            output_key="weighted_llm_score",
        )
        all_rows_method_metrics["weighted_llm_score"] = evaluate_ranking_rows(
            weighted_rows_all,
            score_key="weighted_llm_score",
            ks=args.rerank_ks,
        )
        _print_metrics("weighted_llm_score.all_rows", all_rows_method_metrics["weighted_llm_score"])
    else:
        print("weighted_llm_score skipped: no requested LLM score fields found")
    _print_method_comparison_table(
        title="method_comparison.all_rows",
        method_metrics=all_rows_method_metrics,
        ks=args.rerank_ks,
        baseline_method="cosine_similarity",
    )

    validation_method_metrics: dict[str, dict[str, float | int]] = {}
    if not args.train_lambdarank:
        _write_comparison_plots(
            base_output_path=Path(args.comparison_plot_path),
            all_rows_method_metrics=all_rows_method_metrics,
            validation_method_metrics=validation_method_metrics,
            ks=args.rerank_ks,
            disabled=args.disable_comparison_plot,
        )
        return

    feature_fields = ["cosine_similarity", *weighted_fields]
    model, train_rows, validation_rows = train_lambdarank(
        weighted_rows_all,
        feature_fields=feature_fields,
        validation_fraction=args.validation_fraction,
        seed=args.random_seed,
    )
    scored_validation_rows = score_rows_with_model(
        validation_rows,
        model,
        feature_fields=feature_fields,
        output_key="lambdarank_score",
    )
    print(f"LambdaRank train rows: {len(train_rows)}")
    print(f"LambdaRank validation rows: {len(validation_rows)}")
    _print_split_diagnostics(train_rows=train_rows, validation_rows=validation_rows)

    validation_method_metrics["cosine_similarity"] = evaluate_ranking_rows(
        validation_rows,
        score_key="cosine_similarity",
        ks=args.rerank_ks,
    )
    _print_metrics("cosine_similarity.validation", validation_method_metrics["cosine_similarity"])

    if weighted_fields:
        weighted_validation_rows = add_weighted_score(
            validation_rows,
            {field: 1.0 for field in weighted_fields},
            output_key="weighted_llm_score",
        )
        validation_method_metrics["weighted_llm_score"] = evaluate_ranking_rows(
            weighted_validation_rows,
            score_key="weighted_llm_score",
            ks=args.rerank_ks,
        )
        _print_metrics(
            "weighted_llm_score.validation",
            validation_method_metrics["weighted_llm_score"],
        )
    validation_method_metrics["lambdarank_score"] = evaluate_ranking_rows(
        scored_validation_rows,
        score_key="lambdarank_score",
        ks=args.rerank_ks,
    )
    _print_metrics(
        "lambdarank_score.validation",
        validation_method_metrics["lambdarank_score"],
    )
    _print_method_comparison_table(
        title="method_comparison.validation",
        method_metrics=validation_method_metrics,
        ks=args.rerank_ks,
        baseline_method="cosine_similarity",
    )
    _save_lambdarank_artifacts(
        train_rows=train_rows,
        validation_rows=validation_rows,
        scored_validation_rows=scored_validation_rows,
        model=model,
        train_output_path=Path(args.lambdarank_train_output_path),
        validation_output_path=Path(args.lambdarank_validation_output_path),
        scored_validation_output_path=Path(args.lambdarank_scored_validation_output_path),
        model_output_path=Path(args.lambdarank_model_output_path),
    )
    _write_comparison_plots(
        base_output_path=Path(args.comparison_plot_path),
        all_rows_method_metrics=all_rows_method_metrics,
        validation_method_metrics=validation_method_metrics,
        ks=args.rerank_ks,
        disabled=args.disable_comparison_plot,
    )


def _available_fields(rows: list[dict], fields: list[str]) -> list[str]:
    return [field for field in fields if any(field in row for row in rows)]


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"[{name}]")
    for metric_name, value in metrics.items():
        print(metric_name, value)


def _print_method_comparison_table(
    *,
    title: str,
    method_metrics: Mapping[str, Mapping[str, float | int]],
    ks: Sequence[int],
    baseline_method: str,
) -> None:
    if not method_metrics:
        return

    metric_columns: list[str] = []
    for k in ks:
        metric_columns.append(f"recall@{k}")
        metric_columns.append(f"ndcg@{k}")
    max_k = max(ks) if ks else 0
    delta_column = f"delta_ndcg@{max_k}" if max_k else "delta"

    headers = ["method", *metric_columns, delta_column]
    baseline_metrics = method_metrics.get(baseline_method, {})
    rows: list[list[str]] = []
    for method_name, metrics in method_metrics.items():
        row = [method_name]
        for metric_name in metric_columns:
            row.append(_format_float(metrics.get(metric_name, 0.0)))
        if max_k:
            ndcg_key = f"ndcg@{max_k}"
            baseline_value = float(baseline_metrics.get(ndcg_key, 0.0))
            current_value = float(metrics.get(ndcg_key, 0.0))
            row.append(f"{current_value - baseline_value:+.4f}")
        else:
            row.append("0.0000")
        rows.append(row)

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print(f"[{title}]")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _format_float(value: float | int) -> str:
    return f"{float(value):.4f}"


def _print_split_diagnostics(*, train_rows: list[dict], validation_rows: list[dict]) -> None:
    train_groups = len({row.get("candidate_id") for row in train_rows})
    validation_groups = len({row.get("candidate_id") for row in validation_rows})
    print(
        "[diagnostics] "
        f"train_groups={train_groups}, validation_groups={validation_groups}, "
        f"train_rows_per_group={len(train_rows) / train_groups if train_groups else 0:.2f}, "
        f"validation_rows_per_group={len(validation_rows) / validation_groups if validation_groups else 0:.2f}"
    )
    if validation_groups < 20:
        print(
            "[diagnostics] validation group count is small; "
            "metric comparison can be noisy on benchmark samples."
        )


def _write_comparison_plots(
    *,
    base_output_path: Path,
    all_rows_method_metrics: Mapping[str, Mapping[str, float | int]],
    validation_method_metrics: Mapping[str, Mapping[str, float | int]],
    ks: Sequence[int],
    disabled: bool,
) -> None:
    if disabled:
        return

    all_rows_path = _with_stem_suffix(base_output_path, "all_rows")
    _save_method_comparison_plot(
        output_path=all_rows_path,
        title="Method Comparison (All Rows)",
        method_metrics=all_rows_method_metrics,
        ks=ks,
    )

    if validation_method_metrics:
        validation_path = _with_stem_suffix(base_output_path, "validation")
        _save_method_comparison_plot(
            output_path=validation_path,
            title="Method Comparison (Validation)",
            method_metrics=validation_method_metrics,
            ks=ks,
        )


def _with_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _save_method_comparison_plot(
    *,
    output_path: Path,
    title: str,
    method_metrics: Mapping[str, Mapping[str, float | int]],
    ks: Sequence[int],
) -> None:
    if not method_metrics or not ks:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks_list = list(ks)

    fig, (ax_recall, ax_ndcg) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method_name, metrics in method_metrics.items():
        recalls = [float(metrics.get(f"recall@{k}", 0.0)) for k in ks_list]
        ndcgs = [float(metrics.get(f"ndcg@{k}", 0.0)) for k in ks_list]
        ax_recall.plot(ks_list, recalls, marker="o", label=method_name)
        ax_ndcg.plot(ks_list, ndcgs, marker="o", label=method_name)

    ax_recall.set_title("Recall@K")
    ax_ndcg.set_title("NDCG@K")
    for axis in (ax_recall, ax_ndcg):
        axis.set_xlabel("K")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
        axis.set_xticks(ks_list)
    ax_ndcg.legend(loc="lower right")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def _save_lambdarank_artifacts(
    *,
    train_rows: list[dict],
    validation_rows: list[dict],
    scored_validation_rows: list[dict],
    model: object,
    train_output_path: Path,
    validation_output_path: Path,
    scored_validation_output_path: Path,
    model_output_path: Path,
) -> None:
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_rows, train_output_path)
    write_jsonl(validation_rows, validation_output_path)
    write_jsonl(scored_validation_rows, scored_validation_output_path)
    with model_output_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved LambdaRank train rows to {train_output_path}")
    print(f"Saved LambdaRank validation rows to {validation_output_path}")
    print(f"Saved LambdaRank scored validation rows to {scored_validation_output_path}")
    print(f"Saved LambdaRank model to {model_output_path}")


if __name__ == "__main__":
    main()
