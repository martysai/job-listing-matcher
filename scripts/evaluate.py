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
    DEFAULT_CANDIDATES_SCHEMA_PATH,
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
from sara_retrieve_rerank.schema_features import (
    SCHEMA_FEATURE_FIELDS,
    enrich_rows_with_schema_features,
    load_candidate_schema_map,
    rows_have_schema_features,
)

BM25_FEATURE_FIELD = "bm25_score"


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
    parser.add_argument(
        "--use-bm25-feature",
        action="store_true",
        help=(
            "Include `bm25_score` as a LambdaRank feature when the rerank feature rows "
            "carry it (produced by --retriever={bm25,hybrid} in run_retrieval.py)."
        ),
    )
    parser.add_argument(
        "--candidates-schema-path",
        default=str(DEFAULT_CANDIDATES_SCHEMA_PATH),
        help=(
            "Path to candidates_with_schema.jsonl. When present and rows do not "
            "already carry schema columns, they are enriched on-the-fly so the "
            "LambdaRank ablation variants can be trained."
        ),
    )
    parser.add_argument(
        "--vacancies-path-for-schema",
        default=str(DEFAULT_VACANCIES_PATH),
        help="Path to vacancies JSONL used for on-the-fly schema enrichment.",
    )
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

    rows = _ensure_schema_features(rows, args=args)
    has_schema_features = rows_have_schema_features(rows)
    if has_schema_features:
        print(f"Schema features detected: {len(SCHEMA_FEATURE_FIELDS)} columns available")

    weighted_fields = _available_fields(rows, args.weighted_score_fields)
    weighted_rows_all = rows
    if weighted_fields:
        weighted_rows_all = add_weighted_score(
            rows,
            {field: 1.0 for field in weighted_fields},
            output_key="weighted_llm_score",
        )

    validation_method_metrics: dict[str, dict[str, float | int]] = {}
    if not args.train_lambdarank:
        # Without training, just report the cosine baseline so callers can
        # sanity-check retrieval features without LightGBM.
        metrics = evaluate_ranking_rows(
            weighted_rows_all,
            score_key="cosine_similarity",
            ks=args.rerank_ks,
        )
        _print_metrics("cosine_similarity.all_rows", metrics)
        return

    variants = _build_lambdarank_variants(
        rows=rows,
        weighted_fields=weighted_fields,
        has_schema_features=has_schema_features,
        use_bm25_feature=args.use_bm25_feature,
    )

    # Always train the main variant first because that one defines the train /
    # validation split that the other variants reuse.
    main_variant_name = "lambdarank_score"
    if main_variant_name not in variants:
        raise RuntimeError(
            f"Internal error: main variant {main_variant_name} not configured."
        )
    main_variant = variants[main_variant_name]
    print(f"[variant {main_variant_name}] features={main_variant['features']}")
    main_model, train_rows, validation_rows = train_lambdarank(
        weighted_rows_all,
        feature_fields=main_variant["features"],
        validation_fraction=args.validation_fraction,
        seed=args.random_seed,
    )
    print(f"LambdaRank train rows: {len(train_rows)}")
    print(f"LambdaRank validation rows: {len(validation_rows)}")
    _print_split_diagnostics(train_rows=train_rows, validation_rows=validation_rows)

    scored_validation_rows = score_rows_with_model(
        validation_rows,
        main_model,
        feature_fields=main_variant["features"],
        output_key=main_variant_name,
    )

    if weighted_fields:
        scored_validation_rows = add_weighted_score(
            scored_validation_rows,
            {field: 1.0 for field in weighted_fields},
            output_key="weighted_llm_score",
        )

    # Cosine baseline on the same split.
    validation_method_metrics["cosine_similarity"] = evaluate_ranking_rows(
        validation_rows,
        score_key="cosine_similarity",
        ks=args.rerank_ks,
    )
    _print_metrics("cosine_similarity.validation", validation_method_metrics["cosine_similarity"])

    if weighted_fields:
        validation_method_metrics["weighted_llm_score"] = evaluate_ranking_rows(
            scored_validation_rows,
            score_key="weighted_llm_score",
            ks=args.rerank_ks,
        )
        _print_metrics(
            "weighted_llm_score.validation",
            validation_method_metrics["weighted_llm_score"],
        )

    # Ablation variants reuse the main train/validation split for a fair
    # comparison. Each variant produces its own model and prediction column.
    ablation_models: dict[str, object] = {main_variant_name: main_model}
    for variant_name, spec in variants.items():
        if variant_name == main_variant_name:
            continue
        print(f"[variant {variant_name}] features={spec['features']}")
        ablation_model, _ablation_train, _ablation_validation = train_lambdarank(
            weighted_rows_all,
            feature_fields=spec["features"],
            validation_fraction=args.validation_fraction,
            seed=args.random_seed,
        )
        ablation_models[variant_name] = ablation_model
        scored_validation_rows = score_rows_with_model(
            scored_validation_rows,
            ablation_model,
            feature_fields=spec["features"],
            output_key=variant_name,
        )
        validation_method_metrics[variant_name] = evaluate_ranking_rows(
            scored_validation_rows,
            score_key=variant_name,
            ks=args.rerank_ks,
        )
        _print_metrics(f"{variant_name}.validation", validation_method_metrics[variant_name])

    # Always show the main lambdarank metric last in the per-method block to
    # keep it visually distinct after the ablations.
    validation_method_metrics[main_variant_name] = evaluate_ranking_rows(
        scored_validation_rows,
        score_key=main_variant_name,
        ks=args.rerank_ks,
    )
    _print_metrics(
        f"{main_variant_name}.validation",
        validation_method_metrics[main_variant_name],
    )

    ordered_validation_metrics = _order_method_metrics(
        validation_method_metrics,
        preferred_order=_validation_method_order(
            include_weighted=bool(weighted_fields),
            ablation_variant_names=[
                name for name in variants if name != main_variant_name
            ],
        ),
    )
    _print_method_comparison_table(
        title="method_comparison.validation",
        method_metrics=ordered_validation_metrics,
        ks=args.rerank_ks,
        baseline_method="cosine_similarity",
    )
    _save_lambdarank_artifacts(
        train_rows=train_rows,
        validation_rows=validation_rows,
        scored_validation_rows=scored_validation_rows,
        model=main_model,
        train_output_path=Path(args.lambdarank_train_output_path),
        validation_output_path=Path(args.lambdarank_validation_output_path),
        scored_validation_output_path=Path(args.lambdarank_scored_validation_output_path),
        model_output_path=Path(args.lambdarank_model_output_path),
    )
    _write_comparison_plots(
        base_output_path=Path(args.comparison_plot_path),
        validation_method_metrics=ordered_validation_metrics,
        ks=args.rerank_ks,
        disabled=args.disable_comparison_plot,
    )


def _available_fields(rows: list[dict], fields: list[str]) -> list[str]:
    return [field for field in fields if any(field in row for row in rows)]


def _ensure_schema_features(
    rows: list[dict],
    *,
    args: argparse.Namespace,
) -> list[dict]:
    """If rows lack schema features, enrich them in-place when sources exist."""
    if not rows or rows_have_schema_features(rows):
        return rows
    schema_path = getattr(args, "candidates_schema_path", None)
    vacancies_path = getattr(args, "vacancies_path_for_schema", None)
    if not schema_path or not vacancies_path:
        return rows
    candidate_schema_by_id = load_candidate_schema_map(schema_path)
    if not candidate_schema_by_id:
        return rows
    if not Path(vacancies_path).exists():
        print(
            f"Schema features skipped: vacancies path {vacancies_path} not found"
        )
        return rows
    vacancies = load_jsonl(vacancies_path)
    vacancies_by_id = {
        str(vacancy.get("dataset_id") or vacancy.get("id")): vacancy
        for vacancy in vacancies
    }
    print(
        f"Enriching {len(rows)} rows with schema features from {schema_path}"
    )
    return enrich_rows_with_schema_features(
        rows,
        candidate_schema_by_id=candidate_schema_by_id,
        vacancies_by_id=vacancies_by_id,
    )


def _build_lambdarank_variants(
    *,
    rows: list[dict],
    weighted_fields: list[str],
    has_schema_features: bool,
    use_bm25_feature: bool,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return the ablation variants to train.

    Each entry is `{variant_name: {"features": tuple_of_field_names}}`. The
    `lambdarank_score` variant is the main model and is always trained first
    (in the caller). `_no_llm_expert` and `_no_schema` are ablations.
    """
    has_bm25 = any(BM25_FEATURE_FIELD in row for row in rows)
    base = ["cosine_similarity"]
    if has_bm25 and use_bm25_feature:
        base.append(BM25_FEATURE_FIELD)
    elif has_bm25 and not use_bm25_feature:
        print(
            f"[features] rows carry {BM25_FEATURE_FIELD} but --use-bm25-feature "
            "is not set; LambdaRank will ignore it."
        )

    schema_fields = list(SCHEMA_FEATURE_FIELDS) if has_schema_features else []

    variants: dict[str, dict[str, tuple[str, ...]]] = {}

    # Main variant: cosine + bm25 (optional) + LLM experts + schema features.
    main_features = tuple(base + list(weighted_fields) + schema_fields)
    variants["lambdarank_score"] = {"features": main_features}

    # Ablation: drop LLM experts.
    if weighted_fields:
        no_llm_features = tuple(base + schema_fields)
        if no_llm_features != main_features and len(no_llm_features) >= 1:
            variants["lambdarank_no_llm_expert"] = {"features": no_llm_features}

    # Ablation: drop schema features.
    if schema_fields:
        no_schema_features = tuple(base + list(weighted_fields))
        if no_schema_features != main_features and len(no_schema_features) >= 1:
            variants["lambdarank_no_schema"] = {"features": no_schema_features}

    return variants


def _validation_method_order(
    *,
    include_weighted: bool,
    ablation_variant_names: list[str],
) -> list[str]:
    """Return the column ordering for the validation comparison table.

    `lambdarank_score` is intentionally last as requested by the user so the
    main model anchors the right side of the table.
    """
    order: list[str] = ["cosine_similarity"]
    if include_weighted:
        order.append("weighted_llm_score")
    order.extend(ablation_variant_names)
    order.append("lambdarank_score")
    return order


def _order_method_metrics(
    method_metrics: dict[str, dict[str, float | int]],
    *,
    preferred_order: list[str],
) -> dict[str, dict[str, float | int]]:
    ordered: dict[str, dict[str, float | int]] = {}
    for method in preferred_order:
        if method in method_metrics:
            ordered[method] = method_metrics[method]
    for method, metrics in method_metrics.items():
        if method not in ordered:
            ordered[method] = metrics
    return ordered


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
    validation_method_metrics: Mapping[str, Mapping[str, float | int]],
    ks: Sequence[int],
    disabled: bool,
) -> None:
    if disabled or not validation_method_metrics:
        return

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
