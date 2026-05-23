from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "backend" / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

import argparse

from sara_retrieve_rerank.config import (
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_CANDIDATES_SCHEMA_PATH,
    DEFAULT_MATCHES_OUTPUT_PATH,
    DEFAULT_RERANK_FEATURES_OUTPUT_PATH,
    DEFAULT_SCORED_RERANK_FEATURES_OUTPUT_PATH,
    DEFAULT_VACANCIES_PATH,
    LLM_PROVIDER,
    LLM_MODEL,
    RERANK_SCORING_VALIDATION_FRACTION,
    RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE,
    RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE,
)
from sara_retrieve_rerank.data import load_jsonl, write_jsonl
from sara_retrieve_rerank.llm_logging import (
    setup_llm_jsonl_logger,
    wrap_invoke_with_logging,
)
from sara_retrieve_rerank.reranking import (
    DEFAULT_LLM_SCORE_FIELDS,
    build_pair_feature_rows,
    make_rate_limited_invoker,
    prepare_rows_for_llm_scoring,
    score_feature_rows_with_experts,
)
from sara_retrieve_rerank.schema_features import (
    enrich_rows_with_schema_features,
    load_candidate_schema_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add LLM expert scores to reranker feature rows.")
    parser.add_argument("--features-path", default=str(DEFAULT_RERANK_FEATURES_OUTPUT_PATH))
    parser.add_argument("--matches-path", default=str(DEFAULT_MATCHES_OUTPUT_PATH))
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument(
        "--candidates-schema-path",
        default=str(DEFAULT_CANDIDATES_SCHEMA_PATH),
        help=(
            "Path to candidates_with_schema.jsonl. When present, scored rows "
            "are enriched with schema-based tabular reranker features."
        ),
    )
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_SCORED_RERANK_FEATURES_OUTPUT_PATH))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=os.getenv("RERANK_LLM_MODEL", LLM_MODEL))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--benchmark-user-fraction", type=float, default=None)
    parser.add_argument("--benchmark-seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Persist partial progress every N newly scored rows (0 disables checkpoints). "
            "Useful for resuming with --rewrite-mode after interruption."
        ),
    )
    parser.add_argument(
        "--rewrite-mode",
        action="store_true",
        help=(
            "Reuse already-scored candidate/vacancy pairs from --output-path and "
            "score only missing pairs."
        ),
    )
    parser.add_argument(
        "--llm-log-path",
        default=os.getenv("RERANK_LLM_LOG_PATH"),
        help=(
            "When set, write a JSONL audit trail of every LLM call (prompt, "
            "response, latency, errors) to this path."
        ),
    )
    return parser.parse_args()


def main() -> None:
    _load_dotenv(ROOT / ".env")
    args = parse_args()

    candidates = load_jsonl(args.candidates_path)
    rows = load_or_create_feature_rows(
        features_path=Path(args.features_path),
        matches_path=Path(args.matches_path),
        candidates=candidates,
    )
    vacancies = load_jsonl(args.vacancies_path)
    candidate_schema_by_id = load_candidate_schema_map(args.candidates_schema_path)
    if candidate_schema_by_id:
        vacancies_by_id = {
            str(vacancy.get("dataset_id") or vacancy.get("id")): vacancy
            for vacancy in vacancies
        }
        rows = enrich_rows_with_schema_features(
            rows,
            candidate_schema_by_id=candidate_schema_by_id,
            vacancies_by_id=vacancies_by_id,
        )
        print(
            f"Enriched {len(rows)} feature rows with schema features "
            f"from {args.candidates_schema_path}"
        )
    else:
        print(
            f"Schema features skipped: no candidate schema found at {args.candidates_schema_path}"
        )
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    benchmark_user_fraction = (
        args.benchmark_user_fraction
        if args.benchmark_user_fraction is not None
        else _env_float("RERANK_BENCHMARK_USER_FRACTION", 0.05)
    )
    if args.benchmark_mode:
        rows, sampled_candidate_count, all_candidate_count = _sample_rows_by_candidate_fraction(
            rows,
            user_fraction=benchmark_user_fraction,
            seed=args.benchmark_seed,
        )
        print(
            "Benchmark mode enabled: "
            f"candidate_groups={sampled_candidate_count}/{all_candidate_count} "
            f"({benchmark_user_fraction:.2%}), rows={len(rows)}"
        )
        if args.output_path == str(DEFAULT_SCORED_RERANK_FEATURES_OUTPUT_PATH):
            benchmark_output_path = Path(args.output_path)
            args.output_path = str(
                benchmark_output_path.with_name(
                    f"{benchmark_output_path.stem}_benchmark{benchmark_output_path.suffix}"
                )
            )
            print(f"Benchmark output path: {args.output_path}")

    request_delay_seconds = _env_float("RERANK_REQUEST_DELAY_SECONDS", 2.0)
    max_retries = _env_int("RERANK_RATE_LIMIT_MAX_RETRIES", 6)
    backoff_base_seconds = _env_float("RERANK_RATE_LIMIT_BACKOFF_BASE_SECONDS", 2.0)
    backoff_max_seconds = _env_float("RERANK_RATE_LIMIT_BACKOFF_MAX_SECONDS", 60.0)
    max_workers = (
        args.max_concurrency
        if args.max_concurrency is not None
        else _env_int("RERANK_MAX_CONCURRENCY", _env_int("RERANK_MAX_WORKERS", 1))
    )
    if max_workers < 1:
        raise ValueError("max concurrency must be >= 1")
    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else _env_int("RERANK_MICRO_BATCH_SIZE", 1)
    )
    if micro_batch_size < 1:
        raise ValueError("micro batch size must be >= 1")
    scoring_validation_fraction = _env_float(
        "RERANK_SCORING_VALIDATION_FRACTION",
        RERANK_SCORING_VALIDATION_FRACTION,
    )
    train_max_negatives_per_candidate = _env_optional_int(
        "RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE",
        RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE,
    )
    validation_max_negatives_per_candidate = _env_optional_int(
        "RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE",
        RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE,
    )
    provider = args.provider or os.getenv("RERANK_LLM_PROVIDER", LLM_PROVIDER)
    api_base = args.api_base if args.api_base is not None else os.getenv("RERANK_LLM_API_BASE")
    resolved_model = resolve_model_for_provider(provider, args.model)

    rows, scoring_stats = prepare_rows_for_llm_scoring(
        rows,
        validation_fraction=scoring_validation_fraction,
        seed=42,
        train_max_negatives_per_candidate=train_max_negatives_per_candidate,
        validation_max_negatives_per_candidate=validation_max_negatives_per_candidate,
    )
    print(
        "LLM scoring rows prepared: "
        f"input={scoring_stats['input_rows']}, positive={scoring_stats['positive_rows']}, "
        f"train_scored={scoring_stats['train_rows_for_scoring']}, "
        f"validation_scored={scoring_stats['validation_rows_for_scoring']}, "
        f"total_scored={scoring_stats['total_rows_for_scoring']}"
    )

    print(f"Loaded {len(rows)} rows to score")
    print(
        "LLM scoring negative sampling: "
        f"train_max_negatives={train_max_negatives_per_candidate}, "
        f"validation_max_negatives={validation_max_negatives_per_candidate}"
    )
    print(f"LLM provider: {provider}")
    print(f"LLM model: {resolved_model}")
    if api_base:
        print(f"LLM API base: {api_base}")
    print(f"LLM request delay: {request_delay_seconds:.2f}s")
    print(
        "Rate-limit retries: "
        f"max_retries={max_retries}, backoff_base={backoff_base_seconds:.2f}s, backoff_max={backoff_max_seconds:.2f}s"
    )
    print(f"Scoring concurrency: {max_workers}")
    if max_workers > 1:
        print("Warning: multiple workers can increase provider rate-limit errors.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")

    prefilled_rows: list[dict] | None = None
    missing_positions: list[int] = []
    rows_for_scoring = rows
    rewrite_base_rows: list[dict] | None = None
    output_path = Path(args.output_path)
    _validate_checkpoint_mode(
        checkpoint_every=args.checkpoint_every,
        rewrite_mode=args.rewrite_mode,
        output_path=output_path,
    )
    if args.rewrite_mode:
        scorable_rows = _filter_rows_with_known_pairs(rows, candidates, vacancies)
        skipped_rows = len(rows) - len(scorable_rows)
        if skipped_rows > 0:
            print(
                "Rewrite mode: "
                f"skipping {skipped_rows} rows with missing candidate/vacancy records."
            )
        rewrite_base_rows = []
        if output_path.exists():
            cached_rows = load_jsonl(output_path)
            rewrite_base_rows = [dict(row) for row in cached_rows]
            (
                prefilled_rows,
                rows_for_scoring,
                missing_positions,
                cache_stats,
            ) = _prepare_rewrite_rows(
                rows=scorable_rows,
                cached_rows=cached_rows,
                score_fields=DEFAULT_LLM_SCORE_FIELDS,
            )
            print(
                "Rewrite cache: "
                f"source_rows={cache_stats['cached_source_rows']}, "
                f"cache_pairs={cache_stats['cache_pairs']}, "
                f"reused_rows={cache_stats['reused_rows']}, "
                f"rows_to_score={cache_stats['rows_to_score']}"
            )
            print(
                "Rewrite mode addition: "
                f"preserving existing {len(rewrite_base_rows)} rows in output."
            )
        else:
            rows_for_scoring = scorable_rows
            print(
                "Rewrite mode: cache file not found at "
                f"{output_path}; scoring all {len(rows_for_scoring)} eligible rows."
            )

    if args.checkpoint_every > 0:
        print(
            "Checkpointing enabled: "
            f"every {args.checkpoint_every} newly scored rows -> {output_path}"
        )

    if not rows_for_scoring:
        if rewrite_base_rows is not None:
            scored_rows = rewrite_base_rows
        elif prefilled_rows is None:
            scored_rows: list[dict] = []
        else:
            scored_rows = prefilled_rows
        _write_jsonl_atomic(scored_rows, output_path)
        print(f"Saved {len(scored_rows)} scored reranker feature rows to {args.output_path}")
        return

    llm = create_llm(model=resolved_model, api_base=api_base)
    invoke = make_rate_limited_invoker(
        request_delay_seconds=request_delay_seconds,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
        on_retry=_on_rate_limit_retry,
    )
    if args.llm_log_path:
        llm_logger = setup_llm_jsonl_logger(args.llm_log_path)
        invoke = wrap_invoke_with_logging(
            invoke,
            logger=llm_logger,
            component="rerank_experts",
        )
        print(f"LLM call audit trail: {args.llm_log_path}")
    micro_batch_autotune = _env_bool(
        "RERANK_MICRO_BATCH_AUTOTUNE",
        default=(provider.strip().lower() == "ollama" and micro_batch_size > 1),
    )
    if micro_batch_autotune and micro_batch_size > 1 and rows_for_scoring:
        tune_rows = _env_int("RERANK_MICRO_BATCH_TUNE_ROWS", 24)
        tuned_micro_batch_size = autotune_micro_batch_size(
            rows=rows_for_scoring,
            candidates=candidates,
            vacancies=vacancies,
            llm=llm,
            invoke=invoke,
            requested_micro_batch_size=micro_batch_size,
            tune_rows=tune_rows,
        )
        if tuned_micro_batch_size != micro_batch_size:
            print(
                f"Micro-batch autotune selected {tuned_micro_batch_size} "
                f"(requested {micro_batch_size})"
            )
            micro_batch_size = tuned_micro_batch_size
    print(f"Scoring micro-batch size: {micro_batch_size}")

    captured_scored_rows: list[dict] = []

    def _progress_with_checkpoint(index: int, total: int, scored_row: dict) -> None:
        _print_progress(index, total, scored_row)
        captured_scored_rows.append(dict(scored_row))
        if prefilled_rows is not None:
            if index > len(missing_positions):
                raise RuntimeError(
                    "Rewrite cache merge failed: progress index exceeds missing positions "
                    f"({index} > {len(missing_positions)})."
                )
            prefilled_rows[missing_positions[index - 1]] = dict(scored_row)
        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            _save_checkpoint(
                output_path=output_path,
                captured_scored_rows=captured_scored_rows,
                prefilled_rows=prefilled_rows,
                rewrite_base_rows=rewrite_base_rows,
                score_fields=DEFAULT_LLM_SCORE_FIELDS,
            )

    try:
        scored_subset = score_feature_rows_with_experts(
            rows_for_scoring,
            candidates,
            vacancies,
            llm,
            invoke=invoke,
            progress_callback=_progress_with_checkpoint,
            max_workers=max_workers,
            micro_batch_size=micro_batch_size,
        )
    except KeyboardInterrupt:
        if captured_scored_rows:
            _save_checkpoint(
                output_path=output_path,
                captured_scored_rows=captured_scored_rows,
                prefilled_rows=prefilled_rows,
                rewrite_base_rows=rewrite_base_rows,
                score_fields=DEFAULT_LLM_SCORE_FIELDS,
            )
            print(
                "Interrupted: saved partial scored rows to "
                f"{output_path} ({len(captured_scored_rows)} newly scored this run)."
            )
        raise

    if rewrite_base_rows is not None:
        scored_rows = _merge_rows_additive(rewrite_base_rows, scored_subset)
    elif prefilled_rows is None:
        scored_rows = scored_subset
    else:
        if len(scored_subset) != len(missing_positions):
            raise RuntimeError(
                "Rewrite cache merge failed: scored row count does not match missing positions "
                f"({len(scored_subset)} vs {len(missing_positions)})."
            )
        for index, scored_row in enumerate(scored_subset):
            prefilled_rows[missing_positions[index]] = scored_row
        scored_rows = prefilled_rows

    _write_jsonl_atomic(scored_rows, output_path)
    print(f"Saved {len(scored_rows)} scored reranker feature rows to {args.output_path}")


def create_llm(*, model: str, api_base: str | None):
    """Build the LangChain runnable used by the rerank scoring loop.

    Routes through ``llm_router.make_chat_model`` so primary→fallback failover
    is wired in automatically (GitHub Models → Anthropic → Mistral by
    default).  The ``model`` argument is retained for backward compatibility
    with the CLI; we use ``llm_router.config.tier_for_model_string`` to map
    it to a router tier.
    """
    from llm_router import make_chat_model
    from llm_router.config import tier_for_model_string

    return make_chat_model(
        tier=tier_for_model_string(model), temperature=0, api_base=api_base
    )


def resolve_model_for_provider(provider: str, model: str) -> str:
    normalized_provider = (provider or "").strip().lower()
    normalized_model = (model or "").strip()
    if not normalized_model:
        raise ValueError("model must be a non-empty string")
    if normalized_provider == "ollama" and not normalized_model.startswith(
        ("ollama/", "ollama_chat/")
    ):
        return f"ollama_chat/{normalized_model}"
    return normalized_model


def load_or_create_feature_rows(
    *,
    features_path: Path,
    matches_path: Path,
    candidates: list[dict],
) -> list[dict]:
    if features_path.exists():
        return load_jsonl(features_path)

    if matches_path.exists():
        matches = load_jsonl(matches_path)
        rows = build_pair_feature_rows(candidates, matches)
        write_jsonl(rows, features_path)
        print(
            "Feature rows were missing. Built and saved "
            f"{len(rows)} rows to {features_path} from {matches_path}."
        )
        return rows

    legacy_features_path = Path("data/processed/rerank_features_top100.jsonl")
    if legacy_features_path.exists():
        print(
            f"Using legacy feature rows at {legacy_features_path} "
            f"(default path {features_path} not found)."
        )
        return load_jsonl(legacy_features_path)

    legacy_matches_path = Path("data/processed/candidate_vacancy_matches_top100.jsonl")
    if not legacy_matches_path.exists():
        raise FileNotFoundError(
            f"Feature rows not found at '{features_path}' and matches not found at '{matches_path}'. "
            "Run run_retrieval.py first or pass --matches-path to an existing matches JSONL."
        )

    print(
        f"Using legacy matches at {legacy_matches_path} "
        f"(default path {matches_path} not found)."
    )
    matches = load_jsonl(legacy_matches_path)
    rows = build_pair_feature_rows(candidates, matches)
    write_jsonl(rows, features_path)
    print(
        "Feature rows were missing. Built and saved "
        f"{len(rows)} rows to {features_path} from {legacy_matches_path}."
    )
    return rows


def _print_progress(index: int, total: int, row: dict) -> None:
    if index == 1 or index == total or index % 25 == 0:
        print(
            f"[progress] {index}/{total} rows scored "
            f"(candidate_id={row.get('candidate_id')}, vacancy_id={row.get('vacancy_id')})"
        )


def _pair_key(row: dict) -> tuple[str, str] | None:
    candidate_id = row.get("candidate_id")
    vacancy_id = row.get("vacancy_id")
    if candidate_id is None or vacancy_id is None:
        return None
    return str(candidate_id), str(vacancy_id)


def _has_score_fields(row: dict, score_fields: tuple[str, ...]) -> bool:
    for field in score_fields:
        if field not in row:
            return False
        try:
            float(row[field])
        except (TypeError, ValueError):
            return False
    return True


def _prepare_rewrite_rows(
    *,
    rows: list[dict],
    cached_rows: list[dict],
    score_fields: tuple[str, ...],
) -> tuple[list[dict], list[dict], list[int], dict[str, int]]:
    cached_by_pair: dict[tuple[str, str], dict] = {}
    for cached_row in cached_rows:
        pair_key = _pair_key(cached_row)
        if pair_key is None or pair_key in cached_by_pair:
            continue
        if _has_score_fields(cached_row, score_fields):
            cached_by_pair[pair_key] = cached_row

    merged_rows: list[dict] = []
    rows_to_score: list[dict] = []
    missing_positions: list[int] = []
    for row in rows:
        pair_key = _pair_key(row)
        cached_row = cached_by_pair.get(pair_key) if pair_key is not None else None
        if cached_row is None:
            missing_positions.append(len(merged_rows))
            merged_rows.append(dict(row))
            rows_to_score.append(dict(row))
            continue

        merged_row = dict(row)
        for field in score_fields:
            merged_row[field] = float(cached_row[field])
        merged_rows.append(merged_row)

    stats = {
        "cached_source_rows": len(cached_rows),
        "cache_pairs": len(cached_by_pair),
        "reused_rows": len(merged_rows) - len(rows_to_score),
        "rows_to_score": len(rows_to_score),
    }
    return merged_rows, rows_to_score, missing_positions, stats


def _filter_rows_with_known_pairs(
    rows: list[dict],
    candidates: list[dict],
    vacancies: list[dict],
) -> list[dict]:
    candidates_by_id = {candidate.get("id"): candidate for candidate in candidates}
    vacancies_by_id = {
        str(vacancy.get("dataset_id") or vacancy.get("id")): vacancy for vacancy in vacancies
    }
    scorable_rows: list[dict] = []
    for row in rows:
        candidate = candidates_by_id.get(row.get("candidate_id"))
        vacancy = vacancies_by_id.get(str(row.get("vacancy_id")))
        if candidate is None or vacancy is None:
            continue
        scorable_rows.append(dict(row))
    return scorable_rows


def _rows_with_scores(rows: list[dict], score_fields: tuple[str, ...]) -> list[dict]:
    return [dict(row) for row in rows if _has_score_fields(row, score_fields)]


def _validate_checkpoint_mode(
    *,
    checkpoint_every: int,
    rewrite_mode: bool,
    output_path: Path,
) -> None:
    if checkpoint_every <= 0 or rewrite_mode:
        return
    if output_path.exists():
        raise ValueError(
            "--checkpoint-every with an existing --output-path requires --rewrite-mode "
            "to avoid overwriting previous scored rows."
        )


def _write_jsonl_atomic(items: list[dict], output_path: Path) -> None:
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    write_jsonl(items, temp_path)
    temp_path.replace(output_path)


def _merge_rows_additive(base_rows: list[dict], additions: list[dict]) -> list[dict]:
    merged_rows = [dict(row) for row in base_rows]
    index_by_pair: dict[tuple[str, str], int] = {}
    for index, row in enumerate(merged_rows):
        pair_key = _pair_key(row)
        if pair_key is None or pair_key in index_by_pair:
            continue
        index_by_pair[pair_key] = index

    for row in additions:
        pair_key = _pair_key(row)
        row_copy = dict(row)
        if pair_key is None:
            merged_rows.append(row_copy)
            continue
        existing_index = index_by_pair.get(pair_key)
        if existing_index is None:
            index_by_pair[pair_key] = len(merged_rows)
            merged_rows.append(row_copy)
            continue
        merged_rows[existing_index] = row_copy

    return merged_rows


def _save_checkpoint(
    *,
    output_path: Path,
    captured_scored_rows: list[dict],
    prefilled_rows: list[dict] | None,
    rewrite_base_rows: list[dict] | None,
    score_fields: tuple[str, ...],
) -> None:
    if rewrite_base_rows is not None:
        rows_to_persist = _merge_rows_additive(rewrite_base_rows, captured_scored_rows)
    elif prefilled_rows is not None:
        rows_to_persist = _rows_with_scores(prefilled_rows, score_fields)
    else:
        rows_to_persist = list(captured_scored_rows)
    _write_jsonl_atomic(rows_to_persist, output_path)
    print(f"Checkpoint saved: {len(rows_to_persist)} rows -> {output_path}")


def _on_rate_limit_retry(attempt: int, wait_seconds: float, exc: Exception) -> None:
    print(
        f"[rate-limit] attempt={attempt}: {exc.__class__.__name__}; "
        f"retrying in {wait_seconds:.2f}s"
    )


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw_value!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an int, got {raw_value!r}") from exc


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an int or one of none/null, got {raw_value!r}"
        ) from exc


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be a boolean (true/false/1/0), got {raw_value!r}"
    )


def _sample_rows_by_candidate_fraction(
    rows: list[dict],
    *,
    user_fraction: float,
    seed: int,
) -> tuple[list[dict], int, int]:
    if not 0 < user_fraction <= 1:
        raise ValueError("benchmark user fraction must be in (0, 1]")

    candidate_order: list[str] = []
    seen_candidates: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in seen_candidates:
            seen_candidates.add(candidate_id)
            candidate_order.append(candidate_id)

    if not candidate_order:
        return [], 0, 0

    target_candidates = max(1, round(len(candidate_order) * user_fraction))
    shuffled = list(candidate_order)
    random.Random(seed).shuffle(shuffled)
    selected = set(shuffled[:target_candidates])
    sampled_rows = [row for row in rows if str(row.get("candidate_id")) in selected]
    return sampled_rows, target_candidates, len(candidate_order)


def autotune_micro_batch_size(
    *,
    rows: list[dict],
    candidates: list[dict],
    vacancies: list[dict],
    llm: object,
    invoke,
    requested_micro_batch_size: int,
    tune_rows: int,
) -> int:
    if requested_micro_batch_size <= 1:
        return 1

    sample_size = min(len(rows), max(4, tune_rows))
    sample_rows = rows[:sample_size]

    baseline_tps = _measure_rows_per_second(
        rows=sample_rows,
        candidates=candidates,
        vacancies=vacancies,
        llm=llm,
        invoke=invoke,
        micro_batch_size=1,
    )
    batched_tps = _measure_rows_per_second(
        rows=sample_rows,
        candidates=candidates,
        vacancies=vacancies,
        llm=llm,
        invoke=invoke,
        micro_batch_size=requested_micro_batch_size,
    )
    print(
        "Micro-batch autotune benchmark: "
        f"batch=1 -> {baseline_tps:.2f} rows/s, "
        f"batch={requested_micro_batch_size} -> {batched_tps:.2f} rows/s"
    )

    if batched_tps > baseline_tps * 1.05:
        return requested_micro_batch_size
    return 1


def _measure_rows_per_second(
    *,
    rows: list[dict],
    candidates: list[dict],
    vacancies: list[dict],
    llm: object,
    invoke,
    micro_batch_size: int,
) -> float:
    started_at = time.perf_counter()
    score_feature_rows_with_experts(
        rows,
        candidates,
        vacancies,
        llm,
        invoke=invoke,
        progress_callback=None,
        max_workers=1,
        micro_batch_size=micro_batch_size,
    )
    elapsed = max(0.001, time.perf_counter() - started_at)
    return len(rows) / elapsed


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


if __name__ == "__main__":
    main()
