from __future__ import annotations

import json
import math
import random
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

from sara_retrieve_rerank.config import (
    RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE,
    RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE,
)
from sara_retrieve_rerank.documents import vacancy_to_text

DEFAULT_RERANK_KS = (1, 5, 10)
DEFAULT_LLM_SCORE_FIELDS = (
    "location_score",
    "seniority_score",
    "salary_score",
    "experience_score",
    "general_score",
)
DEFAULT_FEATURE_FIELDS = ("cosine_similarity",) + DEFAULT_LLM_SCORE_FIELDS
# Opt-in feature set used when retrieval rows carry a `bm25_score` column
# (produced by the BM25 or hybrid retriever). Existing callers and tests that
# rely on `DEFAULT_FEATURE_FIELDS` are unaffected.
DEFAULT_FEATURE_FIELDS_WITH_BM25 = (
    "cosine_similarity",
    "bm25_score",
) + DEFAULT_LLM_SCORE_FIELDS
SINGLE_PAIR_JSON_SCORE_PROMPT = """\
You are an expert recruiter assistant scoring candidate-vacancy match quality.

Score the pair on five dimensions from 0.0 to 1.0:
- location_score: region/city plus remote/hybrid/office compatibility
- seniority_score: role level, leadership scope, and seniority fit
- salary_score: compensation compatibility when available
- experience_score: skills, tools, domain, and years of experience match
- general_score: overall candidate-vacancy fit

Return ONLY valid JSON in this exact schema:
{{
  "location_score": 0.0,
  "seniority_score": 0.0,
  "salary_score": 0.0,
  "experience_score": 0.0,
  "general_score": 0.0
}}

Candidate:
{candidate_text}

Vacancy:
{vacancy_text}
"""


@dataclass(frozen=True)
class LLMExpert:
    """Prompt definition for one numeric expert score."""

    name: str
    prompt: str

    @property
    def score_field(self) -> str:
        return f"{self.name}_score"


DEFAULT_LLM_EXPERTS = (
    LLMExpert(
        name="location",
        prompt=(
            "You are a location and work-format matching expert. Score whether the "
            "candidate's location preferences match the vacancy's regions, cities, "
            "and remote/hybrid/office format.\n\n"
            "Return only one number from 0 to 1.\n\n"
            "Candidate:\n{candidate_text}\n\nVacancy:\n{vacancy_text}"
        ),
    ),
    LLMExpert(
        name="seniority",
        prompt=(
            "You are a seniority matching expert. Score whether the candidate's "
            "seniority, role scope, and leadership level match the vacancy.\n\n"
            "Return only one number from 0 to 1.\n\n"
            "Candidate:\n{candidate_text}\n\nVacancy:\n{vacancy_text}"
        ),
    ),
    LLMExpert(
        name="salary",
        prompt=(
            "You are a compensation matching expert. Score whether the candidate's "
            "salary expectations are compatible with the vacancy, using available "
            "salary or compensation hints only.\n\n"
            "Return only one number from 0 to 1.\n\n"
            "Candidate:\n{candidate_text}\n\nVacancy:\n{vacancy_text}"
        ),
    ),
    LLMExpert(
        name="experience",
        prompt=(
            "You are an experience matching expert. Score whether the candidate's "
            "tools, domains, years of experience, and project background match the "
            "vacancy requirements.\n\n"
            "Return only one number from 0 to 1.\n\n"
            "Candidate:\n{candidate_text}\n\nVacancy:\n{vacancy_text}"
        ),
    ),
    LLMExpert(
        name="general",
        prompt=(
            "You are a general candidate-vacancy fit expert. Consider the whole CV "
            "and the whole vacancy, then score the overall fit.\n\n"
            "Return only one number from 0 to 1.\n\n"
            "Candidate:\n{candidate_text}\n\nVacancy:\n{vacancy_text}"
        ),
    ),
)


def build_pair_feature_row(
    candidate: Mapping[str, Any],
    match: Mapping[str, Any],
    *,
    expert_scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build one supervised reranking row from a retrieved candidate-vacancy match."""
    vacancy_id = match.get("vacancy_id")
    source_vacancy_id = candidate.get("source_vacancy_id")
    row = dict(match)
    row["candidate_id"] = candidate.get("id")
    row["vacancy_id"] = vacancy_id
    row["source_vacancy_id"] = source_vacancy_id
    row["retrieval_rank"] = match.get("rank")
    row["label"] = int(bool(source_vacancy_id) and str(vacancy_id) == str(source_vacancy_id))

    if expert_scores:
        for field, score in expert_scores.items():
            row[field] = float(score)

    return row


def build_pair_feature_rows(
    candidates: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build reranking feature rows from existing retrieval output."""
    candidates_by_id = {candidate.get("id"): candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []

    for match in matches:
        candidate = candidates_by_id.get(match.get("candidate_id"))
        if candidate is None:
            continue
        rows.append(build_pair_feature_row(candidate, match))

    return rows


def parse_llm_score(value: Any) -> float:
    """Parse and clamp a numeric LLM score to the [0, 1] interval."""
    if isinstance(value, int | float):
        score = float(value)
    else:
        text = getattr(value, "content", value)
        match = re.search(r"-?\d+(?:\.\d+)?", str(text))
        if not match:
            raise ValueError(f"Could not parse a numeric score from: {text!r}")
        score = float(match.group(0))

    return min(1.0, max(0.0, score))


def score_pair_with_experts(
    candidate: Mapping[str, Any],
    vacancy: Mapping[str, Any],
    llm: Any,
    *,
    experts: Sequence[LLMExpert] = DEFAULT_LLM_EXPERTS,
    invoke: Callable[[Any, str], Any] | None = None,
) -> dict[str, float]:
    """Score one candidate-vacancy pair with a single JSON LLM call."""
    candidate_text = str(candidate.get("text", ""))
    vacancy_text = vacancy_to_text(dict(vacancy))
    call_llm = invoke or _invoke_llm
    expected_fields = _expected_score_fields(experts)

    prompt = SINGLE_PAIR_JSON_SCORE_PROMPT.format(
        candidate_text=candidate_text,
        vacancy_text=vacancy_text,
    )
    response = call_llm(llm, prompt)
    try:
        return parse_llm_scores_json(response, expected_fields=expected_fields)
    except ValueError:
        # Fallback preserves previous behavior if the model breaks JSON format.
        scores: dict[str, float] = {}
        for expert in experts:
            expert_prompt = expert.prompt.format(
                candidate_text=candidate_text,
                vacancy_text=vacancy_text,
            )
            scores[expert.score_field] = parse_llm_score(call_llm(llm, expert_prompt))
        return scores


def score_feature_rows_with_experts(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    vacancies: Sequence[Mapping[str, Any]],
    llm: Any,
    *,
    experts: Sequence[LLMExpert] = DEFAULT_LLM_EXPERTS,
    invoke: Callable[[Any, str], Any] | None = None,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    max_workers: int = 1,
    micro_batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Enrich reranking feature rows with LLM expert scores."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")

    candidates_by_id = {candidate.get("id"): candidate for candidate in candidates}
    vacancies_by_id = {
        str(vacancy.get("dataset_id") or vacancy.get("id")): vacancy for vacancy in vacancies
    }

    valid_rows: list[tuple[int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for index, row in enumerate(rows, start=1):
        candidate = candidates_by_id.get(row.get("candidate_id"))
        vacancy = vacancies_by_id.get(str(row.get("vacancy_id")))
        if candidate is None or vacancy is None:
            continue
        valid_rows.append((index, row, candidate, vacancy))

    total_rows = len(valid_rows)
    if total_rows == 0:
        return []

    scored_rows: list[dict[str, Any]] = []
    if max_workers == 1:
        if micro_batch_size > 1:
            completed = 0
            for batch in _chunked(valid_rows, micro_batch_size):
                batch_scored_rows = _score_rows_batch_with_experts(
                    batch=batch,
                    llm=llm,
                    experts=experts,
                    invoke=invoke,
                )
                for scored_row in batch_scored_rows:
                    completed += 1
                    scored_rows.append(scored_row)
                    if progress_callback is not None:
                        progress_callback(completed, total_rows, scored_row)
            return scored_rows

        for completed, (index, row, candidate, vacancy) in enumerate(valid_rows, start=1):
            scored_row = _score_row_with_experts(
                row=row,
                candidate=candidate,
                vacancy=vacancy,
                llm=llm,
                experts=experts,
                invoke=invoke,
            )
            scored_rows.append(scored_row)
            if progress_callback is not None:
                progress_callback(completed, total_rows, scored_row)
        return scored_rows

    batched_rows = list(_chunked(valid_rows, micro_batch_size))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda batch: _score_rows_batch_with_experts(
                batch=batch,
                llm=llm,
                experts=experts,
                invoke=invoke,
            ),
            batched_rows,
        )
        for batch_scored_rows in results:
            for scored_row in batch_scored_rows:
                scored_rows.append(scored_row)
                if progress_callback is not None:
                    progress_callback(len(scored_rows), total_rows, scored_row)

    return scored_rows


def _score_rows_batch_with_experts(
    *,
    batch: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    llm: Any,
    experts: Sequence[LLMExpert],
    invoke: Callable[[Any, str], Any] | None,
) -> list[dict[str, Any]]:
    if not batch:
        return []
    if len(batch) == 1:
        _index, row, candidate, vacancy = batch[0]
        return [
            _score_row_with_experts(
                row=row,
                candidate=candidate,
                vacancy=vacancy,
                llm=llm,
                experts=experts,
                invoke=invoke,
            )
        ]

    call_llm = invoke or _invoke_llm
    expected_fields = _expected_score_fields(experts)
    prompt = _build_batch_json_prompt(batch, expected_fields=expected_fields)

    response = call_llm(llm, prompt)
    try:
        parsed_scores = _parse_batch_scores_json(
            response,
            expected_fields=expected_fields,
        )
    except ValueError:
        # Robust fallback: keep pipeline progressing by scoring rows independently.
        fallback_rows: list[dict[str, Any]] = []
        for _index, row, candidate, vacancy in batch:
            fallback_rows.append(
                _score_row_with_experts(
                    row=row,
                    candidate=candidate,
                    vacancy=vacancy,
                    llm=llm,
                    experts=experts,
                    invoke=invoke,
                )
            )
        return fallback_rows

    scored_rows: list[dict[str, Any]] = []
    for row_index, (_input_index, row, _candidate, _vacancy) in enumerate(batch):
        item_id = str(row_index)
        item_scores = parsed_scores.get(item_id, {})
        scored_row = dict(row)
        for field in expected_fields:
            scored_row[field] = float(item_scores.get(field, 0.0))
        scored_rows.append(scored_row)

    return scored_rows


def _build_batch_json_prompt(
    batch: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    *,
    expected_fields: Sequence[str],
) -> str:
    schema_lines = ",\n".join(f'      "{field}": 0.0' for field in expected_fields)
    items: list[str] = []
    for row_index, (_input_index, _row, candidate, vacancy) in enumerate(batch):
        candidate_text = str(candidate.get("text", ""))
        vacancy_text = vacancy_to_text(dict(vacancy))
        items.append(
            (
                f"Item {row_index}\n"
                f"item_id: {row_index}\n"
                f"Candidate:\n{candidate_text}\n\n"
                f"Vacancy:\n{vacancy_text}"
            )
        )
    items_text = "\n\n---\n\n".join(items)
    return (
        "Score each item from 0.0 to 1.0 for these fields: "
        f"{', '.join(expected_fields)}.\n"
        "Return ONLY valid JSON in this exact schema:\n"
        "{\n"
        '  "results": [\n'
        "    {\n"
        '      "item_id": "0",\n'
        f"{schema_lines}\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Items:\n"
        f"{items_text}"
    )


def parse_llm_scores_json(value: Any, *, expected_fields: Sequence[str]) -> dict[str, float]:
    parsed = _parse_json_object(value)
    if "results" in parsed and isinstance(parsed["results"], list) and parsed["results"]:
        parsed = parsed["results"][0]

    scores: dict[str, float] = {}
    for field in expected_fields:
        if field not in parsed:
            raise ValueError(f"Field {field!r} missing in LLM JSON response")
        scores[field] = parse_llm_score(parsed[field])
    return scores


def _parse_batch_scores_json(
    value: Any,
    *,
    expected_fields: Sequence[str],
) -> dict[str, dict[str, float]]:
    parsed = _parse_json_object(value)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Batch JSON response must contain a list under 'results'")

    scores_by_item: dict[str, dict[str, float]] = {}
    for item in results:
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("item_id"))
        item_scores: dict[str, float] = {}
        for field in expected_fields:
            if field in item:
                item_scores[field] = parse_llm_score(item[field])
        scores_by_item[item_id] = item_scores

    return scores_by_item


def _parse_json_object(value: Any) -> dict[str, Any]:
    text = str(getattr(value, "content", value)).strip()
    if not text:
        raise ValueError("LLM response is empty; expected JSON")

    parsed = _try_parse_json(text)
    if isinstance(parsed, Mapping):
        return dict(parsed)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in LLM response: {text!r}")
    parsed = _try_parse_json(match.group(0))
    if not isinstance(parsed, Mapping):
        raise ValueError("Parsed JSON is not an object")
    return dict(parsed)


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _expected_score_fields(experts: Sequence[LLMExpert]) -> tuple[str, ...]:
    if experts:
        return tuple(expert.score_field for expert in experts)
    return DEFAULT_LLM_SCORE_FIELDS


def _score_row_with_experts(
    *,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    vacancy: Mapping[str, Any],
    llm: Any,
    experts: Sequence[LLMExpert],
    invoke: Callable[[Any, str], Any] | None,
) -> dict[str, Any]:
    scored_row = dict(row)
    scored_row.update(
        score_pair_with_experts(
            candidate,
            vacancy,
            llm,
            experts=experts,
            invoke=invoke,
        )
    )
    return scored_row


def filter_rows_with_positive_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "candidate_id",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    """Return rows only for candidate groups that contain at least one positive."""
    return filter_groups_with_positive(rows, group_key=group_key, label_key=label_key)


def prepare_rows_for_llm_scoring(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    train_max_negatives_per_candidate: int | None = RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE,
    validation_max_negatives_per_candidate: int | None = RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE,
    split_key: str = "dataset_split",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Prepare rows for expensive LLM scoring with configurable split-wise negative sampling."""
    positive_rows = filter_groups_with_positive(rows)
    train_rows, validation_rows = grouped_train_validation_split(
        positive_rows,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_rows = subsample_training_negatives(
        train_rows,
        max_negatives_per_candidate=train_max_negatives_per_candidate,
        seed=seed,
    )
    validation_rows = subsample_training_negatives(
        validation_rows,
        max_negatives_per_candidate=validation_max_negatives_per_candidate,
        seed=seed,
    )

    prepared_rows: list[dict[str, Any]] = []
    for row in train_rows:
        scored_row = dict(row)
        scored_row[split_key] = "train"
        prepared_rows.append(scored_row)
    for row in validation_rows:
        scored_row = dict(row)
        scored_row[split_key] = "validation"
        prepared_rows.append(scored_row)

    stats = {
        "input_rows": len(rows),
        "positive_rows": len(positive_rows),
        "train_rows_for_scoring": len(train_rows),
        "validation_rows_for_scoring": len(validation_rows),
        "total_rows_for_scoring": len(prepared_rows),
    }
    return prepared_rows, stats


def _invoke_llm(llm: Any, prompt: str) -> Any:
    message = llm.invoke(prompt)
    return getattr(message, "content", message)


def add_weighted_score(
    rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    *,
    output_key: str = "weighted_score",
) -> list[dict[str, Any]]:
    """Add a weighted-average score to every row."""
    scored_rows: list[dict[str, Any]] = []

    for row in rows:
        weighted_sum = 0.0
        total_weight = 0.0
        for field, weight in weights.items():
            value = row.get(field)
            if value is None:
                continue
            weighted_sum += float(value) * float(weight)
            total_weight += abs(float(weight))

        scored_row = dict(row)
        scored_row[output_key] = weighted_sum / total_weight if total_weight else 0.0
        scored_rows.append(scored_row)

    return scored_rows


def rerank_candidate_matches(
    matches: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Reorder one candidate's retrieved matches by a reranker score."""
    sorted_matches = sorted(
        matches,
        key=lambda row: (
            _score_sort_value(row.get(score_key), descending=descending),
            int(row.get("rank") or row.get("retrieval_rank") or 0),
        ),
    )

    reranked: list[dict[str, Any]] = []
    for rank, match in enumerate(sorted_matches, start=1):
        row = dict(match)
        row.setdefault("retrieval_rank", match.get("rank"))
        row["rank"] = rank
        row["rerank_score"] = float(match.get(score_key) or 0.0)
        reranked.append(row)

    return reranked


def rerank_all_matches(
    matches: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    group_key: str = "candidate_id",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Reorder all retrieved matches independently within each candidate group."""
    reranked: list[dict[str, Any]] = []
    for group in _groups_in_input_order(matches, group_key=group_key):
        reranked.extend(
            rerank_candidate_matches(group, score_key=score_key, descending=descending)
        )
    return reranked


def filter_groups_with_positive(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "candidate_id",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    """Keep only candidate groups where the true vacancy is present in retrieved rows."""
    kept: list[dict[str, Any]] = []
    for group in _groups_in_input_order(rows, group_key=group_key):
        if any(float(row.get(label_key, 0)) > 0 for row in group):
            kept.extend(dict(row) for row in group)
    return kept


def grouped_train_validation_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    group_key: str = "candidate_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split reranking rows by candidate group so each query stays intact."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    groups = _groups_in_input_order(rows, group_key=group_key)
    group_indices = list(range(len(groups)))
    random.Random(seed).shuffle(group_indices)

    validation_group_count = max(1, round(len(groups) * validation_fraction)) if groups else 0
    validation_indices = set(group_indices[:validation_group_count])

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        target = validation_rows if index in validation_indices else train_rows
        target.extend(dict(row) for row in group)

    return train_rows, validation_rows


def evaluate_ranking_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    ks: Sequence[int] = DEFAULT_RERANK_KS,
    group_key: str = "candidate_id",
    label_key: str = "label",
    descending: bool = True,
) -> dict[str, float | int]:
    """Evaluate grouped ranking rows with recall@K and ndcg@K."""
    groups = _groups_in_input_order(rows, group_key=group_key)
    metrics: dict[str, float | int] = {"num_groups": len(groups)}

    for k in ks:
        metrics[f"recall@{k}"] = _recall_at_k(
            groups,
            k=k,
            score_key=score_key,
            label_key=label_key,
            descending=descending,
        )
        metrics[f"ndcg@{k}"] = _ndcg_at_k(
            groups,
            k=k,
            score_key=score_key,
            label_key=label_key,
            descending=descending,
        )

    return metrics


def train_lambdarank(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_fields: Sequence[str] = DEFAULT_FEATURE_FIELDS,
    validation_fraction: float = 0.2,
    seed: int = 42,
    train_max_negatives_per_candidate: int | None = RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE,
    params: Mapping[str, Any] | None = None,
):
    """Train a LightGBM LambdaRank model on grouped reranking rows."""
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for LambdaRank training. Install it with "
            "`python -m pip install lightgbm`."
        ) from exc

    positive_rows = filter_groups_with_positive(rows)
    positive_groups = _groups_in_input_order(positive_rows, group_key="candidate_id")
    if len(positive_groups) < 2:
        raise ValueError("LambdaRank training requires at least two candidate groups with positives")

    pre_split_rows = [row for row in positive_rows if row.get("dataset_split") in {"train", "validation"}]
    if pre_split_rows and len(pre_split_rows) == len(positive_rows):
        train_rows = [dict(row) for row in positive_rows if row.get("dataset_split") == "train"]
        validation_rows = [dict(row) for row in positive_rows if row.get("dataset_split") == "validation"]
    else:
        train_rows, validation_rows = grouped_train_validation_split(
            positive_rows,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        train_rows = subsample_training_negatives(
            train_rows,
            max_negatives_per_candidate=train_max_negatives_per_candidate,
            seed=seed,
        )

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "random_state": seed,
        "n_estimators": 100,
        "verbosity": -1,
    }
    if params:
        default_params.update(params)

    model = LGBMRanker(**default_params)
    train_matrix, train_labels, train_group = to_lambdarank_arrays(
        train_rows,
        feature_fields=feature_fields,
    )
    validation_matrix, validation_labels, validation_group = to_lambdarank_arrays(
        validation_rows,
        feature_fields=feature_fields,
    )
    try:
        import numpy as np

        train_matrix = np.asarray(train_matrix, dtype=float)
        train_labels = np.asarray(train_labels, dtype=int)
        validation_matrix = np.asarray(validation_matrix, dtype=float)
        validation_labels = np.asarray(validation_labels, dtype=int)
    except ImportError:
        pass

    fit_kwargs: dict[str, Any] = {"group": train_group}
    if validation_rows:
        fit_kwargs["eval_set"] = [(validation_matrix, validation_labels)]
        fit_kwargs["eval_group"] = [validation_group]

    model.fit(train_matrix, train_labels, **fit_kwargs)
    return model, train_rows, validation_rows


def subsample_training_negatives(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_negatives_per_candidate: int | None,
    seed: int = 42,
    group_key: str = "candidate_id",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    """Keep all positives and at most N negatives per candidate group."""
    if max_negatives_per_candidate is None:
        return [dict(row) for row in rows]
    if max_negatives_per_candidate < 0:
        raise ValueError("max_negatives_per_candidate must be >= 0 or None")

    rng = random.Random(seed)
    sampled_rows: list[dict[str, Any]] = []

    for group in _groups_in_input_order(rows, group_key=group_key):
        positives = [row for row in group if float(row.get(label_key, 0)) > 0]
        negatives = [row for row in group if float(row.get(label_key, 0)) <= 0]

        if len(negatives) > max_negatives_per_candidate:
            negatives = rng.sample(negatives, k=max_negatives_per_candidate)

        selected = positives + negatives
        selected_sorted = sorted(
            selected,
            key=lambda row: int(row.get("retrieval_rank") or row.get("rank") or 0),
        )
        sampled_rows.extend(dict(row) for row in selected_sorted)

    return sampled_rows


def make_rate_limited_invoker(
    *,
    request_delay_seconds: float,
    max_retries: int = 6,
    backoff_base_seconds: float = 2.0,
    backoff_max_seconds: float = 60.0,
    sleep_fn: Callable[[float], None] | None = None,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> Callable[[Any, str], Any]:
    """Create an LLM invoke function with delay and exponential backoff retries."""
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be >= 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if backoff_base_seconds < 0:
        raise ValueError("backoff_base_seconds must be >= 0")
    if backoff_max_seconds < 0:
        raise ValueError("backoff_max_seconds must be >= 0")

    do_sleep = sleep_fn or time.sleep

    def invoke(llm: Any, prompt: str) -> Any:
        attempt = 0
        while True:
            try:
                message = llm.invoke(prompt)
                result = getattr(message, "content", message)
                if request_delay_seconds > 0:
                    do_sleep(request_delay_seconds)
                return result
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt >= max_retries:
                    raise

                wait_seconds = min(backoff_max_seconds, backoff_base_seconds * (2**attempt))
                if on_retry is not None:
                    on_retry(attempt + 1, wait_seconds, exc)
                if wait_seconds > 0:
                    do_sleep(wait_seconds)
                attempt += 1

    return invoke


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort classification for provider 429 / rate-limit exceptions."""
    class_name = exc.__class__.__name__.lower()
    if "ratelimit" in class_name:
        return True

    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message or "429" in message


def to_lambdarank_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_fields: Sequence[str] = DEFAULT_FEATURE_FIELDS,
    label_key: str = "label",
    group_key: str = "candidate_id",
) -> tuple[list[list[float]], list[int], list[int]]:
    """Convert grouped rows to LightGBM-compatible X, y, and group sizes."""
    matrix: list[list[float]] = []
    labels: list[int] = []
    group_sizes: list[int] = []

    for group in _groups_in_input_order(rows, group_key=group_key):
        group_sizes.append(len(group))
        for row in group:
            matrix.append([float(row.get(field) or 0.0) for field in feature_fields])
            labels.append(int(row.get(label_key, 0)))

    return matrix, labels, group_sizes


def score_rows_with_model(
    rows: Sequence[Mapping[str, Any]],
    model: Any,
    *,
    feature_fields: Sequence[str] = DEFAULT_FEATURE_FIELDS,
    output_key: str = "lambdarank_score",
) -> list[dict[str, Any]]:
    """Add model predictions to reranking rows."""
    matrix = [[float(row.get(field) or 0.0) for field in feature_fields] for row in rows]
    if matrix:
        try:
            import numpy as np

            matrix_input: Any = np.asarray(matrix, dtype=float)
        except ImportError:
            matrix_input = matrix
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
                )
                predictions = model.predict(matrix_input, validate_features=False)
        except TypeError:
            predictions = model.predict(matrix_input)
    else:
        predictions = []

    scored_rows: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        scored_row = dict(row)
        scored_row[output_key] = float(prediction)
        scored_rows.append(scored_row)

    return scored_rows


def _score_sort_value(value: Any, *, descending: bool) -> float:
    if value is None:
        return float("inf")

    score = float(value)
    return -score if descending else score


def _groups_in_input_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
) -> list[list[Mapping[str, Any]]]:
    groups_by_key: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    group_order: list[Any] = []

    for row in rows:
        key = row.get(group_key)
        if key not in groups_by_key:
            group_order.append(key)
        groups_by_key[key].append(row)

    return [groups_by_key[key] for key in group_order]


T = TypeVar("T")


def _chunked(values: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _sorted_group(
    group: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    descending: bool,
) -> list[Mapping[str, Any]]:
    return sorted(
        group,
        key=lambda row: (
            _score_sort_value(row.get(score_key), descending=descending),
            int(row.get("retrieval_rank") or row.get("rank") or 0),
        ),
    )


def _recall_at_k(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    k: int,
    score_key: str,
    label_key: str,
    descending: bool,
) -> float:
    if not groups:
        return 0.0

    hits = 0
    for group in groups:
        ranked = _sorted_group(group, score_key=score_key, descending=descending)
        hits += int(any(float(row.get(label_key, 0)) > 0 for row in ranked[:k]))

    return hits / len(groups)


def _ndcg_at_k(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    k: int,
    score_key: str,
    label_key: str,
    descending: bool,
) -> float:
    if not groups:
        return 0.0

    total = 0.0
    for group in groups:
        ranked = _sorted_group(group, score_key=score_key, descending=descending)
        labels = [float(row.get(label_key, 0)) for row in ranked]
        dcg = _dcg(labels[:k])
        ideal_dcg = _dcg(sorted(labels, reverse=True)[:k])
        total += dcg / ideal_dcg if ideal_dcg else 0.0

    return total / len(groups)


def _dcg(labels: Iterable[float]) -> float:
    return sum(label / math.log2(index + 2) for index, label in enumerate(labels))
