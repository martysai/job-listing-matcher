import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_rerank_experts.py"
_SPEC = importlib.util.spec_from_file_location("score_rerank_experts", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_filter_rows_with_known_pairs = _MODULE._filter_rows_with_known_pairs
_has_score_fields = _MODULE._has_score_fields
_merge_rows_additive = _MODULE._merge_rows_additive
_pair_key = _MODULE._pair_key
_prepare_rewrite_rows = _MODULE._prepare_rewrite_rows
_rows_with_scores = _MODULE._rows_with_scores
_save_checkpoint = _MODULE._save_checkpoint
_validate_checkpoint_mode = _MODULE._validate_checkpoint_mode
_write_jsonl_atomic = _MODULE._write_jsonl_atomic


def test_pair_key_returns_none_when_ids_missing():
    assert _pair_key({"candidate_id": "c1"}) is None
    assert _pair_key({"vacancy_id": "v1"}) is None
    assert _pair_key({"candidate_id": "c1", "vacancy_id": "v1"}) == ("c1", "v1")


def test_has_score_fields_requires_all_numeric_scores():
    score_fields = ("location_score", "general_score")

    assert _has_score_fields({"location_score": 0.4, "general_score": "0.9"}, score_fields)
    assert not _has_score_fields({"location_score": 0.4}, score_fields)
    assert not _has_score_fields({"location_score": 0.4, "general_score": "x"}, score_fields)


def test_prepare_rewrite_rows_reuses_cached_pairs_and_keeps_current_row_fields():
    score_fields = ("location_score", "general_score")
    rows = [
        {"candidate_id": "c1", "vacancy_id": "v1", "split": "validation"},
        {"candidate_id": "c2", "vacancy_id": "v2", "split": "validation"},
    ]
    cached_rows = [
        {
            "candidate_id": "c1",
            "vacancy_id": "v1",
            "split": "train",
            "location_score": 0.2,
            "general_score": 0.8,
        },
        {
            "candidate_id": "c2",
            "vacancy_id": "v2",
            "location_score": 0.4,
        },
    ]

    merged_rows, rows_to_score, missing_positions, stats = _prepare_rewrite_rows(
        rows=rows,
        cached_rows=cached_rows,
        score_fields=score_fields,
    )

    assert merged_rows[0]["split"] == "validation"
    assert merged_rows[0]["location_score"] == 0.2
    assert merged_rows[0]["general_score"] == 0.8
    assert rows_to_score == [rows[1]]
    assert missing_positions == [1]
    assert stats == {
        "cached_source_rows": 2,
        "cache_pairs": 1,
        "reused_rows": 1,
        "rows_to_score": 1,
    }


def test_filter_rows_with_known_pairs_filters_missing_candidate_or_vacancy():
    rows = [
        {"candidate_id": "c1", "vacancy_id": "v1"},
        {"candidate_id": "missing", "vacancy_id": "v1"},
        {"candidate_id": "c1", "vacancy_id": "missing"},
    ]
    candidates = [{"id": "c1"}]
    vacancies = [{"dataset_id": "v1"}]

    filtered = _filter_rows_with_known_pairs(rows, candidates, vacancies)

    assert filtered == [{"candidate_id": "c1", "vacancy_id": "v1"}]


def test_rows_with_scores_keeps_only_rows_with_complete_scores():
    rows = [
        {"candidate_id": "c1", "location_score": 0.1, "general_score": 0.2},
        {"candidate_id": "c2", "location_score": 0.2},
    ]

    filtered = _rows_with_scores(rows, ("location_score", "general_score"))

    assert filtered == [{"candidate_id": "c1", "location_score": 0.1, "general_score": 0.2}]


def test_write_jsonl_atomic_writes_target_without_leftover_temp(tmp_path):
    output_path = tmp_path / "scored.jsonl"

    _write_jsonl_atomic([{"candidate_id": "c1"}], output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip() == '{"candidate_id": "c1"}'
    assert not output_path.with_suffix(".jsonl.tmp").exists()


def test_save_checkpoint_uses_scored_prefilled_rows_only(tmp_path):
    output_path = tmp_path / "scored.jsonl"
    prefilled_rows = [
        {"candidate_id": "c1", "vacancy_id": "v1", "location_score": 0.1, "general_score": 0.2},
        {"candidate_id": "c2", "vacancy_id": "v2"},
    ]

    _save_checkpoint(
        output_path=output_path,
        captured_scored_rows=[],
        prefilled_rows=prefilled_rows,
        rewrite_base_rows=None,
        score_fields=("location_score", "general_score"),
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"candidate_id": "c1"' in lines[0]


def test_save_checkpoint_writes_captured_rows_when_prefilled_missing(tmp_path):
    output_path = tmp_path / "scored.jsonl"
    captured = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]

    _save_checkpoint(
        output_path=output_path,
        captured_scored_rows=captured,
        prefilled_rows=None,
        rewrite_base_rows=None,
        score_fields=("location_score", "general_score"),
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_merge_rows_additive_keeps_old_and_adds_new_pairs():
    base_rows = [
        {"candidate_id": "c1", "vacancy_id": "v1", "general_score": 0.1},
        {"candidate_id": "c2", "vacancy_id": "v2", "general_score": 0.2},
    ]
    additions = [
        {"candidate_id": "c2", "vacancy_id": "v2", "general_score": 0.9},
        {"candidate_id": "c3", "vacancy_id": "v3", "general_score": 0.8},
    ]

    merged = _merge_rows_additive(base_rows, additions)

    assert len(merged) == 3
    assert merged[0]["candidate_id"] == "c1"
    assert merged[1]["candidate_id"] == "c2"
    assert merged[1]["general_score"] == 0.9
    assert merged[2]["candidate_id"] == "c3"


def test_save_checkpoint_rewrite_mode_preserves_base_rows(tmp_path):
    output_path = tmp_path / "scored.jsonl"
    base_rows = [
        {"candidate_id": "c1", "vacancy_id": "v1", "general_score": 0.1},
        {"candidate_id": "c2", "vacancy_id": "v2", "general_score": 0.2},
    ]
    captured = [{"candidate_id": "c3", "vacancy_id": "v3", "general_score": 0.3}]

    _save_checkpoint(
        output_path=output_path,
        captured_scored_rows=captured,
        prefilled_rows=None,
        rewrite_base_rows=base_rows,
        score_fields=("location_score", "general_score"),
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_validate_checkpoint_mode_blocks_overwrite_without_rewrite(tmp_path):
    output_path = tmp_path / "scored.jsonl"
    output_path.write_text('{"x": 1}\\n', encoding="utf-8")

    try:
        _validate_checkpoint_mode(
            checkpoint_every=25,
            rewrite_mode=False,
            output_path=output_path,
        )
    except ValueError as exc:
        assert "--rewrite-mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_validate_checkpoint_mode_allows_when_missing_or_rewrite(tmp_path):
    missing_output = tmp_path / "missing.jsonl"
    _validate_checkpoint_mode(
        checkpoint_every=25,
        rewrite_mode=False,
        output_path=missing_output,
    )

    existing_output = tmp_path / "existing.jsonl"
    existing_output.write_text("{}", encoding="utf-8")
    _validate_checkpoint_mode(
        checkpoint_every=25,
        rewrite_mode=True,
        output_path=existing_output,
    )
