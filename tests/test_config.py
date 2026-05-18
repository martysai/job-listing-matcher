"""Test that config.yaml overrides module defaults without breaking imports."""

from __future__ import annotations

import importlib

import sara_retrieve_rerank.config as config_module


def test_config_module_exposes_expected_constants():
    expected = {
        "DEFAULT_CANDIDATES_PATH",
        "DEFAULT_VACANCIES_PATH",
        "DEFAULT_MATCHES_OUTPUT_PATH",
        "DEFAULT_RERANK_FEATURES_OUTPUT_PATH",
        "DEFAULT_SCORED_RERANK_FEATURES_OUTPUT_PATH",
        "DEFAULT_LAMBDARANK_TRAIN_ROWS_OUTPUT_PATH",
        "DEFAULT_LAMBDARANK_VALIDATION_ROWS_OUTPUT_PATH",
        "DEFAULT_LAMBDARANK_VALIDATION_SCORED_OUTPUT_PATH",
        "DEFAULT_LAMBDARANK_MODEL_OUTPUT_PATH",
        "DEFAULT_CHROMA_DIR",
        "EMBEDDING_MODEL",
        "COLLECTION_NAME",
        "HNSW_SPACE",
        "TOP_K",
        "BATCH_SIZE",
        "DEFAULT_EVAL_KS",
        "RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE",
        "RERANK_SCORING_VALIDATION_FRACTION",
        "RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE",
        "LLM_MODEL",
        "LLM_PROVIDER",
    }
    missing = expected - set(dir(config_module))
    assert missing == set(), f"Missing config constants: {missing}"


def test_load_yaml_config_returns_empty_dict_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    assert config_module._load_yaml_config(missing) == {}


def test_load_yaml_config_reads_overrides(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text(
        """
retrieval:
  top_k: 99
  embedding_model: custom/model
""",
        encoding="utf-8",
    )
    data = config_module._load_yaml_config(path)
    assert data["retrieval"]["top_k"] == 99
    assert data["retrieval"]["embedding_model"] == "custom/model"


def test_config_module_reimports_cleanly():
    # Ensure a re-import does not throw even when the project root config file
    # is present or absent. This guards against import-time crashes when the
    # YAML loader is missing or the file is malformed.
    importlib.reload(config_module)
    assert isinstance(config_module.TOP_K, int)
