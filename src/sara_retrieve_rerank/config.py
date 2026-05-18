"""Module-level configuration constants.

Defaults can be overridden by a `config.yaml` file at the project root. The
loader is intentionally tolerant: any missing file, missing key, or invalid
value silently falls back to the in-code default below, so existing imports
and tests continue to work without `config.yaml` present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Read an optional YAML config file. Returns an empty dict on any error."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


_CONFIG = _load_yaml_config(PROJECT_ROOT / "config.yaml")


def _get(section: str, key: str, default: Any) -> Any:
    bucket = _CONFIG.get(section) if isinstance(_CONFIG.get(section), dict) else {}
    value = bucket.get(key, default) if isinstance(bucket, dict) else default
    return value if value is not None else default


def _get_path(section: str, key: str, default: str) -> Path:
    return Path(_get(section, key, default))


DEFAULT_CANDIDATES_PATH = _get_path(
    "paths", "candidates", "data/raw/results_5000_scores_acknowledged.jsonl"
)
DEFAULT_VACANCIES_PATH = _get_path(
    "paths", "vacancies", "data/raw/vacancies_safe_ml_dataset_nozip.jsonl"
)
DEFAULT_MATCHES_OUTPUT_PATH = _get_path(
    "paths", "matches_output", "data/processed/candidate_vacancy_matches_top20.jsonl"
)
DEFAULT_RERANK_FEATURES_OUTPUT_PATH = _get_path(
    "paths", "rerank_features_output", "data/processed/rerank_features_top20.jsonl"
)
DEFAULT_SCORED_RERANK_FEATURES_OUTPUT_PATH = _get_path(
    "paths",
    "scored_rerank_features_output",
    "data/processed/rerank_features_scored_top20.jsonl",
)
DEFAULT_LAMBDARANK_TRAIN_ROWS_OUTPUT_PATH = _get_path(
    "paths",
    "lambdarank_train_rows_output",
    "data/processed/lambdarank_train_rows_top20.jsonl",
)
DEFAULT_LAMBDARANK_VALIDATION_ROWS_OUTPUT_PATH = _get_path(
    "paths",
    "lambdarank_validation_rows_output",
    "data/processed/lambdarank_validation_rows_top20.jsonl",
)
DEFAULT_LAMBDARANK_VALIDATION_SCORED_OUTPUT_PATH = _get_path(
    "paths",
    "lambdarank_validation_scored_output",
    "data/processed/lambdarank_validation_scored_top20.jsonl",
)
DEFAULT_LAMBDARANK_MODEL_OUTPUT_PATH = _get_path(
    "paths", "lambdarank_model_output", "outputs/lambdarank_model_top20.pkl"
)
DEFAULT_CHROMA_DIR = _get_path("paths", "chroma_directory", "data/chroma")

EMBEDDING_MODEL = str(_get("retrieval", "embedding_model", "all-MiniLM-L6-v2"))
COLLECTION_NAME = str(_get("retrieval", "collection_name", "vacancy_retriever"))
HNSW_SPACE = str(_get("retrieval", "hnsw_space", "cosine"))
TOP_K = int(_get("retrieval", "top_k", 20))
BATCH_SIZE = int(_get("retrieval", "batch_size", 2_000))
DEFAULT_EVAL_KS = tuple(_get("retrieval", "eval_ks", (1, 5, 10, 20, 50, 100)))

RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE = int(
    _get("reranking", "train_max_negatives_per_candidate", 1)
)
RERANK_SCORING_VALIDATION_FRACTION = float(
    _get("reranking", "scoring_validation_fraction", 0.2)
)
RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE = int(
    _get("reranking", "validation_max_negatives_per_candidate", 19)
)

LLM_MODEL = str(_get("llm", "model", "mistral/mistral-small-latest"))
LLM_PROVIDER = str(_get("llm", "provider", "litellm"))
