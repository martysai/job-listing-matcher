from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CANDIDATES_PATH = Path("data/raw/results_5000_scores_acknowledged.jsonl")
DEFAULT_VACANCIES_PATH = Path("data/raw/vacancies_safe_ml_dataset_nozip.jsonl")
DEFAULT_MATCHES_OUTPUT_PATH = Path("data/processed/candidate_vacancy_matches_top20.jsonl")
DEFAULT_RERANK_FEATURES_OUTPUT_PATH = Path("data/processed/rerank_features_top20.jsonl")
DEFAULT_SCORED_RERANK_FEATURES_OUTPUT_PATH = Path("data/processed/rerank_features_scored_top20.jsonl")
DEFAULT_LAMBDARANK_TRAIN_ROWS_OUTPUT_PATH = Path("data/processed/lambdarank_train_rows_top20.jsonl")
DEFAULT_LAMBDARANK_VALIDATION_ROWS_OUTPUT_PATH = Path(
    "data/processed/lambdarank_validation_rows_top20.jsonl"
)
DEFAULT_LAMBDARANK_VALIDATION_SCORED_OUTPUT_PATH = Path(
    "data/processed/lambdarank_validation_scored_top20.jsonl"
)
DEFAULT_LAMBDARANK_MODEL_OUTPUT_PATH = Path("outputs/lambdarank_model_top20.pkl")
DEFAULT_CHROMA_DIR = Path("data/chroma")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "vacancy_retriever"
HNSW_SPACE = "cosine"
TOP_K = 20
BATCH_SIZE = 2_000
DEFAULT_EVAL_KS = (1, 5, 10, 20, 50, 100)
RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE = 4
RERANK_SCORING_VALIDATION_FRACTION = 0.2
RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE = 4

LLM_MODEL = "mistral/mistral-small-latest"
LLM_PROVIDER = "litellm"
