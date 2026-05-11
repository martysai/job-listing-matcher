from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CANDIDATES_PATH = Path("data/raw/results_5000_scores_acknowledged.jsonl")
DEFAULT_VACANCIES_PATH = Path("data/raw/vacancies_safe_ml_dataset_nozip.jsonl")
DEFAULT_MATCHES_OUTPUT_PATH = Path("data/processed/candidate_vacancy_matches_top100.jsonl")
DEFAULT_CHROMA_DIR = Path("data/chroma")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "vacancy_retriever"
HNSW_SPACE = "cosine"
TOP_K = 100
BATCH_SIZE = 2_000
DEFAULT_EVAL_KS = (1, 5, 10, 20, 50, 100)

LLM_MODEL = "mistral/mistral-small-latest"
