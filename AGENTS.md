# Agent instructions

Use this repository as a structured refactor of the original Colab notebook.

## Rules

- Keep the first working behavior equivalent to the notebook unless the task explicitly asks for algorithmic changes.
- Put reusable logic in `src/sara_retrieve_rerank/`.
- Keep notebooks thin: orchestration, inspection, plots, and experiment notes only.
- Do not hard-code local absolute paths.
- Keep raw data out of git. Use `data/raw/` for local datasets.
- Add or update tests for pure helpers whenever changing preprocessing, formatting, or metrics code.

## Current pipeline

- `data.py`: JSONL input/output helpers.
- `preprocessing.py`: text cleaning and candidate query formatting.
- `documents.py`: vacancy-to-text, vacancy metadata, LangChain document creation.
- `vector_store.py`: embeddings, Chroma setup, batch indexing.
- `retrieval.py`: top-K matching and export format.
- `evaluation.py`: Recall@K and missed candidate analysis.
- `visualization.py`: simple data exploration plots.

## Suggested implementation style

- Make small, reviewable commits.
- Prefer explicit function arguments over globals.
- Add docstrings when introducing non-obvious ranking/retrieval logic.
- Keep heavy experiment outputs in `outputs/` or `data/processed/`, not in source files.
