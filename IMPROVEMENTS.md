# Improvements backlog

Use this file to give Codex or another coding agent a controlled task list.

## Example prompt

> Read this file and implement only items 1-3. Keep notebook behavior unchanged. Add tests for any pure helper functions you modify. Do not change model hyperparameters unless explicitly requested.

## Backlog

1. Move any remaining notebook-only logic into modules.
2. Add persistent Chroma support and document when to reset vs reuse the index.
3. Add BM25 baseline retrieval.
4. Add hybrid retrieval: embedding score + BM25 score.
5. Add reranking step using an LLM or cross-encoder.
6. Add experiment config files for model names, top-K, paths, and reranker options.
7. Add evaluation report generation into `outputs/`.
8. Add small test fixtures for candidates and vacancies.
9. Add CI or a simple `make test` / `make lint` workflow.
10. Convert the original notebook into a cleaner experiment notebook after modules stabilize.
