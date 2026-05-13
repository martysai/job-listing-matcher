# Sara retrieve + rerank project

This repository is a structured version of the original Colab notebook `sara_retrieve+rerank.ipynb`.

The current pipeline is preserved:

1. Load candidates from `vacancy_conditioned_candidates.jsonl`.
2. Load vacancies from `vacancies_safe_ml_dataset_nozip.jsonl`.
3. Convert vacancies into LangChain `Document` objects.
4. Build a Chroma vector index with Hugging Face embeddings.
5. Retrieve top vacancies for each candidate.
6. Save candidate-vacancy matches.
7. Evaluate Recall@K and inspect misses.

## Project layout

```text
.
├── notebooks/
│   ├── sara_retrieve_rerank_original.ipynb
│   └── sara_retrieve_rerank_refactored.ipynb
├── src/sara_retrieve_rerank/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── documents.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   ├── setup_env.sh
│   ├── check_env.py
│   ├── build_index.py
│   ├── run_retrieval.py
│   └── evaluate.py
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
├── outputs/
├── AGENTS.md
├── IMPROVEMENTS.md
├── requirements.txt
└── pyproject.toml
```

## Data files

Put these files into `data/raw/`:

```text
data/raw/vacancy_conditioned_candidates.jsonl
data/raw/vacancies_safe_ml_dataset_nozip.jsonl
```

The notebook originally expected these files in the working directory. The structured version defaults to `data/raw/`, but all scripts accept CLI path overrides.

## Recommended local setup on macOS / VS Code

Always call pip through the Python interpreter you are actually using. This avoids the common macOS problem where `python3` points to Homebrew Python but `pip` points to a broken `/usr/local` Python.

From the project root:

```bash
cd /Users/r.nesterov/Downloads/sara_retrieve_rerank_project
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/check_env.py
```

Then select this interpreter in VS Code:

```text
Command Palette -> Python: Select Interpreter -> .venv/bin/python
```

Run the script with the venv Python:

```bash
python scripts/evaluate.py
```

Or, without activating the environment:

```bash
.venv/bin/python scripts/evaluate.py
```

You can also run the setup helper:

```bash
bash scripts/setup_env.sh
```

If your default `python3` is too new for one of the ML packages, install Python 3.11 or 3.12 and run:

```bash
PYTHON_BIN=python3.11 bash scripts/setup_env.sh
```

## Run the pipeline

Build an index only:

```bash
python scripts/build_index.py
```

Run retrieval for all candidates and save matches:

```bash
python scripts/run_retrieval.py
```

Default retrieval now writes top-20 matches per candidate (`TOP_K=20`) for faster rerank
feature generation. Evaluation can still compute Recall@K up to 100 because it retrieves
directly from Chroma during `scripts/evaluate.py`.

Evaluate Recall@K:

```bash
python scripts/evaluate.py
```

## Reranking experiment

Create the same top-K retrieval output plus labeled reranker feature rows:

```bash
python scripts/run_retrieval.py --feature-output-path data/processed/rerank_features_top20.jsonl
```

Add LLM expert score columns such as `location_score`, `seniority_score`,
`salary_score`, `experience_score`, and `general_score`:

```bash
python scripts/score_rerank_experts.py --max-rows 100
```

For rate-limit safety, configure `.env` (see `.env.example`) or export:

```bash
export RERANK_REQUEST_DELAY_SECONDS=2.0
export RERANK_RATE_LIMIT_MAX_RETRIES=6
export RERANK_RATE_LIMIT_BACKOFF_BASE_SECONDS=2.0
export RERANK_RATE_LIMIT_BACKOFF_MAX_SECONDS=60.0
export RERANK_MAX_CONCURRENCY=1
export RERANK_MICRO_BATCH_SIZE=4
export RERANK_MICRO_BATCH_AUTOTUNE=true
export RERANK_SCORING_VALIDATION_FRACTION=0.2
export RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE=4
export RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE=4
export RERANK_LLM_PROVIDER=litellm
export RERANK_LLM_MODEL=mistral/mistral-small-latest
```

`scripts/score_rerank_experts.py` auto-loads `.env` from the project root before parsing CLI args.

`score_rerank_experts.py` now:
- scores only candidate groups where retrieval contains at least one positive label,
- for LLM scoring only: subsamples both train and validation groups to
  positive + 4 negatives per candidate by default before any LLM calls,
- asks for all five expert fields in one JSON response per candidate-vacancy pair,
- supports micro-batching multiple pairs into one LLM request (`RERANK_MICRO_BATCH_SIZE`),
- retries rate-limited calls with exponential backoff,
- sleeps between requests,
- prints progress while scoring.

With top-20 retrieval this reduces LLM-scored rows to about `5` rows/group for the
positive-retrieval candidate groups (configurable via env vars above).

For faster and safer runs, keep `RERANK_MAX_CONCURRENCY=1` unless your provider quota is high.
Optionally use a local model (for example Ollama) to avoid hosted API rate limits:

```bash
export RERANK_LLM_PROVIDER=ollama
export RERANK_LLM_MODEL=ollama_chat/qwen2.5:3b-instruct
export RERANK_LLM_API_BASE=http://localhost:11434
export RERANK_MAX_CONCURRENCY=1
export RERANK_REQUEST_DELAY_SECONDS=0
python scripts/score_rerank_experts.py
```

Use quick benchmark mode to test the pipeline on ~5% of users before long runs:

```bash
python scripts/score_rerank_experts.py --benchmark-mode
```

Optional benchmark controls:
- `--benchmark-user-fraction` (default from `RERANK_BENCHMARK_USER_FRACTION=0.05`)
- `--benchmark-seed`
- `--micro-batch-size` and `RERANK_MICRO_BATCH_AUTOTUNE=true` to auto-keep batching only when it is faster on a short local benchmark.

Then compare the original cosine ranking with a weighted LLM baseline:

```bash
python scripts/evaluate.py --rerank-features-path data/processed/rerank_features_scored_top20.jsonl
```

To train and validate a LightGBM LambdaRank model on the cached feature rows:

```bash
python scripts/evaluate.py --rerank-features-path data/processed/rerank_features_scored_top20.jsonl --train-lambdarank
```

When `--train-lambdarank` is enabled, the script now also saves:
- train rows for LightGBM (`data/processed/lambdarank_train_rows_top20.jsonl`)
- validation rows (`data/processed/lambdarank_validation_rows_top20.jsonl`)
- validation rows with `lambdarank_score` (`data/processed/lambdarank_validation_scored_top20.jsonl`)
- fitted model (`outputs/lambdarank_model_top20.pkl`)

It prints metrics for:
- baseline cosine on all rows
- baseline weighted expert average on all rows
- cosine on LambdaRank validation split
- weighted expert average on LambdaRank validation split
- LambdaRank score on the same validation split

It also prints compact method-comparison tables and saves comparison charts by default:
- `outputs/rerank_method_comparison_all_rows.png`
- `outputs/rerank_method_comparison_validation.png` (when `--train-lambdarank` is used)

The reusable reranking helpers live in `src/sara_retrieve_rerank/reranking.py`.
LightGBM is optional for the base retriever, but install the full requirements or
`python -m pip install -e ".[rerank]"` before using `--train-lambdarank`.
LambdaRank training keeps all positives and subsamples negatives to `4` per candidate
by default (`RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE` in config). Validation-group
negative subsampling for LLM scoring is controlled separately with
`RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE`.

## VS Code notebook setup

Open `notebooks/sara_retrieve_rerank_refactored.ipynb` and select `.venv/bin/python` as the kernel.

The first notebook cell now detects the project root and adds `src/` to `sys.path`, so local imports work even before editable install. If dependency imports fail, run the optional install cell in the notebook.

## Colab setup

In Colab, upload this repo or clone it, then run the notebook's optional install cell. It uses the current kernel executable and the detected project root, which is more reliable than hard-coded `../requirements.txt` paths.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_core'`

Dependencies are not installed in the Python interpreter that ran the script. Use:

```bash
python -m pip install -r requirements.txt
```

Do not use plain `pip` unless you are sure it belongs to the same interpreter.

### `ModuleNotFoundError: No module named 'sara_retrieve_rerank'`

The project package is not installed or the notebook kernel does not know about `src/`. Use:

```bash
python -m pip install -e .
```

For notebooks, run the first bootstrap cell and select `.venv/bin/python` as the kernel.

### `zsh: command not found: python`

On macOS, `python` may not exist until a venv is activated. Use `python3` to create the venv, then activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, `python` should work.

## Live Adzuna RAG search

`scripts/adzuna_rag.py` is a small standalone script that emulates a RAG-style
job search over the public Adzuna API. Give it a candidate's free-text job
request and it returns the top live vacancies that match.

Setup credentials (one-time). Adzuna free tier has a monthly call budget, so
the script caches every response under `data/cache/adzuna/`:

```bash
cp .env.example .env
# Then edit .env and set:
#   ADZUNA_APP_ID=<your app id>
#   ADZUNA_APP_KEY=<your app key>
```

If your `.env.example` does not list the Adzuna keys yet, add them by hand:

```text
ADZUNA_APP_ID=
ADZUNA_APP_KEY=
```

You can also pass the credentials on the CLI (`--app-id` / `--app-key`) or
export them as environment variables.

Run a search across the default European endpoints + UK:

```bash
python scripts/adzuna_rag.py --query "junior data scientist remote Python NLP" --top-k 10
```

Restrict the search to specific country endpoints and an optional location:

```bash
python scripts/adzuna_rag.py \
    --query "senior backend engineer Go fintech" \
    --countries gb,de,nl \
    --where london \
    --top-k 15
```

Useful flags:
- `--countries gb,de,fr,nl,it,es,pl` — comma-separated Adzuna country codes.
- `--results-per-page 50` — Adzuna max page size (default 50).
- `--max-days-old 14` — only return jobs posted in the last N days.
- `--output-jsonl outputs/adzuna_results.jsonl` — dump merged results to disk.
- `--no-cache` — skip the on-disk cache (will burn quota; only use to refresh).

Ranking is Adzuna's built-in relevance order. Results are merged across
countries and re-sorted by a derived relevance score so the top matches show
up first.

### Adzuna API — legal use & attribution

Before integrating this script into anything beyond local experimentation,
read Adzuna's [Terms of Service](https://developer.adzuna.com/docs/terms_of_service).
Key points relevant to this repo (paraphrased; the TOS is the source of truth):

- **Permitted use.** "Personal research" is one of the three explicitly
  permitted uses of the API, alongside publishing Adzuna ad listings and
  publishing Jobsworth salary estimates. Educational / personal exploration
  fits squarely inside the personal research clause.
- **Free-tier rate limits.** 25 hits/min, 250/day, 1000/week, 2500/month.
  This script makes one call per country per query and caches every response
  on disk, so an iteration cycle costs ~1 call.
- **Academic / commercial / government organisations.** Any use "by a
  commercial, government or academic organisation including any affiliates or
  individuals" only gets a 14-day trial for validation, after which a written
  licence agreement may be required. Beyond the trial, the data may **not**
  be used "in its original format or in aggregation … to deliver any ongoing
  work or research" without written consent. If you are running this work
  under an institutional affiliation, treat your usage as the trial path and
  contact Adzuna before relying on the data long-term.
- **Mandatory attribution.** When you publish anything derived from the API
  (a paper, a notebook, a screenshot, a blog post, a public repo), you must
  acknowledge Adzuna: reference "The Adzuna API" with a link to
  <https://www.adzuna.co.uk/> (or the relevant local domain). Ad listings
  displayed in a UI need the "Jobs by Adzuna" badge (≥ 116 × 23 px) with
  hyperlinks.
- **Confidentiality / scraping.** "Any usage that appears to be an attempt
  to extract Confidential Information for commercial reuse" is a breach. No
  rebuilding a competing job board from the cached responses.
- **Termination obligation.** "Upon termination of this agreement … an API
  user shall immediately remove all insertion codes and data acquired from
  Adzuna from all pages of its web sites." For this reason raw Adzuna
  responses are kept under the gitignored `data/cache/adzuna/` and
  `outputs/` paths, **not** committed to the repository.

**Project-level attribution.** Job listings retrieved by `scripts/adzuna_rag.py`
are powered by [The Adzuna API](https://www.adzuna.co.uk/).

## Good next refactoring targets

- Add BM25 / hybrid retrieval in a separate module.
- Add reranking as a separate `reranking.py` module.
- Add config files for experiments.
- Add cached Chroma persistence for faster iteration.
- Add small fixtures so tests can run without the full dataset.
