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

## Configuration

Module defaults in [src/sara_retrieve_rerank/config.py](src/sara_retrieve_rerank/config.py)
can be overridden by a [config.yaml](config.yaml) file at the project root.
Any key missing from `config.yaml` (or a missing file altogether) falls back
to the in-code default, so the YAML file is optional and existing scripts
behave the same way without it.

## Run the pipeline

Build an index only:

```bash
python scripts/build_index.py
```

Run retrieval for all candidates and save matches:

```bash
python scripts/run_retrieval.py
```

### Retriever choice (dense / BM25 / hybrid)

`run_retrieval.py` accepts `--retriever={dense,bm25,hybrid}`. The default is
the original dense (Chroma) retriever to preserve existing reproducibility.
BM25 is a sparse keyword retriever (from `rank_bm25`) over the same vacancy
documents. Hybrid fuses dense and BM25 via reciprocal rank fusion (RRF):

```bash
# Pure BM25 (no embedding model needed)
python scripts/run_retrieval.py --retriever bm25 --top-k 20

# Hybrid: dense top-40 + BM25 top-40 fused into top-20
python scripts/run_retrieval.py --retriever hybrid --top-k 20

# Tweak fusion: weight BM25 higher and shrink RRF constant for sharper top-K
python scripts/run_retrieval.py \
  --retriever hybrid \
  --hybrid-weight-bm25 1.5 \
  --hybrid-weight-dense 1.0 \
  --hybrid-rrf-k 30
```

Hybrid matches emit additional fields per row: `bm25_score`, `rrf_score`,
`dense_rank`, and `bm25_rank`. These are persisted by `run_retrieval.py` and
can later be consumed by the reranker — pass `--use-bm25-feature` to
`scripts/evaluate.py --train-lambdarank` to include `bm25_score` as an extra
LambdaRank input feature.

Default retrieval now writes top-20 matches per candidate (`TOP_K=20`) for faster rerank
feature generation. Evaluation can still compute Recall@K up to 100 because it retrieves
directly from Chroma during `scripts/evaluate.py`.

Optional: print Recall@K directly from produced match rows (no extra retrieval pass):

```bash
python scripts/run_retrieval.py --report-recall-ks 20
```

To measure `recall@100`, run retrieval with `--top-k 100` and request both Ks:

```bash
python scripts/run_retrieval.py \
  --top-k 100 \
  --output-path data/processed/candidate_vacancy_matches_top100.jsonl \
  --report-recall-ks 20 100
```

Evaluate Recall@K:

```bash
python scripts/evaluate.py
```

To explicitly print `recall@20` and `recall@100` from Chroma:

```bash
python scripts/evaluate.py --ks 20 100 --misses-k 100
```

### Retrieval metrics snapshot

From cached match files in `data/processed/` (4,995 candidates with ground
truth in every run):

| Retriever | Output file | Recall@20 | Recall@100 |
| --- | --- | ---: | ---: |
| Dense, top-20 | `candidate_vacancy_matches_top20.jsonl` | 0.6683 (3,338 / 4,995) | n/a (max rank 20) |
| Dense, top-100 | `candidate_vacancy_matches_top100.jsonl` | 0.6705 (3,348 / 4,995) | 0.7926 (3,959 / 4,995) |
| Hybrid (dense + BM25), top-100 | `candidate_vacancy_matches_top100_hybrid.jsonl` | **0.7538** (3,765 / 4,995) | **0.8342** (4,167 / 4,995) |

Hybrid retrieval (reciprocal rank fusion of the dense Chroma index and a
BM25 index over the same vacancy documents) lifts Recall@20 from 0.6683
to 0.7538 (**+8.6 percentage points**, +428 ground-truth vacancies pulled
into the top-20).

```bash
# Reproduce the hybrid recall numbers
python scripts/run_retrieval.py \
  --retriever hybrid \
  --top-k 100 \
  --report-recall-ks 20 100 \
  --output-path data/processed/candidate_vacancy_matches_top100_hybrid.jsonl
```

#### Recall summary at a glance

![Retriever recall summary](outputs/miss_analysis/recall_summary.png)

### Where the retriever fails (miss analysis)

Generate the visual breakdown for any two retrieval runs with
[scripts/plot_miss_analysis.py](scripts/plot_miss_analysis.py):

```bash
python scripts/plot_miss_analysis.py \
  --label dense  --matches-path data/processed/candidate_vacancy_matches_top20.jsonl \
  --label hybrid --matches-path data/processed/candidate_vacancy_matches_top100_hybrid.jsonl \
  --k 20 \
  --output-dir outputs/miss_analysis
```

The script reuses the same per-candidate bookkeeping as
[scripts/analyze_misses.py](scripts/analyze_misses.py) (a plain-text
companion) and saves one PNG per breakdown.

**Specializations that are systematically harder to retrieve.** The
share of misses where the ground-truth vacancy carries each specialization,
side-by-side for the dense and hybrid runs:

![Missed specializations](outputs/miss_analysis/miss_specializations.png)

**Regions of the missed vacancies.** Hybrid pulls in a lot more
Russia-tagged vacancies (BM25 helps when CV / vacancy vocabulary
overlaps strongly) but does relatively worse on US-based listings:

![Missed regions](outputs/miss_analysis/miss_regions.png)

**Work format of the missed vacancies.** Misses skew toward remote
vacancies in both retrievers because the CV corpus has heavy
`"remote"` / `"english"` boilerplate which dilutes the signal:

![Missed work format](outputs/miss_analysis/miss_work_format.png)

**Required English level on the missed vacancy.** Hybrid noticeably
reduces the b2-English miss bucket — BM25 keyword anchors help when
candidates list English proficiency explicitly:

![Missed English level](outputs/miss_analysis/miss_english_level.png)

**Locale signals in the *candidate* text of misses.** Bars are the share
of missed candidates whose CV contains the signal token. Hybrid removes
some of the "russian"-heavy long-tail misses but does not change the
profile dramatically:

![Locale signals in misses](outputs/miss_analysis/miss_locale_signals.png)

**Candidate text length distribution (hits vs misses).** Short CVs are
slightly over-represented in the miss set for both retrievers, which is
consistent with cold-start being a problem:

![Candidate text length distribution](outputs/miss_analysis/candidate_text_length.png)

#### Honest takeaways from miss analysis

- Hybrid retrieval is a clear net win at Recall@20 (+8.6pp), but it
  trades blind spots: US-based vacancies make up only 6.8% of hybrid
  misses (vs 20.4% for dense), while Russia jumps from 43.9% → 64.2% of
  the (now smaller) miss set. BM25 over-rewards Cyrillic / brand-token
  matches.
- Both retrievers struggle most on `product_design`, `qa_testing`,
  `marketing`, `backend_dev`, and `frontend_dev`. These are the buckets
  to target first with prompt engineering or query rewriting.
- Required English ≥ B2 dominates the miss set in both runs. Candidates
  who list English fluency explicitly are easier to retrieve; we should
  enrich the candidate query with explicit English / locale tokens when
  they appear in the schema.

## Reproducing the full hybrid + schema-feature pipeline

These steps re-run retrieval, LLM expert scoring, and LambdaRank training
end-to-end. The LambdaRank model is trained on hybrid (dense + BM25)
retrieval and uses four feature groups: `cosine_similarity`, `bm25_score`,
LLM expert scores, and schema-based tabular features parsed from
`data/raw/candidates_with_schema.jsonl`.

```bash
# 1) Hybrid retrieval (dense + BM25 via reciprocal rank fusion).
#    Also writes labeled reranker feature rows enriched with schema features.
python scripts/run_retrieval.py \
  --retriever hybrid \
  --top-k 20 \
  --output-path data/processed/candidate_vacancy_matches_top20_hybrid.jsonl \
  --feature-output-path data/processed/rerank_features_top20.jsonl \
  --candidates-schema-path data/raw/candidates_with_schema.jsonl

# 2) Score each retrieved pair with the LLM expert prompts. Schema
#    features are appended automatically before saving.
python scripts/score_rerank_experts.py \
  --features-path data/processed/rerank_features_top20.jsonl \
  --matches-path data/processed/candidate_vacancy_matches_top20_hybrid.jsonl \
  --output-path data/processed/rerank_features_scored_top20.jsonl \
  --candidates-schema-path data/raw/candidates_with_schema.jsonl

# 3) Train all three LambdaRank variants and print the comparison table.
#    The output table shows cosine_similarity, bm25_score,
#    weighted_llm_score, lambdarank_no_llm_expert, lambdarank_no_schema,
#    and finally lambdarank_score (main model, listed last).
python scripts/evaluate.py \
  --rerank-features-path data/processed/rerank_features_scored_top20.jsonl \
  --use-bm25-feature \
  --train-lambdarank
```

Tip: rerunning step 3 against an older scored file that pre-dates schema
features still works — `evaluate.py` enriches the rows in-memory using
`--candidates-schema-path` and `--vacancies-path-for-schema` (both default
to the standard `data/raw/` paths).

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
export RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE=1
export RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE=19
export RERANK_LLM_PROVIDER=litellm
export RERANK_LLM_MODEL=mistral/mistral-small-latest
```

`scripts/score_rerank_experts.py` auto-loads `.env` from the project root before parsing CLI args.

`score_rerank_experts.py` now:
- scores only candidate groups where retrieval contains at least one positive label,
- for LLM scoring only: subsamples negatives per candidate separately for train
  and validation (`RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE` and
  `RERANK_VALIDATION_MAX_NEGATIVES_PER_CANDIDATE`),
- asks for all five expert fields in one JSON response per candidate-vacancy pair,
- supports micro-batching multiple pairs into one LLM request (`RERANK_MICRO_BATCH_SIZE`),
- retries rate-limited calls with exponential backoff,
- sleeps between requests,
- supports additive resume mode with `--rewrite-mode` and periodic persistence
  with `--checkpoint-every`,
- prints progress while scoring.

With top-20 retrieval, row count and scoring cost depend heavily on the negative
sampling caps above. Use smaller train caps for cheaper LLM scoring and larger
validation caps for stronger reranker evaluation coverage.

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
- baseline BM25 on all rows (when rerank rows carry `bm25_score`)
- baseline weighted expert average on all rows
- cosine, BM25 and weighted expert score on the LambdaRank validation split
- `lambdarank_no_llm_expert` (cosine + BM25 + schema features) on the validation split
- `lambdarank_no_schema` (cosine + BM25 + LLM expert scores) on the validation split
- `lambdarank_score` (cosine + BM25 + LLM expert scores + schema features) on the validation split

It also prints compact method-comparison tables (with `lambdarank_score`
shown last as the main model) and saves comparison charts by default:
- `outputs/rerank_method_comparison_all_rows.png`
- `outputs/rerank_method_comparison_validation.png` (when `--train-lambdarank` is used)

### Schema-based reranker features

Candidate descriptions parsed into structured fields live at
`data/raw/candidates_with_schema.jsonl`. The
[schema_features.py](src/sara_retrieve_rerank/schema_features.py) module
turns each candidate-vacancy pair into 82 numeric columns:

- `cand_*` (one-hot education level, candidate-side preferred remote
  policy, employment type, language / skill / industry counts, salary
  expectation in USD, ...).
- `vac_*` (one-hot grade, English level, work format, work type,
  employee type, company type, salary in USD, vacancy score).
- `pair_*` (boolean / count interaction features: format / acceptable
  format / employment-type / seniority / English / industry / skill /
  position / salary match).

These columns are added automatically when:

- `scripts/run_retrieval.py --feature-output-path ...` writes feature
  rows (controlled by `--candidates-schema-path`, default
  `data/raw/candidates_with_schema.jsonl`).
- `scripts/score_rerank_experts.py` runs LLM scoring (same flag).
- `scripts/evaluate.py --rerank-features-path ...` loads rows that
  pre-date schema enrichment — it enriches in-memory before training.

Pass `--candidates-schema-path /dev/null` to disable enrichment if you
want to test the no-schema baseline directly.

### Latest reranking results (top-20, scored features + schema features)

Command:

```bash
python scripts/evaluate.py \
  --rerank-features-path data/processed/rerank_features_scored_top20.jsonl \
  --use-bm25-feature \
  --train-lambdarank
```

Dataset summary:

| Metric | Value |
| --- | ---: |
| Feature rows loaded | 28,760 |
| Schema feature columns enriched on the fly | 82 |
| Candidate groups (all rows) | 3,333 |
| LambdaRank train rows | 15,420 |
| LambdaRank validation rows | 13,340 |
| LambdaRank train groups | 2,666 |
| LambdaRank validation groups | 667 |
| Train rows per group | 5.78 |
| Validation rows per group | 20.00 |

All rows comparison (baselines only — LambdaRank is trained / evaluated
on the validation split below):

| Method | Recall@1 | NDCG@1 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Δ NDCG@10 vs cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cosine_similarity` | 0.6949 | 0.6949 | 0.9418 | 0.8278 | 0.9817 | 0.8411 | +0.0000 |
| `weighted_llm_score` | 0.5053 | 0.5053 | 0.9415 | 0.7423 | 0.9877 | 0.7576 | -0.0834 |

Validation comparison (same split used for LambdaRank model validation,
with **`lambdarank_score` shown last as the main model**):

| Method | Recall@1 | NDCG@1 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Δ NDCG@10 vs cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cosine_similarity` | 0.5517 | 0.5517 | 0.7901 | 0.6830 | 0.9085 | 0.7206 | +0.0000 |
| `weighted_llm_score` | 0.2729 | 0.2729 | 0.7256 | 0.5035 | 0.9385 | 0.5735 | -0.1471 |
| `lambdarank_no_llm_expert` | 0.8606 | 0.8606 | 0.9805 | 0.9291 | 0.9985 | 0.9351 | +0.2145 |
| `lambdarank_no_schema` | 0.6027 | 0.6027 | 0.9175 | 0.7683 | 0.9790 | 0.7884 | +0.0677 |
| **`lambdarank_score`** | **0.8591** | **0.8591** | **0.9880** | **0.9327** | **0.9985** | **0.9362** | **+0.2156** |

Key takeaways (feature ablation):
- **Schema features carry most of the lift.** The `lambdarank_no_llm_expert`
  variant (cosine + schema features only — no LLM expert scores) gets to
  `ndcg@10 = 0.9351`, almost the entire +0.2156 jump from cosine to the
  full model.
- **LLM expert scores alone are weaker.** `lambdarank_no_schema`
  (cosine + LLM expert scores, no schema) lifts cosine by only +0.0677
  ndcg@10. The LLM features still help in combination with schema
  features — the full model is +0.0011 ndcg@10 above schema-only — but
  the marginal value is small once schema features are in.
- **Cold-start funnel.** Hybrid retrieval finds the true vacancy in
  top-20 for ~75% of labeled candidates. On the reranker validation
  split (top-20 hit groups), the full `lambdarank_score` places the true
  vacancy at rank 1 in **85.9%** of cases and within top-10 in **99.9%**
  of cases.

Saved artifacts:
- `data/processed/lambdarank_train_rows_top20.jsonl`
- `data/processed/lambdarank_validation_rows_top20.jsonl`
- `data/processed/lambdarank_validation_scored_top20.jsonl` (carries
  `cosine_similarity`, `weighted_llm_score`, `lambdarank_no_llm_expert`,
  `lambdarank_no_schema`, `lambdarank_score` plus all 82 schema columns)
- `outputs/lambdarank_model_top20.pkl`
- `outputs/rerank_method_comparison_all_rows.png`
- `outputs/rerank_method_comparison_validation.png`

Comparison plots:

![All rows comparison](outputs/rerank_method_comparison_all_rows.png)
![Validation comparison](outputs/rerank_method_comparison_validation.png)

The reusable reranking helpers live in `src/sara_retrieve_rerank/reranking.py`.
LightGBM is optional for the base retriever, but install the full requirements or
`python -m pip install -e ".[rerank]"` before using `--train-lambdarank`.
LambdaRank training keeps all positives and uses
`RERANK_TRAIN_MAX_NEGATIVES_PER_CANDIDATE` for train-group negative subsampling.
Validation-group negative subsampling for LLM scoring is controlled separately with
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

## LLM call audit logging

Pass `--llm-log-path data/logs/rerank_llm.jsonl` (or set
`RERANK_LLM_LOG_PATH`) when running `scripts/score_rerank_experts.py` to
record every LLM call as a JSONL row containing the prompt, response,
latency, and any error. The helper lives in
[src/sara_retrieve_rerank/llm_logging.py](src/sara_retrieve_rerank/llm_logging.py)
and can be reused to wrap any future LLM-based component (CV parser,
chat agent, query rewriter) without changing call sites.

## Persistent index & high-level pipeline

`src/sara_retrieve_rerank/pipeline.py` exposes a single
`RecommendationPipeline` class that wraps:

- the persistent Chroma index (reused from `data/chroma` when present),
- an in-memory BM25 index built from the same vacancy documents, and
- an optional LambdaRank reranker loaded from a pickle.

```python
from sara_retrieve_rerank.pipeline import RecommendationPipeline

pipeline = RecommendationPipeline(
    vacancies_path="data/raw/vacancies_safe_ml_dataset_nozip.jsonl",
    reranker_model_path="outputs/lambdarank_model_top20.pkl",
)
matches = pipeline.match(
    {"id": "alice", "text": "Senior ML engineer with PyTorch experience"},
    k=10,
    retriever="hybrid",
)
```

The first run builds the Chroma index and writes it to `data/chroma`.
Subsequent runs reuse it, so process startup is fast — this is the seam any
UI / API layer is meant to import.

## Error analysis

After running retrieval, break down the misses by candidate and vacancy
attributes:

```bash
python scripts/analyze_misses.py \
  --candidates-path data/raw/results_5000_scores_acknowledged.jsonl \
  --vacancies-path data/raw/vacancies_safe_ml_dataset_nozip.jsonl \
  --matches-path data/processed/candidate_vacancy_matches_top20.jsonl \
  --k 20
```

The script prints:

- Recall@K and miss count.
- Candidate text length distribution for hits vs. misses (so we can see
  whether short CVs are systematically harder to retrieve for).
- Frequency of locale signals (`remote`, `hybrid`, `visa`, `english`, ...)
  in missed candidates' text.
- The top vacancy fields — `specializations`, `work_format`, `regions`,
  `english_level` — that appear in the ground-truth vacancies of misses,
  so we can see which kinds of roles the retriever blind-spots.

### Known limitations (honest assessment)

- **Cold start / sparse CVs.** Recall@20 currently sits at ~67%. Inspection
  shows short candidate texts (< 200 chars) are over-represented in the
  miss set: with fewer keywords the dense retriever has less to anchor on
  and BM25 has fewer term matches. Hybrid retrieval mitigates this but
  does not eliminate it.
- **Specialization blind spots.** Some specializations are under-represented
  in the training corpus, so cosine similarity gives diffuse top-K lists.
  The error-analysis script surfaces which specializations these are so we
  can target prompt or schema improvements.
- **Offline → online quality drift.** The LambdaRank reranker is trained on
  cached offline matches. Switching to live data (Adzuna) may degrade
  ranking quality until features are recomputed; this is intentional
  scope for the next iteration.

## Tests and CI

Run the full test suite:

```bash
python -m pytest tests/
```

A GitHub Actions workflow at [.github/workflows/tests.yml](.github/workflows/tests.yml)
runs the same suite on every push and pull request against `main`.

## Good next refactoring targets

- Add a query-rewriter LLM step for sparse / short CVs (cold-start fix).
- Train the LambdaRank reranker on hybrid (BM25 + dense) candidate pools.
- Add the Adzuna tool-call so live searches augment the cached corpus.
- Add cross-encoder reranking on the top-50 fused candidates.
