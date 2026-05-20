# `adzuna_rag.py` — quick start

Simple RAG-style live job search over the public [Adzuna API](https://developer.adzuna.com/).
Give it a candidate's free-text job request and it returns the top live vacancies
that match.

## 1. Install

From the repo root:

```bash
python -m pip install -e .[server,rerank]
```

## 2. Credentials

Sign up at <https://developer.adzuna.com/> and put your keys in a local `.env`
at the repo root (already gitignored):

```text
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
```

You can also pass them on the CLI:

```bash
python scripts/adzuna_rag.py --query "..." --app-id ... --app-key ...
```

## 3. Run

Single example — senior C++ roles in London:

```bash
python scripts/adzuna_rag.py \
    --query "find the senior c++ software engineer roles in london" \
    --where london \
    --countries gb \
    --top-k 15 \
    --output-jsonl outputs/adzuna_senior_cpp_london.jsonl
```

The raw Adzuna responses for that command are written to
`outputs/adzuna_senior_cpp_london.jsonl` (50 rows) and the console transcript
to `outputs/adzuna_senior_cpp_london.txt`. Both paths are **gitignored** — see
section 8 below for why we do not commit raw Adzuna data.

Multi-country (default — UK plus several European endpoints):

```bash
python scripts/adzuna_rag.py --query "junior data scientist remote Python NLP" --top-k 10
```

## 4. CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--query` | required | Candidate's free-text job request. |
| `--query-file` | — | Read the query from a text file instead of `--query`. |
| `--countries` | `gb,de,fr,nl,it,es,pl` | Comma-separated Adzuna country codes. Supported: `at au be br ca ch de es fr gb in it mx nl nz pl sg us za`. |
| `--where` | — | Adzuna location filter, e.g. `london`. |
| `--max-days-old` | — | Restrict to jobs posted within N days. |
| `--results-per-page` | `50` | Adzuna page-size cap is 50. |
| `--top-k` | `10` | How many merged results to print. |
| `--strict-and` | off | Require every word in the query to match (Adzuna `what`). Default is `what_or` (any-of) so long free-text queries don't collapse to zero hits. |
| `--no-cache` | off | Bypass the disk cache and re-hit the API. |
| `--cache-dir` | `data/cache/adzuna/` | Where to store cached JSON responses. |
| `--output-jsonl` | — | Path to dump merged results as JSONL. |
| `--app-id` / `--app-key` | env | Override `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`. |

## 5. How it ranks

- Each Adzuna country endpoint is called once per run with `sort_by=relevance`.
- The top of each country's result list is treated as score `1.0`, the bottom as
  near `0.0` (derived relevance), so results from multiple countries can be
  merged into one ranked list.
- No client-side reranker — the ordering you see is Adzuna's own relevance
  judgement of how well each posting matches the candidate query.

## 6. Free-tier friendliness

Adzuna's free tier has a monthly call cap, so the script:

- Makes **one HTTP call per country per run** (page 1, up to 50 results).
- Caches every raw response under `data/cache/adzuna/` keyed by
  `(country, query, page, where, max_days_old, strict_and)`.
- On a repeat run with the same parameters, hits cache and makes **zero** API
  calls. Use `--no-cache` only when you want fresh results.

The console line `Adzuna calls made this run: N` tells you exactly how many
calls you burned.

## 7. Output schema (JSONL)

One JSON object per line in `--output-jsonl`. Fields:

```json
{
  "id": "5721597110",
  "country": "gb",
  "title": "Senior Engineer",
  "company": "Kier Group",
  "location": "South Bank, South East London",
  "description": "...",
  "salary_min": 72438.05,
  "salary_max": 72438.05,
  "contract_time": "full_time",
  "contract_type": "permanent",
  "category": "Engineering Jobs",
  "created": "2026-05-07T12:42:42Z",
  "redirect_url": "https://www.adzuna.co.uk/jobs/land/ad/...",
  "adzuna_relevance": 1.0
}
```

## 8. Legal use & attribution

This script hits the public [Adzuna API](https://developer.adzuna.com/) under
its [Terms of Service](https://developer.adzuna.com/docs/terms_of_service). In
short:

- **Personal research** is explicitly permitted on the free tier
  (25/min, 250/day, 1000/week, 2500/month).
- Use by an **academic, commercial, or government organisation** (or its
  affiliates / individuals) is limited to a **14-day validation trial**;
  beyond that, a written licence may be required, and the data may not be
  used "in its original format or in aggregation … to deliver any ongoing
  work or research" without Adzuna's written consent.
- **Attribution is required** wherever Adzuna data is published. Reference
  "The Adzuna API" with a link to <https://www.adzuna.co.uk/> (or the
  relevant local domain). UI ad listings need the "Jobs by Adzuna" badge.
- **Termination clause**: on termination, all Adzuna data must be removed
  from your sites. For this reason cached responses and example output dumps
  in this repo live under gitignored paths (`data/cache/adzuna/`, `outputs/`)
  and are **not** committed.

Job listings retrieved with this script are powered by
[The Adzuna API](https://www.adzuna.co.uk/).

## 9. Troubleshooting

- **Zero results** on a long free-text query: that usually means `--strict-and`
  is on. Without it, the script uses `what_or` and should always return
  something for a reasonable query.
- **Mojibake (`?` or `�`) in the terminal**: PowerShell can't render some
  non-ASCII characters from job descriptions. Use `--output-jsonl` to get
  clean UTF-8 results in a file.
- **HTTP 401 / 403**: bad `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`. Verify them at
  <https://developer.adzuna.com/admin/access_details>.
- **HTTP 429**: you hit the monthly call limit. Wait, or use the cached
  responses already on disk.
