"""Simple RAG-style vacancy search using the Adzuna public API.

Use case: a candidate writes a free-text job request (e.g. "junior data scientist
remote, Python, NLP") and this script retrieves matching live vacancies from
Adzuna across one or more country endpoints, then prints the top results.

Ranking is Adzuna's built-in relevance order (the API already returns jobs
sorted by relevance to the query). Results are merged across countries and
re-sorted by a small derived relevance score so the top matches surface first.

The Adzuna free tier has a monthly call budget, so this script:
- makes one HTTP call per country (page 1, 50 results),
- caches every response on disk under data/cache/adzuna/,
- reuses cached responses on repeated runs.

Credentials are read from ADZUNA_APP_ID / ADZUNA_APP_KEY in the environment
or in a local .env file. They can also be passed on the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import os
import textwrap

from sara_retrieve_rerank.adzuna import (
    AdzunaJob,
    SUPPORTED_COUNTRIES,
    search_many_countries,
)


DEFAULT_COUNTRIES = "gb,de,fr,nl,it,es,pl"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "adzuna"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple RAG over the Adzuna API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        help="Candidate job request as free text. Required unless --query-file is set.",
    )
    parser.add_argument(
        "--query-file",
        help="Path to a text file whose contents are used as the candidate query.",
    )
    parser.add_argument(
        "--countries",
        default=DEFAULT_COUNTRIES,
        help=(
            "Comma-separated Adzuna country codes. "
            f"Default: {DEFAULT_COUNTRIES}. Supported: {','.join(sorted(SUPPORTED_COUNTRIES))}."
        ),
    )
    parser.add_argument(
        "--where",
        default=None,
        help="Optional location filter passed to Adzuna (e.g. 'london').",
    )
    parser.add_argument(
        "--max-days-old",
        type=int,
        default=None,
        help="Optional age filter in days.",
    )
    parser.add_argument(
        "--results-per-page",
        type=int,
        default=50,
        help="Adzuna max is 50.",
    )
    parser.add_argument(
        "--strict-and",
        action="store_true",
        help=(
            "Require every word in the query to match (Adzuna `what` AND). "
            "Default is any-of (`what_or`), which keeps free-text queries from "
            "collapsing to zero hits."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many merged top results to print.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Where to cache raw Adzuna JSON responses.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk cache (will burn quota on every run).",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path to dump merged results as JSONL.",
    )
    parser.add_argument(
        "--app-id",
        default=None,
        help="Override ADZUNA_APP_ID env var.",
    )
    parser.add_argument(
        "--app-key",
        default=None,
        help="Override ADZUNA_APP_KEY env var.",
    )
    return parser.parse_args()


def _resolve_query(args: argparse.Namespace) -> str:
    if args.query and args.query_file:
        raise SystemExit("Pass either --query or --query-file, not both.")
    if args.query_file:
        return Path(args.query_file).read_text(encoding="utf-8").strip()
    if args.query:
        return args.query.strip()
    raise SystemExit("Either --query or --query-file is required.")


def _resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    app_id = args.app_id or os.environ.get("ADZUNA_APP_ID")
    app_key = args.app_key or os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise SystemExit(
            "Missing Adzuna credentials. Set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env, "
            "in your environment, or pass --app-id / --app-key."
        )
    return app_id, app_key


def _format_job(job: AdzunaJob, rank: int) -> str:
    salary = ""
    if job.salary_min or job.salary_max:
        lo = f"{int(job.salary_min):,}" if job.salary_min else "?"
        hi = f"{int(job.salary_max):,}" if job.salary_max else "?"
        salary = f" | salary {lo}-{hi}"
    contract = ""
    if job.contract_time or job.contract_type:
        contract = f" | {job.contract_time or ''} {job.contract_type or ''}".rstrip()
    description = textwrap.shorten(job.description, width=280, placeholder=" …")
    return (
        f"#{rank:>2}  [{job.country}]  {job.title}\n"
        f"      {job.company} — {job.location}{salary}{contract}\n"
        f"      {description}\n"
        f"      {job.redirect_url}"
    )


def main() -> None:
    _load_dotenv(ROOT / ".env")
    args = parse_args()
    query = _resolve_query(args)
    app_id, app_key = _resolve_credentials(args)

    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    print(f"Query: {query!r}")
    print(f"Countries: {countries}")
    if args.where:
        print(f"Where filter: {args.where}")
    if cache_dir:
        print(f"Cache dir: {cache_dir}")
    else:
        print("Cache: disabled")

    jobs, metas = search_many_countries(
        app_id=app_id,
        app_key=app_key,
        countries=countries,
        query=query,
        results_per_page=args.results_per_page,
        where=args.where,
        max_days_old=args.max_days_old,
        strict_and=args.strict_and,
        cache_dir=cache_dir,
    )

    api_calls = sum(0 if m["from_cache"] else 1 for m in metas)
    print()
    print(f"Adzuna calls made this run: {api_calls} (of {len(metas)} countries; rest from cache)")
    for meta in metas:
        source = "cache" if meta["from_cache"] else "live"
        total = meta.get("count_total")
        print(
            f"  {meta['country']}: {meta['count_returned']} returned, "
            f"~{total} total available, source={source}"
        )

    jobs.sort(key=lambda j: j.adzuna_relevance or 0.0, reverse=True)
    top = jobs[: args.top_k]

    print()
    print(f"Top {len(top)} matches across {len(countries)} country endpoints:")
    print("-" * 80)
    for rank, job in enumerate(top, start=1):
        print(_format_job(job, rank))
        print()

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for job in jobs:
                fh.write(json.dumps(job.to_dict(), ensure_ascii=False))
                fh.write("\n")
        print(f"Wrote {len(jobs)} merged results to {out_path}")


if __name__ == "__main__":
    main()
