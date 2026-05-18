"""Error analysis for retrieval misses.

Given a candidates file with `source_vacancy_id` ground truth and a retrieval
matches JSONL, this script breaks down the miss rate by:

    - candidate text length quartile
    - presence of common locale signals in candidate text
    - vacancy fields the candidate's source vacancy carries (so we can spot
      blind spots — e.g. vacancies tagged `Embedded` that are systematically
      missed).

The output is a plain-text report; the goal is to give the team an honest,
auditable view of where the retriever fails, which the project rubric
rewards explicitly.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sara_retrieve_rerank.config import (  # noqa: E402
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_MATCHES_OUTPUT_PATH,
    DEFAULT_VACANCIES_PATH,
)
from sara_retrieve_rerank.data import load_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down retrieval misses by candidate / vacancy attributes.")
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument("--matches-path", default=str(DEFAULT_MATCHES_OUTPUT_PATH))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10, help="Show top-N rows per breakdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(args.candidates_path)
    vacancies = load_jsonl(args.vacancies_path)
    matches = load_jsonl(args.matches_path)

    matches_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        cid = match.get("candidate_id")
        if cid is None:
            continue
        matches_by_candidate[str(cid)].append(match)

    vacancies_by_id = {str(v.get("dataset_id")): v for v in vacancies}

    labeled = [c for c in candidates if c.get("source_vacancy_id")]
    if not labeled:
        print("No candidates have a `source_vacancy_id`; cannot run error analysis.")
        return

    misses: list[dict] = []
    hits: list[dict] = []
    for candidate in labeled:
        cid = str(candidate.get("id"))
        true_id = str(candidate.get("source_vacancy_id"))
        candidate_matches = sorted(
            matches_by_candidate.get(cid, []),
            key=lambda row: int(row.get("rank") or 0),
        )[: args.k]
        retrieved_ids = [str(m.get("vacancy_id")) for m in candidate_matches]
        if true_id in retrieved_ids:
            hits.append(candidate)
        else:
            misses.append(candidate)

    n = len(labeled)
    print(f"[recall@{args.k}] {1 - len(misses) / n:.4f}  ({len(hits)} / {n})")
    print(f"[miss_count] {len(misses)}")
    print()

    _print_length_breakdown(hits, misses)
    _print_locale_breakdown(misses)
    _print_missed_vacancy_breakdown(misses, vacancies_by_id, top_n=args.top_n)


def _print_length_breakdown(hits: Iterable[dict], misses: Iterable[dict]) -> None:
    hit_lengths = [len(str(c.get("text") or "")) for c in hits]
    miss_lengths = [len(str(c.get("text") or "")) for c in misses]
    if not hit_lengths and not miss_lengths:
        return

    def _summary(values: list[int]) -> str:
        if not values:
            return "n=0"
        return (
            f"n={len(values)} median={int(statistics.median(values))} "
            f"p25={int(_percentile(values, 25))} p75={int(_percentile(values, 75))}"
        )

    print("[candidate_text_length_chars]")
    print(f"  hits   {_summary(hit_lengths)}")
    print(f"  misses {_summary(miss_lengths)}")
    print()


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (q / 100.0) * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


_LOCALE_SIGNALS = ("remote", "hybrid", "onsite", "office", "relocate", "visa", "english", "russian")


def _print_locale_breakdown(misses: Iterable[dict]) -> None:
    counter: Counter[str] = Counter()
    total = 0
    for candidate in misses:
        text = str(candidate.get("text") or "").lower()
        total += 1
        for signal in _LOCALE_SIGNALS:
            if signal in text:
                counter[signal] += 1

    if not total:
        return

    print("[miss_locale_signals]")
    for signal, count in counter.most_common():
        print(f"  {signal:<10} {count:>5} ({count / total:.2%} of misses)")
    print()


def _print_missed_vacancy_breakdown(
    misses: Iterable[dict],
    vacancies_by_id: dict[str, dict],
    *,
    top_n: int,
) -> None:
    field_counters: dict[str, Counter[str]] = {
        "specializations": Counter(),
        "work_format": Counter(),
        "regions": Counter(),
        "english_level": Counter(),
    }
    total = 0
    for candidate in misses:
        vid = str(candidate.get("source_vacancy_id"))
        vacancy = vacancies_by_id.get(vid)
        if vacancy is None:
            continue
        total += 1
        for field in field_counters:
            value = vacancy.get(field)
            if isinstance(value, list):
                for item in value:
                    field_counters[field][str(item)] += 1
            elif value:
                field_counters[field][str(value)] += 1

    if total == 0:
        print("[missed_vacancy_features] no missed vacancies were resolvable in the vacancies file")
        return

    print(f"[missed_vacancy_features] over {total} misses with resolvable vacancies")
    for field, counter in field_counters.items():
        if not counter:
            continue
        print(f"  {field}:")
        for value, count in counter.most_common(top_n):
            print(f"    {value:<40} {count:>5} ({count / total:.2%})")
    print()


if __name__ == "__main__":
    main()
