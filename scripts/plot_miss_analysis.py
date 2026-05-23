"""Render miss-analysis figures for one or more retrieval runs.

Reuses the same per-candidate / per-vacancy bookkeeping as
`scripts/analyze_misses.py` but saves a directory of PNGs instead of
printing a text report. The plots are designed for the README so the
reader can quickly see *where* the retriever struggles (which vacancy
specializations, work formats, regions, English-level requirements show
up disproportionately in the miss set) and how a different retriever
changes that profile.

Example
-------

    python scripts/plot_miss_analysis.py \
        --label dense  --matches-path data/processed/candidate_vacancy_matches_top20.jsonl \
        --label hybrid --matches-path data/processed/candidate_vacancy_matches_top100_hybrid.jsonl \
        --k 20 \
        --output-dir outputs/miss_analysis

The script accepts ``--label`` and ``--matches-path`` in pairs so that
several retrieval runs can be compared on the same chart.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "backend" / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

from sara_retrieve_rerank.config import (  # noqa: E402
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_VACANCIES_PATH,
)
from sara_retrieve_rerank.data import load_jsonl  # noqa: E402

LOCALE_SIGNALS: tuple[str, ...] = (
    "remote",
    "hybrid",
    "onsite",
    "office",
    "relocate",
    "visa",
    "english",
    "russian",
)

# Vacancy categorical fields we summarize for the miss set.
VACANCY_FIELDS = ("specializations", "work_format", "regions", "english_level")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot miss-analysis charts for retrieval runs.")
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--vacancies-path", default=str(DEFAULT_VACANCIES_PATH))
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for the matching --matches-path. Pass once per run.",
    )
    parser.add_argument(
        "--matches-path",
        action="append",
        required=True,
        help="Path to a matches JSONL produced by run_retrieval.py.",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/miss_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.label) != len(args.matches_path):
        raise SystemExit("Pass --label and --matches-path the same number of times.")

    candidates = load_jsonl(args.candidates_path)
    vacancies = load_jsonl(args.vacancies_path)
    vacancies_by_id = {str(v.get("dataset_id")): v for v in vacancies}
    labeled = [c for c in candidates if c.get("source_vacancy_id")]
    if not labeled:
        raise SystemExit("No labeled candidates; cannot analyze misses.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for label, matches_path in zip(args.label, args.matches_path, strict=True):
        matches = load_jsonl(matches_path)
        analysis = analyze_run(
            label=label,
            matches=matches,
            labeled_candidates=labeled,
            vacancies_by_id=vacancies_by_id,
            k=args.k,
        )
        runs.append(analysis)
        print_run_summary(analysis)

    plot_recall_summary(runs, output_dir / "recall_summary.png")
    plot_miss_breakdown(
        runs,
        field="specializations",
        title="Missed-vacancy specializations",
        output_path=output_dir / "miss_specializations.png",
        top_n=args.top_n,
    )
    plot_miss_breakdown(
        runs,
        field="regions",
        title="Missed-vacancy regions",
        output_path=output_dir / "miss_regions.png",
        top_n=args.top_n,
    )
    plot_miss_breakdown(
        runs,
        field="work_format",
        title="Missed-vacancy work format",
        output_path=output_dir / "miss_work_format.png",
        top_n=None,
    )
    plot_miss_breakdown(
        runs,
        field="english_level",
        title="Missed-vacancy English level",
        output_path=output_dir / "miss_english_level.png",
        top_n=None,
    )
    plot_locale_signals(runs, output_dir / "miss_locale_signals.png")
    plot_text_length_distribution(runs, output_dir / "candidate_text_length.png")
    print(f"Saved miss-analysis figures to {output_dir}")


def analyze_run(
    *,
    label: str,
    matches: list[dict],
    labeled_candidates: list[dict],
    vacancies_by_id: dict[str, dict],
    k: int,
) -> dict:
    """Compute miss / hit sets and per-field counters for one retrieval run."""
    matches_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        cid = match.get("candidate_id")
        if cid is None:
            continue
        matches_by_candidate[str(cid)].append(match)

    misses: list[dict] = []
    hits: list[dict] = []
    for candidate in labeled_candidates:
        cid = str(candidate.get("id"))
        true_id = str(candidate.get("source_vacancy_id"))
        candidate_matches = sorted(
            matches_by_candidate.get(cid, []),
            key=lambda row: int(row.get("rank") or 0),
        )[:k]
        retrieved_ids = [str(m.get("vacancy_id")) for m in candidate_matches]
        if true_id in retrieved_ids:
            hits.append(candidate)
        else:
            misses.append(candidate)

    field_counters: dict[str, Counter[str]] = {field: Counter() for field in VACANCY_FIELDS}
    resolvable_misses = 0
    for candidate in misses:
        vid = str(candidate.get("source_vacancy_id"))
        vacancy = vacancies_by_id.get(vid)
        if vacancy is None:
            continue
        resolvable_misses += 1
        for field in VACANCY_FIELDS:
            value = vacancy.get(field)
            if isinstance(value, list):
                for item in value:
                    field_counters[field][str(item)] += 1
            elif value:
                field_counters[field][str(value)] += 1

    locale_counter: Counter[str] = Counter()
    for candidate in misses:
        text = str(candidate.get("text") or "").lower()
        for signal in LOCALE_SIGNALS:
            if signal in text:
                locale_counter[signal] += 1

    hit_lengths = [len(str(c.get("text") or "")) for c in hits]
    miss_lengths = [len(str(c.get("text") or "")) for c in misses]

    return {
        "label": label,
        "labeled": len(labeled_candidates),
        "hits": len(hits),
        "misses": len(misses),
        "resolvable_misses": resolvable_misses,
        "recall_at_k": (len(hits) / len(labeled_candidates)) if labeled_candidates else 0.0,
        "field_counters": field_counters,
        "locale_counter": locale_counter,
        "hit_lengths": hit_lengths,
        "miss_lengths": miss_lengths,
    }


def print_run_summary(analysis: dict) -> None:
    label = analysis["label"]
    recall = analysis["recall_at_k"]
    print(
        f"[{label}] recall@K={recall:.4f}  "
        f"hits={analysis['hits']}/{analysis['labeled']}  misses={analysis['misses']}"
    )


def plot_recall_summary(runs: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in runs]
    recalls = [r["recall_at_k"] for r in runs]
    misses = [r["misses"] for r in runs]

    fig, (ax_recall, ax_misses) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    bars = ax_recall.bar(labels, recalls, color=["#5b8def", "#f5a25d"][: len(labels)])
    ax_recall.set_ylim(0.0, 1.0)
    ax_recall.set_ylabel("Recall@K")
    ax_recall.set_title("Recall@K on labeled candidates")
    for bar, value in zip(bars, recalls):
        ax_recall.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
        )
    ax_recall.grid(axis="y", alpha=0.2)

    bars = ax_misses.bar(labels, misses, color=["#5b8def", "#f5a25d"][: len(labels)])
    ax_misses.set_ylabel("# missed candidates")
    ax_misses.set_title("Miss count")
    for bar, value in zip(bars, misses):
        ax_misses.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_misses.grid(axis="y", alpha=0.2)
    fig.suptitle("Retriever performance overview")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_miss_breakdown(
    runs: list[dict],
    *,
    field: str,
    title: str,
    output_path: Path,
    top_n: int | None,
) -> None:
    """Horizontal bar chart of absolute miss counts per category, per run.

    Percentages can be misleading when the two miss sets have very
    different sizes (e.g. dense misses are 1657 vs hybrid 1230) — raw
    counts make the absolute-improvement story easier to read.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Union of top-N keys across runs gives the rows we plot.
    keys: list[str] = []
    seen: set[str] = set()
    for run in runs:
        counter = run["field_counters"][field]
        items = counter.most_common(top_n) if top_n else counter.most_common()
        for key, _count in items:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    if not keys:
        return
    if top_n is not None:
        keys = keys[:top_n]

    n_keys = len(keys)
    n_runs = len(runs)
    bar_height = 0.8 / n_runs
    fig_height = max(3.5, 0.45 * n_keys + 1)
    fig, ax = plt.subplots(figsize=(9, fig_height), constrained_layout=True)
    palette = ["#5b8def", "#f5a25d", "#7bbf63", "#c474ff"]

    y_positions = list(range(n_keys))
    for run_index, run in enumerate(runs):
        counter = run["field_counters"][field]
        counts = [counter.get(key, 0) for key in keys]
        offsets = [
            position + (run_index - (n_runs - 1) / 2) * bar_height for position in y_positions
        ]
        bars = ax.barh(
            offsets,
            counts,
            height=bar_height,
            label=f"{run['label']} (miss n={run['resolvable_misses']})",
            color=palette[run_index % len(palette)],
        )
        for bar, count in zip(bars, counts):
            if count <= 0:
                continue
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{count}",
                va="center",
                fontsize=8,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_xlabel("# missed candidates")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_locale_signals(runs: list[dict], output_path: Path) -> None:
    """Bar chart of locale-signal frequency in candidate text for misses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = list(LOCALE_SIGNALS)
    n_runs = len(runs)
    bar_width = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    palette = ["#5b8def", "#f5a25d", "#7bbf63", "#c474ff"]
    positions = list(range(len(signals)))

    for run_index, run in enumerate(runs):
        counter = run["locale_counter"]
        counts = [counter.get(signal, 0) for signal in signals]
        offsets = [
            position + (run_index - (n_runs - 1) / 2) * bar_width for position in positions
        ]
        bars = ax.bar(
            offsets,
            counts,
            width=bar_width,
            label=f"{run['label']} (miss n={run['misses']})",
            color=palette[run_index % len(palette)],
        )
        for bar, count in zip(bars, counts):
            if count <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(signals)
    ax.set_ylabel("# missed candidates with signal in text")
    ax.set_title("Locale signals in miss-set candidate text")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_text_length_distribution(runs: list[dict], output_path: Path) -> None:
    """Compare hit vs miss candidate text length per run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]
    max_len = 0
    for run in runs:
        if run["hit_lengths"]:
            max_len = max(max_len, max(run["hit_lengths"]))
        if run["miss_lengths"]:
            max_len = max(max_len, max(run["miss_lengths"]))
    bins = _make_bins(max_len)

    for ax, run in zip(axes, runs):
        ax.hist(
            run["hit_lengths"],
            bins=bins,
            alpha=0.6,
            label=f"hits (n={len(run['hit_lengths'])})",
            color="#7bbf63",
        )
        ax.hist(
            run["miss_lengths"],
            bins=bins,
            alpha=0.6,
            label=f"misses (n={len(run['miss_lengths'])})",
            color="#d96c6c",
        )
        ax.set_xlabel("Candidate text length (chars)")
        ax.set_ylabel("# candidates")
        ax.set_title(f"{run['label']}: hit vs miss text length")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right")
    fig.suptitle("Candidate text length distribution")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _make_bins(max_len: int) -> list[int]:
    if max_len <= 0:
        return [0, 1]
    step = max(200, int(math.ceil(max_len / 30 / 100.0) * 100))
    return list(range(0, max_len + step, step))


def _percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (q / 100.0) * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: Iterable[int]) -> str:
    values = list(values)
    if not values:
        return "n=0"
    return (
        f"n={len(values)} median={int(statistics.median(values))} "
        f"p25={int(_percentile(values, 25))} p75={int(_percentile(values, 75))}"
    )


if __name__ == "__main__":
    main()
