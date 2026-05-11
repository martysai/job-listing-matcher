from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt


def plot_vacancies_stats(vacancies: list[dict], name: str, top_n: int = 20) -> Counter:
    """Plot the most common values for a vacancy list field."""
    counts = Counter(
        value
        for vacancy in vacancies
        for value in vacancy.get(name, [])
    )

    print(len(counts))
    print(counts)

    top_items = counts.most_common(top_n)
    labels = [item[0] for item in top_items]
    values = [item[1] for item in top_items]

    plt.figure(figsize=(8, 6))
    plt.barh(labels, values)
    plt.xlabel("Number of vacancies")
    plt.ylabel(name)
    plt.title(f"Top {top_n} {name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return counts
