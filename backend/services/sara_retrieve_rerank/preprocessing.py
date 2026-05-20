from __future__ import annotations

from bs4 import BeautifulSoup


def clean_html(html: str | None) -> str:
    """Convert HTML to plain text, preserving spaces between tags."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def join_list(value) -> str:
    """Format list-like metadata values as a comma-separated string."""
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def candidate_to_query(candidate: dict) -> str:
    """Return the text used to search vacancies for a candidate."""
    return candidate.get("text", "")
