"""
Adapter layer between the FastAPI route and the sara_retrieve_rerank ML pipeline.

Reads three env vars:
  VACANCIES_PATH          (required) path to vacancy JSONL corpus
  CHROMA_DIR              (optional) Chroma persist dir, default data/chroma
  LAMBDARANK_MODEL_PATH   (optional) trained .pkl reranker; absent = no reranking
"""

import asyncio
import os

from sara_retrieve_rerank.pipeline import RecommendationPipeline

_SCORE_MIN = 0.55
_SCORE_MAX = 0.97
_SCORE_DEGENERATE = 0.80
_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "RUB": "₽"}


class RecommenderService:
    def __init__(self):
        vacancies_path = os.environ["VACANCIES_PATH"]
        chroma_dir = os.environ.get("CHROMA_DIR", "data/chroma")
        model_path = os.environ.get("LAMBDARANK_MODEL_PATH") or None

        self.pipeline = RecommendationPipeline(
            vacancies_path=vacancies_path,
            persist_directory=chroma_dir,
            reranker_model_path=model_path,
        )
        self._vacancy_lookup: dict[str, dict] = {
            str(v["dataset_id"]): v for v in self.pipeline.vacancies
        }

    async def search(self, profile: dict, top_k: int = 10) -> list[dict]:
        candidate = {"text": _build_query(profile)}
        matches = await asyncio.to_thread(self.pipeline.match, candidate, k=top_k)
        jobs = [_format_job(m, self._vacancy_lookup) for m in matches]
        _normalize_scores(jobs)
        return jobs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_query(profile: dict) -> str:
    """Convert structured profile to a retrieval query string."""
    jd = profile.get("job_description", {})
    cd = profile.get("candidate_description", {})
    parts = []
    if jd.get("desired_positions"):
        parts.append(", ".join(jd["desired_positions"]))
    skills = cd.get("skills", []) + cd.get("languages", []) + jd.get("desired_tech_stack", [])
    if skills:
        parts.append(", ".join(dict.fromkeys(skills)))
    wm = jd.get("preferred_work_mode") or {}
    if wm.get("preferred_remote_policy"):
        parts.append(", ".join(wm["preferred_remote_policy"]))
    locations = jd.get("preferred_locations", [])
    if locations:
        parts.append(", ".join(locations))
    return " | ".join(parts)


def _format_job(match: dict, lookup: dict) -> dict:
    """Map a pipeline match row + raw vacancy to the API response schema."""
    vacancy_id = str(match.get("vacancy_id", ""))
    vac = lookup.get(vacancy_id, {})

    # Tags come back from Chroma metadata as a comma-joined string.
    raw_tags = match.get("tags") or ""
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else []

    # Location: prefer cities, fall back to regions (both are comma-joined strings).
    location_raw = match.get("cities") or match.get("regions") or vac.get("location")
    location = _pretty_location(location_raw)

    # Salary: raw vacancy field may be absent; format to readable string if present.
    salary = _format_salary(vac.get("salary"))

    # Summary: prefer short description, fall back to full text, truncate.
    summary_raw = vac.get("tldr_sanitized") or vac.get("text_sanitized") or ""
    summary = summary_raw[:300] if summary_raw else ""

    # Score: prefer reranker score, then fusion score, then cosine similarity.
    match_score = float(
        match.get("lambdarank_score")
        or match.get("rrf_score")
        or match.get("cosine_similarity")
        or 0.0
    )

    return {
        "id":          vacancy_id,
        "title":       match.get("title") or vac.get("title", ""),
        "company":     vac.get("company_name") or vac.get("company", ""),
        "location":    location,
        "salary":      salary,
        "tags":        tags,
        "summary":     summary,
        "url":         vac.get("url", "#"),
        "match_score": match_score,
    }


def _pretty_location(raw) -> str:
    """Turn raw 'united_kingdom, berlin' → 'United Kingdom, Berlin'."""
    if not raw:
        return "Remote"
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    out = []
    for p in parts:
        words = p.replace("_", " ").split()
        pretty = " ".join(
            w if (len(w) <= 3 and w.isupper()) else w.capitalize() for w in words
        )
        out.append(pretty)
    return ", ".join(out) if out else "Remote"


def _format_salary(raw) -> str | None:
    """Format dict salary {min,max,currency,salary_in_usd} or pass through string."""
    if not raw:
        return None
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return str(raw)
    cur = (raw.get("currency") or "").upper()
    sym = _CURRENCY_SYMBOLS.get(cur, cur + " " if cur else "")
    lo, hi = raw.get("min"), raw.get("max")
    usd = raw.get("salary_in_usd")
    if lo is not None and hi is not None:
        return f"{sym}{int(lo):,} – {int(hi):,}"
    if lo is not None:
        return f"from {sym}{int(lo):,}"
    if hi is not None:
        return f"up to {sym}{int(hi):,}"
    if usd is not None:
        return f"~${int(usd):,}"
    return None


def _normalize_scores(jobs: list[dict]) -> None:
    """Min-max rescale match_score in place to [_SCORE_MIN, _SCORE_MAX]."""
    if not jobs:
        return
    scores = [j.get("match_score", 0.0) for j in jobs]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-9:
        for j in jobs:
            j["match_score"] = _SCORE_DEGENERATE
        return
    out_span = _SCORE_MAX - _SCORE_MIN
    for j in jobs:
        norm = (j.get("match_score", 0.0) - lo) / span
        j["match_score"] = _SCORE_MIN + norm * out_span
