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
        return [_format_job(m, self._vacancy_lookup) for m in matches]


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
    location = match.get("cities") or match.get("regions") or vac.get("location", "Remote")

    # Salary: raw vacancy field may be absent; format to string if present.
    salary_raw = vac.get("salary")
    salary = str(salary_raw) if salary_raw else None

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
