"""
Adapter layer between the FastAPI route and your existing ML code.

PLUG IN YOUR CODE HERE:
- Replace the imports and stub calls with your actual vector store,
  retriever, and reranker.
- The interface contract this file exposes to the rest of the app is:
    RecommenderService.search(profile: dict, top_k: int) -> list[dict]
"""

# ── Your ML imports go here ────────────────────────────────────────────────────
# from your_ml_module import VectorStore, Retriever, Reranker
# ──────────────────────────────────────────────────────────────────────────────

import asyncio
import uuid


class RecommenderService:
    def __init__(self):
        # Initialise your components here, e.g.:
        # self.store     = VectorStore(path="...")
        # self.retriever = Retriever(self.store)
        # self.reranker  = Reranker(model="...")
        pass

    async def search(self, profile: dict, top_k: int = 10) -> list[dict]:
        """
        1. Build a query string / embedding from the user profile.
        2. Retrieve candidate jobs from the vector store.
        3. Rerank and return top_k results.

        Replace the stub below with your actual implementation.
        """

        # ── Stub: remove when connecting real ML code ──────────────────────────
        await asyncio.sleep(0.1)  # simulate latency
        jobs = _stub_jobs(profile, top_k)
        # ──────────────────────────────────────────────────────────────────────

        # ── Real implementation skeleton ───────────────────────────────────────
        # query = _build_query(profile)
        # candidates = await asyncio.to_thread(self.retriever.retrieve, query, k=top_k * 3)
        # ranked = await asyncio.to_thread(self.reranker.rerank, query, candidates, k=top_k)
        # jobs = [_format_job(j) for j in ranked]
        # ──────────────────────────────────────────────────────────────────────

        return jobs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_query(profile: dict) -> str:
    """Convert structured profile to a retrieval query string."""
    parts = []
    if profile.get("title"):
        parts.append(profile["title"])
    if profile.get("skills"):
        parts.append(", ".join(profile["skills"]))
    if profile.get("location"):
        parts.append(profile["location"])
    if profile.get("notes"):
        parts.append(profile["notes"])
    return " | ".join(parts)


def _format_job(raw: dict) -> dict:
    """Map your ML layer's job schema to the API response schema."""
    return {
        "id":          raw.get("id", str(uuid.uuid4())),
        "title":       raw["title"],
        "company":     raw["company"],
        "location":    raw.get("location", "Remote"),
        "salary":      raw.get("salary"),
        "tags":        raw.get("tags", []),
        "summary":     raw.get("description", "")[:300],
        "url":         raw.get("url", "#"),
        "match_score": float(raw.get("score", 0.0)),
    }


# ── Stub data (delete once real ML is connected) ───────────────────────────────

def _stub_jobs(profile: dict, n: int) -> list[dict]:
    skills = profile.get("skills", ["Python"])
    title  = profile.get("title", "Software Engineer")
    loc    = profile.get("location", "Remote")

    templates = [
        ("Acme Corp",    0.97, 90_000),
        ("Beta Systems", 0.91, 85_000),
        ("Gamma Inc",    0.88, 95_000),
        ("Delta Labs",   0.84, 80_000),
        ("Epsilon Tech", 0.79, 75_000),
    ]

    return [
        {
            "id":          str(uuid.uuid4()),
            "title":       f"{title} {'I' * (i % 3 + 1)}",
            "company":     company,
            "location":    loc,
            "salary":      f"${salary:,}–${salary + 20_000:,} / yr",
            "tags":        skills[:3] + ["Remote-friendly"],
            "summary":     (
                f"Join {company} as a {title}. You'll work with "
                + ", ".join(skills[:3])
                + " in a fast-paced, collaborative environment."
            ),
            "url":         f"https://example.com/jobs/{i}",
            "match_score": score - i * 0.03,
        }
        for i, (company, score, salary) in enumerate(templates[:n])
    ]
