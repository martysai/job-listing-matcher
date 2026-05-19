from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.recommender import RecommenderService

router = APIRouter()
recommender = RecommenderService()


class UserProfile(BaseModel):
    """Structured profile extracted from the conversation."""

    title: str | None = None          # e.g. "Senior Python Engineer"
    skills: list[str] = []            # e.g. ["Python", "FastAPI", "PostgreSQL"]
    location: str | None = None       # e.g. "Berlin" or "Remote"
    experience_years: int | None = None
    job_type: str | None = None       # "full-time" | "part-time" | "contract"
    salary_min: int | None = None
    notes: str | None = None          # freeform extra context


class JobRecommendationRequest(BaseModel):
    profile: UserProfile
    top_k: int = 10


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    salary: str | None = None
    tags: list[str] = []
    summary: str
    url: str
    match_score: float  # 0-1, from reranker


@router.post("/jobs/recommend", response_model=list[Job])
async def recommend_jobs(request: JobRecommendationRequest):
    """
    Query the vector store with the user profile.
    Returns reranked job listings sorted by relevance.
    """
    try:
        jobs = await recommender.search(
            profile=request.profile.model_dump(exclude_none=True),
            top_k=request.top_k,
        )
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
