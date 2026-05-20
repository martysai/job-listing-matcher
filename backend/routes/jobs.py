from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.recommender import RecommenderService
from services.sara_candidate_poll import JobAndCandidateDescription

router = APIRouter()
recommender = RecommenderService()


class JobRecommendationRequest(BaseModel):
    profile: JobAndCandidateDescription
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
