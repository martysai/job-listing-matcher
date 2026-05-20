from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FIXED_TERM = "fixed_term"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    SELF_EMPLOYED = "self_employed"


class CandidateDescription(BaseModel):
    """
    Factual and relatively stable information about the candidate.
    Used for matching and ranking against job postings.
    """
    years_experience: Optional[int] = Field(
        default=None,
        description="Total years of professional work experience",
    )
    languages: List[str] = Field(
        default_factory=list,
        description="Languages the candidate can speak or work in",
    )
    education_level: Optional[str] = Field(
        default=None,
        description="Highest completed education level (e.g. bachelor's, master's, PhD)",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Technical and soft skills of the candidate",
    )


class CompanyPreference(BaseModel):
    industry: Optional[str] = Field(
        default=None,
        description="Preferred industry or sector (e.g. fintech, healthcare)",
    )
    location: Optional[str] = Field(
        default=None,
        description="Preferred company or job location",
    )


class CompensationPreference(BaseModel):
    salary_min: Optional[int] = Field(
        default=None,
        description="Minimum desired monthly salary",
    )
    salary_max: Optional[int] = Field(
        default=None,
        description="Maximum desired monthly salary",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Salary currency (e.g. USD, EUR, RUB)",
    )
    is_gross: Optional[bool] = Field(
        default=None,
        description="True if gross, False if net, None if unspecified",
    )
    benefits: List[str] = Field(
        default_factory=list,
        description="Desired benefits (e.g. health insurance, stock options)",
    )


class WorkModePreference(BaseModel):
    employment_type: List[EmploymentType] = Field(
        description="Type of employment (full_time, contract, freelance, etc.)",
    )
    preferred_remote_policy: List[str] = Field(
        default_factory=list,
        description="Preferred work mode: remote, hybrid, onsite",
    )
    acceptable_remote_policy: List[str] = Field(
        default_factory=list,
        description="Acceptable work modes (remote, hybrid, onsite)",
    )


class JobDescription(BaseModel):
    """
    What the candidate is looking for in a job.
    This includes preferences, constraints, and search filters.
    """
    desired_positions: List[str] = Field(
        default_factory=list,
        description="Desired job titles or role names",
    )
    # This will likely rarely be used, but can provide better accuracy for certain candidates.
    desired_tech_stack: List[str] = Field(
        default_factory=list,
        description="Technologies the candidate wants to work with",
    )
    preferred_domains: List[str] = Field(
        default_factory=list,
        description="Subject areas or disciplines the candidate wants to work in",
    )
    preferred_activities: List[str] = Field(
        default_factory=list,
        description="Activities or responsibilities the candidate wants to perform",
    )
    preferred_companies: List[CompanyPreference] = Field(
        default_factory=list,
        description="Target industries, locations, or specific companies",
    )
    desired_compensation_monthly: Optional[CompensationPreference] = Field(
        default=None,
        description="Salary and benefits expectations",
    )
    preferred_work_mode: Optional[WorkModePreference] = Field(
        default=None,
        description="Employment type and work arrangement preferences",
    )


class JobAndCandidateDescription(BaseModel):
    candidate_description: CandidateDescription = Field(
        description="Factual and relatively stable information about the candidate",
    )
    job_description: JobDescription = Field(
        description="What the candidate is looking for in a job",
    )
