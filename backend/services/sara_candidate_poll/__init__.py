"""Structured job-search profile parser backed by Mistral."""

from .models import (
    CandidateDescription,
    CompensationPreference,
    EmploymentType,
    JobAndCandidateDescription,
    JobDescription,
    WorkModePreference,
)
from .parser import parse_job_request

__all__ = [
    "parse_job_request",
    "JobAndCandidateDescription",
    "CandidateDescription",
    "JobDescription",
    "CompensationPreference",
    "WorkModePreference",
    "EmploymentType",
]
