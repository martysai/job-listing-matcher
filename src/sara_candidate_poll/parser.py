from .client import parse_structured, _DEFAULT_MODEL
from .models import JobAndCandidateDescription
from .prompts import build_full_prompt


def parse_job_request(
    user_input: str,
    model: str = _DEFAULT_MODEL,
) -> JobAndCandidateDescription:
    """
    Parse a free-text job search message or resume snippet into a structured profile.

    Args:
        user_input: Raw text from the user (job request, resume excerpt, etc.)
        model: Mistral model ID to use for parsing.

    Returns:
        JobAndCandidateDescription with candidate facts and job preferences extracted.
    """
    full_prompt = build_full_prompt(user_input)
    return parse_structured(full_prompt, JobAndCandidateDescription, model=model)
