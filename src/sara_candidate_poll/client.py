import os
from typing import Type, TypeVar

from dotenv import load_dotenv
from mistralai import Mistral
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "mistral-small-latest"


def _get_client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY environment variable is not set")
    return Mistral(api_key=api_key)


def parse_structured(
    full_prompt: str,
    response_format: Type[T],
    model: str = _DEFAULT_MODEL,
) -> T:
    """Send full_prompt as a user message and parse the response into response_format."""
    client = _get_client()
    messages = [{"role": "user", "content": full_prompt}]
    response = client.chat.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    return response.choices[0].message.parsed
