"""Concrete provider adapters used by the router."""

from .base import Provider, StructuredResult
from .github_models import GitHubModelsProvider
from .mistral import MistralProvider

__all__ = [
    "Provider",
    "StructuredResult",
    "MistralProvider",
    "GitHubModelsProvider",
]
