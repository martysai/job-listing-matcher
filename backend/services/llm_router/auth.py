"""Load LLM provider credentials from local files.

Two loaders live here:

* ``load_github_pat()`` — GitHub PAT for GitHub Models (env
  ``GITHUB_MODELS_KEY_PATH``, default ``./key.txt``).
* ``load_anthropic_key()`` — Anthropic API key for Claude (env
  ``ANTHROPIC_API_KEY_PATH``, default ``~/anthropic_api_personal_key.txt``).

Tokens are read on demand from disk so they never enter environment
variables, process listings, or shell history.  The functions are cheap so
callers can re-read on each request without caching stale tokens.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KEY_PATH = "key.txt"
ENV_VAR = "GITHUB_MODELS_KEY_PATH"

ANTHROPIC_DEFAULT_KEY_PATH = "~/anthropic_api_personal_key.txt"
ANTHROPIC_ENV_VAR = "ANTHROPIC_API_KEY_PATH"

_PAT_PREFIXES = ("github_pat_", "ghp_", "ghs_", "gho_")
_ANTHROPIC_PREFIX = "sk-ant-"


class GitHubModelsAuthError(RuntimeError):
    """Raised when the GitHub PAT cannot be loaded or looks malformed."""


class AnthropicAuthError(RuntimeError):
    """Raised when the Anthropic API key cannot be loaded or looks malformed."""


def resolve_key_path() -> Path:
    """Return the configured GitHub key path (env var or default)."""
    return Path(os.environ.get(ENV_VAR, DEFAULT_KEY_PATH))


def resolve_anthropic_key_path() -> Path:
    """Return the configured Anthropic key path (env var or default)."""
    raw = os.environ.get(ANTHROPIC_ENV_VAR, ANTHROPIC_DEFAULT_KEY_PATH)
    return Path(raw).expanduser()


def _read_token(path: Path, *, label: str, error_cls: type[RuntimeError]) -> str:
    if not path.exists():
        raise error_cls(f"{label} file not found: {path}")
    if not path.is_file():
        raise error_cls(f"{label} path is not a file: {path}")

    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise error_cls(f"{label} file is empty: {path}")
    if any(ws in token for ws in (" ", "\n", "\t")):
        raise error_cls(f"{label} looks malformed (contains internal whitespace)")
    return token


def load_github_pat(path: Path | str | None = None) -> str:
    """Read and lightly validate a GitHub PAT.

    Args:
        path: Override the path; defaults to ``resolve_key_path()``.

    Raises:
        GitHubModelsAuthError: If the file is missing, empty, or contains
            whitespace inside the token.
    """
    key_path = Path(path) if path is not None else resolve_key_path()
    token = _read_token(key_path, label="GitHub PAT", error_cls=GitHubModelsAuthError)

    if not token.startswith(_PAT_PREFIXES):
        logger.debug(
            "llm_router: GitHub PAT does not match known prefixes %r; using it anyway",
            _PAT_PREFIXES,
        )

    return token


def load_anthropic_key(path: Path | str | None = None) -> str:
    """Read and lightly validate an Anthropic API key.

    Args:
        path: Override the path; defaults to ``resolve_anthropic_key_path()``.

    Raises:
        AnthropicAuthError: If the file is missing, empty, or contains
            whitespace inside the token.
    """
    key_path = Path(path).expanduser() if path is not None else resolve_anthropic_key_path()
    token = _read_token(key_path, label="Anthropic API key", error_cls=AnthropicAuthError)

    if not token.startswith(_ANTHROPIC_PREFIX):
        logger.debug(
            "llm_router: Anthropic key does not start with %r; using it anyway",
            _ANTHROPIC_PREFIX,
        )

    return token
