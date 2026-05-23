"""Load the GitHub PAT for GitHub Models from a local file.

The token is read on demand from a file path (``GITHUB_MODELS_KEY_PATH``,
default ``./key.txt``) so it never enters environment variables, process
listings, or shell history.  The function is intentionally cheap so callers
can re-read on each request without caching stale tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_KEY_PATH = "key.txt"
ENV_VAR = "GITHUB_MODELS_KEY_PATH"

_PAT_PREFIXES = ("github_pat_", "ghp_", "ghs_", "gho_")


class GitHubModelsAuthError(RuntimeError):
    """Raised when the GitHub PAT cannot be loaded or looks malformed."""


def resolve_key_path() -> Path:
    """Return the configured key path (env var or default)."""
    return Path(os.environ.get(ENV_VAR, DEFAULT_KEY_PATH))


def load_github_pat(path: Path | str | None = None) -> str:
    """Read and lightly validate a GitHub PAT.

    Args:
        path: Override the path; defaults to ``resolve_key_path()``.

    Raises:
        GitHubModelsAuthError: If the file is missing, empty, or contains
            whitespace inside the token.
    """
    key_path = Path(path) if path is not None else resolve_key_path()

    if not key_path.exists():
        raise GitHubModelsAuthError(f"GitHub PAT file not found: {key_path}")
    if not key_path.is_file():
        raise GitHubModelsAuthError(f"GitHub PAT path is not a file: {key_path}")

    token = key_path.read_text(encoding="utf-8").strip()
    if not token:
        raise GitHubModelsAuthError(f"GitHub PAT file is empty: {key_path}")
    if any(ws in token for ws in (" ", "\n", "\t")):
        raise GitHubModelsAuthError(
            "GitHub PAT looks malformed (contains internal whitespace)"
        )

    if not token.startswith(_PAT_PREFIXES):
        # Soft check: warn through exception message only when used; do not
        # reject so users can supply alternate token types if GitHub Models
        # ever broadens its accepted formats.
        pass

    return token
