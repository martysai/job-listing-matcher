"""
Minimal GPT-4o client via GitHub Models.

The GitHub PAT is read from a local file you specify with --key_path so the
secret never appears on the command line, in shell history, in environment
dumps, or in this script.

Usage:
    python gpt4o_demo.py --key_path key.txt --prompt "Explain RAG in one sentence."
    python gpt4o_demo.py --key_path key.txt --prompt "Tell me a joke" --stream
    python gpt4o_demo.py --key_path key.txt --model openai/gpt-4o-mini --prompt "Hi"

Requires: pip install openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAI, APIError, AuthenticationError, RateLimitError

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_SYSTEM = "You are a concise, helpful assistant."


def load_token(key_path: Path) -> str:
    """Read a GitHub PAT from a local file, with friendly error messages."""
    if not key_path.exists():
        sys.exit(f"error: key file not found: {key_path}")
    if not key_path.is_file():
        sys.exit(f"error: key path is not a file: {key_path}")

    token = key_path.read_text(encoding="utf-8").strip()
    if not token:
        sys.exit(f"error: key file is empty: {key_path}")
    if "\n" in token or " " in token:
        sys.exit("error: key file looks malformed (contains whitespace inside the token)")
    if not token.startswith(("github_pat_", "ghp_", "ghs_", "gho_")):
        print(
            "warning: token does not look like a GitHub PAT "
            "(expected prefix github_pat_/ghp_/ghs_/gho_)",
            file=sys.stderr,
        )
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call GPT-4o (or any GitHub Models model) using a PAT loaded from a file."
    )
    parser.add_argument(
        "--key_path",
        required=True,
        type=Path,
        help="Path to a .txt file whose contents are your GitHub PAT.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User message to send to the model.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=f"System prompt. Default: {DEFAULT_SYSTEM!r}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"GitHub Models model id. Default: {DEFAULT_MODEL!r}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-2.0). Default: 0.7",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens in the response. Default: 1024",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens to stdout as they arrive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = load_token(args.key_path)

    client = OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=token)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    try:
        if args.stream:
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
        else:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            print(resp.choices[0].message.content)
            usage = resp.usage
            if usage is not None:
                print(
                    f"\n[tokens] prompt={usage.prompt_tokens} "
                    f"completion={usage.completion_tokens} total={usage.total_tokens}",
                    file=sys.stderr,
                )
    except AuthenticationError:
        sys.exit("error: authentication failed - the PAT was rejected. "
                 "Confirm it has the 'models:read' permission and has not been revoked.")
    except RateLimitError:
        sys.exit("error: rate limit hit. Wait a minute or switch to a lower-tier model.")
    except APIError as e:
        sys.exit(f"error: API call failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
