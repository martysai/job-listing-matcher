from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "backend" / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Project root: {ROOT}")

required_modules = [
    "sara_retrieve_rerank",
    "langchain_core",
    "langchain_chroma",
    "langchain_huggingface",
    "bs4",
    "sentence_transformers",
    "chromadb",
]
optional_modules = [
    ("lightgbm", "LambdaRank reranker training"),
]

missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    print("Missing modules:")
    for name in missing:
        print(f"  - {name}")
    print("\nInstall with:")
    print("  python -m pip install -r requirements.txt")
    print("  python -m pip install -e .")
    raise SystemExit(1)

print("Environment OK: all required modules are importable.")

missing_optional = [
    (name, feature) for name, feature in optional_modules if importlib.util.find_spec(name) is None
]
if missing_optional:
    print("Optional modules missing:")
    for name, feature in missing_optional:
        print(f"  - {name}: needed for {feature}")
