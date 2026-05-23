"""Snapshot the live Chroma DB into a portable zip for shipping to prod.

Bundles three things by default:

  * ``data/chroma/`` — the persistent vector store
  * ``data/raw/adzuna_vacancies.jsonl`` — live Adzuna BM25 audit log
  * ``data/raw/vacancies_safe_ml_dataset_nozip.jsonl`` — seed BM25 corpus
    (the prod backend's ``RecommendationPipeline`` requires this at startup
    via the ``VACANCIES_PATH`` env var)

The zip is safe to copy to the production machine and extract on top of
its ``data/`` directory.  The receiving machine needs no API keys for the
read path — only the same sentence-transformers embedding model
(``sentence-transformers/all-MiniLM-L6-v2``), which HuggingFace fetches
automatically on first use.

Usage::

    python scripts/package_chroma.py
    python scripts/package_chroma.py --label nightly
    python scripts/package_chroma.py --no-jsonl     # skip live Adzuna log
    python scripts/package_chroma.py --no-seed      # skip 46 MB seed corpus
                                                    # (when prod already has it)

The daemon can keep running while this script executes; SQLite uses
copy-on-write under the hood and the snapshot will reflect the state at
script start.  For a strictly consistent snapshot (no in-flight cycle),
stop the daemon first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
JSONL_PATH = REPO_ROOT / "data" / "raw" / "adzuna_vacancies.jsonl"
SEED_PATH  = REPO_ROOT / "data" / "raw" / "vacancies_safe_ml_dataset_nozip.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs"


def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Suffix on the output filename (default: timestamp).",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip the adzuna_vacancies.jsonl raw audit log.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip the 46 MB vacancies_safe_ml_dataset_nozip.jsonl seed corpus "
             "(use when the prod box already has the same file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the zip (default: outputs/).",
    )
    args = parser.parse_args()

    if not CHROMA_DIR.exists():
        print(f"ERROR: {CHROMA_DIR} does not exist — nothing to package.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"chroma-snapshot-{args.label}.zip"

    if out_path.exists():
        print(f"ERROR: {out_path} already exists. Pick a different --label.", file=sys.stderr)
        return 2

    print(f"Packaging Chroma snapshot → {out_path}")
    n_files = 0
    n_bytes = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(CHROMA_DIR.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(REPO_ROOT)
                zf.write(path, arcname=str(arcname))
                n_files += 1
                n_bytes += path.stat().st_size

        if not args.no_jsonl and JSONL_PATH.exists():
            zf.write(JSONL_PATH, arcname=str(JSONL_PATH.relative_to(REPO_ROOT)))
            n_files += 1
            n_bytes += JSONL_PATH.stat().st_size
            print(f"  + {JSONL_PATH.relative_to(REPO_ROOT)} (live Adzuna BM25 audit log)")

        if not args.no_seed and SEED_PATH.exists():
            zf.write(SEED_PATH, arcname=str(SEED_PATH.relative_to(REPO_ROOT)))
            n_files += 1
            n_bytes += SEED_PATH.stat().st_size
            print(f"  + {SEED_PATH.relative_to(REPO_ROOT)} (seed BM25 corpus, required at startup)")

    final_size = out_path.stat().st_size
    print()
    print(f"Files bundled:       {n_files}")
    print(f"Uncompressed total:  {_format_size(n_bytes)}")
    print(f"Compressed zip size: {_format_size(final_size)}")
    print(f"Output path:         {out_path}")
    print()
    print("To install on the production machine:")
    print(f"  1. Copy {out_path.name} to the prod repo root")
    print(f"  2. Extract:  python -c \"import zipfile; zipfile.ZipFile('{out_path.name}').extractall('.')\"")
    print( "  3. (re)start the FastAPI backend — it will pick up the new data/chroma automatically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
