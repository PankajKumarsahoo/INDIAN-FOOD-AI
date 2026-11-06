"""Phase 2a: download the Kaggle dataset INTO this project (data/raw/), not a global cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import shutil

from foodai.config import ensure_dirs, load_config


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    dest = cfg["paths"]["raw_data"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if data exists")
    args = parser.parse_args()

    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"Dataset already present at {dest}. Use --force to re-download.")
        return

    # Make kagglehub cache inside the project BEFORE importing it.
    cache = cfg["paths"]["raw_data"].parent / "_kagglehub_cache"
    os.environ["KAGGLEHUB_CACHE"] = str(cache)
    import kagglehub

    print("Downloading", cfg["dataset"]["kaggle_id"], "...")
    downloaded = Path(kagglehub.dataset_download(cfg["dataset"]["kaggle_id"]))

    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(downloaded), str(dest))
    shutil.rmtree(cache, ignore_errors=True)
    print("Dataset saved to:", dest)
    print("Next: python scripts/02_analyze_dataset.py")


if __name__ == "__main__":
    main()
