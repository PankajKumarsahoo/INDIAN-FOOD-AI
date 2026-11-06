"""Phase 2b: dataset analysis -> CSVs + plots in outputs/analysis. Also detects corrupt images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from foodai.config import ensure_dirs, load_config
from foodai.data import build_dataframe, is_image_ok


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    raw, out = cfg["paths"]["raw_data"], cfg["paths"]["analysis"]

    df = build_dataframe(raw, cfg["dataset"]["image_extensions"])
    print("Total images :", len(df))
    print("Classes      :", df["class_name"].nunique())

    counts = df["class_name"].value_counts()
    print(f"Images/class : min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")

    # corrupt check
    bad = [p for p in tqdm(df["image_path"], desc="Checking images") if not is_image_ok(raw / p)]
    print("Corrupted    :", len(bad))

    df.to_csv(out / "image_labels.csv", index=False)
    counts.to_csv(out / "class_distribution.csv")
    import pandas as pd

    pd.DataFrame({"image_path": bad}).to_csv(out / "corrupt_images.csv", index=False)

    # class distribution plot
    plt.figure(figsize=(15, 8))
    counts.plot(kind="bar")
    plt.title("Food Class Distribution")
    plt.xlabel("Food Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(cfg["paths"]["plots"] / "class_distribution.png", dpi=120)
    plt.close()

    # image sizes (sample for speed)
    sample = random.sample(list(df["image_path"]), min(2000, len(df)))
    sizes = Counter()
    for p in sample:
        try:
            with Image.open(raw / p) as im:
                sizes[im.size] += 1
        except Exception:
            pass
    print("Most common image sizes (sample):", sizes.most_common(5))

    # sample grid
    picks = df.sample(min(12, len(df)), random_state=0)
    plt.figure(figsize=(16, 12))
    for i, (_, r) in enumerate(picks.iterrows()):
        try:
            plt.subplot(3, 4, i + 1)
            plt.imshow(Image.open(raw / r["image_path"]).convert("RGB"))
            plt.title(r["class_name"])
            plt.axis("off")
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(cfg["paths"]["plots"] / "sample_images.png", dpi=100)
    plt.close()

    print("Saved analysis to", out, "and plots to", cfg["paths"]["plots"])
    print("Next: python scripts/03_prepare_data.py")


if __name__ == "__main__":
    main()
