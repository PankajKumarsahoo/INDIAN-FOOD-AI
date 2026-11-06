"""Phase 3: label encoding + stratified train/val/test split -> data/processed/*.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from foodai.config import ensure_dirs, load_config
from foodai.data import build_dataframe, make_splits
from foodai.utils import save_json


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    raw, out = cfg["paths"]["raw_data"], cfg["paths"]["processed"]

    df = build_dataframe(raw, cfg["dataset"]["image_extensions"])

    # drop corrupt images found by script 02 (if it was run)
    corrupt_csv = cfg["paths"]["analysis"] / "corrupt_images.csv"
    if corrupt_csv.exists():
        bad = set(pd.read_csv(corrupt_csv)["image_path"])
        if bad:
            df = df[~df["image_path"].isin(bad)].reset_index(drop=True)
            print(f"Excluded {len(bad)} corrupt images")

    classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    df["label"] = df["class_name"].map(class_to_idx)

    s = cfg["split"]
    train_df, val_df, test_df = make_splits(df, s["train"], s["val"], s["test"], s["seed"])

    # sanity checks
    sets = [set(d["image_path"]) for d in (train_df, val_df, test_df)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "split overlap!"
    for name, d in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        assert d["label"].nunique() == len(classes), f"{name} split is missing classes"

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    save_json({"class_to_idx": class_to_idx, "idx_to_class": {str(k): v for k, v in idx_to_class.items()}},
              out / "class_mapping.json")

    print(f"Classes: {len(classes)} | train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")
    print("Saved to", out)
    print("Next: python scripts/04_train_classifier.py")


if __name__ == "__main__":
    main()
