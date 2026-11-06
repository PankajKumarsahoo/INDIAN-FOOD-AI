"""Phase 6: train YOLO11-seg on data/yolo_seg_dataset (needs annotated images, see script 06)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

from foodai.config import ensure_dirs, load_config


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    ds = cfg["paths"]["yolo_dataset"]
    data_yaml = ds / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit("data.yaml missing. Run scripts/06_prepare_yolo_dataset.py first.")
    if not any((ds / "images" / "train").glob("*")):
        raise SystemExit(f"No training images in {ds / 'images' / 'train'}. "
                         "Segmentation needs mask annotations - see scripts/06_prepare_yolo_dataset.py.")

    from ultralytics import YOLO

    s = cfg["segmentation"]
    model = YOLO(s["weights"])
    model.train(
        data=str(data_yaml), epochs=s["epochs"], imgsz=s["imgsz"], batch=s["batch"],
        patience=s["patience"], project=str(cfg["paths"]["yolo_runs"]), name=s["run_name"],
        exist_ok=True,
    )
    print(model.val())

    best = cfg["paths"]["yolo_runs"] / s["run_name"] / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, cfg["paths"]["models"] / "yolo11_food_seg.pt")
        print("Segmentation weights copied to models/yolo11_food_seg.pt")


if __name__ == "__main__":
    main()
