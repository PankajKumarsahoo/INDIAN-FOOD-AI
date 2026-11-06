"""Phase 5: build the YOLO-seg folder layout + data.yaml, and (optionally) convert a COCO export.

The classification dataset has NO pixel masks, so it cannot train segmentation by itself.
You need annotations from one of: a Roboflow/CVAT/Label Studio COCO export, SAM-generated
pseudo-labels, or manual annotation. Then:

  python scripts/06_prepare_yolo_dataset.py --coco path/to/_annotations.coco.json \
         --images path/to/images --split train
  (repeat with --split val and --split test)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil

import yaml

from foodai.config import ensure_dirs, load_config
from foodai.utils import load_json


def write_data_yaml(dataset_dir, class_names):
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    out = dataset_dir / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    print("Wrote", out)


def coco_to_yolo_seg(coco_json_path, src_images_dir, images_out, labels_out):
    with open(coco_json_path, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    cat_idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for iid, info in images.items():
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem
        lines = []
        for a in anns.get(iid, []):
            seg = a.get("segmentation")
            if not seg or isinstance(seg, dict):
                continue  # RLE masks unsupported; polygons only
            for poly in seg:
                coords = []
                for i in range(0, len(poly), 2):
                    coords += [f"{poly[i] / w:.6f}", f"{poly[i + 1] / h:.6f}"]
                lines.append(f"{cat_idx[a['category_id']]} " + " ".join(coords))
        (labels_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        src = Path(src_images_dir) / info["file_name"]
        if src.exists():
            shutil.copy(src, images_out / src.name)
    print(f"Converted {len(images)} images -> {labels_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", help="COCO segmentation JSON to convert")
    parser.add_argument("--images", help="folder with the images referenced by the COCO JSON")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--class-source", choices=["classification", "coco"], default="classification",
                        help="class names for data.yaml: the classifier's classes, or the COCO categories")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    ds = cfg["paths"]["yolo_dataset"]
    for s in ("train", "val", "test"):
        (ds / "images" / s).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / s).mkdir(parents=True, exist_ok=True)

    names = None
    if args.coco:
        if not args.images:
            raise SystemExit("--images is required with --coco")
        coco_to_yolo_seg(args.coco, args.images, ds / "images" / args.split, ds / "labels" / args.split)
        if args.class_source == "coco":
            with open(args.coco, encoding="utf-8") as f:
                cats = sorted(json.load(f)["categories"], key=lambda c: c["id"])
            names = [c["name"] for c in cats]

    if names is None:
        mapping = load_json(cfg["paths"]["processed"] / "class_mapping.json")
        names = [mapping["idx_to_class"][str(i)] for i in range(len(mapping["idx_to_class"]))]
    write_data_yaml(ds, names)
    print("YOLO dataset layout ready at", ds)


if __name__ == "__main__":
    main()
