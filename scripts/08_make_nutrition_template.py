"""Phase 8 helper: list every class and which ones still lack nutrition data (values are per 100 g)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from foodai.config import ensure_dirs, load_config
from foodai.nutrition import load_nutrition_db
from foodai.utils import load_json, normalize_name

COLUMNS = ["kcal", "protein", "carbs", "fat", "fiber", "sugar", "sodium_mg", "serving_g"]


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    mapping = load_json(cfg["paths"]["processed"] / "class_mapping.json")
    classes = [mapping["idx_to_class"][str(i)] for i in range(len(mapping["idx_to_class"]))]
    db = load_nutrition_db(cfg["paths"]["nutrition_db"])

    out = cfg["paths"]["reports"] / "nutrition_template.csv"
    missing = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name"] + COLUMNS)
        for c in classes:
            row = db.get(normalize_name(c))
            if row:
                w.writerow([c] + [row.get(k, "") for k in COLUMNS])
            else:
                w.writerow([c] + [""] * len(COLUMNS))
                missing += 1
    print(f"{len(classes)} classes, {len(classes) - missing} have nutrition data, {missing} missing.")
    print("Fill the blanks in", out)
    print("  Required (per 100 g): kcal, protein, carbs, fat.  Optional: fiber, sugar, sodium_mg, serving_g")
    print("  Then run: python scripts/09_import_nutrition_csv.py")


if __name__ == "__main__":
    main()
