"""Phase 8 helper: turn the filled-in outputs/reports/nutrition_template.csv into nutrition_db.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from foodai.config import load_config
from foodai.nutrition import OPTIONAL, REQUIRED, atwater_kcal
from foodai.utils import normalize_name, save_json


def num(value):
    value = (value or "").strip()
    return float(value) if value else None


def main():
    cfg = load_config()
    src = cfg["paths"]["reports"] / "nutrition_template.csv"
    db = {"_comment": "Per-100g: kcal, protein, carbs, fat (required); fiber, sugar, sodium_mg (optional); serving_g = typical serving in grams."}
    skipped, warnings = 0, []
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                entry = {k: float(r[k]) for k in REQUIRED}
            except (ValueError, TypeError, KeyError):
                skipped += 1  # required value blank: skip this row
                continue
            for k in OPTIONAL + ("serving_g",):
                v = num(r.get(k))
                if v is not None:
                    entry[k] = v
            db[normalize_name(r["class_name"])] = entry

            est = atwater_kcal(entry["protein"], entry["carbs"], entry["fat"])
            if est > 0 and abs(est - entry["kcal"]) / est > 0.30:
                warnings.append(f"  {r['class_name']}: kcal={entry['kcal']:.0f} but macros imply about {est:.0f}")
            if entry.get("fiber") is not None and entry["fiber"] > entry["carbs"]:
                warnings.append(f"  {r['class_name']}: fiber ({entry['fiber']}) is more than carbs ({entry['carbs']})")

    save_json(db, cfg["paths"]["nutrition_db"])
    print(f"Wrote {len(db) - 1} entries to {cfg['paths']['nutrition_db']} ({skipped} rows skipped: required values blank)")
    if warnings:
        print("\nPlease double-check these rows (numbers look inconsistent):")
        print("\n".join(warnings))


if __name__ == "__main__":
    main()
