"""Per-100g nutrition lookup, loaded from data/nutrition/nutrition_db.json.

Required per food : kcal, protein, carbs, fat      (all per 100 g)
Optional per food : fiber, sugar, sodium_mg        (per 100 g)
                    serving_g                      (typical serving in grams, used when Gemini is unavailable)
"""
from .utils import load_json, normalize_name

REQUIRED = ("kcal", "protein", "carbs", "fat")
OPTIONAL = ("fiber", "sugar", "sodium_mg")


def load_nutrition_db(path):
    try:
        raw = load_json(path)
    except FileNotFoundError:
        return {}
    return {normalize_name(k): v for k, v in raw.items() if not k.startswith("_")}


def calculate_nutrition(food_class, grams, db):
    key = normalize_name(food_class)
    if key not in db:
        return {"error": f"'{food_class}' not in nutrition DB. Add it to data/nutrition/nutrition_db.json "
                         "(see scripts/08_make_nutrition_template.py)."}
    per100 = db[key]
    f = grams / 100.0
    out = {
        "food": food_class,
        "portion_grams": round(grams, 1),
        "kcal": round(per100["kcal"] * f, 1),
        "protein_g": round(per100["protein"] * f, 1),
        "carbs_g": round(per100["carbs"] * f, 1),
        "fat_g": round(per100["fat"] * f, 1),
    }
    if per100.get("fiber") is not None:
        out["fiber_g"] = round(per100["fiber"] * f, 1)
        out["net_carbs_g"] = round(max(per100["carbs"] - per100["fiber"], 0) * f, 1)
    if per100.get("sugar") is not None:
        out["sugar_g"] = round(per100["sugar"] * f, 1)
    if per100.get("sodium_mg") is not None:
        out["sodium_mg"] = round(per100["sodium_mg"] * f, 0)
    return out


def atwater_kcal(protein, carbs, fat):
    """Energy estimate from macros: 4 kcal/g protein, 4 kcal/g carbs, 9 kcal/g fat."""
    return 4 * protein + 4 * carbs + 9 * fat
