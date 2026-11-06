import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodai.nutrition import atwater_kcal, calculate_nutrition  # noqa: E402

DB = {"samosa": {"kcal": 262, "protein": 4.0, "carbs": 30.0, "fat": 14.0, "fiber": 2.5}}


def test_scaling():
    r = calculate_nutrition("Samosa", 200, DB)
    assert r["kcal"] == 524.0 and r["fat_g"] == 28.0 and r["fiber_g"] == 5.0 and r["net_carbs_g"] == 55.0


def test_missing():
    assert "error" in calculate_nutrition("unknown_dish", 100, DB)


def test_atwater():
    assert atwater_kcal(4, 30, 14) == 262
