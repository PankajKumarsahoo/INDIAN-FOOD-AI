"""End-to-end predictor: classify -> (segment) -> portion -> nutrition."""
import time
from pathlib import Path

import torch
from PIL import Image

from .config import load_config
from .data import get_transforms
from .model import load_classifier
from .nutrition import calculate_nutrition, load_nutrition_db
from .portion import estimate_portion_grams
from .utils import normalize_name
from .utils import get_device


class FoodPredictor:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.device = get_device()
        models = self.cfg["paths"]["models"]

        clf_path = models / "food_classifier_convnext_tiny.pth"
        if not clf_path.exists():
            raise FileNotFoundError(f"{clf_path} not found. Train first: python scripts/04_train_classifier.py")
        self.model, self.idx_to_class, _ = load_classifier(clf_path, self.device)

        c = self.cfg["classifier"]
        _, self.tf = get_transforms(c["image_size"], c["resize_size"])
        self.nutrition_db = load_nutrition_db(self.cfg["paths"]["nutrition_db"])

        self.seg_model = None
        seg_path = models / "yolo11_food_seg.pt"
        if seg_path.exists():
            from ultralytics import YOLO

            self.seg_model = YOLO(str(seg_path))

    @torch.no_grad()
    def classify(self, image, top_k=3):
        x = self.tf(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.topk(min(top_k, len(self.idx_to_class)))
        return [(self.idx_to_class[int(i)], float(c)) for c, i in zip(conf, idx)]

    def food_area_fraction(self, image):
        """Fraction of the image covered by segmented food (None if no segmentation model)."""
        if self.seg_model is None:
            return None
        res = self.seg_model(image, verbose=False)[0]
        if res.masks is None:
            return 0.0
        return round(float(res.masks.data.any(dim=0).float().mean()), 4)

    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(Path(image))
        image = image.convert("RGB")

        t0 = time.perf_counter()
        top = self.classify(image, self.cfg["inference"]["top_k"])
        t1 = time.perf_counter()
        food, conf = top[0]
        area = self.food_area_fraction(image)
        t2 = time.perf_counter()
        portion = estimate_portion_grams(image, food, self.cfg["inference"]["default_portion_grams"])
        t3 = time.perf_counter()
        grams = portion.get("estimated_grams") or self.cfg["inference"]["default_portion_grams"]
        if portion.get("confidence") == "none":  # Gemini unavailable: use this food's typical serving if known
            serving = self.nutrition_db.get(normalize_name(food), {}).get("serving_g")
            if serving:
                grams = serving
                portion = {**portion, "estimated_grams": serving,
                           "reasoning": portion.get("reasoning", "") + f" Using typical serving of {serving} g for this food."}
        timings = {"classify_s": round(t1 - t0, 2), "segment_s": round(t2 - t1, 2),
                   "portion_s": round(t3 - t2, 2), "total_s": round(t3 - t0, 2)}
        print(f"[timing] {food}: {timings}", flush=True)

        return {
            "food": food,
            "classification_confidence": round(conf, 4),
            "top_k": [{"food": n, "confidence": round(c, 4)} for n, c in top],
            "food_area_fraction": area,
            "portion": portion,
            "nutrition": calculate_nutrition(food, float(grams), self.nutrition_db),
            "timings": timings,
        }
