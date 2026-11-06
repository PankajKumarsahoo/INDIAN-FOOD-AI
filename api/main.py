"""FastAPI service + web UI.   Run from project root:  uvicorn api.main:app --port 8000
Open http://localhost:8000  for the upload page, /docs for the API docs."""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from PIL import Image  # noqa: E402

from foodai.nutrition import calculate_nutrition  # noqa: E402

app = FastAPI(title="Indian Food Nutrition API")
FRONTEND = ROOT / "frontend" / "index.html"
MAX_BYTES = 15 * 1024 * 1024
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from foodai.inference import FoodPredictor

        _predictor = FoodPredictor()
    return _predictor


@app.get("/", include_in_schema=False)
def home():
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "device": str(p.device), "segmentation_loaded": p.seg_model is not None}


# Plain `def` (not async): FastAPI runs it in a worker thread, so model inference
# does not block other requests.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return get_predictor().predict(image)


@app.get("/nutrition")
def nutrition(food: str, grams: float):
    """Recalculate nutrition for a corrected portion size (used by the web UI)."""
    if not 1 <= grams <= 5000:
        raise HTTPException(status_code=400, detail="grams must be between 1 and 5000.")
    return calculate_nutrition(food, grams, get_predictor().nutrition_db)
