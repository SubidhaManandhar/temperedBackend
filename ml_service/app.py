import os
import uuid
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

from models.model_def import build_resnet50_7ch
from inference.predict_pytorch import predict_image

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "originals"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "ela"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "noise"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "masks"), exist_ok=True)

app.mount("/static", StaticFiles(directory=OUTPUTS_DIR), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")

model = build_resnet50_7ch(num_classes=5, pretrained=False)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.to(DEVICE)
model.eval()


@app.get("/")
def root():
    return {"message": "PyTorch ML service running"}


@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    ext = os.path.splitext(image.filename)[1].lower() or ".jpg"
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{ext}")

    with open(tmp_path, "wb") as f:
        f.write(await image.read())

    try:
        result = predict_image(tmp_path, model, DEVICE, OUTPUTS_DIR)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return {
        "stageA_label": result["stageA_label"],
        "stageA_confidence": result["stageA_confidence"],
        "label": result["label"],
        "confidence": result["confidence"],
        "original_url": f"/static/originals/{result['original_file']}" if result["original_file"] else None,
        "ela_url": f"/static/ela/{result['ela_file']}" if result["ela_file"] else None,
        "noise_url": f"/static/noise/{result['noise_file']}" if result["noise_file"] else None,
        "heatmap_url": f"/static/heatmaps/{result['heatmap_file']}" if result["heatmap_file"] else None,
        "mask_url": f"/static/masks/{result['mask_file']}" if result["mask_file"] else None,
        "bbox": result["bbox"],
        "best_layer": result["best_layer"],
    }