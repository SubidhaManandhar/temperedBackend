import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

from inference.gradcam_autolayer import predict_with_gradcam

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "masks"), exist_ok=True)

# Serve outputs as /static
app.mount("/static", StaticFiles(directory=OUTPUTS_DIR), name="static")

# ✅ CHANGE THESE TWO FILENAMES to match what you downloaded from GitHub outputs/models


STAGEA_MODEL_PATH = os.path.join(MODELS_DIR, "best_stageA_binary.h5")
STAGEB_MODEL_PATH = os.path.join(MODELS_DIR, "final_stageB_tamper4.h5")

stageA = tf.keras.models.load_model(STAGEA_MODEL_PATH)
stageB = tf.keras.models.load_model(STAGEB_MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True, "message": "ML service running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    ext = os.path.splitext(image.filename)[1].lower() or ".jpg"
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{ext}")

    with open(tmp_path, "wb") as f:
        f.write(await image.read())

    try:
        result = predict_with_gradcam(
            image_path=tmp_path,
            stageA=stageA,
            stageB=stageB,
            outputs_dir=OUTPUTS_DIR
        )
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    heatmap_url = f"/static/heatmaps/{result['heatmap_file']}" if result["heatmap_file"] else None
    mask_url = f"/static/masks/{result['mask_file']}" if result["mask_file"] else None

    return {
        "stageA_label": result["stageA_label"],
        "stageA_confidence": float(result["stageA_confidence"]),
        "label": result["label"],
        "confidence": float(result["confidence"]),
        "heatmap_url": heatmap_url,
        "mask_url": mask_url,
        "bbox": result["bbox"],
        "best_layer": result["best_layer"]
    }

@app.get("/")
def root():
    return {"message": "ML service running. Use /docs, /health, /predict"}