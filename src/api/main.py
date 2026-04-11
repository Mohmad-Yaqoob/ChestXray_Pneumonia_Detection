import os
import io
import time
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "xray_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "xray_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
PREDICTION_COUNTER = Counter(
    "xray_predictions_total",
    "Total predictions by class",
    ["prediction"]
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Upload a chest X-ray image to get a Normal/Pneumonia prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/mobilenetv2_final.h5")
model = None

def load_model():
    global model
    if model is not None:
        return model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logger.info(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    return model

@app.on_event("startup")
async def startup_event():
    load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Chest X-Ray Pneumonia Detection API",
        "status":  "running",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {
        "status":       "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
    }

@app.get("/model/info")
def model_info():
    m = load_model()
    return {
        "model_path":   MODEL_PATH,
        "input_shape":  str(m.input_shape),
        "output_shape": str(m.output_shape),
        "total_params": m.count_params(),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        image_bytes = await file.read()
        img_array   = preprocess_image(image_bytes)

        m        = load_model()
        raw_pred = float(m.predict(img_array, verbose=0)[0][0])

        label      = "PNEUMONIA" if raw_pred >= 0.5 else "NORMAL"
        confidence = raw_pred if raw_pred >= 0.5 else 1 - raw_pred
        latency    = time.time() - start

        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        PREDICTION_COUNTER.labels(prediction=label).inc()

        logger.info(f"Prediction: {label} ({confidence:.4f}) in {latency:.3f}s")

        return JSONResponse({
            "filename":        file.filename,
            "prediction":      label,
            "confidence":      round(confidence * 100, 2),
            "raw_score":       round(raw_pred, 6),
            "latency_seconds": round(latency, 4),
            "interpretation": (
                "Pneumonia detected. Please consult a doctor."
                if label == "PNEUMONIA"
                else "No pneumonia detected. Chest X-ray appears normal."
            )
        })

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)