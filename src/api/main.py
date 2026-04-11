import os
import io
import time
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "xray_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "xray_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
PREDICTION_DIST = Counter(
    "xray_prediction_distribution",
    "Distribution of predictions",
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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logger.info(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")

@app.on_event("startup")
async def startup_event():
    load_model()

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    pneumonia_probability: float
    normal_probability: float
    model_version: str = "mobilenetv2_v1"

# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes to model-ready numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Chest X-Ray Pneumonia Detection API",
        "status": "running",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start = time.time()

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        image_bytes = await file.read()
        img_array   = preprocess_image(image_bytes)

        # Run inference
        prob = float(model.predict(img_array, verbose=0)[0][0])

        # Class index 1 = PNEUMONIA (ImageDataGenerator sorts alphabetically)
        # NORMAL=0, PNEUMONIA=1
        pneumonia_prob = prob
        normal_prob    = 1.0 - prob
        prediction     = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
        confidence     = pneumonia_prob if prediction == "PNEUMONIA" else normal_prob

        # Record metrics
        latency = time.time() - start
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        PREDICTION_DIST.labels(prediction=prediction).inc()

        logger.info(
            f"Prediction: {prediction} | "
            f"Confidence: {confidence:.4f} | "
            f"Latency: {latency:.3f}s"
        )

        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            pneumonia_probability=round(pneumonia_prob, 4),
            normal_probability=round(normal_prob, 4),
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)