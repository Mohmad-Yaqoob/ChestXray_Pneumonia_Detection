"""
PneumoScan AI — FastAPI Inference Engine
DA5402 MLOps | Mohmad Yaqoob | DA25M017 | IIT Madras

This module exposes the following REST endpoints:
    GET  /          - Root welcome message
    GET  /health    - Health check (model loaded status)
    GET  /ready     - Readiness probe (is model ready to serve?)
    GET  /model/info - Model metadata
    POST /predict   - Chest X-ray inference
    POST /feedback  - Ground truth feedback for drift tracking
    GET  /metrics   - Prometheus metrics
"""

import os
import io
import time
import logging
import numpy as np
from datetime import datetime
from PIL import Image

import mlflow
import mlflow.keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────

# Total prediction requests by endpoint and status
REQUEST_COUNT = Counter(
    "xray_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)

# Inference latency histogram
REQUEST_LATENCY = Histogram(
    "xray_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# Prediction class distribution counter
PREDICTION_COUNTER = Counter(
    "xray_predictions_total",
    "Total predictions by class",
    ["prediction"]
)

# Error rate counter — used for Grafana alerting when error rate > 5%
ERROR_COUNTER = Counter(
    "xray_errors_total",
    "Total prediction errors",
    ["error_type"]
)

# Feedback counter — tracks ground truth labels received for drift detection
FEEDBACK_COUNTER = Counter(
    "xray_feedback_total",
    "Ground truth feedback labels received",
    ["true_label", "predicted_label"]
)

# Model readiness gauge — 1 when model is loaded and ready, 0 otherwise
MODEL_READY = Gauge(
    "xray_model_ready",
    "Whether the model is loaded and ready to serve (1=ready, 0=not ready)"
)

# Drift detection gauge — tracks running mean of raw scores for baseline comparison
SCORE_MEAN = Gauge(
    "xray_score_running_mean",
    "Running mean of raw sigmoid scores (baseline: 0.31 from training data)"
)

# ── App Initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Upload a chest X-ray image to get a Normal/Pneumonia prediction",
    version="1.0.0",
)

# Allow cross-origin requests from the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ─────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "models/mobilenetv2_final.h5")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
model = None

# Running score accumulator for drift detection
_score_sum = 0.0
_score_count = 0

# Baseline mean raw score from training data (computed during v3 training)
BASELINE_SCORE_MEAN = 0.31


def load_model():
    """
    Load the TensorFlow model from disk.

    Uses a global variable to avoid reloading on every request.
    Sets the MODEL_READY Prometheus gauge to 1 on success.

    Returns:
        tf.keras.Model: The loaded Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist at MODEL_PATH.
    """
    global model
    if model is not None:
        return model
    if not os.path.exists(MODEL_PATH):
        MODEL_READY.set(0)
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logger.info(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    MODEL_READY.set(1)
    logger.info(f"Model loaded successfully — {model.count_params():,} parameters")
    return model


@app.on_event("startup")
async def startup_event():
    """
    Application startup handler.

    Loads the model, logs baseline statistics to MLflow for drift
    detection reference, and registers the model with MLflow if not
    already registered.
    """
    load_model()
    _log_baseline_to_mlflow()


def _log_baseline_to_mlflow():
    """
    Log dataset baseline statistics to MLflow for drift detection.

    These statistics were computed from the v3 training dataset and
    serve as the reference point for detecting data drift in production.
    The baseline values are logged as MLflow parameters so they appear
    in the experiment tracking UI alongside model metrics.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("chest-xray-pneumonia-detection")

        # Start a passive run to log baseline stats — does not interfere
        # with training runs already logged
        with mlflow.start_run(run_name="production-baseline-stats", nested=True):
            # Dataset composition baseline (v3 proper 80/10/10 split)
            mlflow.log_param("baseline_train_normal", 1266)
            mlflow.log_param("baseline_train_pneumonia", 3418)
            mlflow.log_param("baseline_val_normal", 158)
            mlflow.log_param("baseline_val_pneumonia", 427)

            # Score distribution baseline from validation set
            # These values describe the expected distribution of raw sigmoid
            # outputs on normal production data
            mlflow.log_param("baseline_score_mean", BASELINE_SCORE_MEAN)
            mlflow.log_param("baseline_score_std", 0.38)
            mlflow.log_param("baseline_normal_ratio", 0.32)
            mlflow.log_param("baseline_pneumonia_ratio", 0.68)

            # Model performance baseline
            mlflow.log_param("baseline_normal_accuracy", 0.774)
            mlflow.log_param("baseline_pneumonia_accuracy", 0.988)
            mlflow.log_param("baseline_val_auc", 0.993)
            mlflow.log_param("baseline_threshold", 0.65)

            mlflow.set_tag("run_type", "production-baseline")
            mlflow.set_tag("model_version", "v3")
            mlflow.set_tag("logged_at", datetime.now().strftime("%Y-%m-%d %H:%M"))

        logger.info("Baseline statistics logged to MLflow successfully")
    except Exception as e:
        # Non-fatal — system continues even if MLflow is unreachable
        logger.warning(f"Could not log baseline to MLflow: {e}")

# ── Image Preprocessing ───────────────────────────────────────────────────────

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a normalised numpy array for model inference.

    Resizes the image to 224x224, converts to RGB, normalises pixel
    values to [0, 1], and adds a batch dimension.

    Args:
        image_bytes (bytes): Raw bytes of the uploaded image file.

    Returns:
        np.ndarray: Shape (1, 224, 224, 3) float32 array.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    """Root endpoint — returns API welcome message."""
    return {
        "message": "Chest X-Ray Pneumonia Detection API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns the current API health status and whether the model is loaded.
    Used by Docker healthcheck and the Streamlit frontend status indicator.
    """
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/ready")
def ready():
    """
    Readiness probe endpoint.

    Indicates whether the service is fully ready to serve prediction
    requests. Returns 200 only when the model is loaded and warm.
    Returns 503 if the model has not finished loading.

    Used by orchestration systems to determine when to route traffic
    to this instance.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready — model is still loading"
        )
    return {
        "ready": True,
        "model_path": MODEL_PATH,
        "status": "ready to serve"
    }


@app.get("/model/info")
def model_info():
    """
    Model metadata endpoint.

    Returns architecture details of the currently loaded model
    including input/output shapes and total parameter count.
    """
    m = load_model()
    return {
        "model_path": MODEL_PATH,
        "input_shape": str(m.input_shape),
        "output_shape": str(m.output_shape),
        "total_params": m.count_params(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main inference endpoint.

    Accepts a chest X-ray image (JPEG or PNG) and returns a binary
    classification result with confidence score and plain-English
    interpretation.

    Args:
        file (UploadFile): The uploaded chest X-ray image file.

    Returns:
        JSONResponse: Contains filename, prediction, confidence,
                      raw_score, latency_seconds, and interpretation.

    Raises:
        HTTPException 400: If the file type is not JPEG or PNG.
        HTTPException 500: If model inference fails.
    """
    global _score_sum, _score_count
    start = time.time()

    # Validate file type before processing
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        ERROR_COUNTER.labels(error_type="invalid_file_type").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        m = load_model()
        raw_pred = float(m.predict(img_array, verbose=0)[0][0])

        # Apply decision threshold — 0.65 instead of default 0.5 to
        # reduce false positives on the NORMAL class
        THRESHOLD = 0.65
        label = "PNEUMONIA" if raw_pred >= THRESHOLD else "NORMAL"
        confidence = raw_pred if raw_pred >= THRESHOLD else 1 - raw_pred
        latency = time.time() - start

        # Update Prometheus metrics
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        PREDICTION_COUNTER.labels(prediction=label).inc()

        # Update running score mean for drift detection
        _score_sum += raw_pred
        _score_count += 1
        running_mean = _score_sum / _score_count
        SCORE_MEAN.set(running_mean)

        # Log warning if running mean drifts more than 0.1 from baseline
        drift = abs(running_mean - BASELINE_SCORE_MEAN)
        if _score_count >= 10 and drift > 0.1:
            logger.warning(
                f"Possible data drift detected — running score mean: "
                f"{running_mean:.3f}, baseline: {BASELINE_SCORE_MEAN:.3f}, "
                f"drift: {drift:.3f}"
            )

        logger.info(f"Prediction: {label} ({confidence:.4f}) in {latency:.3f}s")

        return JSONResponse({
            "filename": file.filename,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": round(raw_pred, 6),
            "latency_seconds": round(latency, 4),
            "interpretation": (
                "Pneumonia detected. Please consult a doctor."
                if label == "PNEUMONIA"
                else "No pneumonia detected. Chest X-ray appears normal."
            )
        })

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(
    filename: str,
    predicted_label: str,
    true_label: str
):
    """
    Ground truth feedback endpoint.

    Accepts the true label for a previously predicted image. This
    implements the feedback loop required for tracking real-world
    model performance and detecting label drift over time.

    In a production system, these feedback records would be stored
    to a database and used to trigger model retraining when
    accuracy drops below the acceptance threshold.

    Args:
        filename (str): The filename of the image that was predicted.
        predicted_label (str): What the model predicted (NORMAL/PNEUMONIA).
        true_label (str): The actual ground truth label.

    Returns:
        dict: Confirmation with match status and logged labels.
    """
    # Validate labels
    valid_labels = {"NORMAL", "PNEUMONIA"}
    if true_label not in valid_labels or predicted_label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Labels must be NORMAL or PNEUMONIA. Got: true={true_label}, predicted={predicted_label}"
        )

    # Track feedback in Prometheus for Grafana visibility
    FEEDBACK_COUNTER.labels(
        true_label=true_label,
        predicted_label=predicted_label
    ).inc()

    match = predicted_label == true_label
    logger.info(
        f"Feedback received — file: {filename}, "
        f"predicted: {predicted_label}, true: {true_label}, "
        f"correct: {match}"
    )

    return {
        "received": True,
        "filename": filename,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "correct": match
    }


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.

    Returns all instrumented metrics in Prometheus text exposition
    format. Scraped automatically by Prometheus every 15 seconds.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)