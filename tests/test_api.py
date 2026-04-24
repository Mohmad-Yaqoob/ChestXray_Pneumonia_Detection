"""
Unit tests for PneumoScan AI — FastAPI Inference Engine
DA5402 MLOps | Mohmad Yaqoob | DA25M017

Run with:
    pytest tests/test_api.py -v
"""

import pytest
import requests
import os
import io
from PIL import Image

BASE_URL = "http://localhost:8000"

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_dummy_image(mode="RGB", size=(224, 224), fmt="JPEG"):
    """Create an in-memory dummy image for testing."""
    img = Image.new(mode, size, color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def get_test_image(label="PNEUMONIA"):
    """Return a real test image path if available, else None."""
    base = "data/processed/test"
    folder = os.path.join(base, label)
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith((".jpg", ".jpeg", ".png"))]
        if files:
            return os.path.join(folder, files[0])
    return None


# ── TC-01: Root endpoint ─────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_status_code(self):
        """TC-01a: GET / returns 200"""
        r = requests.get(f"{BASE_URL}/")
        assert r.status_code == 200

    def test_message_field_exists(self):
        """TC-01b: GET / response contains message field"""
        r = requests.get(f"{BASE_URL}/")
        assert "message" in r.json()

    def test_response_is_json(self):
        """TC-01c: GET / returns valid JSON"""
        r = requests.get(f"{BASE_URL}/")
        assert r.headers["content-type"].startswith("application/json")


# ── TC-02: Health endpoint ───────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_status_code(self):
        """TC-02a: GET /health returns 200"""
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200

    def test_status_field(self):
        """TC-02b: health status is healthy"""
        r = requests.get(f"{BASE_URL}/health")
        assert r.json()["status"] == "healthy"

    def test_model_loaded(self):
        """TC-02c: model_loaded is true"""
        r = requests.get(f"{BASE_URL}/health")
        assert r.json()["model_loaded"] is True

    def test_model_path_present(self):
        """TC-02d: model_path field is non-empty string"""
        r = requests.get(f"{BASE_URL}/health")
        assert isinstance(r.json()["model_path"], str)
        assert len(r.json()["model_path"]) > 0


# ── TC-03: Model info endpoint ───────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_status_code(self):
        """TC-03a: GET /model/info returns 200"""
        r = requests.get(f"{BASE_URL}/model/info")
        assert r.status_code == 200

    def test_total_params(self):
        """TC-03b: total_params is a positive integer"""
        r = requests.get(f"{BASE_URL}/model/info")
        assert r.json()["total_params"] > 0

    def test_input_shape(self):
        """TC-03c: input_shape field present"""
        r = requests.get(f"{BASE_URL}/model/info")
        assert "input_shape" in r.json()

    def test_output_shape(self):
        """TC-03d: output_shape field present"""
        r = requests.get(f"{BASE_URL}/model/info")
        assert "output_shape" in r.json()


# ── TC-04 / TC-05: Predict endpoint — valid images ──────────────────────────

class TestPredictEndpoint:
    def test_predict_dummy_image_returns_200(self):
        """TC-04a: POST /predict with dummy image returns 200"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert r.status_code == 200

    def test_predict_response_has_prediction_field(self):
        """TC-04b: response contains prediction field"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert "prediction" in r.json()

    def test_predict_prediction_is_valid_label(self):
        """TC-04c: prediction is NORMAL or PNEUMONIA"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert r.json()["prediction"] in ["NORMAL", "PNEUMONIA"]

    def test_predict_confidence_range(self):
        """TC-04d: confidence is between 0 and 100"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        conf = r.json()["confidence"]
        assert 0.0 <= conf <= 100.0

    def test_predict_raw_score_range(self):
        """TC-04e: raw_score is between 0 and 1"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert 0.0 <= r.json()["raw_score"] <= 1.0

    def test_predict_latency_positive(self):
        """TC-04f: latency_seconds is a positive float"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert r.json()["latency_seconds"] > 0

    def test_predict_interpretation_present(self):
        """TC-04g: interpretation field is non-empty string"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert isinstance(r.json()["interpretation"], str)
        assert len(r.json()["interpretation"]) > 0

    def test_predict_filename_echoed(self):
        """TC-04h: filename matches uploaded filename"""
        buf = make_dummy_image()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("mytest.jpg", buf, "image/jpeg")})
        assert r.json()["filename"] == "mytest.jpg"

    def test_predict_png_image(self):
        """TC-05a: POST /predict accepts PNG format"""
        buf = make_dummy_image(fmt="PNG")
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.png", buf, "image/png")})
        assert r.status_code == 200

    def test_predict_real_pneumonia_image(self):
        """TC-05b: Real PNEUMONIA image classified correctly"""
        path = get_test_image("PNEUMONIA")
        if path is None:
            pytest.skip("No test PNEUMONIA images found in data/processed/test/PNEUMONIA/")
        with open(path, "rb") as f:
            r = requests.post(f"{BASE_URL}/predict", files={"file": (os.path.basename(path), f, "image/jpeg")})
        assert r.status_code == 200
        assert r.json()["prediction"] == "PNEUMONIA"

    def test_predict_real_normal_image(self):
        """TC-05c: Real NORMAL image returns valid response"""
        path = get_test_image("NORMAL")
        if path is None:
            pytest.skip("No test NORMAL images found in data/processed/test/NORMAL/")
        with open(path, "rb") as f:
            r = requests.post(f"{BASE_URL}/predict", files={"file": (os.path.basename(path), f, "image/jpeg")})
        assert r.status_code == 200
        assert r.json()["prediction"] in ["NORMAL", "PNEUMONIA"]


# ── TC-06: No file uploaded ──────────────────────────────────────────────────

class TestPredictErrorHandling:
    def test_no_file_returns_422(self):
        """TC-06: POST /predict with no file returns 422"""
        r = requests.post(f"{BASE_URL}/predict")
        assert r.status_code == 422

    def test_empty_file_handled(self):
        """TC-07a: POST /predict with empty bytes handled gracefully"""
        buf = io.BytesIO(b"")
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("empty.jpg", buf, "image/jpeg")})
        assert r.status_code in [400, 422, 500]

    def test_text_file_rejected(self):
        """TC-07b: POST /predict with plain text file handled gracefully"""
        buf = io.BytesIO(b"this is not an image")
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("bad.txt", buf, "text/plain")})
        assert r.status_code in [400, 422, 500]


# ── TC-08: Metrics endpoint ──────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        """TC-08a: GET /metrics returns 200"""
        r = requests.get(f"{BASE_URL}/metrics")
        assert r.status_code == 200

    def test_metrics_contains_xray_requests(self):
        """TC-08b: xray_requests_total metric present"""
        r = requests.get(f"{BASE_URL}/metrics")
        assert "xray_requests_total" in r.text

    def test_metrics_contains_xray_predictions(self):
        """TC-08c: xray_predictions_total metric present"""
        r = requests.get(f"{BASE_URL}/metrics")
        assert "xray_predictions_total" in r.text

    def test_metrics_contains_latency(self):
        """TC-08d: xray_request_latency_seconds metric present"""
        r = requests.get(f"{BASE_URL}/metrics")
        assert "xray_request_latency_seconds" in r.text

    def test_metrics_content_type(self):
        """TC-08e: metrics endpoint returns text/plain format"""
        r = requests.get(f"{BASE_URL}/metrics")
        assert "text/plain" in r.headers["content-type"]


# ── TC: Response time ────────────────────────────────────────────────────────

class TestPerformance:
    def test_predict_under_2_seconds(self):
        """Performance: prediction completes within 2 seconds"""
        import time
        buf = make_dummy_image()
        start = time.time()
        r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        elapsed = time.time() - start
        assert r.status_code == 200
        assert elapsed < 2.0, f"Prediction took {elapsed:.2f}s — exceeds 2s limit"

    def test_health_check_fast(self):
        """Performance: health check completes within 1 second"""
        import time
        start = time.time()
        r = requests.get(f"{BASE_URL}/health")
        elapsed = time.time() - start
        assert elapsed < 1.0