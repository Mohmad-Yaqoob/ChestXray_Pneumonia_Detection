"""
PneumoScan AI — Streamlit Frontend
DA5402 MLOps | Mohmad Yaqoob | DA25M017 | IIT Madras

This module provides the web-based user interface for the Chest X-Ray
Pneumonia Detection system. It communicates with the FastAPI backend
exclusively via REST API calls — no shared memory or direct model access.

Pages:
    - Single Scan: Upload and analyse one X-ray image
    - Batch Testing: Analyse multiple images at once
    - Prediction History: View all predictions in the current session
    - ML Pipeline: Visualize the end-to-end MLOps pipeline
"""

import io
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────

# Base URL for the FastAPI backend (Docker internal network)
API_URL = "http://fastapi:8000"

# ── Page Setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS Styles ─────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #050d1a; }
section[data-testid="stSidebar"] { background: #080f1f !important; border-right: 1px solid #1a2744; }
section[data-testid="stSidebar"] * { color: #a8bdd4 !important; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }
.hero-header { background: linear-gradient(135deg, #0a1628 0%, #0d1f3e 50%, #091425 100%); border: 1px solid #1e3a5f; border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden; }
.hero-title { font-size: 2.2rem; font-weight: 600; color: #e8f0fe; margin: 0 0 0.4rem 0; }
.hero-subtitle { font-size: 1rem; color: #6b8aad; margin: 0; font-weight: 300; }
.hero-badge { display: inline-block; background: rgba(37,99,235,0.15); border: 1px solid rgba(37,99,235,0.4); color: #60a5fa; font-size: 0.75rem; padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem; font-family: 'DM Mono', monospace; }
.metric-card { background: #0a1628; border: 1px solid #1a2f50; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 600; color: #60a5fa; display: block; font-family: 'DM Mono', monospace; }
.metric-label { font-size: 0.78rem; color: #4a6580; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; display: block; }
.upload-zone { background: #080f1f; border: 2px dashed #1e3a5f; border-radius: 16px; padding: 2rem; text-align: center; }
.result-card-normal { background: linear-gradient(135deg, #052213 0%, #072e18 100%); border: 1px solid #166534; border-radius: 16px; padding: 2rem; }
.result-card-pneumonia { background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 100%); border: 1px solid #7f1d1d; border-radius: 16px; padding: 2rem; }
.result-label-normal { font-size: 1.8rem; font-weight: 600; color: #4ade80; }
.result-label-pneumonia { font-size: 1.8rem; font-weight: 600; color: #f87171; }
.plain-english-box { background: #0a1628; border-left: 3px solid #3b82f6; border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem; margin-top: 1rem; }
.plain-english-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #3b82f6; margin-bottom: 0.5rem; }
.plain-english-text { font-size: 1rem; color: #c4d4e8; line-height: 1.6; }
.history-item { background: #0a1628; border: 1px solid #1a2f50; border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between; }
.history-badge-normal { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-weight: 500; }
.history-badge-pneumonia { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: #f87171; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-weight: 500; }
.section-header { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: #3b6494; margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #0f1e35; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display: inline-block; margin-right: 6px; box-shadow: 0 0 6px rgba(34,197,94,0.5); }
.model-info-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #0f1e35; font-size: 0.85rem; }
.model-info-key { color: #4a6580; }
.model-info-val { color: #a8bdd4; font-family: 'DM Mono', monospace; font-size: 0.8rem; }
.pipeline-layer { background: #0a1628; border: 1px solid #1a2f50; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.5rem; }
.pipeline-layer-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: #3b6494; margin-bottom: 0.8rem; }
.pipeline-boxes { display: flex; gap: 10px; flex-wrap: wrap; }
.pipeline-box { background: #111d35; border: 1px solid #1e3a5f; border-radius: 8px; padding: 10px 16px; font-size: 0.82rem; color: #c4d4e8; font-weight: 500; flex: 1; min-width: 120px; text-align: center; }
.pipeline-box-sub { font-size: 0.7rem; color: #4a6580; margin-top: 3px; }
.pipeline-arrow { text-align: center; color: #3b6494; font-size: 1.2rem; margin: 4px 0; }
.dvc-box { background: #080f1f; border: 1px solid #1a2f50; border-radius: 8px; padding: 8px 14px; font-size: 0.78rem; color: #6b8aad; font-family: 'DM Mono', monospace; display: inline-block; margin: 4px; }
.dvc-arrow { color: #3b6494; font-size: 1rem; margin: 0 4px; }
.stButton button { background: #1d4ed8 !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; padding: 0.5rem 1.5rem !important; }
.stFileUploader > div { background: #080f1f !important; border: 1px solid #1a2f50 !important; border-radius: 12px !important; }
.stProgress > div > div { background: #1d4ed8 !important; }
div[data-testid="stExpander"] { background: #0a1628 !important; border: 1px solid #1a2f50 !important; border-radius: 10px !important; }
h1, h2, h3 { color: #e8f0fe !important; }
p, li { color: #7a95b0; }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialisation ──────────────────────────────────────────────

# Initialise prediction history list in session state on first load
if "history" not in st.session_state:
    st.session_state.history = []

# ── Helper Functions ──────────────────────────────────────────────────────────


def check_api():
    """
    Check if the FastAPI backend is reachable and healthy.

    Returns:
        dict: JSON response from /health if successful, None otherwise.
    """
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def predict(file_bytes, filename):
    """
    Send an image to the FastAPI /predict endpoint and return the result.

    Args:
        file_bytes (bytes): Raw image bytes to send.
        filename (str): Original filename, echoed back by API.

    Returns:
        dict: Prediction result with prediction, confidence, raw_score,
              latency_seconds, and interpretation. None if request fails.
    """
    try:
        files = {"file": (filename, file_bytes, "image/jpeg")}
        start = time.time()
        r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        latency = time.time() - start

        if r.status_code == 200:
            result = r.json()
            # Override latency with client-side measurement for accuracy
            result["latency_seconds"] = round(latency, 3)
            return result
        return None
    except Exception:
        return None


def confidence_gauge_html(confidence, prediction):
    """
    Generate an SVG circular confidence gauge as an HTML string.

    Args:
        confidence (float): Confidence percentage (0-100).
        prediction (str): NORMAL or PNEUMONIA — determines gauge colour.

    Returns:
        str: HTML string containing the SVG gauge element.
    """
    # Green for NORMAL, red for PNEUMONIA
    color = "#4ade80" if prediction == "NORMAL" else "#f87171"
    track_color = "#0a1628"

    # Calculate SVG stroke-dasharray for circular progress indicator
    circumference = 2 * 3.14159 * 54
    dash = (confidence / 100) * circumference
    gap = circumference - dash

    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:1rem 0;">
        <svg width="140" height="140" viewBox="0 0 140 140">
            <circle cx="70" cy="70" r="54" fill="none" stroke="{track_color}" stroke-width="12"/>
            <circle cx="70" cy="70" r="54" fill="none" stroke="{color}" stroke-width="12"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                stroke-dashoffset="{circumference/4:.1f}"
                stroke-linecap="round"
                style="transition: stroke-dasharray 0.8s ease;"/>
            <text x="70" y="65" text-anchor="middle" fill="{color}"
                font-size="22" font-weight="600" font-family="DM Mono, monospace">{confidence:.1f}%</text>
            <text x="70" y="85" text-anchor="middle" fill="#4a6580"
                font-size="11" font-family="DM Sans, sans-serif">confidence</text>
        </svg>
    </div>
    """


def plain_english(prediction, confidence):
    """
    Generate a plain-English explanation of the prediction result.

    Tiered by confidence so non-technical users understand certainty level.

    Args:
        prediction (str): NORMAL or PNEUMONIA.
        confidence (float): Confidence percentage (0-100).

    Returns:
        str: Human-readable explanation string.
    """
    if prediction == "NORMAL":
        if confidence >= 90:
            return "✅ The scan looks healthy. No signs of pneumonia were detected. The lungs appear clear and normal."
        elif confidence >= 75:
            return "✅ The scan appears normal with good confidence. No clear signs of pneumonia detected."
        else:
            return "🟡 The scan leans toward normal, but confidence is moderate. A doctor's review is recommended."
    else:
        if confidence >= 90:
            return "⚠️ Signs of pneumonia were detected with high confidence. Please consult a doctor immediately."
        elif confidence >= 75:
            return "⚠️ Possible pneumonia detected. Medical consultation is strongly recommended."
        else:
            return "🟡 The scan shows possible signs of pneumonia, but confidence is moderate. Please see a doctor."


# ── API Status Check ──────────────────────────────────────────────────────────

# Check API health once at page load
api_status = check_api()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div style="font-size:1.2rem;font-weight:600;color:#e8f0fe;">🫁 PneumoScan AI</div>
        <div style="font-size:0.75rem;color:#3b6494;margin-top:2px;">Medical Imaging Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Live API status indicator
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    if api_status:
        st.markdown(
            '<div style="font-size:0.85rem;color:#4a6580;"><span class="status-dot"></span>API Online — model loaded</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="font-size:0.85rem;color:#f87171;">⚫ API Offline</div>', unsafe_allow_html=True)

    # Static model metadata panel
    st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-info-row"><span class="model-info-key">Architecture</span><span class="model-info-val">MobileNetV2</span></div>
    <div class="model-info-row"><span class="model-info-key">Version</span><span class="model-info-val">v3 — proper split</span></div>
    <div class="model-info-row"><span class="model-info-key">Input</span><span class="model-info-val">224 × 224 px</span></div>
    <div class="model-info-row"><span class="model-info-key">NORMAL acc.</span><span class="model-info-val">77.4%</span></div>
    <div class="model-info-row"><span class="model-info-key">PNEUMONIA acc.</span><span class="model-info-val">98.8%</span></div>
    <div class="model-info-row" style="border:none;"><span class="model-info-key">Val AUC</span><span class="model-info-val">99.3%</span></div>
    """, unsafe_allow_html=True)

    # Page navigation — now includes ML Pipeline tab
    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Single Scan", "Batch Testing", "Prediction History", "ML Pipeline"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-header">Disclaimer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.75rem;color:#3b6494;line-height:1.6;">For educational purposes only. Not a substitute for professional medical diagnosis.</div>',
        unsafe_allow_html=True
    )

# ── Hero Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">MobileNetV2 · v3 Retrain · AUC 99.3%</div>
    <div class="hero-title">🫁 Chest X-Ray Pneumonia Detection</div>
    <div class="hero-subtitle">AI-powered analysis of chest radiographs — upload a scan to receive an instant assessment</div>
</div>
""", unsafe_allow_html=True)

# ── Page: Single Scan ─────────────────────────────────────────────────────────

if page == "Single Scan":
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Upload X-Ray Image</div>', unsafe_allow_html=True)

        # File uploader — accepts JPG and PNG formats only
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        # Clear previous result when image is removed or a different image is uploaded
        if uploaded is None:
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_filename", None)
        elif uploaded.name != st.session_state.get("last_filename", ""):
            st.session_state.pop("last_result", None)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_column_width=True, caption=f"📄 {uploaded.name}")

            # Trigger prediction when user clicks the analyse button
            if st.button("🔍 Analyse This Scan"):
                with st.spinner("Analysing scan..."):
                    uploaded.seek(0)
                    result = predict(uploaded.read(), uploaded.name)

                if result:
                    # Store result in session state for rendering
                    st.session_state.last_result = result
                    st.session_state.last_filename = uploaded.name
                    # Append to prediction history log
                    entry = {
                        "filename": uploaded.name,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "latency": result.get("latency_seconds", 0)
                    }
                    st.session_state.history.insert(0, entry)
                else:
                    st.error("Failed to connect to the API. Make sure FastAPI is running.")
        else:
            # Empty state placeholder
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">📡</div>
                <div style="color:#3b6494;font-size:0.9rem;">Drag & drop a chest X-ray here<br>JPG, JPEG, or PNG</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)

        if "last_result" in st.session_state:
            result = st.session_state.last_result
            pred = result["prediction"]
            conf = result["confidence"]

            # Select card style based on prediction class
            card_class = "result-card-normal" if pred == "NORMAL" else "result-card-pneumonia"
            label_class = "result-label-normal" if pred == "NORMAL" else "result-label-pneumonia"
            icon = "✅" if pred == "NORMAL" else "⚠️"

            # Diagnosis label card
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;color:#4a6580;margin-bottom:0.5rem;">Diagnosis</div>
                <div class="{label_class}">{icon} {pred}</div>
            </div>
            """, unsafe_allow_html=True)

            # SVG circular confidence gauge
            st.markdown(confidence_gauge_html(conf, pred), unsafe_allow_html=True)

            # Plain-English explanation for non-technical users
            st.markdown(f"""
            <div class="plain-english-box">
                <div class="plain-english-title">What this means for you</div>
                <div class="plain-english-text">{plain_english(pred, conf)}</div>
            </div>
            """, unsafe_allow_html=True)

            # Collapsible raw technical details for advanced users
            with st.expander("📊 Technical Details"):
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#6b8aad;line-height:2;">
                    <div><span style="color:#4a6580;">File:</span> {st.session_state.get('last_filename','')}</div>
                    <div><span style="color:#4a6580;">Prediction:</span> {result['prediction']}</div>
                    <div><span style="color:#4a6580;">Confidence:</span> {result['confidence']:.2f}%</div>
                    <div><span style="color:#4a6580;">Raw score:</span> {result.get('raw_score', 'N/A')}</div>
                    <div><span style="color:#4a6580;">Latency:</span> {result.get('latency_seconds', 0)*1000:.0f} ms</div>
                    <div><span style="color:#4a6580;">Model:</span> mobilenetv2_v3_final.h5</div>
                    <div><span style="color:#4a6580;">Threshold:</span> 0.65</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#080f1f;border:1px solid #0f1e35;border-radius:16px;padding:3rem;text-align:center;">
                <div style="font-size:3rem;margin-bottom:1rem;">🫁</div>
                <div style="color:#3b6494;font-size:0.9rem;">Upload an X-ray and click Analyse<br>to see the result here</div>
            </div>
            """, unsafe_allow_html=True)

# ── Page: Batch Testing ───────────────────────────────────────────────────────

elif page == "Batch Testing":
    st.markdown('<div class="section-header">Batch Testing — Multiple Images</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a6580;font-size:0.9rem;">Upload multiple chest X-rays at once and get results for all of them in one go.</p>', unsafe_allow_html=True)

    # Multi-file uploader
    batch_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")

    if batch_files:
        st.markdown(f'<div style="color:#60a5fa;font-size:0.85rem;margin-bottom:1rem;">📁 {len(batch_files)} file(s) selected</div>', unsafe_allow_html=True)

        if st.button(f"🔍 Analyse All {len(batch_files)} Scans"):
            results = []
            progress = st.progress(0)
            status = st.empty()

            # Process each uploaded file sequentially
            for i, f in enumerate(batch_files):
                status.markdown(f'<div style="color:#60a5fa;font-size:0.85rem;">Analysing {f.name}...</div>', unsafe_allow_html=True)
                f.seek(0)
                r = predict(f.read(), f.name)
                if r:
                    results.append({"filename": f.name, **r})
                    # Add each result to global history
                    st.session_state.history.insert(0, {
                        "filename": f.name,
                        "prediction": r["prediction"],
                        "confidence": r["confidence"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "latency": r.get("latency_seconds", 0)
                    })
                progress.progress((i + 1) / len(batch_files))

            status.empty()
            progress.empty()

            # Compute summary counts
            normal_count = sum(1 for r in results if r["prediction"] == "NORMAL")
            pneumonia_count = len(results) - normal_count

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{len(results)}</span><span class="metric-label">Total Scanned</span></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#4ade80;">{normal_count}</span><span class="metric-label">Normal</span></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f87171;">{pneumonia_count}</span><span class="metric-label">Pneumonia</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:1.5rem;">Individual Results</div>', unsafe_allow_html=True)

            # Render each result card
            for r in results:
                pred = r["prediction"]
                conf = r["confidence"]
                badge = f'<span class="history-badge-normal">NORMAL</span>' if pred == "NORMAL" else f'<span class="history-badge-pneumonia">PNEUMONIA</span>'
                icon = "✅" if pred == "NORMAL" else "⚠️"
                st.markdown(f"""
                <div class="history-item">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="font-size:1.2rem;">{icon}</div>
                        <div>
                            <div style="color:#c4d4e8;font-size:0.85rem;font-weight:500;">{r['filename']}</div>
                            <div style="color:#4a6580;font-size:0.75rem;margin-top:2px;">{plain_english(pred, conf)}</div>
                        </div>
                    </div>
                    <div style="text-align:right;">{badge}<div style="color:#4a6580;font-size:0.75rem;margin-top:4px;">{conf:.1f}% confidence</div></div>
                </div>
                """, unsafe_allow_html=True)

# ── Page: Prediction History ──────────────────────────────────────────────────

elif page == "Prediction History":
    st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="background:#080f1f;border:1px solid #0f1e35;border-radius:16px;padding:3rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">📋</div>
            <div style="color:#3b6494;font-size:0.9rem;">No predictions yet.<br>Run a scan from Single Scan or Batch Testing.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Compute session-level summary
        normal_total = sum(1 for h in st.session_state.history if h["prediction"] == "NORMAL")
        pneumonia_total = len(st.session_state.history) - normal_total

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><span class="metric-value">{len(st.session_state.history)}</span><span class="metric-label">Total Scans</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#4ade80;">{normal_total}</span><span class="metric-label">Normal</span></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f87171;">{pneumonia_total}</span><span class="metric-label">Pneumonia</span></div>', unsafe_allow_html=True)

        # Clear history resets the session state list and reruns the page
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">All Predictions (newest first)</div>', unsafe_allow_html=True)

        for h in st.session_state.history:
            pred = h["prediction"]
            badge = f'<span class="history-badge-normal">NORMAL</span>' if pred == "NORMAL" else f'<span class="history-badge-pneumonia">PNEUMONIA</span>'
            icon = "✅" if pred == "NORMAL" else "⚠️"
            latency_ms = round(h.get("latency", 0) * 1000)
            st.markdown(f"""
            <div class="history-item">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="font-size:1.2rem;">{icon}</div>
                    <div>
                        <div style="color:#c4d4e8;font-size:0.85rem;font-weight:500;">{h['filename']}</div>
                        <div style="color:#4a6580;font-size:0.75rem;margin-top:2px;">{plain_english(pred, h['confidence'])}</div>
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0;margin-left:1rem;">
                    {badge}
                    <div style="color:#4a6580;font-size:0.75rem;margin-top:4px;">{h['confidence']:.1f}% · {latency_ms}ms · {h['time']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Page: ML Pipeline Visualization ──────────────────────────────────────────

elif page == "ML Pipeline":
    st.markdown('<div class="section-header">End-to-End MLOps Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a6580;font-size:0.9rem;margin-bottom:1.5rem;">This page visualizes the complete machine learning pipeline — from raw data to live predictions.</p>', unsafe_allow_html=True)

    # Render each pipeline layer as a styled card with component boxes
    st.markdown("""
    <div class="pipeline-layer">
        <div class="pipeline-layer-title">📦 Data Layer</div>
        <div class="pipeline-boxes">
            <div class="pipeline-box">Kaggle Dataset<div class="pipeline-box-sub">5,856 chest X-rays</div></div>
            <div class="pipeline-box">Airflow DAG<div class="pipeline-box-sub">download → unzip → validate → preprocess → dvc_add</div></div>
            <div class="pipeline-box">DVC Versioning<div class="pipeline-box-sub">data/processed.dvc</div></div>
            <div class="pipeline-box">Git + Git LFS<div class="pipeline-box-sub">Source control</div></div>
        </div>
    </div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-layer">
        <div class="pipeline-layer-title">🧠 Training Layer</div>
        <div class="pipeline-boxes">
            <div class="pipeline-box">MobileNetV2<div class="pipeline-box-sub">TensorFlow 2.19 · ImageNet weights</div></div>
            <div class="pipeline-box">Kaggle GPU P100<div class="pipeline-box-sub">Phase 1: frozen · Phase 2: fine-tune</div></div>
            <div class="pipeline-box">80/10/10 Split<div class="pipeline-box-sub">1266 / 158 / 159 NORMAL</div></div>
            <div class="pipeline-box">MLflow Tracking<div class="pipeline-box-sub">3 experiments · params + metrics</div></div>
        </div>
    </div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-layer">
        <div class="pipeline-layer-title">🚀 Serving Layer</div>
        <div class="pipeline-boxes">
            <div class="pipeline-box">FastAPI :8000<div class="pipeline-box-sub">POST /predict · GET /health · GET /metrics</div></div>
            <div class="pipeline-box">Streamlit :8501<div class="pipeline-box-sub">This UI · REST calls only</div></div>
            <div class="pipeline-box">MLflow :5000<div class="pipeline-box-sub">Experiment tracking UI</div></div>
        </div>
    </div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-layer">
        <div class="pipeline-layer-title">📊 Monitoring Layer</div>
        <div class="pipeline-boxes">
            <div class="pipeline-box">Prometheus :9090<div class="pipeline-box-sub">xray_requests_total · xray_predictions_total · latency</div></div>
            <div class="pipeline-box">Grafana :3000<div class="pipeline-box-sub">Live dashboard · 10s refresh</div></div>
            <div class="pipeline-box">Node Exporter :9100<div class="pipeline-box-sub">Host CPU · memory · disk</div></div>
        </div>
    </div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-layer">
        <div class="pipeline-layer-title">🐳 Infrastructure Layer</div>
        <div class="pipeline-boxes">
            <div class="pipeline-box">Docker Compose<div class="pipeline-box-sub">6 containers · xray-network</div></div>
            <div class="pipeline-box">Named Volumes<div class="pipeline-box-sub">Persistent MLflow + Grafana data</div></div>
            <div class="pipeline-box">Health Checks<div class="pipeline-box-sub">Auto-restart on failure</div></div>
            <div class="pipeline-box">MLproject<div class="pipeline-box-sub">Reproducible environments</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # DVC DAG showing model version history
    st.markdown('<div class="section-header" style="margin-top:2rem;">DVC Model Versioning DAG</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#080f1f;border:1px solid #1a2f50;border-radius:12px;padding:1.5rem;margin-top:0.5rem;">
        <div style="font-size:0.75rem;color:#3b6494;margin-bottom:1rem;text-transform:uppercase;letter-spacing:1px;">Model Evolution tracked by DVC</div>
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
            <div class="dvc-box">data/processed.dvc</div>
            <span class="dvc-arrow">→</span>
            <div class="dvc-box">best_model.h5.dvc<br><span style="font-size:0.65rem;color:#3b6494;">v1 checkpoint</span></div>
            <span class="dvc-arrow">→</span>
            <div class="dvc-box">mobilenetv2_final.h5.dvc<br><span style="font-size:0.65rem;color:#3b6494;">v1 — biased</span></div>
            <span class="dvc-arrow">→</span>
            <div class="dvc-box">mobilenetv2_v2_balanced.h5.dvc<br><span style="font-size:0.65rem;color:#3b6494;">v2 — attempt</span></div>
            <span class="dvc-arrow">→</span>
            <div class="dvc-box" style="border-color:#4ade80;color:#4ade80;">mobilenetv2_v3_final.h5.dvc<br><span style="font-size:0.65rem;">v3 — PRODUCTION ✅</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live metrics fetched directly from FastAPI /metrics endpoint
    st.markdown('<div class="section-header" style="margin-top:2rem;">Live System Metrics</div>', unsafe_allow_html=True)
    try:
        metrics_response = requests.get(f"{API_URL}/metrics", timeout=3)
        if metrics_response.status_code == 200:
            lines = metrics_response.text.split("\n")

            # Parse custom xray_* metric values from Prometheus text format
            total_requests = 0
            normal_count = 0
            pneumonia_count = 0

            for line in lines:
                if line.startswith("xray_requests_total{") and "success" in line:
                    total_requests = int(float(line.split()[-1]))
                if line.startswith("xray_predictions_total{") and "NORMAL" in line:
                    normal_count = int(float(line.split()[-1]))
                if line.startswith("xray_predictions_total{") and "PNEUMONIA" in line:
                    pneumonia_count = int(float(line.split()[-1]))

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{total_requests}</span><span class="metric-label">Total Predictions</span></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#4ade80;">{normal_count}</span><span class="metric-label">Normal Results</span></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#f87171;">{pneumonia_count}</span><span class="metric-label">Pneumonia Detected</span></div>', unsafe_allow_html=True)
        else:
            st.info("Metrics unavailable — FastAPI may be starting up.")
    except Exception:
        st.info("Could not fetch live metrics. Make sure the FastAPI container is running.")

    # External MLOps tool link cards
    st.markdown('<div class="section-header" style="margin-top:2rem;">MLOps Tool Links — Click to Open</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown("""
        <a href="http://localhost:5000" target="_blank" style="text-decoration:none;">
        <div class="pipeline-box" style="padding:1rem;cursor:pointer;transition:border 0.2s;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#1e3a5f'">
            <div style="font-size:1.4rem;">📈</div>
            <div style="margin-top:6px;font-weight:600;color:#c4d4e8;">MLflow</div>
            <div class="pipeline-box-sub">localhost:5000</div>
            <div style="margin-top:8px;font-size:0.72rem;color:#3b82f6;">Experiment tracking · 3 runs</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <a href="http://localhost:9090" target="_blank" style="text-decoration:none;">
        <div class="pipeline-box" style="padding:1rem;cursor:pointer;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#1e3a5f'">
            <div style="font-size:1.4rem;">🔥</div>
            <div style="margin-top:6px;font-weight:600;color:#c4d4e8;">Prometheus</div>
            <div class="pipeline-box-sub">localhost:9090</div>
            <div style="margin-top:8px;font-size:0.72rem;color:#3b82f6;">5 custom xray_* metrics</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <a href="http://localhost:3000" target="_blank" style="text-decoration:none;">
        <div class="pipeline-box" style="padding:1rem;cursor:pointer;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#1e3a5f'">
            <div style="font-size:1.4rem;">📊</div>
            <div style="margin-top:6px;font-weight:600;color:#c4d4e8;">Grafana</div>
            <div class="pipeline-box-sub">localhost:3000</div>
            <div style="margin-top:8px;font-size:0.72rem;color:#3b82f6;">Live dashboard · 10s refresh</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <a href="http://localhost:8080" target="_blank" style="text-decoration:none;">
        <div class="pipeline-box" style="padding:1rem;cursor:pointer;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#1e3a5f'">
            <div style="font-size:1.4rem;">🌀</div>
            <div style="margin-top:6px;font-weight:600;color:#c4d4e8;">Airflow</div>
            <div class="pipeline-box-sub">localhost:8080</div>
            <div style="margin-top:8px;font-size:0.72rem;color:#3b82f6;">Pipeline management · DAG runs</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with t5:
        st.markdown("""
        <a href="http://localhost:8000/docs" target="_blank" style="text-decoration:none;">
        <div class="pipeline-box" style="padding:1rem;cursor:pointer;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#1e3a5f'">
            <div style="font-size:1.4rem;">⚡</div>
            <div style="margin-top:6px;font-weight:600;color:#c4d4e8;">FastAPI</div>
            <div class="pipeline-box-sub">localhost:8000</div>
            <div style="margin-top:8px;font-size:0.72rem;color:#3b82f6;">REST API docs · 6 endpoints</div>
        </div>
        </a>
        """, unsafe_allow_html=True)