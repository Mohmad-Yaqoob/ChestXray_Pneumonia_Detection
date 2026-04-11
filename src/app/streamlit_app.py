import streamlit as st
import requests
import io
from PIL import Image
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app uses a fine-tuned **MobileNetV2** CNN "
        "to classify chest X-rays as **Normal** or **Pneumonia**."
    )
    st.markdown("---")
    st.markdown("### Model Info")
    try:
        r = requests.get(f"{api_url}/model/info", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.json(info)
        else:
            st.warning("Could not fetch model info")
    except Exception:
        st.warning("Start FastAPI to see model info")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.markdown(
    "Upload a chest X-ray image to detect pneumonia instantly. "
    "This tool is for **educational purposes only** and does not replace "
    "professional medical diagnosis."
)
st.markdown("---")

# ── API Health check ──────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Upload X-Ray Image")
with col2:
    try:
        health = requests.get(f"{api_url}/health", timeout=3)
        if health.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline")

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray (PA or AP view)"
)

if uploaded_file is not None:
    # Show image
    col_img, col_result = st.columns(2)

    with col_img:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=uploaded_file.name)
        st.caption(f"Size: {image.size[0]}x{image.size[1]} px")

    with col_result:
        st.subheader("Prediction")

        with st.spinner("Analyzing X-ray..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)

                # Call API
                start = time.time()
                response = requests.post(
                    f"{api_url}/predict",
                    files={"file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )},
                    timeout=30
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    interpretation = result["interpretation"]

                    # Display result
                    if prediction == "PNEUMONIA":
                        st.error(f"**{prediction}** Detected")
                        st.metric("Confidence", f"{confidence}%")
                        st.warning(interpretation)
                    else:
                        st.success(f"**{prediction}**")
                        st.metric("Confidence", f"{confidence}%")
                        st.info(interpretation)

                    # Confidence bar
                    st.markdown("**Confidence Score**")
                    st.progress(int(confidence))

                    # Details expander
                    with st.expander("View Details"):
                        st.json(result)
                        st.caption(f"UI round-trip: {elapsed:.3f}s")

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Sample images section ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Try Sample Images")
st.markdown(
    "No image handy? Use one from the test set located at "
    "`data/processed/test/NORMAL/` or `data/processed/test/PNEUMONIA/`"
)

# ── Batch testing section ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Batch Testing"):
    st.markdown("Test multiple images at once")
    batch_files = st.file_uploader(
        "Upload multiple X-rays",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch"
    )

    if batch_files and st.button("Run Batch Prediction"):
        results = []
        progress = st.progress(0)

        for i, bf in enumerate(batch_files):
            try:
                r = requests.post(
                    f"{api_url}/predict",
                    files={"file": (bf.name, bf.getvalue(), bf.type)},
                    timeout=30
                )
                if r.status_code == 200:
                    d = r.json()
                    results.append({
                        "File":       d["filename"],
                        "Prediction": d["prediction"],
                        "Confidence": f"{d['confidence']}%",
                        "Latency":    f"{d['latency_seconds']}s"
                    })
            except Exception as e:
                results.append({"File": bf.name, "Prediction": f"Error: {e}",
                                 "Confidence": "-", "Latency": "-"})
            progress.progress((i + 1) / len(batch_files))

        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # Summary
            total    = len(df)
            pneum    = len(df[df["Prediction"] == "PNEUMONIA"])
            normal   = len(df[df["Prediction"] == "NORMAL"])
            st.markdown(f"**Summary:** {total} images — "
                        f"{pneum} Pneumonia, {normal} Normal")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with MobileNetV2 + FastAPI + Streamlit | "
    "For educational purposes only | "
    "Not a substitute for professional medical advice"
)