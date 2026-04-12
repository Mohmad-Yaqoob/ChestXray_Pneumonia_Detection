# 🫁 Chest X-Ray Pneumonia Detection System

> An end-to-end MLOps project that detects pneumonia from chest X-rays using a fine-tuned MobileNetV2 CNN — fully containerized and production-ready.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.11-red)
![DVC](https://img.shields.io/badge/DVC-3.67-purple)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Services](#services)
- [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
- [API Reference](#api-reference)
- [MLOps Practices](#mlops-practices)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

Pneumonia is one of the leading causes of death worldwide, responsible for over **2 million deaths annually**. Early and accurate diagnosis is critical — but it depends heavily on the availability of trained radiologists to interpret chest X-rays.

In resource-limited settings such as rural hospitals and clinics in developing countries, radiologists are scarce. This leads to critical delays in diagnosis and treatment.

**This project builds an AI-powered system that can classify a chest X-ray as Normal or Pneumonia in under 2 seconds — making expert-level screening accessible anywhere.**

---

## Project Overview

| Property | Details |
|---|---|
| Domain | Medical Imaging, Healthcare AI |
| Task | Binary Classification (Normal vs Pneumonia) |
| Dataset | [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Model | Fine-tuned MobileNetV2 |
| Training Platform | Kaggle (GPU P100) |
| Deployment | Docker Compose (local) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  Kaggle Dataset → Airflow Pipeline → DVC Versioning         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     TRAINING LAYER                          │
│  MobileNetV2 Fine-tuning → MLflow Experiment Tracking       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     SERVING LAYER                           │
│  FastAPI Inference Engine ← Streamlit Web Frontend          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     MONITORING LAYER                        │
│  Prometheus Metrics → Grafana Dashboard                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     INFRA LAYER                             │
│  Docker Compose — 6 containerized services                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Model | MobileNetV2 + TensorFlow | CNN for X-ray classification |
| Data Pipeline | Apache Airflow | Automated data ingestion |
| Data Versioning | DVC | Track datasets and models |
| Experiment Tracking | MLflow | Log metrics, params, artifacts |
| Inference API | FastAPI + Uvicorn | REST API for predictions |
| Frontend | Streamlit | Web UI for doctors/users |
| Monitoring | Prometheus + Grafana | Real-time metrics dashboard |
| Containerization | Docker + Docker Compose | Full stack deployment |
| Version Control | Git + GitHub | Code versioning |

---

## Project Structure

```
ChestXray_Pneumonia_Detection/
├── airflow/
│   ├── dags/
│   │   └── xray_pipeline.py        # 5-task Airflow DAG
│   └── logs/
├── data/
│   ├── raw/                        # Downloaded from Kaggle (DVC)
│   └── processed/                  # Train/val/test splits (DVC)
│       ├── train/
│       │   ├── NORMAL/             # ~1341 images
│       │   └── PNEUMONIA/          # ~3875 images
│       ├── val/
│       └── test/
├── docker/
│   ├── Dockerfile.fastapi
│   ├── Dockerfile.streamlit
│   └── requirements.docker.txt
├── models/
│   ├── mobilenetv2_final.h5        # Final trained model (DVC)
│   └── best_model.h5               # Best checkpoint (DVC)
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           └── dashboards/
├── notebooks/
│   └── train_mobilenetv2.ipynb
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI app
│   ├── app/
│   │   └── streamlit_app.py        # Streamlit frontend
│   └── model/
│       └── train.py                # Training script
├── tests/
│   └── test_api.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Model Performance

The model was trained on the Kaggle Chest X-Ray dataset using two-phase transfer learning:

| Metric | Score |
|---|---|
| Test AUC | **93.6%** |
| Test Recall | **99.2%** |
| Test Accuracy | 79.2% |
| Test Precision | 73.4% |

> **Why Recall matters most:** In medical diagnosis, missing a pneumonia case (false negative) is far more dangerous than a false alarm. A 99.2% recall means the model catches virtually every pneumonia case.

### Training Strategy

**Phase 1 — Frozen base (10 epochs)**
- MobileNetV2 base frozen (ImageNet weights preserved)
- Only custom head trained
- Learning rate: 1e-4

**Phase 2 — Fine-tuning (10 epochs)**
- Top 30 layers of base unfrozen
- Lower learning rate: 1e-5
- EarlyStopping + ReduceLROnPlateau callbacks

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Git
- 8GB RAM minimum

### 1. Clone the repository

```bash
git clone https://github.com/Mohmad-Yaqoob/ChestXray_Pneumonia_Detection.git
cd ChestXray_Pneumonia_Detection
```

### 2. Pull model files (DVC)

```bash
pip install dvc
dvc pull
```

### 3. Launch the full stack

```bash
docker compose up -d
```

### 4. Verify all services

```bash
docker compose ps
```

All 6 containers should show `Up`:
```
xray-fastapi        Up (healthy)
xray-streamlit      Up
xray-mlflow         Up
xray-prometheus     Up
xray-grafana        Up
xray-node-exporter  Up
```

### 5. Open the app

Go to **http://localhost:8501** and upload a chest X-ray!

---

## Services

| Service | URL | Credentials |
|---|---|---|
| Streamlit Web App | http://localhost:8501 | — |
| FastAPI Swagger Docs | http://localhost:8000/docs | — |
| FastAPI Health Check | http://localhost:8000/health | — |
| MLflow Tracking UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana Dashboard | http://localhost:3000 | admin / admin |

---

## Phase-by-Phase Breakdown

### Phase 1 — Repository & Environment Setup
- GitHub repository initialized
- Python virtual environment with all dependencies
- Git + DVC initialized
- `.gitignore` configured to exclude data, models, secrets

### Phase 2 — Airflow Data Pipeline
Automated 5-task DAG:
```
download_dataset → unzip_dataset → validate_dataset → preprocess_dataset → dvc_add_data
```
- Downloads chest X-ray dataset from Kaggle API
- Validates all 6 expected folders exist
- Copies images to clean `processed/` structure
- Tracks output with DVC

### Phase 3 — Model Training (Kaggle GPU)
- MobileNetV2 with ImageNet weights
- Custom head: GlobalAveragePooling → BatchNorm → Dense(256) → Dropout → Dense(64) → Sigmoid
- Data augmentation: rotation, zoom, horizontal flip, shift
- MLflow tracking: all hyperparameters, per-epoch metrics, model artifacts
- Trained on Kaggle P100 GPU (~20 minutes)

### Phase 4 — FastAPI Inference Engine
- `POST /predict` — upload X-ray, get prediction + confidence
- `GET /health` — model status check
- `GET /metrics` — Prometheus-format metrics
- `GET /model/info` — model metadata
- Prometheus counters for request count, latency, prediction distribution

### Phase 5 — Streamlit Frontend
- Single image upload with instant prediction
- Confidence score with visual progress bar
- Batch testing for multiple images
- API health indicator in header
- Model info in sidebar

### Phase 6 — Prometheus + Grafana Monitoring
Metrics tracked:
- `xray_requests_total` — total predictions by status
- `xray_request_latency_seconds` — inference latency histogram
- `xray_predictions_total` — predictions by class (NORMAL/PNEUMONIA)

Grafana dashboard panels:
- Total predictions (stat)
- Average latency (stat)
- Pneumonia vs Normal counts (stat)
- Request rate over time (timeseries)
- Latency over time (timeseries)

### Phase 7 — Docker Compose Deployment
All 6 services on a shared `xray-network`:
- FastAPI and Streamlit built from custom Dockerfiles
- MLflow, Prometheus, Grafana, Node Exporter from official images
- Named volumes for persistent data
- Health checks and auto-restart policies

---

## API Reference

### `POST /predict`

Upload a chest X-ray image and receive a prediction.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.jpg"
```

**Response:**
```json
{
  "filename": "chest_xray.jpg",
  "prediction": "PNEUMONIA",
  "confidence": 99.78,
  "raw_score": 0.997781,
  "latency_seconds": 1.524,
  "interpretation": "Pneumonia detected. Please consult a doctor."
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/mobilenetv2_final.h5"
}
```

---

## MLOps Practices

| Practice | Implementation |
|---|---|
| Data Versioning | DVC tracks datasets and model files |
| Experiment Tracking | MLflow logs every training run |
| Reproducibility | Docker ensures identical environments |
| Automated Pipeline | Airflow DAG for data ingestion |
| Monitoring | Prometheus + Grafana for production metrics |
| API-first | FastAPI serves model as REST endpoint |
| Health Checks | Docker healthcheck on FastAPI container |
| Separation of concerns | Data / Training / Serving / Monitoring in separate layers |

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — Paul Mooney, Kaggle

- Training: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- Validation: 16 images (8 Normal, 8 Pneumonia)
- Test: 624 images (234 Normal, 390 Pneumonia)

Dataset link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## Disclaimer

> This tool is built for **educational and research purposes only**. It is **not a medical device** and should **not be used as a substitute for professional medical diagnosis**. Always consult a qualified healthcare professional for medical advice.

---

## Acknowledgements

- Dataset: Kermany et al., Cell 2018
- Base model: MobileNetV2 — Howard et al., Google
- Course: DA5402 — MLOps, IIT Madras M.Tech Program
- Framework: TensorFlow / Keras, FastAPI, Streamlit, Apache Airflow

---

*Built with passion as part of IIT Madras M.Tech MLOps coursework — Bismillah to Alhamdulillah 🤲*