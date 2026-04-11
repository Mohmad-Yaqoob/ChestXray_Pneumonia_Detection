from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import zipfile
import shutil
import logging

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/mnt/d/IIT Madras/2nd Mtech/MLOps/DA5402-Assignments/project/ChestXray_Pneumonia_Detection"
RAW_DIR      = os.path.join(PROJECT_ROOT, "data/raw")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data/processed")
DATASET      = "paultimothymooney/chest-xray-pneumonia"

logger = logging.getLogger(__name__)

# ── Task functions ─────────────────────────────────────────────────────────────

def download_dataset():
    """Download chest x-ray dataset from Kaggle."""
    import kaggle
    os.makedirs(RAW_DIR, exist_ok=True)
    logger.info(f"Downloading {DATASET} to {RAW_DIR}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=RAW_DIR, unzip=False)
    logger.info("Download complete")


def unzip_dataset():
    """Unzip the downloaded dataset."""
    zip_path = os.path.join(RAW_DIR, "chest-xray-pneumonia.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    logger.info(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    logger.info("Unzip complete")


def validate_dataset():
    """Check expected folder structure exists after extraction."""
    expected = [
        "chest_xray/train/NORMAL",
        "chest_xray/train/PNEUMONIA",
        "chest_xray/val/NORMAL",
        "chest_xray/val/PNEUMONIA",
        "chest_xray/test/NORMAL",
        "chest_xray/test/PNEUMONIA",
    ]
    for rel_path in expected:
        full_path = os.path.join(RAW_DIR, rel_path)
        if not os.path.isdir(full_path):
            raise ValueError(f"Missing expected folder: {full_path}")
        count = len(os.listdir(full_path))
        logger.info(f"{rel_path}: {count} images")
    logger.info("Validation passed")


def preprocess_dataset():
    """
    Copy images into processed/ with a clean flat structure.
    processed/
      train/NORMAL/
      train/PNEUMONIA/
      val/NORMAL/
      val/PNEUMONIA/
      test/NORMAL/
      test/PNEUMONIA/
    """
    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    for split in splits:
        for cls in classes:
            src = os.path.join(RAW_DIR, "chest_xray", split, cls)
            dst = os.path.join(PROC_DIR, split, cls)
            os.makedirs(dst, exist_ok=True)

            files = [f for f in os.listdir(src)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for fname in files:
                shutil.copy2(os.path.join(src, fname),
                             os.path.join(dst, fname))

            logger.info(f"Copied {len(files)} files → {dst}")

    logger.info("Preprocessing complete")


def dvc_add_data():
    """Track processed data with DVC."""
    import subprocess

    dvc_bin = "/mnt/d/IIT Madras/2nd Mtech/MLOps/DA5402-Assignments/project/ChestXray_Pneumonia_Detection/venv/bin/dvc"
    proc_path = os.path.join(PROJECT_ROOT, "data", "processed")

    if not os.path.exists(dvc_bin):
        raise FileNotFoundError(f"DVC binary not found at {dvc_bin}")

    if not os.path.isdir(proc_path):
        raise FileNotFoundError(f"data/processed not found at {proc_path}")

    logger.info(f"Using DVC at: {dvc_bin}")
    logger.info(f"Tracking: {proc_path}")

    result = subprocess.run(
        [dvc_bin, "add", proc_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    logger.info("STDOUT: " + result.stdout)
    if result.returncode != 0:
        logger.error("STDERR: " + result.stderr)
        raise RuntimeError(f"DVC add failed:\n{result.stderr}")

    logger.info("DVC tracking added successfully")
# ── DAG definition ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="xray_data_pipeline",
    default_args=default_args,
    description="Download, validate and preprocess chest X-ray dataset",
    schedule_interval=None,       # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["xray", "data", "mlops"],
) as dag:

    t1 = PythonOperator(task_id="download_dataset",  python_callable=download_dataset)
    t2 = PythonOperator(task_id="unzip_dataset",     python_callable=unzip_dataset)
    t3 = PythonOperator(task_id="validate_dataset",  python_callable=validate_dataset)
    t4 = PythonOperator(task_id="preprocess_dataset",python_callable=preprocess_dataset)
    t5 = PythonOperator(task_id="dvc_add_data",      python_callable=dvc_add_data)

    t1 >> t2 >> t3 >> t4 >> t5