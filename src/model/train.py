# src/model/train.py

import os
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
NUM_CLASSES = 1          # binary: Normal vs Pneumonia
DATA_DIR    = "data/processed"   # override in Colab
MODEL_DIR   = "models"
MLFLOW_URI  = "http://localhost:5000"  # override in Colab

# ── Data generators ───────────────────────────────────────────────────────────

def build_generators(data_dir):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
    )
    val = val_test_gen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
    )
    test = val_test_gen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
    )
    return train, val, test


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze base first, fine-tune later
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model, base


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir=DATA_DIR, model_dir=MODEL_DIR, mlflow_uri=MLFLOW_URI):
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("chest-xray-pneumonia")

    train_gen, val_gen, test_gen = build_generators(data_dir)
    model, base = build_model()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_auc",
        ),
    ]

    with mlflow.start_run(run_name="mobilenetv2-frozen-base"):
        # Log params
        mlflow.log_params({
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "base_model": "MobileNetV2",
            "base_trainable": False,
        })

        # Phase 1 — train head only
        print("\n=== Phase 1: Training head (base frozen) ===")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
        )

        # Log metrics per epoch
        for epoch, (acc, val_acc, auc, val_auc) in enumerate(zip(
            history.history["accuracy"],
            history.history["val_accuracy"],
            history.history["auc"],
            history.history["val_auc"],
        )):
            mlflow.log_metrics({
                "train_accuracy": acc,
                "val_accuracy": val_acc,
                "train_auc": auc,
                "val_auc": val_auc,
            }, step=epoch)

        # Phase 2 — fine-tune top layers of base
        print("\n=== Phase 2: Fine-tuning top 30 layers ===")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=LR / 10),
            loss="binary_crossentropy",
            metrics=["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")],
        )

        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks,
        )

        # Evaluate on test set
        print("\n=== Evaluating on test set ===")
        results = model.evaluate(test_gen)
        test_metrics = dict(zip(model.metrics_names, results))
        print(test_metrics)

        mlflow.log_metrics({
            "test_accuracy":  test_metrics["accuracy"],
            "test_auc":       test_metrics["auc"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
        })

        # Save model
        model_path = os.path.join(model_dir, "mobilenetv2_final.h5")
        model.save(model_path)
        mlflow.tensorflow.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_path)

        print(f"\nModel saved to {model_path}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC:      {test_metrics['auc']:.4f}")

    return model


if __name__ == "__main__":
    train()