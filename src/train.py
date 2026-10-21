import json
from pathlib import Path
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from src.architecture import build_model, compile_model
from src.config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    FINE_TUNE_AT,
    LEARNING_RATE,
    MANIFEST_SPLIT_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_EXPORT_PATH,
    MODELS_DIR,
    SEED,
)
from src.metrics import (
    evaluate_predictions,
    generate_classification_report_text,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from src.preprocessing import create_dataset_from_manifest


def train_and_log(
    run_name: str,
    base_trainable: bool = False,
    fine_tune_at: int = None,
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = DROPOUT_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> str:
    """Entrena una variante de la red y registra todos los artefactos, parámetros y métricas en MLflow."""
    # Configurar MLflow Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if not MANIFEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "No se encontró manifest_split.csv. Ejecuta 'python -m src.dataset' primero."
        )

    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    df_train = df_split[df_split["split"] == "train"]
    df_val = df_split[df_split["split"] == "val"]
    df_test = df_split[df_split["split"] == "test"]

    ds_train = create_dataset_from_manifest(
        df_train, is_training=True, batch_size=batch_size
    )
    ds_val = create_dataset_from_manifest(
        df_val, is_training=False, batch_size=batch_size
    )
    ds_test = create_dataset_from_manifest(
        df_test, is_training=False, batch_size=batch_size
    )

    # Construir y compilar modelo
    model, _ = build_model(
        base_trainable=base_trainable,
        fine_tune_at=fine_tune_at,
        dropout_rate=dropout_rate,
    )
    model = compile_model(model, learning_rate=learning_rate)

    # Callbacks
    checkpoint_path = MODELS_DIR / f"{run_name}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Registrar Parámetros
        mlflow.log_params(
            {
                "run_name": run_name,
                "architecture": "ResNet50",
                "base_trainable": base_trainable,
                "fine_tune_at": fine_tune_at if fine_tune_at else "None",
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": SEED,
            }
        )

        # 2. Entrenar
        print(f"\n Iniciando entrenamiento: {run_name}...")
        history = model.fit(
            ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks
        )

        # Registrar métricas por época
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics(
                {
                    "train_loss": history.history["loss"][epoch],
                    "train_accuracy": history.history["accuracy"][epoch],
                    "val_loss": history.history["val_loss"][epoch],
                    "val_accuracy": history.history["val_accuracy"][epoch],
                },
                step=epoch + 1,
            )

        # 3. Evaluación en el conjunto de prueba (Test)
        y_true = []
        y_pred_probs = []
        for images, labels in ds_test:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred_probs.extend(preds.flatten())

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)

        test_metrics = evaluate_predictions(y_true, y_pred_probs)
        mlflow.log_metrics(
            {f"test_{k}": float(v) for k, v in test_metrics.items()}
        )

        # 4. Generar y guardar Artefactos
        artifacts_dir = MODELS_DIR / f"artifacts_{run_name}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        cm_path = artifacts_dir / "confusion_matrix.png"
        roc_path = artifacts_dir / "roc_curve.png"
        curves_path = artifacts_dir / "training_curves.png"
        report_path = artifacts_dir / "classification_report.txt"

        plot_confusion_matrix(y_true, y_pred_probs, cm_path)
        plot_roc_curve(y_true, y_pred_probs, roc_path)
        plot_training_curves(history.history, curves_path)
        generate_classification_report_text(y_true, y_pred_probs, report_path)

        mlflow.log_artifacts(str(artifacts_dir))

        # Información de entrada esperada
        input_info = {
            "input_shape": list((batch_size, 224, 224, 3)),
            "data_type": "float32",
        }
        with open(artifacts_dir / "input_example_info.json", "w") as f:
            json.dump(input_info, f)
        mlflow.log_artifact(str(artifacts_dir / "input_example_info.json"))

        # 5. Registrar Modelo en MLflow
        mlflow.keras.log_model(model, artifact_path="model")

        # Guardar como exportación local para FastAPI
        model.save(MODEL_EXPORT_PATH)
        print(f" Modelo exportado localmente en: {MODEL_EXPORT_PATH}")
        print(f" Run {run_name} registrado en MLflow con ID: {run.info.run_id}")

        return run.info.run_id


if __name__ == "__main__":
    # Ejecutar Run 1 de prueba (Fase A: Congelado)
    train_and_log(
        run_name="Run_1_Frozen_Baseline",
        base_trainable=False,
        learning_rate=1e-3,
        epochs=3,  # 3 épocas para verificar que el pipeline completo funcione rápido
    )
    
    
# ┌────────────────────────────────────────────────────────────────────────┐
# │ RUN                                                                    │
# └────────────────────────────────────────────────────────────────────────┘

# python -m src.train