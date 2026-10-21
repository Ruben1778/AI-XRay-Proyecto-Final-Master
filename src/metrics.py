import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5
) -> dict:
    """Calcula las métricas principales de evaluación para clasificación binaria."""
    y_pred = (y_pred_probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_pred_probs)),
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path,
    threshold: float = 0.5,
):
    """Genera y guarda el gráfico de la matriz de confusión."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
    )
    plt.title("Matriz de Confusión")
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Predicción del Modelo")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, save_path: Path):
    """Genera y guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_val = roc_auc_score(y_true, y_pred_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (Recall / TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(history: dict, save_path: Path):
    """Genera y guarda las curvas de evolución del entrenamiento (Loss y Accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Curva de Loss
    axes[0].plot(history.get("loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Pérdida (Loss) por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Curva de Accuracy
    axes[1].plot(history.get("accuracy", []), label="Train Accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="Val Accuracy")
    axes[1].set_title("Exactitud (Accuracy) por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_classification_report_text(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path,
    threshold: float = 0.5,
) -> str:
    """Genera y guarda el reporte detallado en archivo de texto."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    report_str = classification_report(
        y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("INFORME DE CLASIFICACIÓN DETALLADO\n")
        f.write("=" * 40 + "\n\n")
        f.write(report_str)

    return report_str


if __name__ == "__main__":
    # Prueba del módulo con datos sintéticos
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.85, 0.4, 0.95, 0.3, 0.7, 0.6])

    metrics = evaluate_predictions(y_true, y_probs)
    print(" Métricas de prueba calculadas correctamente:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")
        
        
# ┌────────────────────────────────────────────────────────────────────────┐
# │ RUN                                                                    │
# └────────────────────────────────────────────────────────────────────────┘

# python -m src.architecture