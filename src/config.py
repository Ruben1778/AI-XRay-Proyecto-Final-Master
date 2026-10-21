import os
from pathlib import Path


# ┌────────────────────────────────────────────────────────────────────────┐
# │ 1. RUTAS Y DIRECTORIOS DEL PROYECTO                                    │
# └────────────────────────────────────────────────────────────────────────┘

# Ruta base del proyecto (AI-XRay)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta de los datos 
DEFAULT_DATA_DIR = BASE_DIR / "data"

DATA_DIR = Path(os.getenv("AIXRAY_DATA_DIR", DEFAULT_DATA_DIR))

# Rutas para guardar artefactos y modelos
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Archivo local del mejor modelo exportado
MODEL_EXPORT_PATH = MODELS_DIR / "aixray_model.keras"

# Ruta para manifiestos generados
MANIFEST_PATH = DEFAULT_DATA_DIR / "manifest.csv"
MANIFEST_SPLIT_PATH = DEFAULT_DATA_DIR / "manifest_split.csv"

# ┌────────────────────────────────────────────────────────────────────────┐
# │ 02. CONFIGURACIÓN DE EXPERIMENTO & SEMILLAS                            │
# └────────────────────────────────────────────────────────────────────────┘
SEED = 42

# MLflow Tracking
MLFLOW_EXPERIMENT_NAME = "AI-XRay"
MLFLOW_TRACKING_URI = "sqlite:///" + str(BASE_DIR / "mlflow.db")

# ┌────────────────────────────────────────────────────────────────────────┐
# │ 03. PARÁMETROS DE IMAGEN Y MODELO                                      │
# └────────────────────────────────────────────────────────────────────────┘


IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 1  # Clasificación binaria: 0 = NORMAL, 1 = PNEUMONIA

# ┌────────────────────────────────────────────────────────────────────────┐
# │ 04. HIPERPARÁMETROS Y ENTRENAMIENTO POR DEFECTO                                   │
# └────────────────────────────────────────────────────────────────────────┘

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

# Subconjunto de demostración (2,000 imágenes balanceadas para la demo de clase)
USE_SUBSET_DEMO = True
SUBSET_SIZE_PER_CLASS = 1000

# Capa a partir de la cual se descongelan pesos en ResNet50 para Fine-Tuning (Fase B)
FINE_TUNE_AT = 143