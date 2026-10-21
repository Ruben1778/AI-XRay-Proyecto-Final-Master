import io
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from src.config import IMG_SIZE, MODEL_EXPORT_PATH


class ModelService:

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Carga el modelo local exportado en memoria."""
        if MODEL_EXPORT_PATH.exists():
            try:
                self.model = tf.keras.models.load_model(MODEL_EXPORT_PATH)
                print(f" Modelo cargado con éxito desde: {MODEL_EXPORT_PATH}")
            except Exception as e:
                print(f" Error cargando el modelo: {e}")
                self.model = None
        else:
            print(
                f"⚠️ No se encontró el modelo en {MODEL_EXPORT_PATH}. Entrena un modelo primero."
            )

    def predict_image(self, image_bytes: bytes) -> tuple[str, float, float]:
        """Procesos los bytes de la imagen y realiza la inferencia."""
        if self.model is None:
            raise RuntimeError("El modelo no está cargado en memoria.")

        # Abrir imagen, convertir a RGB y redimensionar
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)

        # Preprocesamiento con estadísticas de ResNet50
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Inferencia
        raw_score = float(self.model.predict(img_array, verbose=0)[0][0])
        prediction = "PNEUMONIA" if raw_score >= 0.5 else "NORMAL"
        confidence = raw_score if prediction == "PNEUMONIA" else (1.0 - raw_score)

        return prediction, round(confidence, 4), round(raw_score, 4)


# Instancia global exportada explícitamente
model_service = ModelService()