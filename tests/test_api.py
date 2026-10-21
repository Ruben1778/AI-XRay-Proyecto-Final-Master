import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verifica que el endpoint /health responda 200 y el modelo esté cargado."""
    response = client.get("/health")
    assert response.status_code == 200, f"Respuesta inesperada: {response.text}"
    data = response.json()
    assert data["status"] == "ok", f"El estado fue 'error'. Detalle: {data}"
    assert data["model_loaded"] is True

def test_predict_rejects_non_image():
    """Verifica que /predict rechace archivos que no son imágenes."""
    files = {"file": ("test.txt", b"contenido de texto invalido", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_predict_valid_image():
    """Verifica una predicción exitosa con una imagen en memoria."""
    from PIL import Image
    import io

    # Crear una imagen RGB válida de 224x224
    img = Image.new("RGB", (224, 224), color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    files = {"file": ("test_xray.jpeg", img_byte_arr, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200, f"Error en /predict: {response.text}"
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert "confidence" in data or "probability" in data
    assert "disclaimer" in data