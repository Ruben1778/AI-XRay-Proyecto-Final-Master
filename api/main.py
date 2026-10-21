from api.model_service import model_service
from api.schemas import HealthResponse, PredictionResponse
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(
    title="AI-XRay API",
    description="Servicio REST de Clasificación de Radiografías de Tórax Pediátricas (NORMAL / PNEUMONIA)",
    version="1.0.0",
)

MEDICAL_DISCLAIMER = (
    "SISTEMA DE IA ACADÉMICO PARA APOYO EN INVESTIGACIÓN. "
    "NO CONSTITUYE UN DIAGNÓSTICO MÉDICO NI SUSTITUYE LA EVALUACIÓN DE UN PROFESIONAL DE LA SALUD."
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verifica el estado del servicio y confirma si el modelo está cargado en memoria."""
    is_loaded = model_service.model is not None
    return HealthResponse(
        status="ok" if is_loaded else "error",
        model_loaded=is_loaded,
        model_version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Recibe una imagen de radiografía (JPEG/PNG) y devuelve la predicción con su nivel de confianza."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Debe subir una imagen JPEG o PNG.",
        )

    try:
        contents = await file.read()
        prediction, confidence, raw_score = model_service.predict_image(
            contents
        )

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            raw_score=raw_score,
            model_version="1.0.0",
            disclaimer=MEDICAL_DISCLAIMER,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error procesando la imagen: {str(e)}"
        )