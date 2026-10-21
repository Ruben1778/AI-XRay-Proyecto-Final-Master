from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    model_version: str = Field(..., example="1.0.0")


class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="Clase predicha",
        example="PNEUMONIA",
    )
    confidence: float = Field(
        ...,
        description="Nivel de confianza entre 0 y 1",
        example=0.9421,
    )
    raw_score: float = Field(
        ...,
        description="Probabilidad cruda de la salida sigmoide",
        example=0.9421,
    )
    model_version: str = Field(..., example="1.0.0")
    disclaimer: str = Field(
        ...,
        example="HERRAMIENTA DE APOYO DE INVESTIGACIÓN. NO REEMPLAZA EL DIAGNÓSTICO MÉDICO.",
    )