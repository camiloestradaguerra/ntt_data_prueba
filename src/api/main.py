from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import AvailableCountriesResponse, ForecastRequest, MultiForecastResponse
from src.api.services import ForecastingService

service = ForecastingService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Carga artefactos de MLOps al arrancar el servicio."""
    try:
        service.load_artifacts()
    except Exception as e:
        print(f"❌ Error al cargar artefactos de MLOps: {e}")
        sys.exit(1)
    yield


app = FastAPI(
    title="API de Pronóstico de Demanda de Café",
    version="1.0.0",
    description="Servicio MLOps para inferencia en tiempo real mediante modelos autoregresivos.",
    lifespan=lifespan,
)

# Configuración de CORS para permitir peticiones desde Streamlit / Agente GenAI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "Coffee Forecast API"}


@app.get("/countries", response_model=AvailableCountriesResponse, status_code=status.HTTP_200_OK)
def list_countries() -> AvailableCountriesResponse:
    """Lista todos los países que el modelo de producción puede proyectar.

    Fuente única de verdad de nombres de país (tal como aparecen en
    `data_long.parquet`, p.ej. "Viet Nam" y no "Vietnam") -- la UI y el
    agente LLM la consultan para poblar sus selectores y evitar pedir
    proyecciones de países mal escritos o fuera del dataset.
    """
    try:
        return AvailableCountriesResponse(countries=service.get_available_countries())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo obtener la lista de países: {str(e)}",
        )


@app.post("/predict", response_model=MultiForecastResponse, status_code=status.HTTP_200_OK)
def predict(payload: ForecastRequest) -> MultiForecastResponse:
    """Proyecta demanda futura para uno o más países.

    Un país inválido (mal escrito o fuera del dataset) NUNCA hace fallar la
    respuesta completa con 404: `ForecastingService.predict_countries` ya
    captura ese caso por país y lo devuelve dentro de `results` con su
    propio campo `error` (ver `CountryForecastResponse` en
    `src/api/schemas.py`), para que un solo nombre incorrecto en un lote no
    le impida a la UI / al agente LLM procesar el resto de países válidos.
    Esta ruta solo devuelve un error HTTP ante un fallo real e inesperado
    del servicio (p.ej. artefactos no cargados).
    """
    try:
        results = service.predict_countries(
            countries=payload.countries,
            end_year=payload.end_year,
        )
        return MultiForecastResponse(**results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el proceso de inferencia: {str(e)}",
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=port, reload=True)