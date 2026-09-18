from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import ForecastRequest, MultiForecastResponse
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


@app.post("/predict", response_model=MultiForecastResponse, status_code=status.HTTP_200_OK)
def predict(payload: ForecastRequest) -> MultiForecastResponse:
    try:
        results = service.predict_countries(
            countries=payload.countries,
            end_year=payload.end_year,
        )
        return MultiForecastResponse(**results)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el proceso de inferencia: {str(e)}",
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=port, reload=True)