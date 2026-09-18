import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Coffee Market MLOps API",
    description="API REST para consulta de pronósticos e inferencia del mercado de café",
    version="1.0.0"
)

# Permitir solicitudes desde cualquier origen (Streamlit, Agente, ngrok)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "data/04_models/lgbm_model.joblib"
PREDICTIONS_PATH = "data/06_predictions"

@app.get("/", tags=["Health Check"])
def root():
    """Verifica el estado de salud de la API."""
    return {"status": "ok", "message": "API de MLOps funcionando correctamente"}

@app.get("/health", tags=["Health Check"])
def health_check():
    """Confirma la presencia del modelo cargado en disco."""
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_exists else "unhealthy",
        "model_loaded": model_exists
    }

@app.get("/forecast/summary", tags=["Forecasting"])
def get_forecast_summary():
    """
    Retorna el resumen consolidado de las proyecciones generadas
    por el pipeline de MLOps.
    """
    try:
        if not os.path.exists(PREDICTIONS_PATH):
            raise HTTPException(status_code=404, detail="La carpeta de predicciones no existe.")

        files = [f for f in os.listdir(PREDICTIONS_PATH) if f.endswith(".parquet") or f.endswith(".csv")]
        if not files:
            raise HTTPException(status_code=404, detail="No se encontraron archivos de pronóstico.")
        
        file_path = os.path.join(PREDICTIONS_PATH, files[0])
        df_pred = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)

        # Convertir NaNs a None para garantizar un JSON válido
        df_cleaned = df_pred.where(pd.notnull(df_pred), None)
        records = df_cleaned.head(100).to_dict(orient="records")

        return {
            "total_records": len(df_pred),
            "sample_data": records
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer pronósticos: {str(e)}")