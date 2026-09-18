from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# 1. Importación dinámica del módulo de Feature Engineering (Paso 2) e inyección en __main__
_feature_module = importlib.import_module("src.pipelines.2_feature_engineering.main")
CategoricalEncoder = _feature_module.CategoricalEncoder
TimeSeriesFeatureEngineer = _feature_module.TimeSeriesFeatureEngineer

if hasattr(_feature_module, "CategoricalEncoder"):
    setattr(sys.modules["__main__"], "CategoricalEncoder", CategoricalEncoder)
if hasattr(_feature_module, "TimeSeriesFeatureEngineer"):
    setattr(sys.modules["__main__"], "TimeSeriesFeatureEngineer", TimeSeriesFeatureEngineer)

# 2. Importación dinámica del módulo de Entrenamiento (Paso 3) e inyección en __main__
_training_module = importlib.import_module("src.pipelines.3_training.main")
if hasattr(_training_module, "DemandForecastModel"):
    setattr(sys.modules["__main__"], "DemandForecastModel", _training_module.DemandForecastModel)


class ForecastingService:
    """Servicio de inferencia recursiva para predicción de series de tiempo."""

    def __init__(
        self,
        data_path: Path = Path("data/02_processed/data_long.parquet"),
        model_path: Path = Path("data/04_models/lgbm_model.joblib"),
        artifacts_dir: Path = Path("data/03_features/artifacts"),
    ) -> None:
        self.data_path = data_path
        self.model_path = model_path
        self.artifacts_dir = artifacts_dir

        self.model: Any = None
        self.config: dict[str, Any] = {}
        self.encoder: CategoricalEncoder | None = None
        self.historical_df: pd.DataFrame | None = None

    def load_artifacts(self) -> None:
        """Carga en memoria el modelo, la configuración y el dataset histórico."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")

        # Carga de artefactos resolviendo referencias de clases en __main__
        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.artifacts_dir / "categorical_encoder.joblib")

        with open(self.artifacts_dir / "feature_config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.historical_df = pd.read_parquet(self.data_path)

    def predict_country(self, country: str, end_year: int) -> dict[str, Any]:
        """Ejecuta el bucle de pronóstico recursivo en memoria para un país (2020 -> end_year)."""
        if self.historical_df is None or self.model is None or self.encoder is None:
            raise RuntimeError("El servicio no ha sido inicializado. Ejecute load_artifacts() primero.")

        df_country = self.historical_df[self.historical_df["Country"] == country].copy()
        if df_country.empty:
            raise ValueError(f"El país '{country}' no existe en el registro histórico.")

        coffee_type = df_country["Coffee_type"].iloc[0]
        max_hist_year = int(df_country["Year"].max())

        feature_engineer = TimeSeriesFeatureEngineer()
        current_df = df_country.copy()

        for target_year in range(max_hist_year + 1, end_year + 1):
            new_row = pd.DataFrame([{
                "Country": country,
                "Coffee_type": coffee_type,
                "Year": target_year,
                "Consumption": np.nan,
            }])
            current_df = pd.concat([current_df, new_row], ignore_index=True)

            transformed_df = feature_engineer.transform(current_df)
            encoded_df = self.encoder.transform(transformed_df)

            target_row = encoded_df[encoded_df["Year"] == target_year]
            X_pred = target_row[self.config["feature_columns"]]

            delta_y_hat = self.model.predict(X_pred)[0]

            prev_y = current_df.loc[current_df["Year"] == target_year - 1, "Consumption"].values[0]
            pred_y = max(0.0, float(prev_y + delta_y_hat))

            current_df.loc[current_df["Year"] == target_year, "Consumption"] = pred_y

        points = []
        for _, row in current_df.iterrows():
            yr = int(row["Year"])
            val = float(row["Consumption"])
            points.append({
                "year": yr,
                "consumption": val,
                "is_forecast": yr > max_hist_year,
            })

        return {
            "country": country,
            "coffee_type": coffee_type,
            "historical_end_year": max_hist_year,
            "forecast_end_year": end_year,
            "data": points,
        }

    def predict_countries(self, countries: list[str], end_year: int) -> dict[str, Any]:
        """Ejecuta la inferencia recursiva para una lista de múltiples países."""
        results = []
        for country in countries:
            country_res = self.predict_country(country=country, end_year=end_year)
            results.append(country_res)
        return {"results": results}