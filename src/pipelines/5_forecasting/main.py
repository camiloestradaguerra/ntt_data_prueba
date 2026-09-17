from __future__ import annotations

import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("5_forecasting")


class DemandForecastModel:
    """Clase wrapper para deserializar el modelo entrenado en el Paso 3."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        categorical_features: list[str] | None = None,
        diff_target_col: str = "Consumption_diff",
        random_state: int = 42,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        self.feature_columns = feature_columns or []
        self.categorical_features = categorical_features or []
        self.diff_target_col = diff_target_col
        self.random_state = random_state
        self.hyperparams = hyperparams or {}
        self.model: Any = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo no contiene un regresor entrenado válidamente.")
        return self.model.predict(self._prepare_features(df))


@dataclass(frozen=True)
class ForecastConfig:
    """Configuración para la inferencia recursiva fuera de muestra."""

    start_forecast_year: int = 2021
    end_forecast_year: int = 2025
    rolling_window: int = 3
    lags: tuple[int, ...] = (1, 2, 3)


class DataLoader:
    """Carga de artefactos y datasets necesarios para inferencia."""

    @staticmethod
    def load_dataset(data_path: Path) -> pd.DataFrame:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset base no encontrado en: {data_path.resolve()}")
        df = pd.read_parquet(data_path)
        logger.info("Dataset base cargado: %s filas", len(df))
        return df

    @staticmethod
    def load_model(model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path.resolve()}")
        model = joblib.load(model_path)
        logger.info("Modelo de pronóstico cargado desde: %s", model_path.resolve())
        return model

    @staticmethod
    def load_feature_config(artifacts_dir: Path) -> dict[str, Any]:
        config_path = artifacts_dir / "feature_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuración de features no encontrada en: {config_path.resolve()}")
        with open(config_path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        return config


class RecursiveForecaster:
    """Ejecuta proyecciones multinivel realimentando iterativamente predicciones pasadas."""

    def __init__(self, model: Any, feature_config: dict[str, Any], config: ForecastConfig, df: pd.DataFrame) -> None:
        self.model = model
        self.feature_columns: list[str] = feature_config["feature_columns"]
        self.categorical_features: list[str] = feature_config["categorical_features"]
        self.config = config

        # Detección de nombres de columnas
        self.group_col = feature_config.get("group_col") or ("Country" if "Country" in df.columns else "Country_Code")
        self.target_col = feature_config.get("target_col") or ("Consumption" if "Consumption" in df.columns else "Demand")
        self.year_col = feature_config.get("year_col", "Year")

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sorted = df.sort_values([self.group_col, self.year_col]).copy()

        historical_df = df_sorted.loc[df_sorted[self.year_col] < self.config.start_forecast_year].copy()
        max_lookback = max(self.config.lags + (self.config.rolling_window,))

        working_series: dict[str, list[float]] = {
            group: group_data.tail(max_lookback)[self.target_col].tolist()
            for group, group_data in historical_df.groupby(self.group_col)
        }

        # Selección de columnas estáticas excluyendo la llave de agrupación del filtro de subconjunto
        possible_static = ["Coffee_type", f"{self.group_col}_encoded", "Coffee_type_encoded"]
        lookup_cols = [c for c in possible_static if c in df_sorted.columns]
        
        static_lookup = (
            df_sorted.drop_duplicates(subset=[self.group_col])
            .set_index(self.group_col)[lookup_cols]
        )

        forecast_rows: list[dict[str, Any]] = []
        years_to_forecast = list(range(self.config.start_forecast_year, self.config.end_forecast_year + 1))

        logger.info(
            "Iniciando inferencia recursiva (%s-%s) agrupada por '%s'...",
            years_to_forecast[0],
            years_to_forecast[-1],
            self.group_col,
        )

        for forecast_year in years_to_forecast:
            step_rows = []
            for group, series in working_series.items():
                row_dict: dict[str, Any] = {
                    self.group_col: group,
                    self.year_col: forecast_year,
                }

                for col in lookup_cols:
                    row_dict[col] = static_lookup.loc[group, col]

                for lag in self.config.lags:
                    row_dict[f"lag_{lag}"] = series[-lag]

                recent_window = series[-self.config.rolling_window:]
                row_dict[f"rolling_mean_{self.config.rolling_window}"] = float(np.mean(recent_window))
                row_dict[f"rolling_std_{self.config.rolling_window}"] = (
                    float(np.std(recent_window, ddof=1)) if len(recent_window) > 1 else 0.0
                )

                step_rows.append(row_dict)

            step_df = pd.DataFrame(step_rows)

            if isinstance(self.model, DemandForecastModel):
                pred_diff = self.model.predict(step_df)
            else:
                X_step = step_df[self.feature_columns].copy()
                for col in self.categorical_features:
                    if col in X_step.columns:
                        X_step[col] = X_step[col].astype("category")
                pred_diff = self.model.predict(X_step)

            step_df["predicted_diff"] = pred_diff
            step_df["forecast_demand"] = step_df["lag_1"] + pred_diff

            for group, pred_val in zip(step_df[self.group_col], step_df["forecast_demand"]):
                working_series[group].append(float(pred_val))

            forecast_rows.extend(step_df.to_dict(orient="records"))

        forecast_results = pd.DataFrame(forecast_rows)
        logger.info("Inferencia recursiva completada: %s registros proyectados", len(forecast_results))
        return forecast_results


class ArtifactExporter:
    """Genera y exporta los artefactos finales de inferencia."""

    @staticmethod
    def export(
        forecast_df: pd.DataFrame,
        base_df: pd.DataFrame,
        model_path: Path,
        output_dir: Path,
        config: ForecastConfig,
        group_col: str,
        target_col: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Proyección completa (forecast_2021_2025.csv)
        main_predictions_path = output_dir / "forecast_2021_2025.csv"
        cols_to_export = [group_col, "Year", "forecast_demand", "predicted_diff", "lag_1"]
        available_cols = [c for c in cols_to_export if c in forecast_df.columns]

        forecast_df[available_cols].to_csv(main_predictions_path, index=False)
        logger.info("1/3 Artefacto principal guardado en: %s", main_predictions_path.resolve())

        # 2. Metadatos de auditoría (forecast_metadata.json)
        metadata_path = output_dir / "forecast_metadata.json"
        metadata = {
            "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_source": str(model_path.resolve()),
            "forecast_start_year": config.start_forecast_year,
            "forecast_end_year": config.end_forecast_year,
            "horizon_years": config.end_forecast_year - config.start_forecast_year + 1,
            "total_projected_records": len(forecast_df),
            "unique_countries_count": int(forecast_df[group_col].nunique()),
            "status": "SUCCESS",
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("2/3 Artefacto de metadatos guardado en: %s", metadata_path.resolve())

        # 3. Resumen acumulado por país con CAGR (forecast_summary_by_country.csv)
        summary_path = output_dir / "forecast_summary_by_country.csv"

        target_name = target_col if target_col in base_df.columns else "Demand"
        last_hist_year = int(base_df.loc[base_df["Year"] < config.start_forecast_year, "Year"].max())

        base_last_hist = (
            base_df.loc[base_df["Year"] == last_hist_year]
            .groupby(group_col)[target_name]
            .first()
            .to_dict()
        )

        summary_rows = []
        for group, group_df in forecast_df.groupby(group_col):
            initial_val = base_last_hist.get(group, group_df["lag_1"].iloc[0])
            final_val = group_df.loc[group_df["Year"] == config.end_forecast_year, "forecast_demand"].values[0]
            horizon = config.end_forecast_year - last_hist_year

            if initial_val > 0 and final_val > 0 and horizon > 0:
                cagr = float(((final_val / initial_val) ** (1.0 / horizon) - 1.0) * 100.0)
            else:
                cagr = 0.0

            summary_rows.append(
                {
                    group_col: group,
                    f"baseline_{last_hist_year}_demand": initial_val,
                    "forecast_2025_demand": final_val,
                    "total_accumulated_forecast_2021_2025": float(group_df["forecast_demand"].sum()),
                    "mean_annual_forecast": float(group_df["forecast_demand"].mean()),
                    "cagr_pct": round(cagr, 2),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by="total_accumulated_forecast_2021_2025", ascending=False
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info("3/3 Artefacto de resumen guardado en: %s", summary_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 5 MLOps: Pronóstico recursivo multinivel (2021-2025)."
    )
    parser.add_argument("--input_path", type=Path, required=True, help="Ruta al dataset de características.")
    parser.add_argument("--model_path", type=Path, required=True, help="Ruta al modelo serializado (.joblib).")
    parser.add_argument("--input_artifacts_dir", type=Path, required=True, help="Directorio con feature_config.json.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directorio de destino para artefactos finales.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ForecastConfig()

    df = DataLoader.load_dataset(args.input_path)
    model = DataLoader.load_model(args.model_path)
    feature_config = DataLoader.load_feature_config(args.input_artifacts_dir)

    forecaster = RecursiveForecaster(model=model, feature_config=feature_config, config=config, df=df)
    forecast_df = forecaster.forecast(df)

    ArtifactExporter.export(
        forecast_df=forecast_df,
        base_df=df,
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config,
        group_col=forecaster.group_col,
        target_col=forecaster.target_col,
    )


if __name__ == "__main__":
    main()