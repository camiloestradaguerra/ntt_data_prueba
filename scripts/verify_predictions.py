"""Script independiente de verificacion de predicciones LightGBM por pais.

No forma parte del pipeline oficial (Paso 4 - src/pipelines/4_evaluation);
es una utilidad de QA para auditar visualmente el comportamiento del modelo
para un pais especifico, separando entrenamiento (1990-2015) y validacion
(2016-2019) con valores reales vs. predichos.

Uso:
    python verify_predictions.py --country Colombia
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("verify_predictions")


class DemandForecastModel:
    """Replica exacta del wrapper usado al serializar el modelo en el Paso 3.

    joblib necesita esta clase definida (con el mismo nombre y atributos)
    para poder deserializar `lgbm_model.joblib`.
    """

    def __init__(
        self,
        feature_columns: list[str],
        categorical_features: list[str],
        diff_target_col: str,
        random_state: int = 42,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.diff_target_col = diff_target_col
        self.random_state = random_state
        self.hyperparams = hyperparams or {}
        self.model: Any = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for col in self.categorical_features:
            X[col] = X[col].astype("category")
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo no contiene un regresor entrenado validamente.")
        return self.model.predict(self._prepare_features(df))


@dataclass(frozen=True)
class VerificationConfig:
    """Parametros de la ventana de verificacion offline para un pais."""

    country: str
    train_start: int = 1990
    train_end: int = 2015
    val_start: int = 2016
    val_end: int = 2019
    year_col: str = "Year"
    country_col: str = "Country"
    target_col: str = "Consumption"


def load_artifacts(
    features_path: Path, model_path: Path, feature_config_path: Path
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    if not features_path.exists():
        raise FileNotFoundError(f"Dataset de features no encontrado: {features_path.resolve()}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path.resolve()}")
    if not feature_config_path.exists():
        raise FileNotFoundError(f"feature_config.json no encontrado: {feature_config_path.resolve()}")

    df = pd.read_parquet(features_path)
    model = joblib.load(model_path)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        feature_config: dict[str, Any] = json.load(f)

    logger.info("Dataset de features cargado: %s filas, %s columnas", *df.shape)
    logger.info("Modelo cargado: %s", type(model).__name__)
    return df, model, feature_config


def split_country_data(df: pd.DataFrame, cfg: VerificationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa la serie de un pais en ventana de entrenamiento y validacion."""
    country_df = df[df[cfg.country_col] == cfg.country].sort_values(cfg.year_col).reset_index(drop=True)

    if country_df.empty:
        available = sorted(df[cfg.country_col].unique())
        raise ValueError(f"Pais '{cfg.country}' no encontrado. Paises disponibles: {available}")

    train_df = country_df[country_df[cfg.year_col].between(cfg.train_start, cfg.train_end)].copy()
    val_df = country_df[country_df[cfg.year_col].between(cfg.val_start, cfg.val_end)].copy()

    if train_df.empty:
        raise ValueError(f"No hay datos de entrenamiento ({cfg.train_start}-{cfg.train_end}) para {cfg.country}.")
    if val_df.empty:
        raise ValueError(f"No hay datos de validacion ({cfg.val_start}-{cfg.val_end}) para {cfg.country}.")

    logger.info(
        "Pais '%s' -> entrenamiento: %s filas (%s-%s) | validacion: %s filas (%s-%s)",
        cfg.country,
        len(train_df),
        cfg.train_start,
        cfg.train_end,
        len(val_df),
        cfg.val_start,
        cfg.val_end,
    )
    return train_df, val_df


def predict_validation(model: Any, feature_config: dict[str, Any], val_df: pd.DataFrame) -> np.ndarray:
    """Genera las predicciones de nivel (Consumption) para la ventana de validacion.

    El modelo predice el delta interanual (Consumption_diff); el nivel se
    reconstruye sumando el ultimo valor observado (lag_1), igual que en el
    paso oficial de evaluacion (src/pipelines/4_evaluation).
    """
    feature_columns: list[str] = feature_config["feature_columns"]
    categorical_features: list[str] = feature_config["categorical_features"]

    if isinstance(model, DemandForecastModel):
        diff_preds = model.predict(val_df)
    else:
        X_val = val_df[feature_columns].copy()
        for col in categorical_features:
            X_val[col] = X_val[col].astype("category")
        diff_preds = model.predict(X_val)

    level_preds = val_df["lag_1"].to_numpy() + diff_preds
    return level_preds


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "WAPE": round(wape, 4)}


def plot_verification(
    cfg: VerificationConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_preds: np.ndarray,
    output_path: Path,
) -> None:
    """Grafico de series de tiempo con 2 subplots (serie completa y zoom a validacion).

    Colores: azul = entrenamiento, verde = valores reales, negro = predicciones.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax = axes[0]
    ax.plot(
        train_df[cfg.year_col], train_df[cfg.target_col],
        color="blue", marker="o", markersize=4, linewidth=1.5,
        label=f"Entrenamiento ({cfg.train_start}-{cfg.train_end})",
    )
    ax.plot(
        val_df[cfg.year_col], val_df[cfg.target_col],
        color="green", marker="o", markersize=5, linewidth=1.5,
        label=f"Real ({cfg.val_start}-{cfg.val_end})",
    )
    ax.plot(
        val_df[cfg.year_col], val_preds,
        color="black", marker="x", markersize=6, linewidth=1.5, linestyle="--",
        label=f"Prediccion modelo ({cfg.val_start}-{cfg.val_end})",
    )
    ax.axvline(cfg.train_end + 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Consumo de cafe -- {cfg.country}: serie completa {cfg.train_start}-{cfg.val_end}")
    ax.set_ylabel("Consumo")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(
        val_df[cfg.year_col], val_df[cfg.target_col],
        color="green", marker="o", markersize=6, linewidth=2, label="Real",
    )
    ax2.plot(
        val_df[cfg.year_col], val_preds,
        color="black", marker="x", markersize=8, linewidth=2, linestyle="--", label="Prediccion",
    )
    ax2.set_title(f"Zoom -- ventana de validacion ({cfg.val_start}-{cfg.val_end})")
    ax2.set_xlabel("Ano")
    ax2.set_ylabel("Consumo")
    ax2.set_xticks(val_df[cfg.year_col].tolist())
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle("Verificacion independiente del modelo LightGBM", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Grafico guardado en: %s", output_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script de verificacion independiente: valida predicciones del modelo LightGBM para un pais."
    )
    parser.add_argument("--country", type=str, default="Colombia", help="Nombre del pais a verificar.")
    parser.add_argument(
        "--features_path", type=Path,
        default=Path("data/03_features/data_features.parquet"),
        help="Ruta al dataset de features (Paso 2/3).",
    )
    parser.add_argument(
        "--model_path", type=Path,
        default=Path("data/04_models/lgbm_model.joblib"),
        help="Ruta al modelo LightGBM serializado.",
    )
    parser.add_argument(
        "--feature_config_path", type=Path,
        default=Path("data/03_features/artifacts/feature_config.json"),
        help="Ruta al feature_config.json generado en el Paso 2.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("reports/verification"),
        help="Directorio de salida para metricas y grafico.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VerificationConfig(country=args.country)

    df, model, feature_config = load_artifacts(args.features_path, args.model_path, args.feature_config_path)
    train_df, val_df = split_country_data(df, cfg)
    val_preds = predict_validation(model, feature_config, val_df)

    y_true = val_df[cfg.target_col].to_numpy()
    metrics = compute_metrics(y_true, val_preds)

    logger.info(
        "Metricas de verificacion (%s, %s-%s) -> MAE: %.2f | RMSE: %.2f | WAPE: %.4f",
        cfg.country, cfg.val_start, cfg.val_end, metrics["MAE"], metrics["RMSE"], metrics["WAPE"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = cfg.country.replace(" ", "_").replace("/", "-")

    metrics_path = args.output_dir / f"verification_metrics_{safe_name}.json"
    report = {
        "country": cfg.country,
        "train_window": [cfg.train_start, cfg.train_end],
        "val_window": [cfg.val_start, cfg.val_end],
        "metrics": metrics,
        "detail": [
            {
                "year": int(year),
                "real": float(real),
                "predicted": round(float(pred), 2),
            }
            for year, real, pred in zip(val_df[cfg.year_col], y_true, val_preds)
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logger.info("Reporte de metricas guardado en: %s", metrics_path.resolve())

    plot_path = args.output_dir / f"verification_plot_{safe_name}.png"
    plot_verification(cfg, train_df, val_df, val_preds, plot_path)


if __name__ == "__main__":
    main()
