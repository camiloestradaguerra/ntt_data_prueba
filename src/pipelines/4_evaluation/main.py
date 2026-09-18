from __future__ import annotations

import os
import warnings

# Evitar alertas de entorno de joblib en Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("4_evaluation")

# Rol que DEBE tener el modelo cargado por este paso. Ver DemandForecastModel
# en src/pipelines/3_training/main.py: "eval" = entrenado solo con
# <= EvalConfig.val_start_year - 1 (holdout honesto). El modelo "prod" (todo
# el historial) NUNCA debe pasar por este script: sus métricas sobre
# 2016-2019 serían in-sample, no una evaluación real.
EXPECTED_MODEL_ROLE = "eval"


class DemandForecastModel:
    """Clase wrapper requerida para deserializar modelos creados en el Paso 3.

    Debe reflejar exactamente los atributos de la clase homónima en
    src/pipelines/3_training/main.py (incluye los metadatos de auditoría
    `model_role` / `train_year_range`), o joblib no podrá reconstruir el
    objeto serializado.
    """

    def __init__(
        self,
        feature_columns: list[str],
        categorical_features: list[str],
        diff_target_col: str,
        random_state: int = 42,
        hyperparams: dict[str, Any] | None = None,
        model_role: str = "unspecified",
    ) -> None:
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.diff_target_col = diff_target_col
        self.random_state = random_state
        self.hyperparams = hyperparams or {}
        self.model: Any = None
        self.model_role = model_role
        self.train_year_range: tuple[int, int] | None = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for col in self.categorical_features:
            X[col] = X[col].astype("category")
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo no contiene un regresor entrenado válidamente.")
        return self.model.predict(self._prepare_features(df))


@dataclass(frozen=True)
class EvalConfig:
    """Configuración de la ventana de evaluación offline."""

    year_col: str = "Year"
    val_start_year: int = 2016
    val_end_year: int = 2019


class DataLoader:
    """Carga de artefactos y datasets para auditoría."""

    @staticmethod
    def load_dataset(data_path: Path) -> pd.DataFrame:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado en: {data_path.resolve()}")
        df = pd.read_parquet(data_path)
        logger.info("Dataset de evaluación cargado: %s filas", len(df))
        return df

    @staticmethod
    def load_model(model_path: Path) -> Any:
        """Carga el modelo serializado y verifica que sea el modelo EVAL.

        Guardrail anti-fuga de datos: este paso solo tiene sentido si el
        modelo nunca vio la ventana de validación durante su entrenamiento.
        Si alguien apunta por error `--eval_model_path` al artefacto de
        producción (`lgbm_prod.joblib`, entrenado con 1990-2019), este
        método corta la ejecución en vez de reportar métricas infladas
        silenciosamente.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo serializado no encontrado en: {model_path.resolve()}")
        model = joblib.load(model_path)
        logger.info("Modelo cargado exitosamente desde: %s", model_path.resolve())

        role = getattr(model, "model_role", None)
        train_year_range = getattr(model, "train_year_range", None)
        if role is None:
            logger.warning(
                "El modelo cargado no tiene metadato 'model_role' (artefacto generado con una versión "
                "anterior del pipeline). No se puede verificar automáticamente que sea el modelo de "
                "evaluación -- confirma manualmente que %s fue entrenado solo con años <= %s.",
                model_path.name,
                EvalConfig.val_start_year - 1,
            )
        elif role != EXPECTED_MODEL_ROLE:
            raise ValueError(
                f"Modelo incorrecto para este paso: '{model_path.name}' tiene model_role='{role}' "
                f"(rango de años entrenados: {train_year_range}), pero 4_evaluation requiere "
                f"model_role='{EXPECTED_MODEL_ROLE}'. Probablemente apuntaste por error al modelo de "
                "producción (lgbm_prod.joblib), que fue entrenado incluyendo la ventana de validación "
                "y por lo tanto produciría métricas infladas (in-sample). Usa lgbm_eval.joblib."
            )
        else:
            logger.info(
                "Verificación OK -> modelo con model_role='%s', entrenado con años %s (no incluye la "
                "ventana de validación %s-%s).",
                role,
                train_year_range,
                EvalConfig.val_start_year,
                EvalConfig.val_end_year,
            )
        return model

    @staticmethod
    def load_feature_config(artifacts_dir: Path) -> dict[str, Any]:
        config_path = artifacts_dir / "feature_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuración de features no encontrada en: {config_path.resolve()}")
        with open(config_path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        return config


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Métrica de negocio WAPE = sum(|y - pred|) / sum(|y|)."""
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


class ModelEvaluator:
    """Calcula las métricas formales de desempeño y genera predicciones."""

    def __init__(self, model: Any, feature_config: dict[str, Any], config: EvalConfig) -> None:
        self.model = model
        self.feature_columns: list[str] = feature_config["feature_columns"]
        self.categorical_features: list[str] = feature_config["categorical_features"]
        self.target_col: str = feature_config["target_col"]
        self.diff_target_col: str = feature_config["diff_target_col"]
        self.config = config

    def evaluate(self, df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
        # Filtrar la ventana de prueba (2016 - 2019)
        eval_df = df.loc[
            df[self.config.year_col].between(self.config.val_start_year, self.config.val_end_year)
        ].copy()

        if eval_df.empty:
            raise ValueError(f"No hay registros en el rango {self.config.val_start_year}-{self.config.val_end_year}")

        # Inferencia adaptativa (Soporta objeto wrapper o regresor directo)
        if isinstance(self.model, DemandForecastModel):
            diff_preds = self.model.predict(eval_df)
        else:
            X_eval = eval_df[self.feature_columns].copy()
            for col in self.categorical_features:
                X_eval[col] = X_eval[col].astype("category")
            diff_preds = self.model.predict(X_eval)

        # Reconstrucción del nivel real de demanda
        level_preds = eval_df["lag_1"].to_numpy() + diff_preds

        eval_df["predicted_diff"] = diff_preds
        eval_df["predicted_demand"] = level_preds

        # Métricas de error
        y_true = eval_df[self.target_col].to_numpy()
        mae = float(mean_absolute_error(y_true, level_preds))
        rmse = float(np.sqrt(mean_squared_error(y_true, level_preds)))
        wape = weighted_absolute_percentage_error(y_true, level_preds)

        metrics = {"MAE": mae, "RMSE": rmse, "WAPE": wape}
        logger.info(
            "Métricas Oficiales de Evaluación -- modelo EVAL, holdout honesto (%s-%s) -> "
            "MAE: %.2f | RMSE: %.2f | WAPE: %.4f",
            self.config.val_start_year,
            self.config.val_end_year,
            mae,
            rmse,
            wape,
        )
        return metrics, eval_df


class EvaluationArtifactManager:
    """Persistencia de los reportes y datasets de evaluación."""

    @staticmethod
    def save(
        metrics: dict[str, float],
        eval_df: pd.DataFrame,
        output_dir: Path,
        model: Any,
        target_col: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar métricas en JSON, etiquetadas con la procedencia del modelo
        report = {
            "model_role": getattr(model, "model_role", "unknown"),
            "train_year_range": list(getattr(model, "train_year_range", None) or []) or None,
            "note": "Métricas calculadas con el modelo EVAL (holdout honesto, no vio 2016-2019 en entrenamiento).",
            "metrics": metrics,
        }
        with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        # 2. Guardar predicciones vs. valores reales, por país -- este parquet es la
        #    ÚNICA fuente de verdad para la serie "predicción de evaluación 2016-2019"
        #    que consume app.py (Streamlit); evita recalcular la predicción con otra
        #    copia del modelo en otro script y que ambos se desincronicen.
        #    NOTA: las columnas reales del dataset de features son "Country" y el
        #    nombre configurado de target_col (p.ej. "Consumption"), no
        #    "Country_Code"/"Demand" -- se guardan tal cual existen para que el
        #    consumidor (app.py) pueda filtrar por país sin adivinar nombres.
        cols_to_save = ["Country", "Coffee_type", "Year", target_col, "predicted_demand", "lag_1", "predicted_diff"]
        available_cols = [c for c in cols_to_save if c in eval_df.columns]
        missing_cols = [c for c in cols_to_save if c not in eval_df.columns]
        if missing_cols:
            logger.warning(
                "Columnas esperadas no encontradas en eval_df y omitidas del parquet de predicciones: %s",
                missing_cols,
            )
        eval_df[available_cols].to_parquet(output_dir / "evaluation_predictions.parquet", index=False)

        logger.info("Artefactos de evaluación guardados correctamente en: %s", output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 4 MLOps: Evaluación offline del modelo (holdout honesto).")
    parser.add_argument("--input_path", type=Path, required=True, help="Ruta al dataset de características.")
    parser.add_argument(
        "--eval_model_path",
        type=Path,
        required=True,
        help="Ruta al modelo de EVALUACIÓN serializado (lgbm_eval.joblib), entrenado solo con <=2015. "
        "NUNCA apuntar aquí al modelo de producción (lgbm_prod.joblib): este paso lo rechazará.",
    )
    parser.add_argument("--input_artifacts_dir", type=Path, required=True, help="Directorio con feature_config.json.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directorio para reportes de evaluación.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig()

    df = DataLoader.load_dataset(args.input_path)
    model = DataLoader.load_model(args.eval_model_path)
    feature_config = DataLoader.load_feature_config(args.input_artifacts_dir)

    evaluator = ModelEvaluator(model=model, feature_config=feature_config, config=config)
    metrics, eval_df = evaluator.evaluate(df)

    EvaluationArtifactManager.save(
        metrics=metrics,
        eval_df=eval_df,
        output_dir=args.output_dir,
        model=model,
        target_col=feature_config["target_col"],
    )


if __name__ == "__main__":
    main()
