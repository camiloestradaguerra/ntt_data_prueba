from __future__ import annotations

import os
import warnings

# Silenciar alerta de joblib/loky en Windows cuando wmic.exe no está presente
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("3_training")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class TrainConfig:
    """Configuración inmutable de entrenamiento y validación temporal."""

    year_col: str = "Year"
    train_end_year: int = 2015
    val_start_year: int = 2016
    val_end_year: int = 2019  # Ajustado al último año real del dataset
    random_state: int = 42
    cv_n_splits: int = 3
    cv_val_years: int = 2
    cv_min_train_years: int = 10


class DataLoader:
    """Carga el dataset de características generado en el Paso 2."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Archivo de entrada no encontrado en: {self.data_path.resolve()}")
        df = pd.read_parquet(self.data_path)
        logger.info("Dataset de características cargado: %s filas, %s columnas", *df.shape)
        return df


class ArtifactLoader:
    """Carga metadatos y contratos del pipeline."""

    @staticmethod
    def load_feature_config(artifacts_dir: Path) -> dict[str, Any]:
        config_path = artifacts_dir / "feature_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuración de features no encontrada en: {config_path.resolve()}")
        with open(config_path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        logger.info("Configuración de características cargada exitosamente desde: %s", config_path.name)
        return config


class TemporalSplitter:
    """Particiona la serie por año (ventana expandida), garantizando cero leakage."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = df.loc[df[self.config.year_col] <= self.config.train_end_year].copy()
        val_df = df.loc[
            df[self.config.year_col].between(self.config.val_start_year, self.config.val_end_year)
        ].copy()
        logger.info(
            "División temporal -> Train: %s filas (<=%s) | Validación: %s filas (%s-%s)",
            len(train_df),
            self.config.train_end_year,
            len(val_df),
            self.config.val_start_year,
            self.config.val_end_year,
        )
        return train_df, val_df


class ExpandingWindowCVSplitter:
    """Genera pliegues cronológicos dentro del periodo de entrenamiento."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        first_year = int(df[self.config.year_col].min())
        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        for split_idx in range(self.config.cv_n_splits):
            val_end = self.config.train_end_year - split_idx * self.config.cv_val_years
            val_start = val_end - self.config.cv_val_years + 1
            train_end = val_start - 1

            if (train_end - first_year + 1) < self.config.cv_min_train_years:
                break

            fold_train = df.loc[df[self.config.year_col] <= train_end]
            fold_val = df.loc[df[self.config.year_col].between(val_start, val_end)]
            folds.append((fold_train, fold_val))

        folds.reverse()
        logger.info("Construidos %s folds temporales para validación cruzada", len(folds))
        return folds


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE = sum(|y - pred|) / sum(|y|). Métrica clave de negocio."""
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE y WAPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = weighted_absolute_percentage_error(np.asarray(y_true), np.asarray(y_pred))
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape}


def reconstruct_level_from_diff(lag_1: np.ndarray, diff_predictions: np.ndarray) -> np.ndarray:
    """Reconstruye nivel: level_t = level_(t-1) + predicted_delta_t."""
    return np.asarray(lag_1, dtype=float) + np.asarray(diff_predictions, dtype=float)


class DemandForecastModel:
    """Modelo global LightGBM en espacio de diferencias (YoY delta).

    IMPORTANTE -- Dos roles, dos artefactos, NUNCA intercambiables:

    - ``model_role="eval"``  -> entrenado SOLO con años <= 2015. Es el único
      modelo con el que tiene sentido calcular métricas de validación sobre
      2016-2019, porque nunca vio esos años durante el entrenamiento. Se
      serializa como ``lgbm_eval.joblib`` y lo consume EXCLUSIVAMENTE el
      Paso 4 (``src/pipelines/4_evaluation``).
    - ``model_role="prod"``  -> re-entrenado con TODO el historial disponible
      (incluye 2016-2019) para maximizar la información al pronosticar el
      futuro (>=2020). Sus predicciones sobre 2016-2019 NO son una medida de
      desempeño honesta (el modelo ya vio esos datos) y por eso nunca debe
      usarse para reportar métricas. Se serializa como ``lgbm_prod.joblib``
      y lo consume EXCLUSIVAMENTE la API (``src/api/services.py``).

    ``model_role`` y ``train_year_range`` son metadatos de auditoría (no
    afectan la inferencia); permiten que cualquier consumidor detecte en
    tiempo de carga si se equivocó de artefacto.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "subsample_freq": 1,
    }

    YEAR_COL = "Year"

    def __init__(
        self,
        feature_columns: list[str],
        categorical_features: list[str],
        diff_target_col: str,
        random_state: int = 42,
        hyperparams: Optional[dict[str, Any]] = None,
        model_role: str = "unspecified",
    ) -> None:
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.diff_target_col = diff_target_col
        self.random_state = random_state
        self.hyperparams = {**self.DEFAULT_PARAMS, **(hyperparams or {})}
        self.model: Optional[lgb.LGBMRegressor] = None

        # --- Metadatos de auditoría (ver docstring de la clase) ---
        self.model_role = model_role
        self.train_year_range: Optional[tuple[int, int]] = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for col in self.categorical_features:
            X[col] = X[col].astype("category")
        return X

    def fit(self, train_df: pd.DataFrame) -> "DemandForecastModel":
        X_train = self._prepare_features(train_df)
        y_train = train_df[self.diff_target_col]

        self.model = lgb.LGBMRegressor(
            **self.hyperparams,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train, categorical_feature=self.categorical_features)

        if self.YEAR_COL in train_df.columns:
            self.train_year_range = (int(train_df[self.YEAR_COL].min()), int(train_df[self.YEAR_COL].max()))

        logger.info(
            "[modelo:%s] Entrenado con %s filas | rango de años visto: %s",
            self.model_role,
            len(train_df),
            self.train_year_range,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        return self.model.predict(self._prepare_features(df))


class HyperparameterTuner:
    """Búsqueda bayesiana de hiperparámetros con Optuna sobre la CV temporal."""

    def __init__(
        self,
        config: TrainConfig,
        feature_columns: list[str],
        categorical_features: list[str],
        target_col: str,
        diff_target_col: str,
        n_trials: int = 40,
    ) -> None:
        self.config = config
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.diff_target_col = diff_target_col
        self.n_trials = n_trials

    def _objective(self, trial: optuna.Trial, folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample_freq": 1,
        }

        fold_wapes = []
        for fold_train, fold_val in folds:
            model = DemandForecastModel(
                feature_columns=self.feature_columns,
                categorical_features=self.categorical_features,
                diff_target_col=self.diff_target_col,
                random_state=self.config.random_state,
                hyperparams=params,
                model_role="cv_trial",
            )
            model.fit(fold_train)
            diff_pred = model.predict(fold_val)
            level_pred = reconstruct_level_from_diff(fold_val["lag_1"].to_numpy(), diff_pred)
            fold_wapes.append(
                weighted_absolute_percentage_error(fold_val[self.target_col].to_numpy(), level_pred)
            )
        return float(np.mean(fold_wapes))

    def tune(self, train_df: pd.DataFrame) -> dict[str, Any]:
        folds = ExpandingWindowCVSplitter(self.config).split(train_df)
        if not folds:
            logger.warning("Historial insuficiente para CV; usando parámetros por defecto.")
            return dict(DemandForecastModel.DEFAULT_PARAMS)

        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lgbm_wape_tuning")
        study.optimize(lambda trial: self._objective(trial, folds), n_trials=self.n_trials)

        logger.info("Ajuste de hiperparámetros completado -> Mejor CV WAPE: %.4f", study.best_value)
        logger.info("Mejores hiperparámetros: %s", study.best_params)
        return study.best_params


class ArtifactManager:
    """Serialización y persistencia de los DOS artefactos de modelo y sus metadatos.

    Genera:
      - ``<output_eval_model_path>``  (p.ej. ``lgbm_eval.joblib``): modelo
        entrenado solo con 1990-2015. Consumido por el Paso 4 (evaluación).
      - ``<output_prod_model_path>``  (p.ej. ``lgbm_prod.joblib``): modelo
        entrenado con 1990-2019. Consumido por la API de producción.
      - ``metrics.json``: métricas de validación HONESTAS (siempre calculadas
        con el modelo eval, nunca con el de producción).
      - ``best_params.json``: hiperparámetros ganadores de Optuna.
      - ``model_registry.json``: manifiesto legible que documenta explícitamente
        el rol, rango de años y ruta de cada artefacto -- para que en
        producción quede claro (sin tener que leer código) cuál modelo es
        cuál y por qué no son intercambiables.
    """

    @staticmethod
    def save_results(
        output_eval_model_path: Path,
        output_prod_model_path: Path,
        artifacts_dir: Path,
        eval_model: DemandForecastModel,
        prod_model: DemandForecastModel,
        metrics: dict[str, float],
        best_params: dict[str, Any],
    ) -> None:
        output_eval_model_path.parent.mkdir(parents=True, exist_ok=True)
        output_prod_model_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Modelo EVAL (holdout honesto -> solo para src/pipelines/4_evaluation)
        joblib.dump(eval_model, output_eval_model_path)
        logger.info(
            "[modelo:eval] Guardado en: %s (años %s) -- USAR SOLO para métricas de validación offline.",
            output_eval_model_path.resolve(),
            eval_model.train_year_range,
        )

        # 2. Modelo PROD (todo el historial -> solo para la API / servicio de inferencia)
        joblib.dump(prod_model, output_prod_model_path)
        logger.info(
            "[modelo:prod] Guardado en: %s (años %s) -- USAR SOLO para inferencia en producción (API). "
            "Sus predicciones sobre 2016-2019 NO son una métrica valida (ya vio esos datos en entrenamiento).",
            output_prod_model_path.resolve(),
            prod_model.train_year_range,
        )

        # 3. Métricas honestas (siempre provienen del modelo eval)
        with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # 4. Hiperparámetros ganadores (compartidos por ambos modelos)
        with open(artifacts_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)

        # 5. Manifiesto de artefactos: documentación explícita para producción/auditoría
        registry = {
            "eval_model": {
                "path": str(output_eval_model_path),
                "role": "eval",
                "train_year_range": list(eval_model.train_year_range) if eval_model.train_year_range else None,
                "consumers": ["src/pipelines/4_evaluation"],
                "notes": (
                    "Entrenado unicamente con anios <= 2015. Es el UNICO modelo valido para "
                    "reportar metricas de validacion sobre 2016-2019. No usar en la API."
                ),
                "validation_metrics_2016_2019": metrics,
            },
            "prod_model": {
                "path": str(output_prod_model_path),
                "role": "prod",
                "train_year_range": list(prod_model.train_year_range) if prod_model.train_year_range else None,
                "consumers": ["src/api/services.py", "src/api/main.py", "app.py (fallback local)"],
                "notes": (
                    "Re-entrenado con el 100% del historial (incluye 2016-2019) para maximizar "
                    "la informacion disponible al pronosticar el futuro (>=2020). NO usar sus "
                    "predicciones sobre 2016-2019 como metrica de desempeno: son in-sample."
                ),
            },
        }
        with open(artifacts_dir / "model_registry.json", "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4, ensure_ascii=False)

        logger.info("Artefactos exportados en: %s (ver model_registry.json)", artifacts_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 3 MLOps: Entrenamiento y Evaluación del modelo Time Series Forecasting."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Ruta explícita al dataset de características (formato parquet).",
    )
    parser.add_argument(
        "--input_artifacts_dir",
        type=Path,
        required=True,
        help="Directorio explícito con artefactos del Paso 2 (feature_config.json).",
    )
    parser.add_argument(
        "--output_eval_model_path",
        type=Path,
        required=True,
        help="Ruta de destino para el modelo de EVALUACIÓN (.joblib), entrenado solo con <=2015. "
        "Lo consume exclusivamente src/pipelines/4_evaluation.",
    )
    parser.add_argument(
        "--output_prod_model_path",
        type=Path,
        required=True,
        help="Ruta de destino para el modelo de PRODUCCIÓN (.joblib), entrenado con todo el historial. "
        "Lo consume exclusivamente la API (src/api/services.py).",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
        help="Directorio explícito de destino para métricas, hiperparámetros y el manifiesto de modelos.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=40,
        help="Número de iteraciones para Optuna.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig()

    df = DataLoader(args.input_path).load()
    feature_config = ArtifactLoader.load_feature_config(args.input_artifacts_dir)

    feature_columns: list[str] = feature_config["feature_columns"]
    categorical_features: list[str] = feature_config["categorical_features"]
    target_col: str = feature_config["target_col"]
    diff_target_col: str = feature_config["diff_target_col"]

    required_cols = feature_columns + [target_col, diff_target_col]
    clean_df = df.dropna(subset=required_cols).reset_index(drop=True)

    train_df, val_df = TemporalSplitter(train_config).split(clean_df)

    tuner = HyperparameterTuner(
        config=train_config,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        target_col=target_col,
        diff_target_col=diff_target_col,
        n_trials=args.n_trials,
    )
    best_params = tuner.tune(train_df)

    # ------------------------------------------------------------------
    # 1. Modelo EVAL -- entrenado SOLO con <=2015 (holdout honesto).
    #    Este es el ÚNICO modelo cuyas métricas sobre 2016-2019 son
    #    válidas, porque nunca vio esos años en entrenamiento.
    # ------------------------------------------------------------------
    logger.info("Entrenando modelo EVAL (holdout honesto, años <= %s)...", train_config.train_end_year)
    eval_model = DemandForecastModel(
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        diff_target_col=diff_target_col,
        random_state=train_config.random_state,
        hyperparams=best_params,
        model_role="eval",
    )
    eval_model.fit(train_df)

    val_diff_preds = eval_model.predict(val_df)
    val_level_preds = reconstruct_level_from_diff(val_df["lag_1"].to_numpy(), val_diff_preds)
    metrics = regression_report(val_df[target_col].to_numpy(), val_level_preds)

    logger.info(
        "Métricas de Validación HONESTAS (modelo eval, %s-%s) -> MAE: %.2f | RMSE: %.2f | WAPE: %.4f",
        train_config.val_start_year,
        train_config.val_end_year,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["WAPE"],
    )

    # ------------------------------------------------------------------
    # 2. Modelo PROD -- re-entrenado con el 100% del historial (1990-2019)
    #    para servir el mejor pronóstico posible en producción (>=2020).
    #    ADVERTENCIA: al incluir 2016-2019 en su entrenamiento, sus
    #    predicciones sobre esos años NO son una métrica de desempeño
    #    válida (in-sample). Nunca evaluar este modelo; nunca reportar
    #    sus métricas como si fueran las del modelo eval.
    # ------------------------------------------------------------------
    logger.info(
        "Entrenando modelo PROD (100%% del historial, incluye %s-%s) para inferencia en producción...",
        train_config.val_start_year,
        train_config.val_end_year,
    )
    prod_model = DemandForecastModel(
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        diff_target_col=diff_target_col,
        random_state=train_config.random_state,
        hyperparams=best_params,
        model_role="prod",
    )
    prod_model.fit(clean_df)
    logger.warning(
        "El modelo PROD fue entrenado incluyendo los años de validación (%s-%s). "
        "NO usar sus predicciones en ese rango como métrica de desempeño -- "
        "usar únicamente el modelo eval (%s) para eso.",
        train_config.val_start_year,
        train_config.val_end_year,
        args.output_eval_model_path.name,
    )

    # 3. Guardar ambos modelos + métricas + manifiesto de auditoría
    ArtifactManager.save_results(
        output_eval_model_path=args.output_eval_model_path,
        output_prod_model_path=args.output_prod_model_path,
        artifacts_dir=args.artifacts_dir,
        eval_model=eval_model,
        prod_model=prod_model,
        metrics=metrics,
        best_params=best_params,
    )


if __name__ == "__main__":
    main()
