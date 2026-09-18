from __future__ import annotations

import difflib
import importlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("api.services")

# Rol que DEBE tener el modelo que sirve la API. Ver DemandForecastModel en
# src/pipelines/3_training/main.py: "prod" = re-entrenado con el 100% del
# historial (1990-2019). El modelo "eval" (solo <=2015) NUNCA debe servir
# inferencias en producción: dejaría fuera 2016-2019 como información y
# está pensado únicamente para métricas offline (src/pipelines/4_evaluation).
EXPECTED_MODEL_ROLE = "prod"

# 1. Importación dinámica de los módulos que definen las clases serializadas
#    en los artefactos joblib (Paso 2: encoders/features; Paso 3: modelo).
_feature_module = importlib.import_module("src.pipelines.2_feature_engineering.main")
CategoricalEncoder = _feature_module.CategoricalEncoder
TimeSeriesFeatureEngineer = _feature_module.TimeSeriesFeatureEngineer

_training_module = importlib.import_module("src.pipelines.3_training.main")
DemandForecastModel = _training_module.DemandForecastModel

# 2. Resolución robusta de clases al deserializar (independiente de __main__).
#
#    Los artefactos (lgbm_prod.joblib, categorical_encoder.joblib, etc.) se
#    generaron ejecutando los scripts de los pasos 2 y 3 como `__main__`
#    (p.ej. `python src/pipelines/3_training/main.py`), así que pickle los
#    serializó referenciando el módulo "__main__". Al cargarlos aquí, el
#    proceso real casi nunca tiene ese mismo `__main__`: la API se lanza con
#    `uvicorn src.api.main:app --reload`, donde `__main__` es el propio CLI
#    de uvicorn, no `src.api.main` ni este módulo. Inyectar las clases en
#    `sys.modules["__main__"]` (técnica anterior) por lo tanto NO llega al
#    lugar correcto y joblib falla con "Can't get attribute 'X'" o, peor,
#    reconstruye un objeto vacío/incompleto (p.ej. un CategoricalEncoder sin
#    `.transform()`).
#
#    La solución robusta -- válida sin importar cómo se lance el proceso
#    (uvicorn CLI, `--reload`, gunicorn, `python -m`, etc.) -- es parchear
#    `find_class` para resolver por NOMBRE de clase, ignorando el módulo de
#    origen registrado en el pickle.
#
#    `pickle.Unpickler` en CPython es la implementación acelerada en C
#    (`_pickle.Unpickler`), un tipo inmutable que no permite parchear sus
#    métodos directamente. `joblib.load` usa su propio
#    `joblib.numpy_pickle.NumpyUnpickler`, una subclase de Python normal (por
#    lo tanto sí parcheable) que hereda de `pickle._Unpickler`/`Unpickler`
#    sin sobrescribir `find_class`. Parcheamos esa subclase directamente
#    -- es la que realmente deserializa `lgbm_prod.joblib`,
#    `categorical_encoder.joblib`, etc. Como red de seguridad adicional, si
#    la API alguna vez usa `pickle.load` puro sobre un stream compatible,
#    también parcheamos la implementación pura de Python (`pickle._Unpickler`),
#    que sí es un tipo mutable.
_PICKLE_CLASS_OVERRIDES: dict[str, type] = {
    "DemandForecastModel": DemandForecastModel,
    "CategoricalEncoder": CategoricalEncoder,
    "TimeSeriesFeatureEngineer": TimeSeriesFeatureEngineer,
}


def _make_patched_find_class(orig_find_class):  # noqa: ANN001
    def _patched_find_class(self, module: str, name: str) -> Any:  # noqa: ANN001
        if name in _PICKLE_CLASS_OVERRIDES:
            return _PICKLE_CLASS_OVERRIDES[name]
        return orig_find_class(self, module, name)

    return _patched_find_class


try:
    import joblib.numpy_pickle as _joblib_numpy_pickle

    _joblib_numpy_pickle.NumpyUnpickler.find_class = _make_patched_find_class(
        _joblib_numpy_pickle.NumpyUnpickler.find_class
    )
except Exception:  # pragma: no cover - defensivo, no debe romper el arranque de la API
    logger.exception("No se pudo parchear joblib.numpy_pickle.NumpyUnpickler.find_class")

# Red de seguridad: la implementación pura de Python de pickle SÍ es mutable.
pickle._Unpickler.find_class = _make_patched_find_class(pickle._Unpickler.find_class)  # type: ignore[attr-defined]


class ForecastingService:
    """Servicio de inferencia recursiva para predicción de series de tiempo.

    IMPORTANTE: este servicio debe cargar únicamente el modelo de
    PRODUCCIÓN (``lgbm_prod.joblib``), entrenado con todo el historial
    disponible (1990-2019). El modelo de evaluación (``lgbm_eval.joblib``,
    solo <=2015) queda reservado para ``src/pipelines/4_evaluation`` y no
    debe usarse aquí: además de estar entrenado con menos información,
    mezclar los dos artefactos rompería la separación eval/prod que evita
    reportar métricas infladas (ver ``model_registry.json`` en
    ``data/04_models/artifacts/``).
    """

    def __init__(
        self,
        data_path: Path = Path("data/02_processed/data_long.parquet"),
        model_path: Path = Path("data/04_models/lgbm_prod.joblib"),
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

        self._verify_model_role()

    def _verify_model_role(self) -> None:
        """Guardrail anti-fuga de datos: rechaza servir un modelo que no sea 'prod'.

        Si por error se apunta ``model_path`` al artefacto de evaluación
        (entrenado solo con <=2015), la API estaría sirviendo un modelo con
        menos información sin que nadie lo note. Se corta en el arranque
        (``load_artifacts`` se llama desde el ``lifespan`` de FastAPI) en
        vez de fallar silenciosamente en producción.
        """
        role = getattr(self.model, "model_role", None)
        train_year_range = getattr(self.model, "train_year_range", None)

        if role is None:
            logger.warning(
                "El modelo cargado (%s) no tiene metadato 'model_role' (artefacto generado con una "
                "versión anterior del pipeline). No se puede verificar automáticamente que sea el "
                "modelo de producción -- confirma manualmente que fue entrenado con todo el historial.",
                self.model_path.name,
            )
            return

        if role != EXPECTED_MODEL_ROLE:
            raise ValueError(
                f"Modelo incorrecto para la API: '{self.model_path.name}' tiene model_role='{role}' "
                f"(rango de años entrenados: {train_year_range}), pero el servicio de inferencia "
                f"requiere model_role='{EXPECTED_MODEL_ROLE}'. Probablemente apuntaste por error al "
                "modelo de evaluación (lgbm_eval.joblib), que fue entrenado solo hasta 2015. Usa "
                "lgbm_prod.joblib."
            )

        logger.info(
            "Modelo de producción cargado correctamente -> model_role='%s', entrenado con años %s.",
            role,
            train_year_range,
        )

    def get_available_countries(self) -> list[str]:
        """Lista (ordenada) de todos los países presentes en el histórico cargado.

        Es la ÚNICA fuente de verdad de qué nombres de país acepta la API --
        tanto la UI (para poblar el selector) como el agente LLM deberían
        usar exactamente estos nombres, tal como aparecen en
        `data_long.parquet` (p.ej. "Viet Nam", no "Vietnam"), para no volver
        a toparse con un país "inexistente" por una simple diferencia de
        escritura.
        """
        if self.historical_df is None:
            raise RuntimeError("El servicio no ha sido inicializado. Ejecute load_artifacts() primero.")
        return sorted(self.historical_df["Country"].dropna().unique().tolist())

    def predict_country(self, country: str, end_year: int) -> dict[str, Any]:
        """Ejecuta el bucle de pronóstico recursivo en memoria para un país (2020 -> end_year)."""
        if self.historical_df is None or self.model is None or self.encoder is None:
            raise RuntimeError("El servicio no ha sido inicializado. Ejecute load_artifacts() primero.")

        df_country = self.historical_df[self.historical_df["Country"] == country].copy()
        if df_country.empty:
            available = self.get_available_countries()
            suggestions = difflib.get_close_matches(country, available, n=3, cutoff=0.6)
            hint = f" ¿Quisiste decir: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(
                f"El país '{country}' no existe en el registro histórico.{hint} "
                "Usa GET /countries para ver la lista completa de países soportados."
            )

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
        """Ejecuta la inferencia recursiva para una lista de múltiples países.

        IMPORTANTE: un país inválido (mal escrito o fuera del dataset) ya NO
        aborta el lote completo con una excepción -- se captura por país y
        se agrega a `results` con su propio campo `error`, dejando que los
        demás países del lote se resuelvan con normalidad. Esto es lo que le
        permite a `src/api/main.py` responder siempre 200 (nunca 404 por un
        solo país problemático) y a la UI / al agente LLM seguir mostrando
        el resto del lote en vez de fallar por completo (ver
        `CountryForecastResponse.error` en `src/api/schemas.py`).
        """
        results = []
        for country in countries:
            try:
                country_res = self.predict_country(country=country, end_year=end_year)
                results.append(country_res)
            except ValueError as e:
                logger.warning("No se pudo generar la proyección para '%s': %s", country, e)
                results.append({
                    "country": country,
                    "coffee_type": None,
                    "historical_end_year": None,
                    "forecast_end_year": end_year,
                    "data": [],
                    "error": str(e),
                })
        return {"results": results}
