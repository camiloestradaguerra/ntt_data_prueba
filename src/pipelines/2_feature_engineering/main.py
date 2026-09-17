from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("2_feature_engineering")


class DataLoader:
    """Carga el dataset procesado en formato Long desde la ruta especificada."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado en la ruta: {self.data_path.resolve()}")
        df = pd.read_parquet(self.data_path)
        logger.info("Dataset Long cargado exitosamente desde %s (%s filas, %s columnas)", self.data_path, *df.shape)
        return df


class TimeSeriesFeatureEngineer:
    """Genera características temporales libres de look-ahead leakage."""

    def __init__(
        self,
        country_col: str = "Country",
        year_col: str = "Year",
        target_col: str = "Consumption",
        diff_target_col: str = "Consumption_diff",
        lags: tuple[int, ...] = (1, 2, 3),
        rolling_window: int = 3,
    ) -> None:
        self.country_col = country_col
        self.year_col = year_col
        self.target_col = target_col
        self.diff_target_col = diff_target_col
        self.lags = lags
        self.rolling_window = rolling_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula lags y estadísticas móviles usando exclusivamente información pasada (t-1)."""
        df = df.sort_values([self.country_col, self.year_col]).reset_index(drop=True)
        by_country = df.groupby(self.country_col)[self.target_col]

        # Lags temporales
        for lag in self.lags:
            df[f"lag_{lag}"] = by_country.shift(lag)

        # Estadísticas móviles aplicadas sobre la serie desplazada t-1
        shifted_target = by_country.shift(1)
        w = self.rolling_window
        df[f"rolling_mean_{w}"] = shifted_target.groupby(df[self.country_col]).transform(
            lambda s: s.rolling(window=w, min_periods=w).mean()
        )
        df[f"rolling_std_{w}"] = shifted_target.groupby(df[self.country_col]).transform(
            lambda s: s.rolling(window=w, min_periods=w).std(ddof=1)
        )

        # Target estacionario (Delta YoY)
        df[self.diff_target_col] = df[self.target_col] - df["lag_1"]

        logger.info(
            "Ingeniería de características completada (Lags %s, Ventana móvil %s)",
            self.lags,
            self.rolling_window,
        )
        return df


class CategoricalEncoder:
    """Codifica identificadores categóricos de forma determinista y persiste sus mapeos."""

    UNSEEN_CODE = -1

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.mappings: dict[str, dict[str, int]] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                categories = sorted(df[col].dropna().unique())
                mapping = {str(category): code for code, category in enumerate(categories)}
                self.mappings[col] = mapping
                df[f"{col}_encoded"] = df[col].map(mapping).fillna(self.UNSEEN_CODE).astype(int)
        logger.info("Columnas categóricas codificadas: %s", self.columns)
        return df


class ArtifactManager:
    """Exporta artefactos de producción y metadatos del esquema de datos."""

    @staticmethod
    def save_artifacts(
        artifacts_dir: Path,
        encoder: CategoricalEncoder,
        feature_columns: list[str],
        categorical_features: list[str],
        target_col: str,
        diff_target_col: str,
    ) -> None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Objeto codificador serializado
        encoder_path = artifacts_dir / "categorical_encoder.joblib"
        joblib.dump(encoder, encoder_path)

        # 2. Mapeos legibles en formato JSON (Auditoría / Servicios de Inferencia)
        mappings_path = artifacts_dir / "encoder_mappings.json"
        with open(mappings_path, "w", encoding="utf-8") as f:
            json.dump(encoder.mappings, f, indent=4, ensure_ascii=False)

        # 3. Esquema de variables y configuración del pipeline
        metadata: dict[str, Any] = {
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "target_col": target_col,
            "diff_target_col": diff_target_col,
        }
        metadata_path = artifacts_dir / "feature_config.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Artefactos guardados exitosamente en directorio: %s", artifacts_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 2 MLOps: Generación de features temporales y persistencia de artefactos."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Ruta explícita al archivo parquet procesado del Paso 1.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Ruta explícita de destino para el dataset con características.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
        help="Directorio explícito donde se guardarán los artefactos serializados y metadatos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Creación explícita de directorios de destino
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    long_df = DataLoader(args.input_path).load()

    # 2. Transformación temporal
    feature_engineer = TimeSeriesFeatureEngineer()
    featured_df = feature_engineer.transform(long_df)

    # 3. Codificación de categóricas
    encoder = CategoricalEncoder(columns=["Country", "Coffee_type"])
    final_df = encoder.fit_transform(featured_df)

    # 4. Definición del contrato de features
    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_std_3",
        "Country_encoded",
        "Coffee_type_encoded",
    ]
    categorical_features = ["Country_encoded", "Coffee_type_encoded"]

    # 5. Persistencia en disco (Dataset + Artefactos)
    final_df.to_parquet(args.output_path, index=False)
    logger.info("Dataset procesado guardado en: %s", args.output_path.resolve())

    ArtifactManager.save_artifacts(
        artifacts_dir=args.artifacts_dir,
        encoder=encoder,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        target_col=feature_engineer.target_col,
        diff_target_col=feature_engineer.diff_target_col,
    )


if __name__ == "__main__":
    main()