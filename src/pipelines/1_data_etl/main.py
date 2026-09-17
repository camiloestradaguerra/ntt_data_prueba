from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("1_data_etl")


class DataLoader:
    """Carga el dataset parquet en formato wide desde el disco."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado en la ruta: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        logger.info("Dataset raw cargado exitosamente: %s filas, %s columnas", *df.shape)
        return df


class ETLProcessor:
    """Transforma el panel wide (columnas 'YYYY/YY') a un formato Long Panel estandarizado."""

    _YEAR_PATTERN = re.compile(r"^\d{4}/\d{2}$")

    def __init__(
        self,
        country_col: str = "Country",
        raw_coffee_type_col: str = "Coffee type",
        coffee_type_col: str = "Coffee_type",
        year_col: str = "Year",
        target_col: str = "Consumption",
    ) -> None:
        self.country_col = country_col
        self.raw_coffee_type_col = raw_coffee_type_col
        self.coffee_type_col = coffee_type_col
        self.year_col = year_col
        self.target_col = target_col

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.rename(columns={self.raw_coffee_type_col: self.coffee_type_col})

        year_columns = [c for c in df.columns if self._YEAR_PATTERN.match(str(c))]
        if not year_columns:
            raise ValueError("No se encontraron columnas de años agrícolas con el patrón 'YYYY/YY'.")

        long_df = df.melt(
            id_vars=[self.country_col, self.coffee_type_col],
            value_vars=year_columns,
            var_name=self.year_col,
            value_name=self.target_col,
        )

        # Transforma '1990/91' -> 1990 (año de inicio del ciclo agrícola)
        long_df[self.year_col] = long_df[self.year_col].str.slice(0, 4).astype(int)
        long_df[self.target_col] = pd.to_numeric(long_df[self.target_col], errors="coerce").fillna(0)

        # --- REGLA DE NEGOCIO: Filtrar ceros estructurales ---
        # Suma todo el consumo del país; si es 0 en todo el periodo, se elimina.
        # Si empezó en 0 y luego tuvo consumo > 0, se conserva completo.
        initial_countries = long_df[self.country_col].nunique()
        country_totals = long_df.groupby(self.country_col)[self.target_col].transform("sum")
        
        long_df = long_df[country_totals > 0].copy()

        removed_count = initial_countries - long_df[self.country_col].nunique()
        logger.info(
            "Filtro aplicado: %s países eliminados por consumo acumulado 0 en todo el histórico.",
            removed_count,
        )
        # ----------------------------------------------------

        long_df = long_df.sort_values(by=[self.country_col, self.year_col]).reset_index(drop=True)

        logger.info(
            "ETL completado: %s filas | %s países activos | Años %s-%s",
            len(long_df),
            long_df[self.country_col].nunique(),
            long_df[self.year_col].min(),
            long_df[self.year_col].max(),
        )
        return long_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 1 MLOps: Carga raw wide dataset y realiza transformación a Long Panel."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/01_raw/coffee_db.parquet"),
        help="Ruta al archivo parquet original de entrada (formato Wide).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/02_processed/data_long.parquet"),
        help="Ruta de destino para el dataset procesado (formato Long).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Asegura que la carpeta de salida exista
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ejecución modular del Paso 1
    raw_df = DataLoader(args.input_path).load()
    long_df = ETLProcessor().transform(raw_df)

    # Persistencia en disco para el siguiente paso del pipeline
    long_df.to_parquet(args.output_path, index=False)
    logger.info("Archivo resultante guardado exitosamente en: %s", args.output_path)


if __name__ == "__main__":
    main()