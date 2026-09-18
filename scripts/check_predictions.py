"""Script independiente de verificacion de predicciones LightGBM (multi-pais).

No forma parte del pipeline oficial (src/pipelines/4_evaluation); es una
utilidad de QA que:

  1. Calcula metricas de validacion (2016-2019) para TODOS los paises.
  2. Selecciona los 3 paises con mejor y los 3 con peor desempeno (segun WAPE,
     por ser una metrica de error relativa y por lo tanto comparable entre
     paises con escalas de consumo muy distintas).
  3. Para esos 6 paises, grafica en una grilla de 2x3 subplots:
       - Azul: entrenamiento (1990-2015)
       - Verde: valores reales de validacion (2016-2019)
       - Negro: predicciones del modelo en validacion (2016-2019)
       - Rojo: proyecciones futuras (>=2020), generadas con el mismo bucle de
         inferencia recursiva que usa el servicio oficial (src/api/services.py).

Uso:
    python check_predictions.py
    python check_predictions.py --forecast_end_year 2025 --metric WAPE
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
logger = logging.getLogger("check_predictions")


# --------------------------------------------------------------------------
# Clases replicadas de los pasos 2 y 3 del pipeline. joblib necesita estas
# clases definidas (mismo nombre y atributos) para poder deserializar el
# modelo y el encoder de categoricas.
# --------------------------------------------------------------------------
class DemandForecastModel:
    """Replica del wrapper usado al serializar el modelo (Paso 3)."""

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


class TimeSeriesFeatureEngineer:
    """Replica del generador de features del Paso 2 (lags + medias moviles)."""

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
        df = df.sort_values([self.country_col, self.year_col]).reset_index(drop=True)
        by_country = df.groupby(self.country_col)[self.target_col]

        for lag in self.lags:
            df[f"lag_{lag}"] = by_country.shift(lag)

        shifted_target = by_country.shift(1)
        w = self.rolling_window
        df[f"rolling_mean_{w}"] = shifted_target.groupby(df[self.country_col]).transform(
            lambda s: s.rolling(window=w, min_periods=w).mean()
        )
        df[f"rolling_std_{w}"] = shifted_target.groupby(df[self.country_col]).transform(
            lambda s: s.rolling(window=w, min_periods=w).std(ddof=1)
        )
        df[self.diff_target_col] = df[self.target_col] - df["lag_1"]
        return df


class CategoricalEncoder:
    """Replica del codificador deterministico de categoricas del Paso 2."""

    UNSEEN_CODE = -1

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.mappings: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        for col in self.columns:
            if col in df.columns:
                categories = sorted(df[col].dropna().unique())
                self.mappings[col] = {str(category): code for code, category in enumerate(categories)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            if col in self.mappings:
                mapping = self.mappings[col]
                df[f"{col}_encoded"] = (
                    df[col].astype(str).map(mapping).fillna(self.UNSEEN_CODE).astype(int)
                )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckConfig:
    train_start: int = 1990
    train_end: int = 2015
    val_start: int = 2016
    val_end: int = 2019
    year_col: str = "Year"
    country_col: str = "Country"
    target_col: str = "Consumption"
    top_n: int = 3
    ranking_metric: str = "WAPE"  # MAE | RMSE | WAPE


def load_artifacts(
    features_path: Path, long_path: Path, model_path: Path,
    feature_config_path: Path, encoder_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, Any], CategoricalEncoder]:
    for p in (features_path, long_path, model_path, feature_config_path, encoder_path):
        if not p.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {p.resolve()}")

    features_df = pd.read_parquet(features_path)
    long_df = pd.read_parquet(long_path)
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        feature_config: dict[str, Any] = json.load(f)

    logger.info("data_features: %s filas | data_long: %s filas", len(features_df), len(long_df))
    logger.info("Modelo cargado: %s", type(model).__name__)
    return features_df, long_df, model, feature_config, encoder


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def predict_level(model: Any, feature_config: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Predice el nivel de Consumption reconstruyendo a partir del delta YoY predicho."""
    feature_columns: list[str] = feature_config["feature_columns"]
    categorical_features: list[str] = feature_config["categorical_features"]

    if isinstance(model, DemandForecastModel):
        diff_preds = model.predict(df)
    else:
        X = df[feature_columns].copy()
        for col in categorical_features:
            X[col] = X[col].astype("category")
        diff_preds = model.predict(X)
    return df["lag_1"].to_numpy() + diff_preds


def evaluate_all_countries(
    features_df: pd.DataFrame, model: Any, feature_config: dict[str, Any], cfg: CheckConfig
) -> pd.DataFrame:
    """Calcula MAE, RMSE, WAPE de validacion (2016-2019) para cada pais."""
    val_df = features_df.loc[
        features_df[cfg.year_col].between(cfg.val_start, cfg.val_end)
    ].copy()
    val_preds = predict_level(model, feature_config, val_df)
    val_df["predicted_demand"] = val_preds

    rows = []
    for country, group in val_df.groupby(cfg.country_col):
        y_true = group[cfg.target_col].to_numpy()
        y_pred = group["predicted_demand"].to_numpy()
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        wape = weighted_absolute_percentage_error(y_true, y_pred)
        rows.append({"Country": country, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "WAPE": round(wape, 4), "n_obs": len(group)})

    metrics_df = pd.DataFrame(rows).sort_values(cfg.ranking_metric, ascending=True).reset_index(drop=True)
    return metrics_df


def select_best_worst(metrics_df: pd.DataFrame, cfg: CheckConfig) -> tuple[list[str], list[str], list[str]]:
    """Rankea por `cfg.ranking_metric` e ignora paises con metrica indefinida.

    WAPE queda como NaN cuando la demanda real es 0 en toda la ventana de
    validacion (denominador cero) -- esos paises no son comparables por WAPE
    y se excluyen del ranking (se reportan aparte), en vez de colarse como
    falsos "peores" solo por el NaN.
    """
    excluded = metrics_df.loc[metrics_df[cfg.ranking_metric].isna(), "Country"].tolist()
    ranked = metrics_df.dropna(subset=[cfg.ranking_metric]).sort_values(cfg.ranking_metric, ascending=True)
    best = ranked.head(cfg.top_n)["Country"].tolist()
    worst = ranked.tail(cfg.top_n)["Country"].tolist()
    return best, worst, excluded


def recursive_forecast(
    country: str, long_df: pd.DataFrame, model: Any, encoder: CategoricalEncoder,
    feature_config: dict[str, Any], end_year: int,
) -> pd.DataFrame:
    """Pronostico recursivo (>=2020), replicando ForecastingService.predict_country."""
    df_country = long_df[long_df["Country"] == country].copy()
    if df_country.empty:
        raise ValueError(f"Pais '{country}' no encontrado en data_long.")

    coffee_type = df_country["Coffee_type"].iloc[0]
    max_hist_year = int(df_country["Year"].max())

    feature_engineer = TimeSeriesFeatureEngineer()
    current_df = df_country.copy()

    for target_year in range(max_hist_year + 1, end_year + 1):
        new_row = pd.DataFrame([{
            "Country": country, "Coffee_type": coffee_type, "Year": target_year, "Consumption": np.nan,
        }])
        current_df = pd.concat([current_df, new_row], ignore_index=True)

        transformed_df = feature_engineer.transform(current_df)
        encoded_df = encoder.transform(transformed_df)

        target_row = encoded_df[encoded_df["Year"] == target_year]
        X_pred = target_row[feature_config["feature_columns"]].copy()
        for col in feature_config["categorical_features"]:
            X_pred[col] = X_pred[col].astype("category")

        if isinstance(model, DemandForecastModel):
            delta_y_hat = model.predict(target_row)[0]
        else:
            delta_y_hat = model.predict(X_pred)[0]

        prev_y = current_df.loc[current_df["Year"] == target_year - 1, "Consumption"].to_numpy()[0]
        pred_y = max(0.0, float(prev_y + delta_y_hat))
        current_df.loc[current_df["Year"] == target_year, "Consumption"] = pred_y

    forecast_df = current_df[current_df["Year"] > max_hist_year].copy()
    return forecast_df[["Country", "Coffee_type", "Year", "Consumption"]]


def plot_grid(
    countries: list[str], features_df: pd.DataFrame, forecasts: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame, cfg: CheckConfig, output_path: Path,
) -> None:
    """Grilla 2x3: fila 1 = mejores 3 paises, fila 2 = peores 3 paises."""
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10), sharex=False)
    axes = axes.reshape(n_rows, n_cols)

    metrics_lookup = metrics_df.set_index("Country").to_dict(orient="index")

    for idx, country in enumerate(countries):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        country_feat = features_df[features_df[cfg.country_col] == country].sort_values(cfg.year_col)
        train_df = country_feat[country_feat[cfg.year_col].between(cfg.train_start, cfg.train_end)]
        val_df = country_feat[country_feat[cfg.year_col].between(cfg.val_start, cfg.val_end)].copy()
        val_preds = val_df["predicted_demand"] if "predicted_demand" in val_df.columns else None
        future_df = forecasts[country]

        ax.plot(train_df[cfg.year_col], train_df[cfg.target_col],
                color="blue", marker="o", markersize=3, linewidth=1.3, label="Entrenamiento (1990-2015)")
        ax.plot(val_df[cfg.year_col], val_df[cfg.target_col],
                color="green", marker="o", markersize=4, linewidth=1.3, label="Real (2016-2019)")
        ax.plot(val_df[cfg.year_col], val_preds,
                color="black", marker="x", markersize=5, linewidth=1.3, linestyle="--", label="Prediccion (2016-2019)")
        ax.plot(future_df[cfg.year_col], future_df[cfg.target_col],
                color="red", marker="^", markersize=4, linewidth=1.3, linestyle="--", label="Proyeccion futura (>=2020)")

        m = metrics_lookup.get(country, {})
        wape_pct = m.get("WAPE", float("nan")) * 100
        ax.set_title(f"{country}\nWAPE={wape_pct:.2f}% | MAE={m.get('MAE', float('nan')):,.0f}", fontsize=10)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    axes[0][0].annotate("MEJORES 3 (menor WAPE)", xy=(0, 1.18), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", ha="left")
    axes[1][0].annotate("PEORES 3 (mayor WAPE)", xy=(0, 1.18), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", ha="left")

    fig.suptitle("Verificacion multi-pais del modelo LightGBM (mejor/peor desempeno de validacion)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Grafico guardado en: %s", output_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verificacion independiente multi-pais de predicciones del modelo LightGBM."
    )
    parser.add_argument("--features_path", type=Path, default=Path("data/03_features/data_features.parquet"))
    parser.add_argument("--long_path", type=Path, default=Path("data/02_processed/data_long.parquet"))
    parser.add_argument("--model_path", type=Path, default=Path("data/04_models/lgbm_model.joblib"))
    parser.add_argument("--feature_config_path", type=Path, default=Path("data/03_features/artifacts/feature_config.json"))
    parser.add_argument("--encoder_path", type=Path, default=Path("data/03_features/artifacts/categorical_encoder.joblib"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/verification"))
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--metric", type=str, default="WAPE", choices=["MAE", "RMSE", "WAPE"])
    parser.add_argument("--forecast_end_year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CheckConfig(top_n=args.top_n, ranking_metric=args.metric)

    features_df, long_df, model, feature_config, encoder = load_artifacts(
        args.features_path, args.long_path, args.model_path, args.feature_config_path, args.encoder_path,
    )

    metrics_df = evaluate_all_countries(features_df, model, feature_config, cfg)
    best_countries, worst_countries, excluded_countries = select_best_worst(metrics_df, cfg)
    selected_countries = best_countries + worst_countries

    logger.info("Mejores %s paises (%s mas bajo): %s", cfg.top_n, cfg.ranking_metric, best_countries)
    logger.info("Peores %s paises (%s mas alto): %s", cfg.top_n, cfg.ranking_metric, worst_countries)
    if excluded_countries:
        logger.warning(
            "Excluidos del ranking por %s indefinido (demanda real=0 en toda la ventana de validacion): %s",
            cfg.ranking_metric, excluded_countries,
        )

    # Predicciones de validacion para poder graficarlas (reutiliza la misma logica de evaluate_all_countries)
    val_df_all = features_df.loc[features_df[cfg.year_col].between(cfg.val_start, cfg.val_end)].copy()
    val_df_all["predicted_demand"] = predict_level(model, feature_config, val_df_all)
    features_df = features_df.merge(
        val_df_all[[cfg.country_col, cfg.year_col, "predicted_demand"]],
        on=[cfg.country_col, cfg.year_col], how="left",
    )

    logger.info("Generando proyecciones futuras recursivas (>=2020 hasta %s)...", args.forecast_end_year)
    forecasts: dict[str, pd.DataFrame] = {}
    for country in selected_countries:
        forecasts[country] = recursive_forecast(
            country, long_df, model, encoder, feature_config, args.forecast_end_year
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reporte de metricas: ranking completo + seleccion
    report = {
        "ranking_metric": cfg.ranking_metric,
        "val_window": [cfg.val_start, cfg.val_end],
        "best_countries": best_countries,
        "worst_countries": worst_countries,
        "excluded_countries_undefined_metric": excluded_countries,
        "metrics_by_country": metrics_df.to_dict(orient="records"),
    }
    metrics_path = args.output_dir / "check_predictions_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logger.info("Reporte de metricas (todos los paises) guardado en: %s", metrics_path.resolve())

    plot_path = args.output_dir / "check_predictions_grid.png"
    plot_grid(selected_countries, features_df, forecasts, metrics_df, cfg, plot_path)


if __name__ == "__main__":
    main()
