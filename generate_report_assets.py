"""
Generates the figures and metric artifacts embedded in the Markdown report.

Reuses the exact classes defined in `coffee_forecast.py` (single source of
truth for the modeling logic) and only adds visualization/reporting code.

Run directly:
    python generate_report_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from coffee_forecast import (
    CategoricalEncoder,
    DataLoader,
    DemandForecastModel,
    ETLProcessor,
    HyperparameterTuner,
    PipelineConfig,
    RecursiveForecaster,
    TemporalSplitter,
    TimeSeriesFeatureEngineer,
    reconstruct_level_from_diff,
    regression_report,
)

FIG_DIR = Path(__file__).resolve().parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COFFEE_BROWN = "#6f4e37"


def _millions_formatter(x: float, _pos: int) -> str:
    return f"{x / 1e6:,.0f}M"


def run_pipeline_and_collect_artifacts(config: PipelineConfig) -> dict:
    """Re-runs the full pipeline, returning the intermediate objects needed for plots."""
    raw_df = DataLoader(config).load()
    long_df = ETLProcessor(config).transform(raw_df)
    engineered_df = TimeSeriesFeatureEngineer(config).add_lag_and_rolling_features(long_df)

    train_raw, val_raw = TemporalSplitter(config).split(engineered_df)

    encoder = CategoricalEncoder(columns=[config.country_col, config.coffee_type_col])
    encoder.fit(train_raw)
    train_df = encoder.transform(train_raw)
    val_df = encoder.transform(val_raw)
    full_encoded_df = encoder.transform(engineered_df)

    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        f"rolling_mean_{config.rolling_window}",
        f"rolling_std_{config.rolling_window}",
        f"{config.country_col}_encoded",
        f"{config.coffee_type_col}_encoded",
    ]
    categorical_features = [f"{config.country_col}_encoded", f"{config.coffee_type_col}_encoded"]
    required_cols = feature_columns + [config.target_col, config.diff_target_col]

    train_clean = train_df.dropna(subset=required_cols).reset_index(drop=True)
    val_clean = val_df.dropna(subset=required_cols).reset_index(drop=True)

    best_params, study = HyperparameterTuner(config, feature_columns, categorical_features).tune(train_clean)

    model = DemandForecastModel(config, feature_columns, categorical_features, hyperparams=best_params)
    model.fit(train_clean)

    val_diff_predictions = model.predict(val_clean)
    val_clean = val_clean.copy()
    val_clean["Predicted"] = reconstruct_level_from_diff(val_clean["lag_1"], val_diff_predictions)
    metrics = regression_report(val_clean[config.target_col].to_numpy(), val_clean["Predicted"].to_numpy())

    future_df = RecursiveForecaster(config, model).forecast(full_encoded_df)

    importances = pd.Series(
        model.model.feature_importances_, index=feature_columns
    ).sort_values(ascending=True)

    return {
        "long_df": long_df,
        "val_clean": val_clean,
        "metrics": metrics,
        "future_df": future_df,
        "importances": importances,
        "study": study,
        "best_params": best_params,
    }


def plot_validation_actual_vs_pred(val_clean: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    actual, predicted = val_clean["Consumption"], val_clean["Predicted"]
    ax.scatter(actual, predicted, alpha=0.6, edgecolor="k", linewidth=0.3, color=COFFEE_BROWN)
    upper_limit = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, upper_limit], [0, upper_limit], "r--", linewidth=1, label="Predicción perfecta (y = x)")
    ax.set_xlim(0, upper_limit)
    ax.set_ylim(0, upper_limit)
    ax.set_xlabel("Consumo real (kg)")
    ax.set_ylabel("Consumo predicho (kg)")
    ax.set_title("Validación 2016-2019: Real vs. Predicho")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_millions_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_millions_formatter))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color=COFFEE_BROWN)
    ax.set_xlabel("Importancia (ganancia acumulada, LightGBM)")
    ax.set_title("Importancia de variables del modelo")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_countries_forecast(
    long_df: pd.DataFrame, future_df: pd.DataFrame, config: PipelineConfig, out_path: Path, top_n: int = 5
) -> None:
    totals = long_df.groupby(config.country_col)[config.target_col].sum().sort_values(ascending=False)
    top_countries = totals.head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10.colors
    for i, country in enumerate(top_countries):
        hist = long_df[long_df[config.country_col] == country].sort_values(config.year_col)
        fut = future_df[future_df[config.country_col] == country].sort_values(config.year_col)
        ax.plot(hist[config.year_col], hist[config.target_col], color=colors[i], label=country)
        ax.plot(
            fut[config.year_col], fut["Predicted_Consumption"],
            color=colors[i], linestyle="--", marker="o", markersize=3,
        )
    ax.axvline(config.forecast_start_year - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Año")
    ax.set_ylabel("Consumo doméstico (kg)")
    ax.set_title(f"Top {top_n} países: histórico (sólida) + forecast recursivo (punteada)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_millions_formatter))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_global_total_trend(
    long_df: pd.DataFrame, future_df: pd.DataFrame, config: PipelineConfig, out_path: Path
) -> None:
    hist_totals = long_df.groupby(config.year_col)[config.target_col].sum()
    fut_totals = future_df.groupby(config.year_col)["Predicted_Consumption"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist_totals.index, hist_totals.values, color="#2c3e50", label="Consumo global histórico")
    ax.plot(
        fut_totals.index, fut_totals.values, color="#e67e22",
        linestyle="--", marker="o", label="Forecast global 2020-2025",
    )
    ax.axvline(config.forecast_start_year - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Año")
    ax.set_ylabel("Consumo doméstico total (kg)")
    ax.set_title("Consumo global de café: histórico vs. proyección")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_millions_formatter))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    value = metrics["WAPE"] * 100
    bar = ax.bar(["WAPE"], [value], color=COFFEE_BROWN, width=0.4)
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, value + 0.5, f"{value:.2f}%", ha="center")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Métrica de negocio en validación (2016-2019)")
    ax.set_ylim(0, value * 1.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_optuna_convergence(study, out_path: Path) -> None:
    """Plots per-trial CV WAPE and the running-best value found by Optuna."""
    if study is None:
        return
    trials_df = study.trials_dataframe().sort_values("number")
    running_best = trials_df["value"].cummin()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(trials_df["number"], trials_df["value"], alpha=0.5, color="#95a5a6", label="WAPE por trial (CV)")
    ax.plot(trials_df["number"], running_best, color=COFFEE_BROWN, linewidth=2, label="Mejor WAPE acumulado")
    ax.set_xlabel("Trial de Optuna")
    ax.set_ylabel("WAPE promedio (folds CV)")
    ax.set_title("Convergencia de la búsqueda bayesiana (TPE) de hiperparámetros")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = PipelineConfig()
    artifacts = run_pipeline_and_collect_artifacts(config)

    plot_validation_actual_vs_pred(artifacts["val_clean"], FIG_DIR / "validation_actual_vs_pred.png")
    plot_feature_importance(artifacts["importances"], FIG_DIR / "feature_importance.png")
    plot_top_countries_forecast(artifacts["long_df"], artifacts["future_df"], config, FIG_DIR / "top_countries_forecast.png")
    plot_global_total_trend(artifacts["long_df"], artifacts["future_df"], config, FIG_DIR / "global_total_trend.png")
    plot_metrics_bar(artifacts["metrics"], FIG_DIR / "wape_metric.png")
    plot_optuna_convergence(artifacts["study"], FIG_DIR / "optuna_convergence.png")

    print("Best hyperparameters:", artifacts["best_params"])
    print("Validation metrics:", artifacts["metrics"])
    print(f"Figures saved under: {FIG_DIR}")


if __name__ == "__main__":
    main()
