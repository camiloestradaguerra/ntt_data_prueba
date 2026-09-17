"""
Coffee Domestic Consumption — Time Series Forecasting Pipeline
================================================================

Production-grade, leakage-free pipeline that:
  1. Loads the wide-format parquet dataset (Country x Coffee type x crop-year).
  2. Unpivots it into a tidy long panel [Country, Coffee_type, Year, Consumption].
  3. Engineers lag / rolling features using strictly past information.
  4. Splits the data with an expanding-window (time-based) strategy — no random
     shuffling, no KFold.
  5. Tunes LightGBM hyperparameters with Optuna (TPE), validated on nested
     expanding-window time-series folds carved out of the training period only.
  6. Trains a global LightGBM regressor shared across all countries.
  7. Evaluates it with MAE, RMSE and WAPE (the business-critical metric).
  8. Recursively forecasts 2020-2025 by feeding each step's own prediction back
     into the next step's lag/rolling features.

Run directly:
    python coffee_forecast.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("coffee_forecast")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence per-trial noise; we log our own summary


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineConfig:
    """Centralized, immutable configuration for the whole pipeline."""

    data_path: Path = Path(__file__).resolve().parent / "coffee_db.parquet"
    output_dir: Path = Path(__file__).resolve().parent

    country_col: str = "Country"
    raw_coffee_type_col: str = "Coffee type"
    coffee_type_col: str = "Coffee_type"
    year_col: str = "Year"
    target_col: str = "Consumption"
    diff_target_col: str = "Consumption_diff"  # y_t - y_(t-1): avoids tree extrapolation on raw level/Year

    lags: tuple[int, ...] = (1, 2, 3)
    rolling_window: int = 3

    train_end_year: int = 2015
    val_start_year: int = 2016
    val_end_year: int = 2020

    forecast_start_year: int = 2020
    forecast_horizon: int = 5

    random_state: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 15

    # Nested time-series CV for hyperparameter tuning (carved out of train only)
    cv_n_splits: int = 3
    cv_val_years: int = 2
    cv_min_train_years: int = 10
    n_trials: int = 40


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
class DataLoader:
    """Loads the raw wide-format coffee consumption dataset from parquet."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        """Read the parquet file from disk.

        Returns:
            The raw wide-format DataFrame exactly as stored in the parquet file.

        Raises:
            FileNotFoundError: If the configured data path does not exist.
        """
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.config.data_path}")
        df = pd.read_parquet(self.config.data_path)
        logger.info("Loaded raw dataset: %s rows, %s columns", *df.shape)
        return df


# --------------------------------------------------------------------------- #
# 2. ETL: wide -> long
# --------------------------------------------------------------------------- #
class ETLProcessor:
    """Transforms the wide (one-column-per-crop-year) dataset into a tidy long panel."""

    _YEAR_PATTERN = re.compile(r"^\d{4}/\d{2}$")

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Unpivot crop-year columns and normalize the schema.

        `Total_domestic_consumption` is intentionally dropped: it is a static
        sum over ALL years (1990-2019) and would leak future information into
        any row predicted before 2019.

        Args:
            raw_df: Wide-format DataFrame with one column per crop-year.

        Returns:
            Long-format DataFrame [Country, Coffee_type, Year, Consumption],
            sorted strictly by Country then Year.
        """
        cfg = self.config
        df = raw_df.rename(columns={cfg.raw_coffee_type_col: cfg.coffee_type_col})

        year_columns = [c for c in df.columns if self._YEAR_PATTERN.match(str(c))]
        if not year_columns:
            raise ValueError("No crop-year columns matching pattern 'YYYY/YY' were found.")

        long_df = df.melt(
            id_vars=[cfg.country_col, cfg.coffee_type_col],
            value_vars=year_columns,
            var_name=cfg.year_col,
            value_name=cfg.target_col,
        )

        # "1990/91" -> 1990 (start year of the crop cycle, continuous integer)
        long_df[cfg.year_col] = long_df[cfg.year_col].str.slice(0, 4).astype(int)
        long_df[cfg.target_col] = pd.to_numeric(long_df[cfg.target_col], errors="coerce")

        long_df = long_df.sort_values(by=[cfg.country_col, cfg.year_col]).reset_index(drop=True)

        logger.info(
            "ETL complete: %s rows | %s countries | years %s-%s",
            len(long_df),
            long_df[cfg.country_col].nunique(),
            long_df[cfg.year_col].min(),
            long_df[cfg.year_col].max(),
        )
        return long_df


# --------------------------------------------------------------------------- #
# 3. Feature engineering (lags / rolling stats) — leakage-free by construction
# --------------------------------------------------------------------------- #
class TimeSeriesFeatureEngineer:
    """Builds per-country lag and rolling features using exclusively past values."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag_{1,2,3} and rolling mean/std features.

        Every feature at row `t` only ever reads values from `t-1` and earlier,
        guaranteeing zero look-ahead leakage regardless of how the data is
        later split in time.

        Args:
            df: Long-format panel sorted by Country/Year.

        Returns:
            The same DataFrame enriched with lag and rolling-window columns.
        """
        cfg = self.config
        df = df.sort_values([cfg.country_col, cfg.year_col]).reset_index(drop=True)
        by_country = df.groupby(cfg.country_col)[cfg.target_col]

        for lag in cfg.lags:
            df[f"lag_{lag}"] = by_country.shift(lag)

        shifted_target = by_country.shift(1)  # only past values are visible to the rolling window
        window = cfg.rolling_window
        df[f"rolling_mean_{window}"] = shifted_target.groupby(df[cfg.country_col]).transform(
            lambda s: s.rolling(window=window, min_periods=window).mean()
        )
        # ddof=1 explicit: must match RecursiveForecaster's np.std(..., ddof=1) exactly
        df[f"rolling_std_{window}"] = shifted_target.groupby(df[cfg.country_col]).transform(
            lambda s: s.rolling(window=window, min_periods=window).std(ddof=1)
        )

        # First-difference target: trees can't extrapolate raw level/Year, but a
        # (mostly stationary) year-over-year delta sidesteps that limitation.
        df[cfg.diff_target_col] = df[cfg.target_col] - df["lag_1"]
        return df


# --------------------------------------------------------------------------- #
# 4. Categorical encoding — fit strictly on train
# --------------------------------------------------------------------------- #
class CategoricalEncoder:
    """Label-encodes categorical columns, fitting exclusively on training data."""

    UNSEEN_CODE = -1

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._mappings: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Learn category -> integer-code mappings from the training set only."""
        for col in self.columns:
            categories = sorted(df[col].dropna().unique())
            self._mappings[col] = {category: code for code, category in enumerate(categories)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously-fitted mappings; unseen categories map to `UNSEEN_CODE`."""
        df = df.copy()
        for col in self.columns:
            if col not in self._mappings:
                raise RuntimeError(f"Encoder for '{col}' has not been fitted yet.")
            df[f"{col}_encoded"] = (
                df[col].map(self._mappings[col]).fillna(self.UNSEEN_CODE).astype(int)
            )
        return df


# --------------------------------------------------------------------------- #
# 5. Temporal validation split — expanding window, no shuffling
# --------------------------------------------------------------------------- #
class TemporalSplitter:
    """Splits the panel by year (expanding-window), never by random sampling."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train (<= train_end_year) and validation (val range).

        Args:
            df: Feature-engineered long-format panel.

        Returns:
            (train_df, val_df) tuple, both still sorted by Country/Year.
        """
        cfg = self.config
        train_df = df.loc[df[cfg.year_col] <= cfg.train_end_year].copy()
        val_df = df.loc[df[cfg.year_col].between(cfg.val_start_year, cfg.val_end_year)].copy()
        logger.info(
            "Temporal split -> train: %s rows (<=%s) | validation: %s rows (%s-%s)",
            len(train_df), cfg.train_end_year, len(val_df), cfg.val_start_year, cfg.val_end_year,
        )
        return train_df, val_df


# --------------------------------------------------------------------------- #
# 6. Nested expanding-window CV — used only for hyperparameter tuning
# --------------------------------------------------------------------------- #
class ExpandingWindowCVSplitter:
    """Generates chronological, expanding-window folds strictly inside the train period.

    Each fold's validation block sits entirely after its own training block
    (rolling-origin evaluation), so hyperparameter tuning never touches the
    held-out 2016-2020 partition and never shuffles years — the time-series
    equivalent of `TimeSeriesSplit`, but blocked by calendar year rather than
    row position (multiple countries share the same year).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Build up to `cv_n_splits` folds, walking backwards from `train_end_year`.

        Args:
            df: The training partition only (years <= train_end_year).

        Returns:
            List of (fold_train, fold_val) tuples in chronological order.
        """
        cfg = self.config
        first_year = int(df[cfg.year_col].min())

        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for split_idx in range(cfg.cv_n_splits):
            val_end = cfg.train_end_year - split_idx * cfg.cv_val_years
            val_start = val_end - cfg.cv_val_years + 1
            train_end = val_start - 1
            if (train_end - first_year + 1) < cfg.cv_min_train_years:
                break
            fold_train = df.loc[df[cfg.year_col] <= train_end]
            fold_val = df.loc[df[cfg.year_col].between(val_start, val_end)]
            folds.append((fold_train, fold_val))

        folds.reverse()  # chronological order: earliest fold first
        logger.info("Built %s nested CV folds for hyperparameter tuning", len(folds))
        return folds


# --------------------------------------------------------------------------- #
# 7. Metrics
# --------------------------------------------------------------------------- #
def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE = sum(|actual - pred|) / sum(|actual|). The key business metric."""
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE and WAPE for a set of predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = weighted_absolute_percentage_error(np.asarray(y_true), np.asarray(y_pred))
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape}


def reconstruct_level_from_diff(lag_1: np.ndarray, diff_predictions: np.ndarray) -> np.ndarray:
    """Undo first-differencing: level_t = level_(t-1) + predicted_delta_t."""
    return np.asarray(lag_1, dtype=float) + np.asarray(diff_predictions, dtype=float)


# --------------------------------------------------------------------------- #
# 8. Model
# --------------------------------------------------------------------------- #
class DemandForecastModel:
    """Global LightGBM regressor shared across all countries (tabular panel model)."""

    #: Fallback hyperparameters, used only if Optuna tuning is skipped.
    DEFAULT_PARAMS: dict = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "subsample_freq": 1,
    }

    def __init__(
        self,
        config: PipelineConfig,
        feature_columns: list[str],
        categorical_features: list[str],
        hyperparams: Optional[dict] = None,
    ) -> None:
        self.config = config
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.hyperparams = {**self.DEFAULT_PARAMS, **(hyperparams or {})}
        self.model: Optional[lgb.LGBMRegressor] = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for col in self.categorical_features:
            X[col] = X[col].astype("category")
        return X

    def fit(self, train_df: pd.DataFrame) -> "DemandForecastModel":
        """Fit the LightGBM regressor on the training partition only.

        The model is trained on the first-differenced target
        (`diff_target_col`), not the raw level, so it never has to extrapolate
        a monotonically increasing quantity outside the training range.
        """
        cfg = self.config
        X_train = self._prepare_features(train_df)
        y_train = train_df[cfg.diff_target_col]

        self.model = lgb.LGBMRegressor(
            **self.hyperparams,
            random_state=cfg.random_state,
            verbosity=-1,
        )
        self.model.fit(X_train, y_train, categorical_feature=self.categorical_features)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the year-over-year delta (diff space) for the given feature rows.

        Callers must reconstruct the level with `reconstruct_level_from_diff`
        using each row's `lag_1` value.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self.model.predict(self._prepare_features(df))


# --------------------------------------------------------------------------- #
# 9. Hyperparameter tuning — Bayesian search (Optuna/TPE) + time-series CV
# --------------------------------------------------------------------------- #
class HyperparameterTuner:
    """Tunes LightGBM hyperparameters with Optuna, scored on expanding-window CV.

    Why this combination for this problem:
      - The search space is small/mixed (int + float + log-scale) with an
        expensive-ish, noisy objective (WAPE over multiple time folds) — Bayesian
        optimization (Tree-structured Parzen Estimator) converges to good
        regions in far fewer trials than grid/random search.
      - Grid/random search would need `KFold`/random CV to be efficient, which
        is explicitly forbidden here: shuffling years would leak future
        consumption into the training folds. Optuna is CV-search-strategy
        agnostic, so it works cleanly with the custom `ExpandingWindowCVSplitter`.
      - The held-out 2016-2020 partition is NEVER seen during tuning — only
        years <= `train_end_year` are used to build CV folds, preserving an
        honest final generalization estimate.
    """

    def __init__(self, config: PipelineConfig, feature_columns: list[str], categorical_features: list[str]) -> None:
        self.config = config
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features

    def _objective(self, trial: "optuna.Trial", folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> float:
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
            model = DemandForecastModel(self.config, self.feature_columns, self.categorical_features, hyperparams=params)
            model.fit(fold_train)
            diff_pred = model.predict(fold_val)
            level_pred = reconstruct_level_from_diff(fold_val["lag_1"], diff_pred)
            fold_wapes.append(
                weighted_absolute_percentage_error(fold_val[self.config.target_col].to_numpy(), level_pred)
            )
        return float(np.mean(fold_wapes))

    def tune(self, train_df: pd.DataFrame) -> tuple[dict, "optuna.Study"]:
        """Run the Optuna study and return the best hyperparameters found.

        Args:
            train_df: The training partition (years <= train_end_year) with
                lag/rolling/diff-target features already computed.

        Returns:
            (best_params, study) — `study` is kept around for diagnostics/plots.
        """
        folds = ExpandingWindowCVSplitter(self.config).split(train_df)
        if not folds:
            logger.warning("Not enough history to build CV folds; falling back to default hyperparameters.")
            return dict(DemandForecastModel.DEFAULT_PARAMS), None  # type: ignore[return-value]

        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lightgbm_wape_tuning")
        study.optimize(lambda trial: self._objective(trial, folds), n_trials=self.config.n_trials)

        logger.info("Hyperparameter tuning done -> best CV WAPE: %.4f", study.best_value)
        logger.info("Best hyperparameters: %s", study.best_params)
        return study.best_params, study


# --------------------------------------------------------------------------- #
# 10. Recursive multi-step forecasting (2020-2025)
# --------------------------------------------------------------------------- #
class RecursiveForecaster:
    """Projects future years by feeding each step's prediction into the next step's lags."""

    def __init__(self, config: PipelineConfig, model: DemandForecastModel) -> None:
        self.config = config
        self.model = model

    def forecast(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Recursively predict `forecast_horizon` future years per country.

        Args:
            history_df: Fully feature-engineered + encoded panel covering all
                actually observed years (used to seed each country's rolling
                window of past consumption).

        Returns:
            DataFrame [Country, Year, Predicted_Consumption] for the forecast horizon.
        """
        cfg = self.config
        max_lookback = max(cfg.lags + (cfg.rolling_window,))

        history_df = history_df.sort_values([cfg.country_col, cfg.year_col])
        recent_history = history_df.groupby(cfg.country_col).tail(max_lookback)

        working_series: dict[str, list[float]] = {
            country: group.sort_values(cfg.year_col)[cfg.target_col].tolist()
            for country, group in recent_history.groupby(cfg.country_col)
        }

        static_cols = [cfg.coffee_type_col, f"{cfg.country_col}_encoded", f"{cfg.coffee_type_col}_encoded"]
        static_lookup = history_df.drop_duplicates(subset=[cfg.country_col]).set_index(cfg.country_col)[
            static_cols
        ]

        window = cfg.rolling_window
        forecasts: list[pd.DataFrame] = []
        for step in range(cfg.forecast_horizon):
            year = cfg.forecast_start_year + step
            rows = []
            for country, series in working_series.items():
                lag_features = {f"lag_{lag}": series[-lag] for lag in cfg.lags}
                recent_window = series[-window:]
                rows.append(
                    {
                        cfg.country_col: country,
                        cfg.year_col: year,
                        **lag_features,
                        f"rolling_mean_{window}": float(np.mean(recent_window)),
                        f"rolling_std_{window}": float(np.std(recent_window, ddof=1)),
                        static_cols[0]: static_lookup.loc[country, static_cols[0]],
                        static_cols[1]: static_lookup.loc[country, static_cols[1]],
                        static_cols[2]: static_lookup.loc[country, static_cols[2]],
                    }
                )

            step_df = pd.DataFrame(rows)
            diff_predictions = self.model.predict(step_df)
            # model predicts Delta y_t; reconstruct the level so recursion feeds real consumption back in
            step_df["Predicted_Consumption"] = reconstruct_level_from_diff(step_df["lag_1"], diff_predictions)

            for country, prediction in zip(step_df[cfg.country_col], step_df["Predicted_Consumption"]):
                working_series[country].append(float(prediction))

            forecasts.append(step_df[[cfg.country_col, cfg.year_col, "Predicted_Consumption"]])

        result = pd.concat(forecasts, ignore_index=True)
        return result.sort_values([cfg.country_col, cfg.year_col]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 11. Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the full pipeline: load -> ETL -> features -> split -> tune -> fit -> evaluate -> forecast."""
    config = PipelineConfig()

    raw_df = DataLoader(config).load()
    long_df = ETLProcessor(config).transform(raw_df)
    engineered_df = TimeSeriesFeatureEngineer(config).add_lag_and_rolling_features(long_df)

    train_raw, val_raw = TemporalSplitter(config).split(engineered_df)

    encoder = CategoricalEncoder(columns=[config.country_col, config.coffee_type_col])
    encoder.fit(train_raw)  # fit strictly on train, per the no-leakage requirement
    train_df = encoder.transform(train_raw)
    val_df = encoder.transform(val_raw)
    full_encoded_df = encoder.transform(engineered_df)  # for recursive forecasting seed history

    # Note: `Year` is intentionally excluded. Tree-based models cannot
    # extrapolate a raw numeric feature beyond the range seen in training
    # (max seen = train_end_year), which previously caused the model to fall
    # back to the same leaf for every future year, flattening the forecast.
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
    logger.info("Training rows after dropping warm-up NaNs: %s", len(train_clean))
    logger.info("Validation rows: %s", len(val_clean))

    best_params, _ = HyperparameterTuner(config, feature_columns, categorical_features).tune(train_clean)

    model = DemandForecastModel(config, feature_columns, categorical_features, hyperparams=best_params)
    model.fit(train_clean)

    val_diff_predictions = model.predict(val_clean)
    val_level_predictions = reconstruct_level_from_diff(val_clean["lag_1"], val_diff_predictions)
    metrics = regression_report(val_clean[config.target_col].to_numpy(), val_level_predictions)
    logger.info(
        "Validation metrics -> MAE: %.2f | RMSE: %.2f | WAPE: %.4f",
        metrics["MAE"], metrics["RMSE"], metrics["WAPE"],
    )

    future_df = RecursiveForecaster(config, model).forecast(full_encoded_df)
    output_path = config.output_dir / "future_forecast_2020_2025.csv"
    future_df.to_csv(output_path, index=False)
    logger.info("Future forecast (2020-2025) saved to %s", output_path)

    print("\n=== Validation Metrics (2016-2019) ===")
    for name, value in metrics.items():
        print(f"{name}: {value:,.4f}")

    print("\n=== Future Forecast Preview ===")
    print(future_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
