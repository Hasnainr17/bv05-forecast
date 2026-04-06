"""
load_forecast_json_and_csv.py
=============================
City-based Azure App Service–friendly load forecasting module using Ordinary Least Squares (OLS)
with temperature regime splitting.

Flow:
    selected city -> fetch weather -> train city model -> forecast 16 days

Expected usage:
    run_load_forecast_pipeline(city="Toronto")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from weather1 import fetch_and_process_forecast


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Forecasted Output"
VALIDATION_DIR = BASE_DIR / "Validation"
LOG_DIR = BASE_DIR / "Log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = {
    "Toronto": "toronto_his_load.csv",
    "Ottawa": "ottawa_his_load.csv",
    "Hamilton": "hamilton_his_load.csv",
    "Mississauga": "mississauga_his_load.csv",
    "Brampton": "brampton_his_load.csv",
    "London": "london_his_load.csv",
}

DEFAULT_LOG_FILE = "load_forecast_log.txt"


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("load_forecast_csv")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

log_path = LOG_DIR / DEFAULT_LOG_FILE

file_handler = logging.FileHandler(str(log_path), mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


# -----------------------------
# Canonical column names
# -----------------------------
DATE_COL = "Date"
DOW_COL = "Day of the week"
TEMP_COL = "temperature_2m_mean (°C)"
WIND_COL = "wind_speed_10m_mean (km/h)"

RES_TARGET_COL = "Total Residential Consumption"
CI_TARGET_COL = "Total CI Consumption"


# -----------------------------
# OLS helpers
# -----------------------------
def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def ols_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta


# -----------------------------
# Feature engineering
# -----------------------------
def ensure_datetime(df: pd.DataFrame) -> None:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])


def ensure_dow(df: pd.DataFrame) -> None:
    if DOW_COL not in df.columns:
        df[DOW_COL] = pd.to_datetime(df[DATE_COL]).dt.dayofweek + 1
    df[DOW_COL] = pd.to_numeric(df[DOW_COL], errors="coerce")


def build_design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    d = df.copy()

    intercept = np.ones((len(d), 1), dtype=float)
    temp = pd.to_numeric(d[TEMP_COL], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    wind = pd.to_numeric(d[WIND_COL], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)

    dow = d[DOW_COL].astype("Int64")
    dummies = pd.get_dummies(dow, prefix="dow")

    for k in range(1, 8):
        col = f"dow_{k}"
        if col not in dummies.columns:
            dummies[col] = 0

    dummies = dummies[[f"dow_{k}" for k in range(1, 8)]].drop(columns=["dow_1"])

    X = np.hstack([intercept, temp, wind, dummies.to_numpy(dtype=float)])
    feature_names = ["intercept", "temp", "wind"] + list(dummies.columns)
    return X, feature_names


# -----------------------------
# Models
# -----------------------------
@dataclass
class OLSModel:
    beta: np.ndarray
    feature_names: list

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_design_matrix(df)
        return ols_predict(X, self.beta)


@dataclass
class SegmentedOLSModel:
    transition_temp: Optional[float]
    cold_model: OLSModel
    hot_model: OLSModel

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.transition_temp is None:
            return self.cold_model.predict(df)

        temps = pd.to_numeric(df[TEMP_COL], errors="coerce")
        cold_mask = temps <= self.transition_temp
        hot_mask = temps > self.transition_temp

        preds = np.full(len(df), np.nan, dtype=float)

        if cold_mask.any():
            preds[cold_mask.values] = self.cold_model.predict(df.loc[cold_mask])
        if hot_mask.any():
            preds[hot_mask.values] = self.hot_model.predict(df.loc[hot_mask])

        return preds


# -----------------------------
# Historical data
# -----------------------------
def load_historical_csv(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading historical CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if DATE_COL not in df.columns:
        raise ValueError(f"Historical CSV must contain '{DATE_COL}'. Columns: {list(df.columns)}")

    ensure_datetime(df)
    ensure_dow(df)

    df[TEMP_COL] = pd.to_numeric(df.get(TEMP_COL), errors="coerce")
    df[WIND_COL] = pd.to_numeric(df.get(WIND_COL), errors="coerce")

    return df


# -----------------------------
# Transition search
# -----------------------------
def find_transition_temperature(
    df: pd.DataFrame,
    target_col: str,
    min_points_per_segment: int = 30,
    num_candidates: int = 40,
) -> Optional[float]:
    d = df[df[target_col].notna()].copy()
    if len(d) < 2 * min_points_per_segment:
        return None

    temps = pd.to_numeric(d[TEMP_COL], errors="coerce")
    d = d[temps.notna()].copy()
    temps = pd.to_numeric(d[TEMP_COL], errors="coerce")

    if len(d) < 2 * min_points_per_segment:
        return None

    t_min = float(temps.quantile(0.10))
    t_max = float(temps.quantile(0.90))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min >= t_max:
        return None

    candidates = np.linspace(t_min, t_max, num_candidates)

    best_t: Optional[float] = None
    best_sse = float("inf")

    for T in candidates:
        cold = d[temps <= T]
        hot = d[temps > T]

        if len(cold) < min_points_per_segment or len(hot) < min_points_per_segment:
            continue

        Xc, _ = build_design_matrix(cold)
        yc = pd.to_numeric(cold[target_col], errors="coerce").to_numpy(dtype=float)
        bc = ols_fit(Xc, yc)
        sse_c = float(np.sum((yc - (Xc @ bc)) ** 2))

        Xh, _ = build_design_matrix(hot)
        yh = pd.to_numeric(hot[target_col], errors="coerce").to_numpy(dtype=float)
        bh = ols_fit(Xh, yh)
        sse_h = float(np.sum((yh - (Xh @ bh)) ** 2))

        total = sse_c + sse_h
        if total < best_sse:
            best_sse = total
            best_t = float(T)

    return best_t


# -----------------------------
# Training
# -----------------------------
def train_segmented_model(
    df: pd.DataFrame,
    target_col: str,
    min_points_per_segment: int = 30,
) -> SegmentedOLSModel:
    d = df[df[target_col].notna()].copy()
    d = d[d[TEMP_COL].notna() & d[WIND_COL].notna() & d[DOW_COL].notna()].copy()

    if d.empty:
        raise ValueError(f"No training rows available for '{target_col}' within training window.")

    T = find_transition_temperature(d, target_col, min_points_per_segment=min_points_per_segment)

    if T is None:
        X, feature_names = build_design_matrix(d)
        y = pd.to_numeric(d[target_col], errors="coerce").to_numpy(dtype=float)
        beta = ols_fit(X, y)
        base = OLSModel(beta=beta, feature_names=feature_names)
        logger.warning(f"Could not find valid transition temperature for '{target_col}'. Using single OLS model.")
        return SegmentedOLSModel(transition_temp=None, cold_model=base, hot_model=base)

    cold = d[pd.to_numeric(d[TEMP_COL], errors="coerce") <= T]
    hot = d[pd.to_numeric(d[TEMP_COL], errors="coerce") > T]

    if len(cold) < min_points_per_segment or len(hot) < min_points_per_segment:
        X, feature_names = build_design_matrix(d)
        y = pd.to_numeric(d[target_col], errors="coerce").to_numpy(dtype=float)
        beta = ols_fit(X, y)
        base = OLSModel(beta=beta, feature_names=feature_names)
        logger.warning(f"Transition temperature split too small for '{target_col}'. Using single OLS model.")
        return SegmentedOLSModel(transition_temp=None, cold_model=base, hot_model=base)

    Xc, fn = build_design_matrix(cold)
    yc = pd.to_numeric(cold[target_col], errors="coerce").to_numpy(dtype=float)
    bc = ols_fit(Xc, yc)
    cold_model = OLSModel(beta=bc, feature_names=fn)

    Xh, _ = build_design_matrix(hot)
    yh = pd.to_numeric(hot[target_col], errors="coerce").to_numpy(dtype=float)
    bh = ols_fit(Xh, yh)
    hot_model = OLSModel(beta=bh, feature_names=fn)

    logger.info(
        f"Transition temperature for '{target_col}': {T:.2f} °C | cold n={len(cold):,} hot n={len(hot):,}"
    )
    return SegmentedOLSModel(transition_temp=T, cold_model=cold_model, hot_model=hot_model)


def train_models_from_historical_csv(
    csv_path: Path,
    min_points_per_segment: int = 30,
) -> Tuple[SegmentedOLSModel, SegmentedOLSModel]:
    full_df = load_historical_csv(csv_path)
    train_df = full_df[(full_df[DATE_COL] >= "2023-01-01") & (full_df[DATE_COL] <= "2025-06-30")].copy()
    train_df = train_df.dropna(subset=[RES_TARGET_COL, CI_TARGET_COL, TEMP_COL, WIND_COL])

    res_model = train_segmented_model(train_df, RES_TARGET_COL, min_points_per_segment=min_points_per_segment)
    ci_model = train_segmented_model(train_df, CI_TARGET_COL, min_points_per_segment=min_points_per_segment)

    return res_model, ci_model


# -----------------------------
# Forecast weather normalization
# -----------------------------
def normalize_forecast_weather(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    rename_map = {}

    if "date" in d.columns:
        rename_map["date"] = DATE_COL

    if "temperature" in d.columns:
        rename_map["temperature"] = TEMP_COL
    elif "temperature_2m_mean (°C)" in d.columns:
        rename_map["temperature_2m_mean (°C)"] = TEMP_COL

    if "wind_speed" in d.columns:
        rename_map["wind_speed"] = WIND_COL
    elif "wind_speed_10m_mean (km/h)" in d.columns:
        rename_map["wind_speed_10m_mean (km/h)"] = WIND_COL

    d = d.rename(columns=rename_map)

    if DATE_COL not in d.columns:
        raise ValueError(f"Forecast weather must contain a date column. Columns: {list(d.columns)}")

    if TEMP_COL not in d.columns or WIND_COL not in d.columns:
        raise ValueError(
            f"Forecast weather missing required predictors '{TEMP_COL}' and '{WIND_COL}'. Columns: {list(d.columns)}"
        )

    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    d[TEMP_COL] = pd.to_numeric(d[TEMP_COL], errors="coerce")
    d[WIND_COL] = pd.to_numeric(d[WIND_COL], errors="coerce")

    ensure_dow(d)

    d = d[d[TEMP_COL].notna() & d[WIND_COL].notna() & d[DOW_COL].notna()].copy()
    return d


# -----------------------------
# Forecast generation
# -----------------------------
def forecast_daily_load(
    res_model: SegmentedOLSModel,
    ci_model: SegmentedOLSModel,
    forecast_weather_df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    fw = normalize_forecast_weather(forecast_weather_df)

    res_pred = res_model.predict(fw)
    ci_pred = ci_model.predict(fw)

    out = pd.DataFrame({
        "date": fw[DATE_COL].dt.date.astype(str),
        "forecast_residential_load": res_pred,
        "forecast_ci_load": ci_pred,
        "temperature": fw[TEMP_COL].to_numpy(dtype=float),
        "wind_speed": fw[WIND_COL].to_numpy(dtype=float),
        "city": city,
    })
    return out


def save_forecast_outputs(
    forecast_df: pd.DataFrame,
    city: str,
) -> Tuple[Path, Optional[Path]]:
    csv_filename = f"{city.lower()}_forecast_daily_load.csv"
    json_filename = f"{city.lower()}_forecast_daily_load.json"

    csv_path = OUTPUT_DIR / csv_filename
    forecast_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved forecast CSV: {csv_path}")

    json_path = OUTPUT_DIR / json_filename
    safe_df = forecast_df.where(pd.notna(forecast_df), None)
    json_path.write_text(safe_df.to_json(orient="records", indent=2), encoding="utf-8")
    logger.info(f"Saved forecast JSON: {json_path}")

    return csv_path, json_path


# -----------------------------
# Validation
# -----------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    yt, yp = y_true[mask], y_pred[mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    rmspe = float(np.sqrt(np.mean(((yt - yp) / yt) ** 2)) * 100)
    return {"RMSE": rmse, "RMSPE": rmspe}


def perform_validation(
    res_model: SegmentedOLSModel,
    ci_model: SegmentedOLSModel,
    hist_path: Path,
    city: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    df = pd.read_csv(hist_path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Test CSV must contain '{DATE_COL}'. Columns: {list(df.columns)}")

    ensure_datetime(df)
    ensure_dow(df)

    mask = (df[DATE_COL] >= "2025-01-01") & (df[DATE_COL] <= "2025-11-30")
    test_df = df.loc[mask].copy()

    if test_df.empty:
        logger.warning(f"No historical validation data found for {city}.")
        return {}

    test_df[TEMP_COL] = pd.to_numeric(test_df.get(TEMP_COL), errors="coerce")
    test_df[WIND_COL] = pd.to_numeric(test_df.get(WIND_COL), errors="coerce")

    metrics: Dict[str, Dict[str, float]] = {}

    if RES_TARGET_COL in test_df.columns:
        res_eval = test_df[
            test_df[RES_TARGET_COL].notna() & test_df[TEMP_COL].notna() & test_df[WIND_COL].notna()
        ].copy()
        if not res_eval.empty:
            res_true = pd.to_numeric(res_eval[RES_TARGET_COL], errors="coerce").to_numpy(dtype=float)
            res_pred = res_model.predict(res_eval)
            metrics["residential"] = regression_metrics(res_true, res_pred)
        else:
            metrics["residential"] = {"RMSE": float("nan"), "RMSPE": float("nan")}
    else:
        metrics["residential"] = {"RMSE": float("nan"), "RMSPE": float("nan")}

    if CI_TARGET_COL in test_df.columns:
        ci_eval = test_df[
            test_df[CI_TARGET_COL].notna() & test_df[TEMP_COL].notna() & test_df[WIND_COL].notna()
        ].copy()
        if not ci_eval.empty:
            ci_true = pd.to_numeric(ci_eval[CI_TARGET_COL], errors="coerce").to_numpy(dtype=float)
            ci_pred = ci_model.predict(ci_eval)
            metrics["ci"] = regression_metrics(ci_true, ci_pred)
        else:
            metrics["ci"] = {"RMSE": float("nan"), "RMSPE": float("nan")}
    else:
        metrics["ci"] = {"RMSE": float("nan"), "RMSPE": float("nan")}

    logger.info(f"Validation metrics for {city}: {metrics}")
    return metrics


# -----------------------------
# Main pipeline
# -----------------------------
def run_load_forecast_pipeline(
    city: str = "Toronto",
    run_test_if_present: bool = True,
    min_points_per_segment: int = 30,
) -> Tuple[pd.DataFrame, Path, Optional[Path], Optional[Dict[str, Dict[str, float]]]]:
    """
    End-to-end for one selected city:
      1) Fetch forecast weather
      2) Train city-specific segmented models from historical CSV
      3) Forecast residential + C&I daily load
      4) Save outputs
      5) Optionally compute validation metrics
    """
    if city not in LOCATIONS:
        raise ValueError(f"Unsupported city: {city}")

    logger.info(f"Running load forecast pipeline for {city}")

    hist_path = DATA_DIR / LOCATIONS[city]
    if not hist_path.exists():
        raise FileNotFoundError(f"Historical CSV not found for {city}: {hist_path}")

    # weather1.py currently does not take a city argument
    forecast_weather_df = fetch_and_process_forecast()
    if forecast_weather_df is None or forecast_weather_df.empty:
        raise FileNotFoundError(f"No forecast weather data generated for {city}")

    res_model, ci_model = train_models_from_historical_csv(
        hist_path,
        min_points_per_segment=min_points_per_segment,
    )

    out_df = forecast_daily_load(res_model, ci_model, forecast_weather_df, city)
    csv_path, json_path = save_forecast_outputs(out_df, city)

    metrics = None
    if run_test_if_present:
        metrics = perform_validation(res_model, ci_model, hist_path, city)

    return out_df, csv_path, json_path, metrics


if __name__ == "__main__":
    logger.info("Running load_forecast_json_and_csv.py directly...")
    df_out, csv_path, json_path, metrics = run_load_forecast_pipeline(city="Toronto")
    logger.info(f"Done. Rows forecasted: {len(df_out)} | CSV: {csv_path}")
    
    