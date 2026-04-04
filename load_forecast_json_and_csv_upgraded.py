"""
load_forecast_csv.py
====================
Azure App Service–friendly load forecasting module using **Ordinary Least Squares (OLS)** regression.

This version adds **temperature regime splitting**:
- Automatically finds a transition temperature that best splits “cold” vs “hot” regimes
- Fits two separate multiple-regression models (cold & hot) for each load type
- Applies the correct model to forecasted weather rows based on temperature

Core requirements
-----------------
- Train on historical data from 2023–2025 (inclusive)
- Single CSV training file: both Residential and CI loads are columns in the same file,
  sharing the same independent variables.
- Predictors:
    * Intercept (explicit)
    * Temperature
    * Wind speed
    * Day-of-week one-hot encoded (drop one category)
- Outputs:
    * CSV (required)
    * JSON (optional)
- Optional testing:
    * If `data_for_testing.csv` exists, evaluate + write CSV with actual/predicted/errors and metrics.

Azure notes
-----------
- Uses relative paths from the directory of this file.
- Logs to console + a log file in the same directory.
- Importable and callable from `app.py` or similar entry point.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Paths (Azure-friendly)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Forecasted Output"
VALIDATION_DIR = BASE_DIR / "Validation"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = {
    "Toronto": "Toronto_his_load.csv",
    "Ottawa": "Ottawa_his_load.csv",
    "Hamilton": "Hamilton_his_load.csv",
    "Mississauga": "Mississauga_his_load.csv",
    "Brampton": "Brampton_his_load.csv",
    "London": "London_his_load.csv"
}

DEFAULT_LOG_FILE = "load_forecast_log.txt"

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("load_forecast_csv")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(str(BASE_DIR / "Log" / DEFAULT_LOG_FILE), mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


# -----------------------------
# Canonical column names (expected in your CSV)
# -----------------------------
DATE_COL = "Date"
DOW_COL = "Day of the week"
TEMP_COL = "temperature_2m_mean (°C)"
WIND_COL = "wind_speed_10m_mean (km/h)"

RES_TARGET_COL = "Total Residential Consumption"
CI_TARGET_COL = "Total CI Consumption"


# -----------------------------
# OLS (no ML libraries)
# -----------------------------
def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form OLS:
        beta = (X'X)^(-1) X'y

    Uses pseudo-inverse for numerical stability (still OLS).
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def ols_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta


# -----------------------------
# Feature engineering
# -----------------------------
def ensure_datetime(df: pd.DataFrame) -> None:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])


def ensure_dow(df: pd.DataFrame) -> None:
    """
    Ensure day-of-week exists:
    - If missing, compute from Date (Mon=1 ... Sun=7)
    """
    if DOW_COL not in df.columns:
        df[DOW_COL] = pd.to_datetime(df[DATE_COL]).dt.dayofweek + 1
    df[DOW_COL] = pd.to_numeric(df[DOW_COL], errors="coerce")


def build_design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Design matrix X:
      X = [1, temp, wind, dow_2, dow_3, ..., dow_7]
    Drops dow_1 to avoid perfect multicollinearity.

    We *force* dummy columns to exist so training + forecasting align.
    """
    d = df.copy()

    # Intercept
    intercept = np.ones((len(d), 1), dtype=float)

    # Predictors
    temp = pd.to_numeric(d[TEMP_COL], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    wind = pd.to_numeric(d[WIND_COL], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)

    # DOW dummies (force 1..7)
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
    """
    Two-regime model:
      - If temperature <= transition_temp: use cold_model
      - Else: use hot_model
    """
    transition_temp: Optional[float]
    cold_model: OLSModel
    hot_model: OLSModel

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.transition_temp is None:
            # Fallback: single model (cold_model == hot_model)
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
# Data loading & filtering
# -----------------------------
def load_historical_csv(csv_path: Path) -> pd.DataFrame:
    """
    Loads a single CSV containing:
      - Date, temp, wind, (optional DOW)
      - Residential target column
      - CI target column
    """
    logger.info(f"Loading historical CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if DATE_COL not in df.columns:
        raise ValueError(f"Historical CSV must contain '{DATE_COL}'. Columns: {list(df.columns)}")

    ensure_datetime(df)
    ensure_dow(df)

    # Numeric coercion for predictors
    df[TEMP_COL] = pd.to_numeric(df.get(TEMP_COL), errors="coerce")
    df[WIND_COL] = pd.to_numeric(df.get(WIND_COL), errors="coerce")

    return df


# -----------------------------
# Transition temperature search
# -----------------------------
def find_transition_temperature(
    df: pd.DataFrame,
    target_col: str,
    min_points_per_segment: int = 30,
    num_candidates: int = 40,
) -> Optional[float]:

    # Only rows where target is available
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

        # Cold fit
        Xc, fn = build_design_matrix(cold)
        yc = pd.to_numeric(cold[target_col], errors="coerce").to_numpy(dtype=float)
        bc = ols_fit(Xc, yc)
        sse_c = float(np.sum((yc - (Xc @ bc)) ** 2))

        # Hot fit
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
# Training (segmented)
# -----------------------------
def train_segmented_model(
    df: pd.DataFrame,
    target_col: str,
    min_points_per_segment: int = 30,
) -> SegmentedOLSModel:
    """
    Trains a segmented OLS model for a target column.

    If a valid transition temperature cannot be found (insufficient data),
    falls back to a single OLS model (transition_temp=None).
    """
    # Clean rows for this target
    d = df[df[target_col].notna()].copy()
    d = d[d[TEMP_COL].notna() & d[WIND_COL].notna() & d[DOW_COL].notna()].copy()

    if d.empty:
        raise ValueError(
            f"No training rows available for '{target_col}' within 2023–2025. "
            f"Check that '{target_col}' is populated for those years."
        )

    T = find_transition_temperature(d, target_col, min_points_per_segment=min_points_per_segment)

    if T is None:
        # Fallback: single model
        X, feature_names = build_design_matrix(d)
        y = pd.to_numeric(d[target_col], errors="coerce").to_numpy(dtype=float)
        beta = ols_fit(X, y)
        base = OLSModel(beta=beta, feature_names=feature_names)
        logger.warning(f"Could not find valid transition temperature for '{target_col}'. Using single OLS model.")
        return SegmentedOLSModel(transition_temp=None, cold_model=base, hot_model=base)

    # Split and train
    cold = d[pd.to_numeric(d[TEMP_COL], errors="coerce") <= T]
    hot = d[pd.to_numeric(d[TEMP_COL], errors="coerce") > T]

    if len(cold) < min_points_per_segment or len(hot) < min_points_per_segment:
        # Very unlikely if T came from search, but guard anyway
        X, feature_names = build_design_matrix(d)
        y = pd.to_numeric(d[target_col], errors="coerce").to_numpy(dtype=float)
        beta = ols_fit(X, y)
        base = OLSModel(beta=beta, feature_names=feature_names)
        logger.warning(f"Transition temperature found but split too small for '{target_col}'. Using single OLS model.")
        return SegmentedOLSModel(transition_temp=None, cold_model=base, hot_model=base)

    # Cold model
    Xc, fn = build_design_matrix(cold)
    yc = pd.to_numeric(cold[target_col], errors="coerce").to_numpy(dtype=float)
    bc = ols_fit(Xc, yc)
    cold_model = OLSModel(beta=bc, feature_names=fn)

    # Hot model
    Xh, _ = build_design_matrix(hot)
    yh = pd.to_numeric(hot[target_col], errors="coerce").to_numpy(dtype=float)
    bh = ols_fit(Xh, yh)
    hot_model = OLSModel(beta=bh, feature_names=fn)

    logger.info(f"Transition temperature for '{target_col}': {T:.2f} °C | cold n={len(cold):,} hot n={len(hot):,}")
    return SegmentedOLSModel(transition_temp=T, cold_model=cold_model, hot_model=hot_model)


def train_models_from_historical_csv(
    csv_path: Path,
    min_points_per_segment: int = 30,
) -> Tuple[SegmentedOLSModel, SegmentedOLSModel]:
    """
    Loads historical CSV, filters training window, trains segmented models:
      - Residential segmented model
      - CI segmented model
    """
    full_df = load_historical_csv(csv_path)
    train_df = full_df[(full_df[DATE_COL] >= '2023-01-01') & (full_df[DATE_COL] <= '2025-06-30')].copy()
    train_df = train_df.dropna(subset=[RES_TARGET_COL, CI_TARGET_COL, TEMP_COL, WIND_COL])

    res_model = train_segmented_model(train_df, RES_TARGET_COL, min_points_per_segment=min_points_per_segment)
    ci_model = train_segmented_model(train_df, CI_TARGET_COL, min_points_per_segment=min_points_per_segment)

    return res_model, ci_model

def try_get_forecast_df_from_weather_module() -> Optional[pd.DataFrame]:
    """
    Attempts to import and call the fetch_and_process_forecast() function 
    specifically from the get_weather_forecast_json_and_csv module.
    """
    module_name = "get_weather_forecast_json_and_csv"
    
    try:
        mod = importlib.import_module(module_name)
        
        if hasattr(mod, "fetch_and_process_forecast"):
            logger.info(f"Using weather module: {module_name}.fetch_and_process_forecast()")
            df = mod.fetch_and_process_forecast()
            
            if df is None:
                logger.error(f"Module {module_name} returned None.")
                return None
                
            # Normalize expected 'date' column
            if "date" not in df.columns and "time" in df.columns:
                df = df.rename(columns={"time": "date"})
            
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                
            return df
        else:
            logger.error(f"Module {module_name} is missing 'fetch_and_process_forecast' function.")
            
    except Exception as e:
        logger.error(f"Failed to import or execute {module_name}: {e}")
        
    return None


def load_forecast_weather_from_csv(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading forecast weather CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def normalize_forecast_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes forecast weather data into the canonical column names.
    Required: date + temperature + wind
    """
    d = df.copy()

    # Date
    if "date" not in d.columns and "Date" in d.columns:
        d = d.rename(columns={"Date": "date"})
    if "date" not in d.columns:
        raise ValueError(f"Forecast weather must contain 'date' column. Columns: {list(d.columns)}")

    d["date"] = pd.to_datetime(d["date"])
    d = d.rename(columns={"date": DATE_COL})

    # Temperature
    if TEMP_COL not in d.columns:
        for alt in ["temperature_2m_mean", "temperature", "temp"]:
            if alt in d.columns:
                d = d.rename(columns={alt: TEMP_COL})
                break

    # Wind
    if WIND_COL not in d.columns:
        for alt in ["wind_speed_10m_mean", "wind_speed", "wind"]:
            if alt in d.columns:
                d = d.rename(columns={alt: WIND_COL})
                break

    if TEMP_COL not in d.columns or WIND_COL not in d.columns:
        raise ValueError(
            f"Forecast weather missing predictors. Need '{TEMP_COL}' and '{WIND_COL}'. "
            f"Columns: {list(d.columns)}"
        )

    d[TEMP_COL] = pd.to_numeric(d[TEMP_COL], errors="coerce")
    d[WIND_COL] = pd.to_numeric(d[WIND_COL], errors="coerce")

    ensure_dow(d)

    # Drop rows with missing predictors
    d = d[d[TEMP_COL].notna() & d[WIND_COL].notna() & d[DOW_COL].notna()].copy()
    return d


# -----------------------------
# Forecast execution + output
# -----------------------------
def forecast_daily_load(
    res_model: SegmentedOLSModel,
    ci_model: SegmentedOLSModel,
    forecast_weather_df: pd.DataFrame,
) -> pd.DataFrame:
    fw = normalize_forecast_weather(forecast_weather_df)

    res_pred = res_model.predict(fw)
    ci_pred = ci_model.predict(fw)

    out = pd.DataFrame({
        "date": fw[DATE_COL].dt.date.astype(str),
        "forecast_residential_load": res_pred,
        "forecast_ci_load": ci_pred,
        # Useful for debugging/QA:
        "temperature": fw[TEMP_COL].to_numpy(dtype=float),
        "wind_speed": fw[WIND_COL].to_numpy(dtype=float),
    })
    return out


def save_forecast_outputs(
    forecast_df: pd.DataFrame,
    csv_filename: str,
    json_filename: str,
) -> Tuple[Path, Optional[Path]]:
    # --- Define and create the Output subfolder ---
    output_dir = BASE_DIR / "Forecasted Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update paths to point inside the Output folder
    csv_path = output_dir / csv_filename
    forecast_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved forecast CSV: {csv_path}")

    json_path = None
    if json_filename:
        json_path = output_dir / json_filename
        safe_df = forecast_df.where(pd.notna(forecast_df), None)
        json_path.write_text(safe_df.to_json(orient="records", indent=2), encoding="utf-8")
        logger.info(f"Saved forecast JSON: {json_path}")

    return csv_path, json_path

# -----------------------------
# Testing / Validation
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
    output_csv: str,
    hist_path: pd.DataFrame,
    city: str,
    start_date: str,
    end_date: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Optional evaluation. If the file doesn't exist, returns None (no-op).
    """
    # --- Validate date range ---
    if start_date < "2023-01-01" or end_date > "2025-11-30":
        raise ValueError("Date range must be within 2023-01-01 and 2025-11-30")

    if start_date > end_date:
        raise ValueError("Start date must be before end date")
    
    # --- Define and create the Validation subfolder ---
    validation_dir = BASE_DIR / "Validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data for the specific testing range
    df = pd.read_csv(hist_path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Test CSV must contain '{DATE_COL}'. Columns: {list(df.columns)}")
    
    ensure_datetime(df)
    ensure_dow(df)
    
    mask = (df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)
    test_df = df.loc[mask].copy()

    if test_df.empty:
        logger.warning(f"No historical data found for validation range in {city}.")
        return {}


    # Coerce predictors
    test_df[TEMP_COL] = pd.to_numeric(test_df.get(TEMP_COL), errors="coerce")
    test_df[WIND_COL] = pd.to_numeric(test_df.get(WIND_COL), errors="coerce")

    # Predict where we have actuals
    out_rows = []

    metrics: Dict[str, Dict[str, float]] = {}

    # Residential
    if RES_TARGET_COL in test_df.columns:
        res_eval = test_df[test_df[RES_TARGET_COL].notna() & test_df[TEMP_COL].notna() & test_df[WIND_COL].notna()].copy()
        if not res_eval.empty:
            res_true = pd.to_numeric(res_eval[RES_TARGET_COL], errors="coerce").to_numpy(dtype=float)
            res_pred = res_model.predict(res_eval)
            metrics["residential"] = regression_metrics(res_true, res_pred)
            res_block = pd.DataFrame({
                "date": res_eval[DATE_COL].dt.date.astype(str),
                "res_actual": res_true,
                "res_predicted": res_pred,
                "res_error": abs(res_true - res_pred),
                "res_error_percentage": abs(res_true - res_pred) / res_true * 100,
            })
        else:
            res_block = pd.DataFrame(columns=["date", "res_actual", "res_predicted", "res_error"])
            metrics["residential"] = {"RMSE": float("nan"), "RMSPE": float("nan")}
    else:
        res_block = pd.DataFrame(columns=["date", "res_actual", "res_predicted", "res_error"])


    # CI
    if CI_TARGET_COL in test_df.columns:
        ci_eval = test_df[test_df[CI_TARGET_COL].notna() & test_df[TEMP_COL].notna() & test_df[WIND_COL].notna()].copy()
        if not ci_eval.empty:
            ci_true = pd.to_numeric(ci_eval[CI_TARGET_COL], errors="coerce").to_numpy(dtype=float)
            ci_pred = ci_model.predict(ci_eval)
            metrics["ci"] = regression_metrics(ci_true, ci_pred)
            ci_block = pd.DataFrame({
                "date": ci_eval[DATE_COL].dt.date.astype(str),
                "ci_actual": ci_true,
                "ci_predicted": ci_pred,
                "ci_error": abs(ci_true - ci_pred),
                "ci_error_percentage": abs(ci_true - ci_pred) / ci_true * 100,
            })
        else:
            ci_block = pd.DataFrame(columns=["date", "ci_actual", "ci_predicted", "ci_error"])
            metrics["ci"] = {"RMSE": float("nan"), "RMSPE": float("nan")}
    else:
        ci_block = pd.DataFrame(columns=["date", "ci_actual", "ci_predicted", "ci_error"])

    # Merge on date if possible, else concatenate
    if not res_block.empty and not ci_block.empty:
        comparison_df = pd.merge(res_block, ci_block, on="date", how="outer")
    elif not res_block.empty:
        comparison_df = res_block
    else:
        comparison_df = ci_block

    out_path = validation_dir / output_csv
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Comparison']
        

        bold_format = workbook.add_format({'bold': True})

        # Write Metrics to the same sheet
        worksheet.write('L1', 'Location', bold_format)
        worksheet.write('M1', city)
        worksheet.write('L3', 'Metric', bold_format)
        worksheet.write('M3', 'Residential', bold_format)
        worksheet.write('N3', 'CI', bold_format)
        worksheet.write('L4', 'RMSE', bold_format)
        worksheet.write('M4', metrics['residential']['RMSE'])
        worksheet.write('N4', metrics['ci']['RMSE'])
        worksheet.write('L5', 'RMSPE (%)', bold_format)
        worksheet.write('M5', metrics['residential']['RMSPE'])
        worksheet.write('N5', metrics['ci']['RMSPE'])

        # Create Line Graphs
        for i, target in enumerate(['Residential', 'CI']):
            chart = workbook.add_chart({'type': 'line'})
            actual_col = 1 if target == 'Residential' else 5
            pred_col = 2 if target == 'Residential' else 6
            
            chart.add_series({
                'name': f'Actual {target}',
                'categories': ['Comparison', 1, 0, len(comparison_df), 0],
                'values': ['Comparison', 1, actual_col, len(comparison_df), actual_col],
            })
            chart.add_series({
                'name': f'Predicted {target}',
                'values': ['Comparison', 1, pred_col, len(comparison_df), pred_col],
            })
            chart.set_title({'name': f'{city} {target} Load Comparison'})
            worksheet.insert_chart('G8' if i == 0 else 'G23', chart)

                # Auto adjust column width
        for i, col in enumerate(comparison_df.columns):
            max_len = max(
                comparison_df[col].astype(str).map(len).max() + 3,
                len(col)
            ) + 2
            worksheet.set_column(i, i, max_len)

    logger.info(f"Test results for {city} saved to: {out_path}")
    return metrics


# -----------------------------
# End-to-end pipeline
# -----------------------------
def run_load_forecast_pipeline(
    run_test_if_present: bool = True,
    min_points_per_segment: int = 30,
) -> Tuple[pd.DataFrame, Path, Optional[Path], Optional[Dict[str, Dict[str, float]]]]:
    """
    End-to-end:
      1) Train segmented Residential + CI OLS models from historical CSV (2023–2025)
      2) Get forecast weather from weather module; if unavailable, fall back to CSV
      3) Predict daily loads, selecting cold/hot model by temperature
      4) Save forecast outputs
      5) Optionally test if data_for_testing.csv exists
    """

    # 1) Ingest forecast weather
    logger.info("Triggering weather forecast script...")
    subprocess.run(["python", str(BASE_DIR / "get_weather_forecast_json_and_csv.py")], check=True)

    for city, csv_name in LOCATIONS.items():
        logger.info(f"Processing region: {city}")
        hist_path = DATA_DIR / csv_name
        weather_fc_path = DATA_DIR / f"{city}_forecast_daily_weather.csv"
        
        if not hist_path.exists():
            logger.warning(f"File {csv_name} not found. Skipping.")
            continue

        forecast_df = load_forecast_weather_from_csv(weather_fc_path)

        output_csv_filename = f"{city}_forecasted_daily_load.csv"
        output_json_filename = f"{city}_forecasted_daily_load.json"
        test_csv_filename = f"{city}_data_evaluation.xlsx"

        # 2) Train segmented models (2023-01-01 to 2025-06-30)
        res_model, ci_model = train_models_from_historical_csv(
            hist_path,
            min_points_per_segment=min_points_per_segment,
        )

        # 3) Forecast
        out_df = forecast_daily_load(res_model, ci_model, forecast_df)

        # 4) Save
        csv_path, json_path = save_forecast_outputs(out_df, output_csv_filename, output_json_filename)

        # 5) Evaluation (2025-01-01 to 2025-11-30)
        metrics = None
        if run_test_if_present:
            metrics = perform_validation(res_model, ci_model, test_csv_filename, hist_path, city, "2025-01-01", "2025-11-30")

    return out_df, csv_path, json_path, metrics


if __name__ == "__main__":
    logger.info("Running load_forecast_csv.py directly...")
    df_out, csv_path, json_path, metrics = run_load_forecast_pipeline()
    logger.info(f"Done. Rows forecasted: {len(df_out)} | CSV: {csv_path}")
