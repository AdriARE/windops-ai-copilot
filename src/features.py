# src/features.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import logging

# Third-party
import numpy as np
import pandas as pd

# Local
from OLD.config_old import ROLLING_WINDOW_24H

logger = logging.getLogger(__name__)

# ===============================
# VALIDATION
# ===============================

REQUIRED_COLUMNS = {
    "turbine_id",
    "timestamp",
    "power_kw",
    "expected_power_kw",
    "power_gap_kw",
    "power_gap_pct",
    "nacelle_temp_c",
    "gear_oil_temp_c",
    "rotor_rpm",
    "availability",
}


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if (df["rotor_rpm"] < 0).any():
        logger.warning("Negative rotor_rpm values detected — check input data.")


# ===============================
# TREND ESTIMATOR
# ===============================

def _linear_trend(values: np.ndarray) -> float:
    """
    Return the slope of a linear fit over an array of values.

    Uses the closed-form OLS formula instead of np.polyfit for efficiency.
    Returns 0.0 if the slope cannot be computed.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    numerator = ((x - x_mean) * (values - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    return float(numerator / denominator) if denominator != 0 else 0.0


# ===============================
# ROLLING FEATURES
# ===============================

def _add_gap_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling mean and max of power gap percentage over a given window (hours)."""
    grp = df.groupby("turbine_id")["power_gap_pct"]
    df[f"gap_roll_mean_{window}h"] = grp.transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f"gap_roll_max_{window}h"] = grp.transform(
        lambda x: x.rolling(window, min_periods=1).max()
    )
    return df


def _add_temperature_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add rolling mean and linear trend for nacelle and gearbox temperatures.

    Trend is the OLS slope over the rolling window, expressed in degrees per hour.
    """
    min_periods = max(2, window // 4)

    for col, prefix in [
        ("nacelle_temp_c", "nacelle_temp"),
        ("gear_oil_temp_c", "gear_oil_temp"),
    ]:
        df[f"{prefix}_roll_mean_{window}h"] = (
            df.groupby("turbine_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{prefix}_trend_{window}h"] = (
            df.groupby("turbine_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=min_periods)
                       .apply(_linear_trend, raw=True))
        )
    return df


def _add_vibration_proxy_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add rolling mean and trend for rotor RPM as a vibration proxy.

    In the absence of dedicated vibration sensors, rotor RPM captures
    mechanical irregularities relevant to the mechanical risk subscore.
    """
    min_periods = max(2, window // 4)

    df[f"vib_roll_mean_{window}h"] = (
        df.groupby("turbine_id")["rotor_rpm"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    df[f"vib_trend_{window}h"] = (
        df.groupby("turbine_id")["rotor_rpm"]
        .transform(lambda x: x.rolling(window, min_periods=min_periods)
                   .apply(_linear_trend, raw=True))
    )
    return df


# ===============================
# OPERATIONAL FEATURES
# ===============================

def _add_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add yaw error proxy and pitch instability proxy.

    yaw_abs: absolute power gap percentage as a proxy for yaw misalignment.
    pitch_roll_std: rolling std of power gap as a proxy for pitch instability.
    Both are proxies — no dedicated yaw or pitch sensors exist in this simulation.
    """
    w = ROLLING_WINDOW_24H
    df["yaw_abs"] = df["power_gap_pct"].abs()
    df[f"pitch_roll_std_{w}h"] = (
        df.groupby("turbine_id")["power_gap_pct"]
        .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0.0))
    )
    return df


# ===============================
# MAIN FEATURE BUILDER
# ===============================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature set from an hourly aggregated fleet DataFrame.

    Input must contain all columns defined in REQUIRED_COLUMNS.
    Returns a copy with all feature columns appended.
    Rows with NaN in core feature columns are dropped with a warning.
    """
    _validate_columns(df)

    out = df.copy()
    out = out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

    out = _add_gap_features(out, ROLLING_WINDOW_24H)
    out = _add_temperature_features(out, ROLLING_WINDOW_24H)
    out = _add_vibration_proxy_features(out, ROLLING_WINDOW_24H)
    out = _add_operational_features(out)

    core_feature_cols = [
        f"gap_roll_mean_{ROLLING_WINDOW_24H}h",
        f"gap_roll_max_{ROLLING_WINDOW_24H}h",
    ]
    before = len(out)
    out = out.dropna(subset=core_feature_cols)
    dropped = before - len(out)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN in core feature columns.", dropped)

    logger.info("Feature build complete: %d rows, %d columns.", len(out), len(out.columns))
    return out


# ===============================
# FEATURE COLUMN CATALOGUE
# ===============================

def get_feature_columns() -> list[str]:
    """
    Return the list of feature column names produced by build_features().

    Useful for selecting model inputs downstream without hardcoding column names.
    """
    w = ROLLING_WINDOW_24H
    return [
        "power_gap_pct",
        f"gap_roll_mean_{w}h",
        f"gap_roll_max_{w}h",
        f"nacelle_temp_roll_mean_{w}h",
        f"nacelle_temp_trend_{w}h",
        f"gear_oil_temp_roll_mean_{w}h",
        f"gear_oil_temp_trend_{w}h",
        f"vib_roll_mean_{w}h",
        f"vib_trend_{w}h",
        "yaw_abs",
        f"pitch_roll_std_{w}h",
    ]


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    from src.data_generation import load_demo_scenario

    print("Loading green scenario...")
    df_hourly = load_demo_scenario("green")

    print("Building features...")
    df_features = build_features(df_hourly)

    print(f"\nShape: {df_features.shape}")
    print(f"\nFeature columns:\n{get_feature_columns()}")
    print(f"\nSample (WTG-01):")
    sample = df_features[df_features["turbine_id"] == "WTG-01"].head(5)
    print(sample[["timestamp"] + get_feature_columns()].to_string(index=False))