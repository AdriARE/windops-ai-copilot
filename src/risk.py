# src/risk.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import logging

# Third-party
import numpy as np
import pandas as pd

# Local
from OLD.config_old import (
    ROLLING_WINDOW_24H,
    W_ABSOLUTE,
    W_AERODYNAMIC,
    W_ANOMALY,
    W_MECHANICAL,
    W_RELATIVE,
)

logger = logging.getLogger(__name__)

# ===============================
# ABSOLUTE SCORE THRESHOLDS
# ===============================

# Aerodynamic thresholds (based on power gap percentage)
GAP_THRESHOLD_LOW = 0.10     # 10% gap → low risk
GAP_THRESHOLD_HIGH = 0.25    # 25% gap → high risk

# Mechanical thresholds (based on gear oil temperature, degrees C)
TEMP_THRESHOLD_LOW = 65.0
TEMP_THRESHOLD_HIGH = 80.0

# Anomaly score thresholds
ANOMALY_THRESHOLD_LOW = 0.40
ANOMALY_THRESHOLD_HIGH = 0.70

# Risk level boundaries for categorical classification
RISK_LEVEL_BINS = [0.0, 0.25, 0.50, 0.75, 1.01]
RISK_LEVEL_LABELS = ["low", "medium", "high", "critical"]

# ===============================
# STARTUP VALIDATION
# ===============================

def _validate_config() -> None:
    """
    Validate that risk score weights and thresholds are internally consistent.

    Called once at module level to catch misconfiguration early.
    """
    weight_sum = W_AERODYNAMIC + W_MECHANICAL + W_ANOMALY
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(
            "Risk score weights sum to %.6f, expected 1.0. Scores may be skewed.", weight_sum
        )

    if GAP_THRESHOLD_LOW >= GAP_THRESHOLD_HIGH:
        raise ValueError(
            f"GAP_THRESHOLD_LOW ({GAP_THRESHOLD_LOW}) must be less than "
            f"GAP_THRESHOLD_HIGH ({GAP_THRESHOLD_HIGH})."
        )

    if TEMP_THRESHOLD_LOW >= TEMP_THRESHOLD_HIGH:
        raise ValueError(
            f"TEMP_THRESHOLD_LOW ({TEMP_THRESHOLD_LOW}) must be less than "
            f"TEMP_THRESHOLD_HIGH ({TEMP_THRESHOLD_HIGH})."
        )

    if any(w < 0 for w in [W_AERODYNAMIC, W_MECHANICAL, W_ANOMALY]):
        raise ValueError("Risk score weights cannot be negative.")


_validate_config()

# ===============================
# SCORING HELPERS
# ===============================

def _clamp(value: float, low: float, high: float) -> float:
    """Map a value to [0, 1] linearly between low and high thresholds."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. Returns zeros if range is zero."""
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


# ===============================
# ABSOLUTE SUBSCORES
# ===============================

def _aerodynamic_absolute(df: pd.DataFrame) -> pd.Series:
    """
    Absolute aerodynamic risk based on power gap and pitch instability.

    A persistent gap above 25% is considered high risk regardless of fleet state.
    """
    w = ROLLING_WINDOW_24H
    gap_col = f"gap_roll_mean_{w}h"
    pitch_col = f"pitch_roll_std_{w}h"

    for col in [gap_col, pitch_col, "yaw_abs"]:
        if col not in df.columns:
            raise ValueError(f"Missing column for aerodynamic score: '{col}'")

    gap_score = df[gap_col].apply(_clamp, args=(GAP_THRESHOLD_LOW, GAP_THRESHOLD_HIGH))
    pitch_score = df[pitch_col].apply(_clamp, args=(0.0, 0.15))
    yaw_score = df["yaw_abs"].apply(_clamp, args=(GAP_THRESHOLD_LOW, GAP_THRESHOLD_HIGH))

    return (0.60 * gap_score + 0.25 * pitch_score + 0.15 * yaw_score).clip(0.0, 1.0)


def _mechanical_absolute(df: pd.DataFrame) -> pd.Series:
    """
    Absolute mechanical risk based on gear oil temperature and vibration trend.

    Elevated gear oil temperature is the strongest mechanical signal available.
    """
    w = ROLLING_WINDOW_24H
    temp_col = f"gear_oil_temp_roll_mean_{w}h"
    vib_col = f"vib_trend_{w}h"

    for col in [temp_col, vib_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column for mechanical score: '{col}'")

    temp_score = df[temp_col].apply(_clamp, args=(TEMP_THRESHOLD_LOW, TEMP_THRESHOLD_HIGH))
    vib_score = df[vib_col].abs().apply(_clamp, args=(0.0, 2.0))

    return (0.70 * temp_score + 0.30 * vib_score).clip(0.0, 1.0)


def _anomaly_absolute(df: pd.DataFrame) -> pd.Series:
    """
    Absolute anomaly risk based on IsolationForest score and persistence.
    """
    w = ROLLING_WINDOW_24H
    persistence_col = f"anomaly_persistence_{w}h"

    for col in ["anomaly_score", persistence_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column for anomaly score: '{col}'")

    score_part = df["anomaly_score"].apply(
        _clamp, args=(ANOMALY_THRESHOLD_LOW, ANOMALY_THRESHOLD_HIGH)
    )
    persistence_part = df[persistence_col].clip(0.0, 1.0)

    return (0.60 * score_part + 0.40 * persistence_part).clip(0.0, 1.0)


# ===============================
# HYBRID RISK SCORE
# ===============================

def _compute_subscore(absolute: pd.Series, relative: pd.Series) -> pd.Series:
    """Combine absolute and relative components into a single subscore."""
    return (W_ABSOLUTE * absolute + W_RELATIVE * relative).clip(0.0, 1.0)


def _add_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'risk_level' column with categorical classification of risk_score.

    Labels: low / medium / high / critical.
    Uses fixed bins defined in config-level constants.
    """
    out = df.copy()
    out["risk_level"] = pd.cut(
        out["risk_score"],
        bins=RISK_LEVEL_BINS,
        labels=RISK_LEVEL_LABELS,
        right=False,
    ).astype(str)
    return out


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full risk score for each row in the feature DataFrame.

    Adds the following columns:
    - aero_risk, mech_risk, anomaly_risk: family subscores
    - risk_score: final weighted combination in [0, 1]
    - risk_level: categorical label (low / medium / high / critical)

    The score is hybrid: 70% absolute (domain thresholds) + 30% relative
    (fleet-normalized). This prevents healthy fleets from showing false highs
    and degraded fleets from masking critical turbines.
    """
    out = df.copy()

    aero_abs = _aerodynamic_absolute(out)
    mech_abs = _mechanical_absolute(out)
    anomaly_abs = _anomaly_absolute(out)

    aero_rel = _normalize_series(aero_abs)
    mech_rel = _normalize_series(mech_abs)
    anomaly_rel = _normalize_series(anomaly_abs)

    out["aero_risk"] = _compute_subscore(aero_abs, aero_rel)
    out["mech_risk"] = _compute_subscore(mech_abs, mech_rel)
    out["anomaly_risk"] = _compute_subscore(anomaly_abs, anomaly_rel)

    out["risk_score"] = (
        W_AERODYNAMIC * out["aero_risk"]
        + W_MECHANICAL * out["mech_risk"]
        + W_ANOMALY * out["anomaly_risk"]
    ).clip(0.0, 1.0)

    out = _add_risk_levels(out)

    logger.info(
        "Risk scores computed | mean=%.3f | max=%.3f | critical=%.1f%% | rows=%d",
        out["risk_score"].mean(),
        out["risk_score"].max(),
        100 * (out["risk_level"] == "critical").mean(),
        len(out),
    )
    return out


# ===============================
# TURBINE-LEVEL SUMMARY
# ===============================

def summarize_risk_by_turbine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate risk scores to turbine level for fleet view.

    Returns one row per turbine with mean and max risk score,
    plus the most recent values of each subscore.
    Sorted by mean risk score descending.
    """
    required = {"turbine_id", "risk_score", "aero_risk", "mech_risk", "anomaly_risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for turbine summary: {sorted(missing)}")

    latest = (
        df.sort_values("timestamp")
        .groupby("turbine_id")
        .last()
        .reset_index()[["turbine_id", "aero_risk", "mech_risk", "anomaly_risk", "risk_score", "risk_level"]]
        .rename(columns={
            "aero_risk": "aero_risk_latest",
            "mech_risk": "mech_risk_latest",
            "anomaly_risk": "anomaly_risk_latest",
            "risk_score": "risk_score_latest",
            "risk_level": "risk_level_latest",
        })
    )

    agg = (
        df.groupby("turbine_id")["risk_score"]
        .agg(risk_score_mean="mean", risk_score_max="max")
        .reset_index()
    )

    summary = agg.merge(latest, on="turbine_id")
    summary = summary.sort_values("risk_score_mean", ascending=False).reset_index(drop=True)

    logger.info("Turbine risk summary built: %d turbines.", len(summary))
    return summary


# ===============================
# PIPELINE WRAPPER
# ===============================

def run_risk_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute risk scores and return both row-level and turbine-level results.

    Intended for use in notebooks and the Streamlit app.
    Returns (df_with_scores, turbine_summary).
    """
    df_scored = compute_risk_scores(df)
    summary = summarize_risk_by_turbine(df_scored)
    return df_scored, summary


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    from src.anomaly import run_anomaly_pipeline
    from src.data_generation import load_demo_scenario
    from src.features import build_features

    print("Loading red scenario...")
    df_hourly = load_demo_scenario("red")

    print("Building features...")
    df_features = build_features(df_hourly)

    print("Running anomaly detection...")
    df_scored, _ = run_anomaly_pipeline(df_features)

    print("Computing risk scores...")
    df_risk, summary = run_risk_pipeline(df_scored)

    print("\nTurbine risk summary (top 10):")
    print(summary.head(10).to_string(index=False))

    print("\nRisk level distribution:")
    print(df_risk["risk_level"].value_counts().to_string())