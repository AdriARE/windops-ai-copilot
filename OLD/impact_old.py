# src/impact.py

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
    ENERGY_PRICE_EUR_MWH,
    W_CRITICALITY,
    W_LOSS,
    W_RISK,
)

logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION
# ===============================

# Criticality is simulated as a fixed property of each turbine.
# In a real system this would come from asset management data:
# grid connection type, contract obligations, location constraints, etc.
# Scale: 0.0 (least critical) to 1.0 (most critical).
CRITICALITY_SEED = 42
CRITICALITY_MIN = 0.3
CRITICALITY_MAX = 1.0


# ===============================
# INTERNAL HELPERS
# ===============================

def _normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1], safe for NaNs."""
    series = series.fillna(0)
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def _criticality_score_for_turbine(turbine_id: str) -> float:
    """
    Return a deterministic criticality score for a given turbine ID.

    Uses the turbine ID hash as seed so each turbine always gets
    the same score regardless of dataset order or execution context.
    """
    seed = abs(hash(turbine_id)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return round(float(rng.uniform(CRITICALITY_MIN, CRITICALITY_MAX)), 2)


# ===============================
# ENERGY LOSS ESTIMATION
# ===============================

def estimate_energy_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate hourly energy loss per turbine from the power gap.

    Adds two columns:
    - loss_kwh: estimated energy not generated in each hour (kWh)
    - loss_eur: monetary equivalent at the configured energy price

    Loss is only counted when the turbine was available. Planned downtime
    intervals are excluded to avoid counting unavoidable losses as operational.
    """
    required = {"power_gap_kw", "availability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for loss estimation: {sorted(missing)}")

    out = df.copy()
    out["loss_kwh"] = out["power_gap_kw"] * out["availability"] * 1.0
    out["loss_kwh"] = out["loss_kwh"].clip(lower=0.0)
    out["loss_eur"] = out["loss_kwh"] / 1000.0 * ENERGY_PRICE_EUR_MWH

    logger.debug("Energy loss estimated: %.1f MWh total.", out["loss_kwh"].sum() / 1000.0)
    return out


def summarize_loss_by_turbine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate energy loss to turbine level.

    Returns one row per turbine with total and mean hourly loss.
    """
    required = {"turbine_id", "loss_kwh", "loss_eur"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for loss summary: {sorted(missing)}")

    summary = (
        df.groupby("turbine_id", sort=False)
        .agg(
            loss_kwh_total=("loss_kwh", "sum"),
            loss_eur_total=("loss_eur", "sum"),
            loss_kwh_mean_hourly=("loss_kwh", "mean"),
        )
        .reset_index()
    )

    summary["loss_mwh_total"] = summary["loss_kwh_total"] / 1000.0

    logger.info(
        "Loss summary | total fleet loss: %.1f MWh | %.0f EUR",
        summary["loss_mwh_total"].sum(),
        summary["loss_eur_total"].sum(),
    )
    return summary


# ===============================
# CRITICALITY
# ===============================

def assign_criticality(turbine_ids: pd.Series) -> pd.Series:
    """
    Assign a deterministic simulated criticality score to each turbine.

    Each turbine always receives the same score regardless of execution order,
    because the score is derived from a hash of the turbine ID.
    In production this would come from asset management data.
    """
    unique_ids = turbine_ids.unique()
    criticality_map = {tid: _criticality_score_for_turbine(tid) for tid in unique_ids}
    return turbine_ids.map(criticality_map)


# ===============================
# PRIORITY SCORE
# ===============================

def compute_priority_scores(df_risk: pd.DataFrame, df_loss: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a final priority score per turbine combining risk, loss and criticality.

    Inputs:
    - df_risk: turbine-level risk summary from risk.summarize_risk_by_turbine()
    - df_loss: turbine-level loss summary from summarize_loss_by_turbine()

    Returns a merged DataFrame with priority_score and priority_rank columns,
    sorted by priority_score descending.

    Formula:
        priority = W_RISK * risk_norm + W_LOSS * loss_norm + W_CRITICALITY * criticality_norm

    Weights are defined in config and justified via sensitivity analysis in notebook 02.
    """
    required_risk = {"turbine_id", "risk_score_mean"}
    required_loss = {"turbine_id", "loss_mwh_total"}
    for col in required_risk - set(df_risk.columns):
        raise ValueError(f"Missing column in risk summary: '{col}'")
    for col in required_loss - set(df_loss.columns):
        raise ValueError(f"Missing column in loss summary: '{col}'")

    out = df_risk.merge(df_loss[["turbine_id", "loss_mwh_total"]], on="turbine_id", how="left")
    out["loss_mwh_total"] = out["loss_mwh_total"].fillna(0.0)
    out["criticality"] = assign_criticality(out["turbine_id"])

    out["risk_norm"] = _normalize_series(out["risk_score_mean"])
    out["loss_norm"] = _normalize_series(out["loss_mwh_total"])
    out["criticality_norm"] = _normalize_series(out["criticality"])

    out["priority_score"] = (
        W_RISK * out["risk_norm"]
        + W_LOSS * out["loss_norm"]
        + W_CRITICALITY * out["criticality_norm"]
    ).clip(0.0, 1.0)

    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    out["priority_rank"] = out.index + 1

    logger.info(
        "Priority ranking computed | turbines=%d | top=%s | score=%.3f",
        len(out),
        out.loc[0, "turbine_id"],
        out.loc[0, "priority_score"],
    )
    return out


# ===============================
# PIPELINE WRAPPER
# ===============================

def run_impact_pipeline(
    df_hourly: pd.DataFrame,
    df_risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate losses and compute priority scores in a single call.

    Inputs:
    - df_hourly: row-level DataFrame with risk scores from run_risk_pipeline()
    - df_risk_summary: turbine-level risk summary from summarize_risk_by_turbine()

    Returns a turbine-level priority DataFrame ready for the agent and Streamlit.
    """
    logger.info("Running impact pipeline...")
    df_with_loss = estimate_energy_loss(df_hourly)
    loss_summary = summarize_loss_by_turbine(df_with_loss)
    priority = compute_priority_scores(df_risk_summary, loss_summary)
    return priority


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    from src.anomaly import run_anomaly_pipeline
    from src.data_generation import load_demo_scenario
    from src.features import build_features
    from src.risk import run_risk_pipeline

    print("Loading red scenario...")
    df_hourly = load_demo_scenario("red")

    print("Building features...")
    df_features = build_features(df_hourly)

    print("Running anomaly detection...")
    df_anomaly, _ = run_anomaly_pipeline(df_features)

    print("Computing risk scores...")
    df_risk, risk_summary = run_risk_pipeline(df_anomaly)

    print("Running impact pipeline...")
    priority = run_impact_pipeline(df_risk, risk_summary)

    print("\nPriority ranking (all turbines):")
    cols = [
        "priority_rank", "turbine_id", "priority_score",
        "risk_score_mean", "loss_mwh_total", "criticality",
    ]
    print(priority[cols].to_string(index=False))