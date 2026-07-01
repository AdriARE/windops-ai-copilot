# src/prioritization.py

# ===============================
# IMPORTS
# ===============================
# Standard library
import logging

# Third-party
import numpy as np
import pandas as pd

# Local
from src.config import (
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

CRITICALITY_MIN = 0.3
CRITICALITY_MAX = 1.0

# ===============================
# INTERNAL HELPERS
# ===============================
def _normalize_series(series: pd.Series) -> pd.Series:
    """
    Min-max normalize a series to [0, 1], safe for NaNs.
    """
    series = series.fillna(0)

    s_min = series.min()
    s_max = series.max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - s_min) / (s_max - s_min)


def _criticality_score_for_turbine(turbine_id: str) -> float:
    """
    Return a deterministic criticality score for a given turbine ID.
    """
    seed = abs(hash(turbine_id)) % (2**32)
    rng = np.random.default_rng(seed)

    return round(float(rng.uniform(CRITICALITY_MIN, CRITICALITY_MAX)), 2)


# ===============================
# CRITICALITY
# ===============================
def assign_criticality(turbine_ids: pd.Series) -> pd.Series:
    """
    Assign a deterministic simulated criticality score to each turbine.
    """

    unique_ids = turbine_ids.unique()

    criticality_map = {
        turbine_id: _criticality_score_for_turbine(turbine_id)
        for turbine_id in unique_ids
    }

    return turbine_ids.map(criticality_map)


# ===============================
# PRIORITY SCORE
# ===============================
def compute_priority_scores(
    df_risk: pd.DataFrame,
    df_loss: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the final priority score for each turbine.
    """

    required_risk = {"turbine_id", "risk_score_mean"}
    required_loss = {"turbine_id", "loss_mwh_total"}

    missing_risk = required_risk - set(df_risk.columns)
    missing_loss = required_loss - set(df_loss.columns)

    if missing_risk:
        raise ValueError(f"Missing columns in risk summary: {sorted(missing_risk)}")

    if missing_loss:
        raise ValueError(f"Missing columns in loss summary: {sorted(missing_loss)}")

    out = df_risk.merge(
        df_loss[["turbine_id", "loss_mwh_total"]],
        on="turbine_id",
        how="left",
    )

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

    out = (
        out.sort_values("priority_score", ascending=False)
        .reset_index(drop=True)
    )

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
def run_prioritization_pipeline(
    df_risk_summary: pd.DataFrame,
    df_loss_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the final turbine priority ranking.
    """

    logger.info("Running prioritization pipeline...")

    priority = compute_priority_scores(
        df_risk_summary,
        df_loss_summary,
    )

    return priority


# ===============================
# MAIN — manual test
# ===============================
if __name__ == "__main__":

    from src.anomaly import run_anomaly_pipeline
    from src.data_generation import load_demo_scenario
    from src.features import build_features
    from src.impact import run_impact_pipeline
    from src.risk import run_risk_pipeline

    print("Loading red scenario...")
    df_hourly = load_demo_scenario("red")

    print("Building features...")
    df_features = build_features(df_hourly)

    print("Running anomaly detection...")
    df_anomaly, _ = run_anomaly_pipeline(df_features)

    print("Computing risk scores...")
    df_risk, risk_summary = run_risk_pipeline(df_anomaly)

    print("Estimating losses...")
    loss_summary = run_impact_pipeline(df_risk)

    print("Computing priority ranking...")
    priority = run_prioritization_pipeline(
        risk_summary,
        loss_summary,
    )

    print("\nPriority ranking:")

    cols = [
        "priority_rank",
        "turbine_id",
        "priority_score",
        "risk_score_mean",
        "loss_mwh_total",
        "criticality",
    ]

    print(priority[cols].to_string(index=False))