# src/impact.oy

# ===============================
# IMPORTS
# ===============================

# Standard library
import logging

# Third-party
import pandas as pd 

# Local
from src.config import 
ENERGY_PRICE_EUR_MWH

logger = logging.getLogger(__name__)

# ===============================
# ENERGY LOSS ESTIMATION
# ===============================
def estimate_energy_loss(df: pd.Dataframe) -> pd.DataFrame:
    """
    Estimate hourly energy loss per turbine from the power gap.
    
    Adds two columns:
    - loss_kwh
    - loss_eur

    Loss is only counted when the turbine was avaliable.
    """

    required = {"power_gap_kw"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns for loss estimation: {sorted(missing)}")

    out = df.copy()

    out["loss_kwh"] = out["power_gap_kw"] * out["availability"]
    out["loss_kwh"] = out["loss_kwh"].clip(lower=0.0)
    out["loss_eur"] = (
        out["loss_kwh"] / 1000.0 * ENERGY_PRICE_EUR_MWH
    )

    logger.debug(
        "Energy loss estimated: %.1f MWh total.",
        out["loss_kwh"].sum() / 1000.0,
    )

    return out


# ===============================
# LOSS SUMMARY
# ===============================
def summarize_loss_by_turbine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate energy loss to turbine level.
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

    summary["loss_mwh_total"] = (
        summary["loss_kwh_total"] / 1000.0
    )

    logger.info(
        "Loss summary | total fleet loss: %.1f MWh | %.0f EUR",
        summary["loss_mwh_total"].sum(),
        summary["loss_eur_total"].sum(),
    )

    return summary


# ===============================
# PIPELINE WRAPPER
# ===============================
def run_impact_pipeline(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate losses and return a turbine-level loss summary.
    """

    logger.info("Running impact pipeline...")

    df_with_loss = estimate_energy_loss(df_hourly)
    loss_summary = summarize_loss_by_turbine(df_with_loss)

    return loss_summary


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
    df_risk, _ = run_risk_pipeline(df_anomaly)

    print("Running impact pipeline...")
    loss_summary = run_impact_pipeline(df_risk)

    print("\nLoss summary:")
    print(loss_summary.head().to_string(index=False))