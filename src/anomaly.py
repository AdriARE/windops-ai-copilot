# src/anomaly.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import logging

# Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Local
from src.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_RANDOM_STATE,
    ROLLING_WINDOW_24H,
)
from src.features import get_feature_columns

logger = logging.getLogger(__name__)

# ===============================
# ROLLING HELPERS
# ===============================

def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Return rolling mean with min_periods=1."""
    return series.rolling(window, min_periods=1).mean()


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Return rolling sum with min_periods=1."""
    return series.rolling(window, min_periods=1).sum()


# ===============================
# ANOMALY DETECTION
# ===============================

def fit_isolation_forest(df: pd.DataFrame) -> IsolationForest:
    """
    Fit an IsolationForest on the full feature set.

    Uses all turbines and all timesteps as training data.
    Contamination and random state are controlled via config.
    """
    feature_cols = get_feature_columns()
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns for anomaly detection: {sorted(missing)}")

    X = df[feature_cols].dropna()
    if len(X) == 0:
        raise ValueError("No valid rows available for fitting IsolationForest.")

    model = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=ISOLATION_FOREST_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)
    logger.info("IsolationForest fitted on %d samples, %d features.", len(X), len(feature_cols))
    return model


def apply_anomaly_scores(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    """
    Apply a fitted IsolationForest to a feature DataFrame.

    Adds three columns:
    - anomaly_score: normalized score in [0, 1] where 1 = most anomalous
    - anomaly_flag: 1 if flagged as anomalous by the model, 0 otherwise
    - anomaly_persistence_24h: rolling fraction of flagged hours in the last 24h per turbine
    """
    feature_cols = get_feature_columns()
    out = df.copy()

    valid_mask = out[feature_cols].notna().all(axis=1)
    X = out.loc[valid_mask, feature_cols]

    raw_scores = model.decision_function(X)
    predictions = model.predict(X)

    # Normalize to [0, 1]: more negative raw score = more anomalous = score closer to 1
    score_min = raw_scores.min()
    score_max = raw_scores.max()
    score_range = score_max - score_min if score_max != score_min else 1.0
    normalized = (score_max - raw_scores) / score_range

    out.loc[valid_mask, "anomaly_score"] = normalized
    out.loc[valid_mask, "anomaly_flag"] = np.where(predictions == -1, 1, 0)

    out["anomaly_score"] = out["anomaly_score"].fillna(0.0)
    out["anomaly_flag"] = out["anomaly_flag"].fillna(0).astype(int)

    out = _add_anomaly_persistence(out, window=ROLLING_WINDOW_24H)

    logger.info(
        "Anomaly scores applied | flagged=%d / %d (%.1f%%)",
        out["anomaly_flag"].sum(),
        len(out),
        100 * out["anomaly_flag"].mean(),
    )
    return out


def _add_anomaly_persistence(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add anomaly_persistence_24h: rolling fraction of anomalous hours per turbine.

    Computed per turbine to avoid cross-contamination between machines.
    """
    out = df.copy()
    persistence_col = f"anomaly_persistence_{window}h"

    result_parts = []
    for turbine_id, group in out.groupby("turbine_id", sort=False):
        group = group.copy()
        group[persistence_col] = _rolling_sum(
            group["anomaly_flag"].astype(float), window
        ) / window
        result_parts.append(group)

    out = pd.concat(result_parts).sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    return out


# ===============================
# PIPELINE WRAPPER
# ===============================

def run_anomaly_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Fit IsolationForest and apply scores in a single call.

    Intended for use in notebooks and the Streamlit app.
    Returns the enriched DataFrame and the fitted model.
    """
    model = fit_isolation_forest(df)
    df_scored = apply_anomaly_scores(df, model)
    return df_scored, model


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    from src.data_generation import load_demo_scenario
    from src.features import build_features

    print("Loading green scenario...")
    df_hourly = load_demo_scenario("green")

    print("Building features...")
    df_features = build_features(df_hourly)

    print("Running anomaly detection...")
    df_scored, model = run_anomaly_pipeline(df_features)

    flagged = df_scored[df_scored["anomaly_flag"] == 1]
    print(f"\nTotal rows: {len(df_scored)}")
    print(f"Flagged anomalies: {len(flagged)} ({100 * len(flagged) / len(df_scored):.1f}%)")
    print(f"\nSample flagged rows:")
    print(
        flagged[["timestamp", "turbine_id", "anomaly_score", "anomaly_flag"]]
        .head(10)
        .to_string(index=False)
    )