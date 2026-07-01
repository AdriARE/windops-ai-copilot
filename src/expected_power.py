# src/expected_power.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import logging
from typing import Union

# Third-party
import numpy as np
import pandas as pd

# Local
from OLD.config_old import (
    CUT_IN_SPEED,
    CUT_OUT_SPEED,
    RATED_POWER_KW,
    RATED_WIND_SPEED,
)

logger = logging.getLogger(__name__)

# ===============================
# POWER CURVE
# ===============================

def expected_power(wind_speed: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute expected power output (kW) from wind speed using a cubic power curve.

    Assumes a generic 3 MW turbine. Below cut-in and above cut-out: zero output.
    Between cut-in and rated: cubic interpolation. Above rated: capped at rated power.
    """
    ws = np.asarray(wind_speed, dtype=float)
    ws = np.clip(ws, 0, None)

    power = np.where(
        ws < CUT_IN_SPEED, 0.0,
        np.where(
            ws >= CUT_OUT_SPEED, 0.0,
            np.where(
                ws >= RATED_WIND_SPEED,
                RATED_POWER_KW,
                RATED_POWER_KW * ((ws - CUT_IN_SPEED) / (RATED_WIND_SPEED - CUT_IN_SPEED)) ** 3,
            ),
        ),
    )

    return float(power) if power.ndim == 0 else power


def add_expected_power(df: pd.DataFrame, wind_col: str = "wind_speed") -> pd.DataFrame:
    """
    Add 'expected_power_kw' and 'expected_capacity_factor' columns to a DataFrame.

    Returns a copy of the DataFrame with the new columns appended.
    """
    if wind_col not in df.columns:
        raise ValueError(f"Column '{wind_col}' not found in DataFrame.")

    out = df.copy()
    out["expected_power_kw"] = expected_power(out[wind_col].to_numpy())
    out["expected_capacity_factor"] = out["expected_power_kw"] / RATED_POWER_KW
    logger.debug("Added expected power columns to DataFrame (%d rows).", len(out))
    return out


def compute_power_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'power_gap_kw' and 'power_gap_pct' columns to a DataFrame.

    Requires 'power_kw' and 'expected_power_kw' to be present.
    Gap is clipped to zero from below: we only care about underperformance.
    """
    required = {"power_kw", "expected_power_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    out["power_gap_kw"] = (out["expected_power_kw"] - out["power_kw"]).clip(lower=0.0)
    out["power_gap_pct"] = np.where(
        out["expected_power_kw"] > 0,
        out["power_gap_kw"] / out["expected_power_kw"],
        0.0,
    )
    logger.debug("Computed power gap columns (%d rows).", len(out))
    return out


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    test_speeds = [0, 2, 3, 6, 12, 15, 25, 30]
    print("Wind speed → Expected power")
    print("-" * 35)
    for ws in test_speeds:
        print(f"  {ws:>5.1f} m/s  →  {expected_power(ws):>8.1f} kW")