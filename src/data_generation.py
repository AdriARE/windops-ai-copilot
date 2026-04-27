# src/data_generation.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import logging
from typing import Optional

# Third-party
import numpy as np
import pandas as pd

# Local
from src.config import (
    AGGREGATION_FREQ,
    CUT_IN_SPEED,
    CUT_OUT_SPEED,
    FAULT_TYPES,
    N_TURBINES,
    RATED_WIND_SPEED,
    RAW_FREQ_MINUTES,
    SCENARIOS,
    DEFAULT_SCENARIO,
    SIM_DAYS,
)
from src.expected_power import expected_power, add_expected_power, compute_power_gap

logger = logging.getLogger(__name__)

# ===============================
# WIND SPEED SIMULATION
# ===============================

def _simulate_wind_speed(
    n_steps: int,
    rng: np.random.Generator,
    mean: float = 8.0,
    std: float = 3.5,
) -> np.ndarray:
    """
    Simulate wind speed as a mean-reverting AR(1) random walk, clipped to [0, cut-out].

    Uses autocorrelation to produce physically plausible wind series.
    """
    ws = np.empty(n_steps)
    ws[0] = mean
    phi = 0.92  # AR(1) persistence coefficient

    for t in range(1, n_steps):
        ws[t] = phi * ws[t - 1] + (1 - phi) * mean + rng.normal(0, std * (1 - phi**2) ** 0.5)

    return np.clip(ws, 0.0, CUT_OUT_SPEED)


# ===============================
# FAULT INJECTION
# ===============================

def _apply_fault(
    power: np.ndarray,
    wind_speed: np.ndarray,
    fault_type: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reduce power output to simulate a specific fault type.

    severity: 0.0 (no effect) to 1.0 (maximum degradation).
    Returns modified power array.
    """
    power = power.copy()

    if fault_type == "gearbox_degradation":
        loss_factor = 1.0 - (0.15 + 0.25 * severity)
        power *= loss_factor

    elif fault_type == "pitch_malfunction":
        high_wind_mask = wind_speed >= RATED_WIND_SPEED
        loss_factor = 1.0 - (0.20 + 0.30 * severity)
        power[high_wind_mask] *= loss_factor

    elif fault_type == "sensor_drift":
        noise_std = 50.0 + 200.0 * severity
        power += rng.normal(0, noise_std, size=len(power))

    return np.clip(power, 0.0, None)


# ===============================
# SINGLE TURBINE SIMULATION
# ===============================

def _simulate_turbine(
    turbine_id: str,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    fault_type: Optional[str] = None,
    fault_severity: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate raw 10-minute SCADA-like data for a single turbine.

    Returns a DataFrame with columns:
    timestamp, turbine_id, wind_speed, power_kw, rotor_rpm,
    nacelle_temp_c, gear_oil_temp_c, availability.
    """
    n = len(timestamps)
    ws = _simulate_wind_speed(n, rng)

    power = expected_power(ws).copy()
    if fault_type is not None:
        power = _apply_fault(power, ws, fault_type, fault_severity, rng)

    power += rng.normal(0, 15.0, size=n)
    power = np.clip(power, 0.0, None)

    rotor_rpm = np.where(
        ws < CUT_IN_SPEED, 0.0,
        np.clip(5.0 + 1.2 * ws + rng.normal(0, 0.5, n), 0.0, 20.0),
    )
    nacelle_temp = 35.0 + 0.4 * power / 1000.0 + rng.normal(0, 1.5, n)
    gear_oil_temp = 55.0 + 0.3 * power / 1000.0 + rng.normal(0, 2.0, n)

    if fault_type == "gearbox_degradation":
        gear_oil_temp += 8.0 * fault_severity

    # Availability: continuous downtime blocks, more realistic than random per-interval
    availability = np.ones(n)
    n_events = rng.integers(1, 4)
    for _ in range(n_events):
        start = rng.integers(0, n - 6)
        duration = rng.integers(2, 12)
        availability[start:start + duration] = 0

    return pd.DataFrame({
        "timestamp": timestamps,
        "turbine_id": turbine_id,
        "wind_speed": ws,
        "power_kw": power,
        "rotor_rpm": rotor_rpm,
        "nacelle_temp_c": nacelle_temp,
        "gear_oil_temp_c": gear_oil_temp,
        "availability": availability,
    })


# ===============================
# FLEET SIMULATION
# ===============================

def simulate_fleet(
    n_turbines: int = N_TURBINES,
    sim_days: int = SIM_DAYS,
    fault_fraction: float = 0.25,
    fault_severity: float = 0.6,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate raw 10-minute SCADA-like data for a full wind farm fleet.

    Assigns faults randomly to a fraction of turbines.
    Returns a single long-format DataFrame with all turbines combined.
    """
    rng = np.random.default_rng(random_seed)

    steps_per_hour = int(60 / RAW_FREQ_MINUTES)
    periods = sim_days * 24 * steps_per_hour
    timestamps = pd.date_range("2024-01-01", periods=periods, freq=f"{RAW_FREQ_MINUTES}min")

    n_faulty = max(1, int(n_turbines * fault_fraction))
    faulty_ids = set(rng.choice(n_turbines, size=n_faulty, replace=False))

    frames = []
    for i in range(n_turbines):
        turbine_id = f"WTG-{i + 1:02d}"
        if i in faulty_ids:
            fault_type = str(rng.choice(FAULT_TYPES))
            severity = fault_severity
        else:
            fault_type = None
            severity = 0.0

        df_turbine = _simulate_turbine(turbine_id, timestamps, rng, fault_type, severity)
        df_turbine["fault_type"] = fault_type if fault_type else "none"
        frames.append(df_turbine)
        logger.debug("Simulated %s — fault: %s", turbine_id, fault_type or "none")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

    logger.info(
        "Fleet simulation complete | turbines=%d | rows=%d | faulty=%d",
        n_turbines, len(raw), n_faulty,
    )
    return raw


# ===============================
# AGGREGATION
# ===============================

def aggregate_to_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw 10-minute data to hourly frequency.

    Power and sensor columns are averaged. Availability is the fraction of
    available intervals within each hour.
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    agg_funcs = {
        "wind_speed": "mean",
        "power_kw": "mean",
        "rotor_rpm": "mean",
        "nacelle_temp_c": "mean",
        "gear_oil_temp_c": "mean",
        "availability": "mean",
        "fault_type": "first",
    }

    hourly = (
        df.groupby("turbine_id")
        .resample(AGGREGATION_FREQ)
        .agg(agg_funcs)
        .reset_index()
    )

    hourly = add_expected_power(hourly)
    hourly = compute_power_gap(hourly)

    logger.info("Aggregation complete: %d hourly rows.", len(hourly))
    return hourly


# ===============================
# SCENARIO LOADER
# ===============================

def load_demo_scenario(scenario: str = DEFAULT_SCENARIO) -> pd.DataFrame:
    """
    Generate a fleet dataset for a named demo scenario (e.g. 'green', 'red').

    Scenario parameters are defined in config.SCENARIOS.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(SCENARIOS.keys())}")

    params = SCENARIOS[scenario]
    logger.info("Loading scenario '%s': %s", scenario, params["description"])

    raw = simulate_fleet(
        fault_fraction=params["fault_fraction"],
        fault_severity=params["fault_severity"],
    )
    return aggregate_to_hourly(raw)


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    print("Running demo scenario: green")
    df = load_demo_scenario("green")
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nTurbines: {df['turbine_id'].unique()}")