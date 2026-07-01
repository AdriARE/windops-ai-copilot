# src/config.py

# ===============================
# IMPORTS
# ===============================
from pathlib import Path

# ===============================
# PATHS
# ===============================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_DEMO_DIR = DATA_DIR / "demo"
REPORTS_DIR = ROOT_DIR / "reports"

# ===============================
# TURBINE PARAMETERS
# ===============================
RATED_POWER_KW = 3000.0
CUT_IN_SPEED = 3.0       # m/s
RATED_WIND_SPEED = 12.0  # m/s
CUT_OUT_SPEED = 25.0     # m/s

# ===============================
# SIMULATION PARAMETERS
# ===============================
N_TURBINES = 20
SIM_DAYS = 30
RAW_FREQ_MINUTES = 10
AGGREGATION_FREQ = "1h"

# ===============================
# FAULT SIMULATION
# ===============================
FAULT_TYPES = ["gearbox_degradation", "pitch_malfunction", "sensor_drift"]
DEFAULT_FAULT_FRACTION = 0.25   # share of turbines affected in red scenario
DEFAULT_FAULT_SEVERITY = 0.6    # 0.0 (none) to 1.0 (max)

# ===============================
# RISK SCORE WEIGHTS
# ===============================
# Subscore family weights
W_AERODYNAMIC = 0.50
W_MECHANICAL = 0.30
W_ANOMALY = 0.20

# Hybrid scoring: absolute vs relative component
W_ABSOLUTE = 0.70
W_RELATIVE = 0.30

# ===============================
# PRIORITY SCORE WEIGHTS
# ===============================
W_RISK = 0.50
W_LOSS = 0.35
W_CRITICALITY = 0.15

# ===============================
# ANOMALY DETECTION
# ===============================
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_RANDOM_STATE = 42

# ===============================
# FEATURE ENGINEERING
# ===============================
ROLLING_WINDOW_24H = 24   # hours — main rolling window
ROLLING_WINDOW_72H = 72   # hours — for percentile-based features if needed

# ===============================
# DEMO SCENARIOS
# ===============================
SCENARIOS = {
    "green": {
        "fault_fraction": 0.05,
        "fault_severity": 0.2,
        "description": "Fleet mostly healthy, few minor anomalies.",
    },
    "red": {
        "fault_fraction": 0.40,
        "fault_severity": 0.8,
        "description": "Multiple critical turbines, high energy loss.",
    },
}
DEFAULT_SCENARIO = "green"

# ===============================
# DATA SOURCE FLAG
# ===============================
USE_REAL_DATA_IF_AVAILABLE = False  # set True when real SCADA data is available

# ===============================
# AGENT
# ===============================
AGENT_MODEL = "claude-3-5-sonnet-20241022"
ACTION_PLAN_URGENCY_LEVELS = ["low", "medium", "high"]

# ===============================
# REPORTING
# ===============================
ENERGY_PRICE_EUR_MWH = 50.0   # indicative price for loss estimation