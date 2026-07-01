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
CUT_IN_SPEED = 3.0
RATED_WIND_SPEED = 12.0
CUT_OUT_SPEED = 25.0

# ===============================
# SIMULATION PARAMETERS
# ===============================
N_TURBINES = 20
SIM_DAYS = 30
RAW_FREQ_MINUTES = 10
AGGREGATION_FREQ = "1h"

# ===============================
# FAULT TYPES
# ===============================
FAULT_TYPES = [
    "gearbox_degradation",
    "pitch_malfunction",
    "sensor_drift",
    "yaw_misalignment",
]

# ===============================
# DEMO SCENARIOS
# ===============================
# Each scenario defines an explicit fault plan:
# { turbine_id: (fault_type, severity) }
# severity: 0.0 (minimal) to 1.0 (maximum degradation)
# Empty faults dict means a fully healthy fleet.

SCENARIOS = {
    "green": {
        "description": "Fully healthy fleet. Baseline operating condition.",
        "faults": {},
    },
    "gearbox": {
        "description": "Single turbine with advanced gearbox degradation.",
        "faults": {
            "WTG-02": ("gearbox_degradation", 0.9),
        },
    },
    "pitch": {
        "description": "Two turbines with pitch malfunction at different severities.",
        "faults": {
            "WTG-05": ("pitch_malfunction", 0.8),
            "WTG-11": ("pitch_malfunction", 0.6),
        },
    },
    "yaw": {
        "description": "Two turbines with yaw misalignment.",
        "faults": {
            "WTG-03": ("yaw_misalignment", 0.7),
            "WTG-15": ("yaw_misalignment", 0.6),
        },
    },
    "mixed": {
        "description": "Three turbines with different fault types — realistic mixed scenario.",
        "faults": {
            "WTG-02": ("gearbox_degradation", 0.9),
            "WTG-07": ("pitch_malfunction", 0.7),
            "WTG-14": ("sensor_drift", 0.6),
        },
    },
    "red": {
        "description": "Severe degradation across multiple turbines — stress test scenario.",
        "faults": {
            "WTG-01": ("gearbox_degradation", 0.9),
            "WTG-03": ("yaw_misalignment", 0.8),
            "WTG-06": ("pitch_malfunction", 0.9),
            "WTG-09": ("sensor_drift", 0.7),
            "WTG-12": ("gearbox_degradation", 0.8),
            "WTG-15": ("pitch_malfunction", 0.7),
            "WTG-17": ("yaw_misalignment", 0.9),
            "WTG-19": ("sensor_drift", 0.8),
        },
    },
}

DEFAULT_SCENARIO = "green"

# ===============================
# RISK SCORE WEIGHTS
# ===============================
W_AERODYNAMIC = 0.50
W_MECHANICAL = 0.30
W_ANOMALY = 0.20

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
ROLLING_WINDOW_24H = 24
ROLLING_WINDOW_72H = 72

# ===============================
# DATA SOURCE FLAG
# ===============================
USE_REAL_DATA_IF_AVAILABLE = False

# ===============================
# AGENT
# ===============================
AGENT_MODEL = "claude-sonnet-4-6"
ACTION_PLAN_URGENCY_LEVELS = ["low", "medium", "high"]

# ===============================
# REPORTING
# ===============================
ENERGY_PRICE_EUR_MWH = 50.0