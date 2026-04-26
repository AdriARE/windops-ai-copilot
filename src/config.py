# src/config.py

from pathlib import Path

# ── Rutas base ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "demo"
REPORTS_DIR = ROOT_DIR / "reports"

# ── Cache de datos demo ──────────────────────────────────────
DEMO_DATA_PATH = DATA_DIR / "scada_demo.parquet"
CACHE_DEMO_DATA = True

# ── Turbina genérica ─────────────────────────────────────────
RATED_POWER_KW = 3000
CUT_IN_SPEED = 3.0
RATED_WIND_SPEED = 12.0
CUT_OUT_SPEED = 25.0

# ── Simulación ───────────────────────────────────────────────
N_TURBINES = 20
SIM_DAYS = 30
RAW_FREQ_MINUTES = 10
AGG_FREQ_HOURS = 1
RANDOM_SEED = 42

# ── Ventanas de features ─────────────────────────────────────
ROLLING_WINDOW_HOURS = 24
LONG_WINDOW_HOURS = 72

# ── Escenarios demo oficiales ────────────────────────────────
SCENARIOS = {
    "green": {
        "fault_probability": 0.05,
        "affected_turbine_pct": 0.10,
        "fault_severity": 0.3,
    },
    "red": {
        "fault_probability": 0.35,
        "affected_turbine_pct": 0.40,
        "fault_severity": 0.8,
    },
}
DEFAULT_SCENARIO = "green"

# ── Risk score (pesos) ───────────────────────────────────────
RISK_WEIGHTS = {
    "aerodynamic": 0.50,
    "mechanical": 0.30,
    "anomaly": 0.20,
}

SUBSCORE_BLEND = {
    "absolute": 0.70,
    "relative": 0.30,
}

# ── Priorización ─────────────────────────────────────────────
PRIORITY_WEIGHTS = {
    "risk": 0.50,
    "loss": 0.35,
    "criticality": 0.15,
}

# ── Energía y monetización ───────────────────────────────────
ENERGY_PRICE_EUR_MWH = 60.0

# ── Normalización ────────────────────────────────────────────
NORMALIZATION_EPS = 1e-6

# ── Flags de datos ───────────────────────────────────────────
USE_REAL_DATA_IF_AVAILABLE = False