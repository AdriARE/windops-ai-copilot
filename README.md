# WindOps AI Copilot

> Predictive maintenance and fleet prioritisation for wind turbine operations,
> powered by heuristic risk scoring and an LLM-based diagnostic agent.

---

## The Problem

Wind farm operators manage fleets of 10–100+ turbines generating continuous
SCADA streams — wind speed, power output, temperatures, rotor RPM, availability.
The challenge is not collecting data. The challenge is knowing **which turbine
to act on first, and why**.

Current approaches are either too simple (threshold alarms) or too opaque
(black-box ML models that operators cannot trust or explain). Neither integrates
well with the operational reality of a wind farm: maintenance windows are
expensive, downtime is costly, and decisions must be justifiable.

WindOps AI Copilot addresses this by combining:

- A **transparent, heuristic risk score** built from domain knowledge
- An **LLM-based agent** that diagnoses fault patterns and generates
  structured, operational maintenance recommendations
- Full **traceability**: every recommendation links back to the specific
  signals and subscores that drove it

---

## Architecture
Raw SCADA data (10-min)
│
▼
Hourly aggregation
│
▼
Feature engineering          ← rolling stats, proxy signals
│
▼
Anomaly detection             ← IsolationForest
│
▼
Risk scoring                  ← aero · mechanical · anomaly subscores
│
▼
Priority ranking              ← risk + energy loss + asset criticality
│
▼
LangGraph agent               ← tool calls → diagnosis → action plan
│
▼
Streamlit dashboard / CLI / JSON export


The risk score is **hybrid**: 70% absolute (domain thresholds) + 30%
relative (fleet-normalised). This prevents a degraded fleet from masking
its own critical turbines.

---

## Project Structure
windops-ai-copilot/
├── src/                        # Core analytics pipeline
│   ├── config.py               # Constants, scenarios, weights
│   ├── data_generation.py      # SCADA simulation with explicit fault plans
│   ├── expected_power.py       # Cubic power curve model
│   ├── features.py             # Feature engineering
│   ├── anomaly.py              # IsolationForest anomaly detection
│   ├── risk.py                 # Hybrid risk scoring
│   ├── impact.py               # Energy loss estimation
│   ├── prioritization.py       # Priority ranking
│   └── io.py                   # CSV / Markdown export
├── app/
│   ├── agent.py                # LangGraph agent + demo mode fallback
│   ├── app.py                  # Streamlit dashboard
│   └── cli.py                  # Command-line interface
├── notebooks/
│   ├── 01_eda_demo_data.ipynb          # Fleet EDA and pipeline validation
│   ├── 02_features_and_scoring.ipynb   # Features, subscores, sensitivity analysis
│   └── 03_agent_decisions.ipynb        # Agent trace, scenario comparison, export
├── reports/                    # Generated action plans (JSON, CSV, Markdown)
├── requirements.txt
└── README.md

---

## Demo Scenarios

The simulation generates physically plausible SCADA data with explicit
fault plans per turbine. Available scenarios:

| Scenario | Description |
|---|---|
| `green` | Fully healthy fleet — baseline |
| `gearbox` | Single turbine with gearbox degradation |
| `pitch` | Two turbines with pitch malfunction |
| `yaw` | Two turbines with yaw misalignment |
| `mixed` | Three turbines with different fault types |
| `red` | Severe degradation across multiple turbines |

Each fault type produces a distinct signal signature:

| Fault | Power gap | Gear oil temp | Anomaly score | Dominant subscore |
|---|---|---|---|---|
| `gearbox_degradation` | Sustained ↑ | Elevated ↑↑ | Moderate | Mechanical + Aero |
| `pitch_malfunction` | High at rated wind | Normal | Low–Moderate | Aerodynamic |
| `sensor_drift` | Noisy / ambiguous | Normal | High ↑↑ | Anomaly |
| `yaw_misalignment` | Sustained ↑ | Normal | Low | Aerodynamic |

---

## Quickstart

### Prerequisites

```bash
python 3.10+
pip install -r requirements.txt

# Create .env at project root
echo "ANTHROPIC_API_KEY=your_key_here" > .env

Without a key, the agent runs in demo mode using rule-based diagnosis.
The pipeline and all analytics run identically in both modes.
Streamlit dashboard

streamlit run app/app.py

Command-line interface
# Default: mixed scenario, top 3 turbines
python app/cli.py

# Custom scenario and export
python app/cli.py --scenario red --top-n 5 --export

# Pipeline only, no agent
python app/cli.py --scenario green --no-agent

# Help
python app/cli.py --help

Notebooks
Run from the notebooks/ directory or with sys.path pointing to project root.
Start with 01_eda_demo_data.ipynb and follow the sequence.
Risk Score Design
The final risk score combines three subscore families:
Subscore
Weight
Primary signals
Aerodynamic
50%
Power gap (24h rolling mean), yaw proxy, pitch instability
Mechanical
30%
Gear oil temperature (24h rolling mean), vibration trend
Anomaly
20%
IsolationForest score, anomaly persistence (24h)
Each subscore is hybrid:
subscore = 0.70 × absolute_component + 0.30 × relative_component
The absolute component uses domain thresholds (e.g. gear oil temp > 65°C).
The relative component normalises against the current fleet state.
Priority ranking combines risk score (50%), estimated energy loss (35%)
and asset criticality (15%). Weights are configurable in src/config.py.
Agent
The LangGraph agent operates in a ReAct loop with three tools:
get_priority_ranking — retrieves the top N turbines by priority score
get_turbine_details — returns latest subscores and sensor readings for a turbine
submit_action_plan — submits a structured maintenance recommendation
Each action plan includes:
{
  "turbine_id": "WTG-02",
  "urgency": "high",
  "fault_hypothesis": "Gearbox or drivetrain degradation",
  "recommended_action": "Schedule gearbox oil sampling and vibration analysis within 24 hours...",
  "rationale": "Mechanical subscore (0.82) is dominant. Gear oil temperature at 74.3°C..."
}
Demo mode: if ANTHROPIC_API_KEY is not set or is invalid, the agent
falls back to rule-based diagnosis using the same subscore patterns the LLM
is prompted to follow. Output format is identical.
Known Limitations
Data
SCADA data is synthetically generated. Signal patterns are physically
plausible but do not capture real turbine complexity (correlated faults,
seasonal effects, manufacturer-specific curves).
A single generic 3 MW turbine model is used across the fleet.
Analytics
Risk score thresholds are manually defined and would require calibration
against historical failure data in production.
IsolationForest is trained and evaluated on the same dataset (no temporal
split). In production, it would be trained on historical healthy operation
and applied to incoming data in streaming or batch mode.
Agent
In live mode, LLM diagnosis is not formally validated and may produce
unexpected outputs for rare fault combinations.
The agent has no memory across runs and no access to maintenance history
or work order systems.
Action plans are recommendations, not executable commands. Production
integration would require a CMMS connection (SAP PM, Maximo, etc.).
Tech Stack
Layer
Technology
Analytics pipeline
Python, Pandas, Scikit-learn
Anomaly detection
IsolationForest
Agent framework
LangGraph, LangChain Anthropic
LLM
Claude (Anthropic)
Frontend
Streamlit
Notebooks
Jupyter
Author
Adrián Rodríguez Estévez
Naval engineer turned data scientist, with field experience in wind farm installation and project management in Spain and Brazil.

LinkedIn · GitHub