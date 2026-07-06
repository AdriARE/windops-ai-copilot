# WindOps AI Copilot

> Predictive maintenance and fleet prioritisation for wind turbine operations powered by heuristic risk scoring and an LLM-based diagnostic agent.

---

# Overview

WindOps AI Copilot is an end-to-end predictive maintenance project that simulates a wind farm SCADA environment, detects anomalies, estimates operational risk and uses an LLM agent to generate explainable maintenance recommendations.

Unlike traditional black-box predictive maintenance solutions, every recommendation is fully traceable back to the signals and subscores that generated it.

---

# The Problem

Wind farm operators manage fleets of dozens or hundreds of turbines continuously generating SCADA data:

- Wind speed
- Power production
- Rotor speed
- Temperatures
- Availability
- Operational status

The difficult part is not collecting data.

The difficult part is deciding:

- Which turbine should be inspected first?
- Why?
- What evidence supports that decision?

WindOps AI Copilot addresses this using:

- Transparent heuristic risk scoring
- Isolation Forest anomaly detection
- Explainable prioritisation
- LangGraph + Claude diagnostic agent
- Structured maintenance recommendations

---

# Architecture

```text
Raw SCADA data (10 min)
        │
        ▼
Hourly aggregation
        │
        ▼
Feature engineering
        │
        ▼
Isolation Forest
        │
        ▼
Hybrid Risk Score
        │
        ▼
Priority Ranking
        │
        ▼
LangGraph Agent
        │
        ▼
Maintenance Action Plan
```

The final risk score is hybrid:

- **70% Absolute risk** (engineering thresholds)
- **30% Relative risk** (fleet-normalised)

This prevents an unhealthy fleet from masking its own problems.

---

# Project Structure

```text
windops-ai-copilot/
│
├── app/
│   ├── agent.py
│   ├── app.py
│   └── cli.py
│
├── notebooks/
│   ├── 01_eda_demo_data.ipynb
│   ├── 02_features_and_scoring.ipynb
│   └── 03_agent_decisions.ipynb
│
├── reports/
│
├── src/
│   ├── anomaly.py
│   ├── config.py
│   ├── data_generation.py
│   ├── expected_power.py
│   ├── features.py
│   ├── impact.py
│   ├── io.py
│   ├── prioritization.py
│   └── risk.py
│
├── requirements.txt
└── README.md
```

---

# Demo Scenarios

| Scenario | Description |
|----------|-------------|
| `green` | Healthy fleet |
| `gearbox` | Gearbox degradation |
| `pitch` | Pitch malfunction |
| `yaw` | Yaw misalignment |
| `mixed` | Multiple simultaneous faults |
| `red` | Severe fleet degradation |

### Fault signatures

| Fault | Power Gap | Gear Oil Temp | Anomaly Score | Dominant Subscore |
|--------|-----------|---------------|----------------|-------------------|
| Gearbox degradation | High | High | Moderate | Mechanical |
| Pitch malfunction | High | Normal | Low | Aerodynamic |
| Sensor drift | Variable | Normal | High | Anomaly |
| Yaw misalignment | High | Normal | Low | Aerodynamic |

---

# Quick Start

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

If no API key is available, the application automatically switches to **Demo Mode** using rule-based diagnosis.

---

# Running the Streamlit App

```bash
streamlit run app/app.py
```

---

# Command Line Interface

Default execution:

```bash
python app/cli.py
```

Custom scenario:

```bash
python app/cli.py --scenario red --top-n 5 --export
```

Pipeline only:

```bash
python app/cli.py --scenario green --no-agent
```

Help:

```bash
python app/cli.py --help
```

---

# Notebooks

Execute in order:

1. `01_eda_demo_data.ipynb`
2. `02_features_and_scoring.ipynb`
3. `03_agent_decisions.ipynb`

---

# Risk Score

The global score combines three explainable subscores.

| Subscore | Weight | Main Features |
|----------|--------|---------------|
| Aerodynamic | 50% | Power gap, yaw proxy, pitch instability |
| Mechanical | 30% | Gear oil temperature, vibration trend |
| Anomaly | 20% | Isolation Forest score, anomaly persistence |

Each subscore is calculated as:

```text
0.70 × Absolute Component
+
0.30 × Relative Component
```

Priority ranking combines:

- 50% Risk Score
- 35% Estimated Energy Loss
- 15% Asset Criticality

Weights are configurable in `src/config.py`.

---

# LangGraph Agent

The agent follows a ReAct workflow using three tools:

- `get_priority_ranking`
- `get_turbine_details`
- `submit_action_plan`

Example output:

```json
{
  "turbine_id": "WTG-02",
  "urgency": "high",
  "fault_hypothesis": "Gearbox degradation",
  "recommended_action": "Schedule gearbox oil sampling within 24 hours.",
  "rationale": "Mechanical subscore is dominant with elevated gearbox temperature."
}
```

If no Anthropic API key is configured, the project automatically switches to Demo Mode while preserving the same output format.

---

# Known Limitations

## Data

- Synthetic SCADA dataset
- Single generic 3 MW turbine model
- Simplified fault simulation

## Analytics

- Thresholds are manually defined
- Isolation Forest is trained and evaluated on the same dataset
- No historical maintenance records

## Agent

- LLM outputs are not formally validated
- No persistent memory
- No CMMS integration
- Recommendations are advisory only

---

# Tech Stack

| Layer | Technology |
|--------|------------|
| Language | Python |
| Data | Pandas |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| Agent Framework | LangGraph |
| LLM | Claude (Anthropic) |
| Frontend | Streamlit |
| Notebooks | Jupyter |

---

# Author

**Adrián Rodríguez Estévez**

Engineer transitioning into Data Science with professional experience in wind farm installation, commissioning and project management across Spain and Brazil.

- GitHub
- LinkedIn