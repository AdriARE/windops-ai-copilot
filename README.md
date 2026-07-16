# WindOps AI Copilot

> Explainable predictive maintenance for wind turbine fleets using hybrid risk scoring, anomaly detection and an LLM-powered maintenance copilot.

---

# Overview

WindOps AI Copilot is an end-to-end AI application that simulates a wind farm SCADA environment, detects anomalous turbine behaviour, prioritises maintenance actions and generates explainable maintenance recommendations using a LangGraph agent.

The project combines synthetic SCADA generation, anomaly detection, hybrid risk scoring and Generative AI to transform operational data into transparent, evidence-based maintenance decisions.

---

# Features

- Synthetic SCADA data generation with configurable fault scenarios
- Feature engineering pipeline for turbine health indicators
- Isolation Forest anomaly detection
- Hybrid risk scoring combining engineering thresholds and fleet-relative behaviour
- Fleet-wide maintenance prioritisation
- LangGraph maintenance copilot
- Automatic Demo Mode fallback
- Password-protected Live Mode
- Interactive Streamlit dashboard
- Command Line Interface (CLI)
- JSON, CSV and PDF report export

---

# Architecture

```text
                    Raw SCADA Data
                          │
                          ▼
               Hourly Aggregation
                          │
                          ▼
              Feature Engineering
                          │
                          ▼
           Isolation Forest Detection
                          │
                          ▼
              Hybrid Risk Scoring
                          │
                          ▼
              Fleet Prioritisation
                         │
          ┌────────┴──────────┐
          │                                │
          ▼                                ▼
   LangGraph AI Agent              Streamlit Dashboard
          │                                │
          └─────────┬─────────┘
                           ▼
             Maintenance Action Plans
                           │
          ┌─────────┼─────────┐
          ▼               ▼               ▼
        JSON             CSV               PDF
```

The hybrid risk score combines:

- **70% Absolute Risk**, based on engineering thresholds.
- **30% Relative Risk**, normalised against the current fleet.

This approach prevents degraded fleet-wide conditions from masking individual turbine behaviour.

---

# Project Structure

```text
windops-ai-copilot/

├── app/
│   ├── agent.py                 # LangGraph agent
│   ├── app.py                   # Streamlit dashboard
│   └── cli.py                   # Command-line interface
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
│   ├── io.py                    # CSV, Markdown and PDF export
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

| Fault | Dominant Risk | Typical Behaviour |
|--------|---------------|-------------------|
| Gearbox degradation | Mechanical | High power loss with elevated gearbox temperature |
| Pitch malfunction | Aerodynamic | Reduced production despite normal wind conditions |
| Yaw misalignment | Aerodynamic | Persistent power gap caused by rotor misalignment |
| Sensor drift | Anomaly | Abnormal sensor behaviour detected by Isolation Forest |

---

# Quick Start

## Requirements

- Python 3.10 or newer

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
ANTHROPIC_API_KEY=your_api_key
LIVE_MODE_PASSWORD=your_password
```

Without an Anthropic API key the application runs entirely in **Demo Mode** using the built-in rule-based diagnostic engine.

Even with a valid `ANTHROPIC_API_KEY`, the Streamlit application always starts in **Demo Mode**. Real API calls must be explicitly unlocked from the sidebar using `LIVE_MODE_PASSWORD` to prevent unintended token usage.

---

# Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run app/app.py
```

The dashboard provides:

- Fleet overview
- Turbine inspection
- AI-generated maintenance recommendations
- Collapsible **Why** section explaining the rationale behind each action plan
- Agent execution trace
- PDF export
- Demo / Live Mode indicator

---

# Command Line Interface

Run the default scenario:

```bash
python -m app.cli
```

Run a different scenario:

```bash
python -m app.cli --scenario red
```

Analyse five turbines:

```bash
python -m app.cli --top-n 5
```

Export reports:

```bash
python -m app.cli --scenario mixed --top-n 5 --export
```

Run only the analytics pipeline:

```bash
python -m app.cli --no-agent
```

Display help:

```bash
python -m app.cli --help
```

> **Note**
>
> The CLI should be executed as a Python module (`python -m app.cli`) rather than as a script. Running it as a module preserves the package structure and avoids `sys.path` import issues.

---

# Notebooks

The project includes three notebooks documenting the development process.

| Notebook | Purpose |
|-----------|---------|
| **01** | Exploratory Data Analysis of synthetic SCADA data |
| **02** | Feature engineering, anomaly detection and hybrid risk scoring |
| **03** | LangGraph agent workflow and maintenance recommendations |

Run them sequentially for the complete project walkthrough.

---

# Hybrid Risk Score

The final turbine risk score combines three explainable subscores.

| Subscore | Weight | Main Signals |
|----------|--------|--------------|
| Aerodynamic | 50% | Power gap, yaw proxy, pitch instability |
| Mechanical | 30% | Gear oil temperature, vibration trend |
| Anomaly | 20% | Isolation Forest score, anomaly persistence |

Each subscore combines engineering thresholds with fleet-relative behaviour:

```text
Risk = 0.70 × Absolute Component
     + 0.30 × Relative Component
```

The final maintenance priority combines:

- **50% Risk Score**
- **35% Estimated Energy Loss**
- **15% Asset Criticality**

All weights are configurable in `src/config.py`.

---

# LangGraph Agent

The AI Copilot follows a ReAct workflow implemented with LangGraph.

Available tools:

- `get_priority_ranking`
- `get_turbine_details`
- `submit_action_plan`

The application automatically selects the appropriate execution mode:

| Mode | Behaviour |
|------|-----------|
| **Live Mode** | Uses Claude through the Anthropic API |
| **Demo Mode** | Uses the built-in deterministic diagnostic engine |

If the Anthropic API is unavailable or authentication fails, the application automatically falls back to Demo Mode while preserving the same workflow and output format.

Every maintenance recommendation includes its supporting risk subscores, relevant operational signals, tool calls and complete execution trace, allowing engineers to understand how each recommendation was generated.

---

# Example Action Plan

The AI agent generates structured maintenance recommendations in JSON format.

```json

{
  "turbine_id": "WTG-02",
  "urgency": "high",
  "fault_hypothesis": "Gearbox or drivetrain degradation",
  "recommended_action": "1. Schedule gearbox oil sampling and vibration analysis within 24 hours. 2. Monitor gearbox temperature during the next operating cycle. 3. Inspect the lubrication system and bearings during the next maintenance window.",
  "rationale": "Mechanical risk is dominant due to elevated gearbox oil temperature and sustained power loss, indicating probable drivetrain degradation."
}
```

Recommendations are displayed in the Streamlit dashboard and can also be exported as JSON, CSV and PDF reports.

---

# Demo Mode vs Live Mode

| Feature | Demo Mode | Live Mode |
|----------|-----------|-----------|
| Analytics Pipeline | ✅ | ✅ |
| Risk Scoring | ✅ | ✅ |
| Priority Ranking | ✅ | ✅ |
| Dashboard | ✅ | ✅ |
| PDF Export | ✅ | ✅ |
| Agent Trace | ✅ | ✅ |
| Claude API | ❌ | ✅ |

Demo Mode uses the project's deterministic rule-based diagnostic engine.

Live Mode replaces that engine with Claude through the Anthropic API while preserving the same workflow and output structure.

---

# Known Limitations

## Data

- SCADA data is synthetically generated.
- Signal behaviour is physically plausible but does not capture the full complexity of a real wind farm.
- A single generic 3 MW turbine model is currently simulated.

## Analytics

- Risk score thresholds are engineering heuristics rather than learned from historical failures.
- Isolation Forest is trained and evaluated on the same synthetic dataset.
- Historical maintenance records are not yet incorporated into the prioritisation process.

## AI Agent

- Live Mode requires a valid Anthropic API key.
- Live Mode consumes Anthropic API credits.
- The Streamlit dashboard always starts in **Demo Mode** and requires `LIVE_MODE_PASSWORD` to unlock Live Mode.
- **The CLI does not yet enforce this lock.** If a valid Anthropic API key is detected, it automatically runs in Live Mode. Aligning this behaviour with the dashboard is an outstanding task.
- Recommendations are intended to support engineering decisions, not replace engineering judgement.
- No integration with CMMS platforms (SAP PM, IBM Maximo, etc.) is currently implemented.

---

# Tech Stack

| Layer | Technology |
|---------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| AI Framework | LangGraph, LangChain |
| LLM | Claude (Anthropic) |
| Dashboard | Streamlit |
| Visualisation | Matplotlib, Seaborn |
| Reports | JSON, CSV, PDF |
| Environment | python-dotenv |
| Development | Jupyter Notebook |

---

# Future Improvements

The next planned milestones for the project are:

- Apply the same `LIVE_MODE_PASSWORD` protection to the CLI.
- Implement real SCADA ingestion through `src/ingesta.py`.
- Validate the pipeline using the Kelmarsh Wind Farm dataset.
- Add automatic fallback to DeepSeek when Anthropic is unavailable.
- Increase automated test coverage with `pytest`.
- Persist simulated datasets to avoid regenerating scenarios on every execution.
- Deploy the application on Streamlit Community Cloud.
- Record a complete demo video covering the dashboard, analytics pipeline and AI Copilot workflow.

---

# Author

**Adrián Rodríguez Estévez**

Engineer transitioning into Data Science and AI Engineering, with professional experience in wind farm installation, commissioning and project management across Europe and LATAM.

This project combines renewable energy domain expertise with Data Science, Machine Learning and Generative AI to build an explainable predictive maintenance workflow for wind turbine fleets.

**Connect with me**

- GitHub: https://github.com/AdriARE
- LinkedIn: https://www.linkedin.com/in/arestevez/

---

# License

This project is released for educational and portfolio purposes.

Feel free to explore the code, reproduce the analysis and use the project as inspiration for learning or research.
