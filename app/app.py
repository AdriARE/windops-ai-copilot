# app/app.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import sys
import logging
from pathlib import Path

# Third-party
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

# Local — src modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import SCENARIOS
from src.data_generation import load_demo_scenario
from src.features import build_features
from src.anomaly import run_anomaly_pipeline
from src.risk import run_risk_pipeline
from src.impact import run_impact_pipeline
from src.prioritization import run_prioritization_pipeline
from src.expected_power import expected_power
from app.agent import run_agent_auto

logging.basicConfig(level=logging.WARNING)

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="WindOps AI Copilot",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# CONSTANTS
# ===============================

FAULT_COLORS = {
    "none":                "steelblue",
    "gearbox_degradation": "tomato",
    "pitch_malfunction":   "darkorange",
    "sensor_drift":        "purple",
    "yaw_misalignment":    "sienna",
}

URGENCY_COLORS = {
    "high":   "#d9534f",
    "medium": "#f0ad4e",
    "low":    "#5bc0de",
}

URGENCY_ICONS = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "🟢",
}

PLOT_STYLE = "seaborn-v0_8-whitegrid"
plt.style.use(PLOT_STYLE)


# ===============================
# CACHED PIPELINE
# ===============================

@st.cache_data(show_spinner="Running analytics pipeline...")
def run_pipeline(scenario: str) -> dict:
    """Run the full analytics pipeline for a given scenario. Results are cached."""
    df_hourly         = load_demo_scenario(scenario)
    df_features       = build_features(df_hourly)
    df_anomaly, model = run_anomaly_pipeline(df_features)
    df_risk, rs       = run_risk_pipeline(df_anomaly)
    loss_summary      = run_impact_pipeline(df_risk)
    priority          = run_prioritization_pipeline(rs, loss_summary)
    fault_map         = df_hourly.groupby("turbine_id")["fault_type"].first().to_dict()

    return {
        "df_hourly":  df_hourly,
        "df_features": df_features,
        "df_risk":    df_risk,
        "risk_summary": rs,
        "loss_summary": loss_summary,
        "priority":   priority,
        "fault_map":  fault_map,
    }


# ===============================
# SIDEBAR
# ===============================

with st.sidebar:
    st.title("🌬️ WindOps AI Copilot")
    st.markdown("---")

    scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index("mixed"),
        help="Each scenario configures a different fault pattern across the fleet.",
    )
    st.caption(SCENARIOS[scenario]["description"])

    top_n = st.slider("Turbines to analyse", min_value=1, max_value=5, value=3)

    st.markdown("---")
    run_agent = st.button("▶ Run Agent", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("WindOps AI Copilot — portfolio demo")
    st.caption("Data: synthetic SCADA-like simulation")


# ===============================
# LOAD PIPELINE
# ===============================

data = run_pipeline(scenario)

df_hourly   = data["df_hourly"]
df_features = data["df_features"]
df_risk     = data["df_risk"]
priority    = data["priority"]
fault_map   = data["fault_map"]


# ===============================
# TABS
# ===============================

tab_fleet, tab_turbine, tab_copilot = st.tabs(
    ["🗺️ Fleet Overview", "🔍 Turbine Detail", "🤖 Copilot"]
)


# ===============================
# TAB 1 — FLEET OVERVIEW
# ===============================

with tab_fleet:
    st.header(f"Fleet Overview — {scenario.upper()} scenario")

    # --- Key metrics ---
    n_turbines    = df_hourly["turbine_id"].nunique()
    mean_avail    = df_hourly["availability"].mean()
    total_loss    = df_hourly["power_gap_kw"].clip(lower=0).sum() / 1000
    n_faulty      = sum(1 for v in fault_map.values() if v != "none")
    mean_risk     = priority["risk_score_mean"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Turbines",          n_turbines)
    c2.metric("Faulty turbines",   f"{n_faulty} / {n_turbines}")
    c3.metric("Mean availability", f"{mean_avail:.1%}")
    c4.metric("Energy loss",       f"{total_loss:.1f} MWh")
    c5.metric("Mean risk score",   f"{mean_risk:.3f}")

    st.markdown("---")

    # --- Priority ranking bar chart ---
    st.subheader("Priority Ranking")

    analysed_ids = set(priority.head(top_n)["turbine_id"])

    bar_colors = []
    for _, row in priority.iterrows():
        tid = row["turbine_id"]
        ft  = fault_map.get(tid, "none")
        bar_colors.append(FAULT_COLORS.get(ft, "steelblue"))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    bars = ax.bar(priority["turbine_id"], priority["priority_score"],
                  color=bar_colors, edgecolor="white", alpha=0.88)

    # Highlight top N
    for i, (_, row) in enumerate(priority.iterrows()):
        if row["turbine_id"] in analysed_ids:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(1.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Priority score")
    ax.set_title("Turbine priority ranking (bold border = top N selected for agent)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    legend_items = [
        mpatches.Patch(color=c, label=ft.replace("_", " ").title())
        for ft, c in FAULT_COLORS.items()
        if ft in set(fault_map.values())
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Risk heatmap ---
    st.subheader("Risk Score Heatmap")

    risk_pivot = (
        df_risk.groupby("turbine_id")[["aero_risk", "mech_risk", "anomaly_risk", "risk_score"]]
        .mean()
        .sort_values("risk_score", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        risk_pivot,
        annot=True, fmt=".2f",
        cmap="RdYlGn_r", vmin=0, vmax=1,
        linewidths=0.4, ax=ax,
    )
    ax.set_title("Mean risk subscores by turbine")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Priority table ---
    st.subheader("Priority Table")
    display_cols = [c for c in [
        "priority_rank", "turbine_id", "priority_score",
        "risk_score_mean", "loss_mwh_total", "criticality",
    ] if c in priority.columns]
    st.dataframe(
        priority[display_cols].style.format({
            "priority_score":  "{:.3f}",
            "risk_score_mean": "{:.3f}",
            "loss_mwh_total":  "{:.2f}",
            "criticality":     "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ===============================
# TAB 2 — TURBINE DETAIL
# ===============================

with tab_turbine:
    st.header("Turbine Detail")

    turbine_ids = sorted(df_hourly["turbine_id"].unique())
    selected_tid = st.selectbox("Select turbine", turbine_ids)

    fault_type = fault_map.get(selected_tid, "none")
    fault_label = fault_type.replace("_", " ").title()
    color = FAULT_COLORS.get(fault_type, "steelblue")

    st.markdown(f"**Fault type:** `{fault_label}`")

    # Subscore breakdown
    latest_risk = (
        df_risk[df_risk["turbine_id"] == selected_tid]
        .sort_values("timestamp")
        .iloc[-1]
    )

    def safe_metric(key: str) -> float:
        try:
            return round(float(latest_risk.get(key, 0)), 3)
        except (TypeError, ValueError):
            return 0.0

    st.subheader("Latest Risk Subscores")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Risk Score",     safe_metric("risk_score"))
    sc2.metric("Aerodynamic",    safe_metric("aero_risk"))
    sc3.metric("Mechanical",     safe_metric("mech_risk"))
    sc4.metric("Anomaly",        safe_metric("anomaly_risk"))

    st.markdown("---")

    # Power time series
    st.subheader("Power Output — Actual vs Expected")

    subset = df_hourly[df_hourly["turbine_id"] == selected_tid].copy()
    subset["timestamp"] = pd.to_datetime(subset["timestamp"])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(subset["timestamp"], subset["expected_power_kw"],
                    alpha=0.10, color="grey")
    ax.plot(subset["timestamp"], subset["expected_power_kw"],
            linewidth=0.8, color="grey", linestyle="--", label="Expected")
    ax.plot(subset["timestamp"], subset["power_kw"],
            linewidth=0.9, color=color, label="Actual")
    ax.fill_between(
        subset["timestamp"],
        subset["power_kw"], subset["expected_power_kw"],
        where=subset["expected_power_kw"] > subset["power_kw"],
        alpha=0.20, color="red", label="Gap",
    )
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"{selected_tid} — {fault_label}")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Sensor signals
    st.subheader("Sensor Signals")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].plot(subset["timestamp"], subset["gear_oil_temp_c"],
                 linewidth=0.9, color=color)
    axes[0].axhline(65, color="orange", linestyle="--", linewidth=0.8, label="Low threshold")
    axes[0].axhline(80, color="red",    linestyle="--", linewidth=0.8, label="High threshold")
    axes[0].set_title("Gear oil temperature (°C)")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(subset["timestamp"], subset["rotor_rpm"],
                 linewidth=0.9, color=color)
    axes[1].set_title("Rotor RPM")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Anomaly score
    st.subheader("Anomaly Score")

    subset_risk = df_risk[df_risk["turbine_id"] == selected_tid].copy()
    subset_risk["timestamp"] = pd.to_datetime(subset_risk["timestamp"])

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(subset_risk["timestamp"], subset_risk["anomaly_score"],
            linewidth=0.9, color=color)
    ax.axhline(0.5, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Anomaly score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ===============================
# TAB 3 — COPILOT
# ===============================

with tab_copilot:
    st.header("🤖 Copilot — Agent Decisions")

    # Agent state in session
    if "agent_results" not in st.session_state:
        st.session_state["agent_results"] = None
    if "agent_scenario" not in st.session_state:
        st.session_state["agent_scenario"] = None

    # Reset if scenario changed
    if st.session_state["agent_scenario"] != scenario:
        st.session_state["agent_results"] = None

    if run_agent or st.session_state["agent_results"] is not None:

        if run_agent or st.session_state["agent_results"] is None:
            with st.spinner("Agent is analysing the fleet..."):
                import time
                t0 = time.time()
                plans, trace, mode = run_agent_auto(priority, df_risk, top_n=top_n)
                elapsed = time.time() - t0
                tool_calls = sum(1 for s in trace if s["step"] == "tool_call")

            st.session_state["agent_results"] = {
                "plans":      plans,
                "trace":      trace,
                "mode":       mode,
                "elapsed":    elapsed,
                "tool_calls": tool_calls,
            }
            st.session_state["agent_scenario"] = scenario

        results = st.session_state["agent_results"]
        plans      = results["plans"]
        trace      = results["trace"]
        mode       = results["mode"]
        elapsed    = results["elapsed"]
        tool_calls = results["tool_calls"]

        # Mode banner
        if mode == "live":
            st.success("✅ LIVE MODE — responses generated by Claude via Anthropic API")
        else:
            st.info("🔵 DEMO MODE — ANTHROPIC_API_KEY not found or invalid. Using rule-based fallback.")

        # Performance metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Execution time",    f"{elapsed:.2f}s")
        m2.metric("Tool calls",        tool_calls)
        m3.metric("Action plans",      len(plans))

        st.markdown("---")

        # Action plans
        st.subheader("Action Plans")

        if not plans:
            st.warning("No action plans were generated.")
        else:
            for plan in plans:
                urgency = plan.get("urgency", "low")
                icon    = URGENCY_ICONS.get(urgency, "⚪")
                color_h = URGENCY_COLORS.get(urgency, "#aaa")
                tid     = plan.get("turbine_id", "?")

                with st.expander(
                    f"{icon} **{tid}** — {plan.get('fault_hypothesis', '—')} [{urgency.upper()}]",
                    expanded=True,
                ):
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.markdown(f"**Turbine**")
                        st.markdown(f"**Urgency**")
                        st.markdown(f"**Fault hypothesis**")
                    with col_r:
                        st.markdown(f"`{tid}`")
                        st.markdown(
                            f"<span style='color:{color_h};font-weight:bold'>"
                            f"{icon} {urgency.upper()}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(plan.get("fault_hypothesis", "—"))

                    st.markdown("**Recommended action**")
                    st.info(plan.get("recommended_action", "—"))
                    st.markdown("**Rationale**")
                    st.caption(plan.get("rationale", "—"))

        st.markdown("---")

        # Agent trace
        st.subheader("Agent Trace")

        import json

        trace_rows = []
        for i, step in enumerate(trace, 1):
            kind = step.get("step", "unknown")

            if kind == "tool_call":
                args    = step.get("input", {})
                summary = ", ".join(f"{k}={v}" for k, v in args.items())
                trace_rows.append({
                    "#":      i,
                    "Type":   "→ Tool Call",
                    "Tool":   step.get("tool", "—"),
                    "Detail": summary[:100],
                })

            elif kind == "tool_result":
                content = step.get("content", "")
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        summary = f"{len(parsed)} record(s)"
                    elif isinstance(parsed, dict):
                        summary = ", ".join(
                            f"{k}: {str(v)[:25]}"
                            for k, v in list(parsed.items())[:3]
                        )
                    else:
                        summary = str(parsed)[:100]
                except Exception:
                    summary = str(content)[:100]
                trace_rows.append({
                    "#":      i,
                    "Type":   "← Tool Result",
                    "Tool":   "—",
                    "Detail": summary,
                })

            elif kind == "ai_message":
                content = step.get("content", "").strip()
                if content:
                    trace_rows.append({
                        "#":      i,
                        "Type":   "◆ Agent",
                        "Tool":   "—",
                        "Detail": content[:120],
                    })

        if trace_rows:
            st.dataframe(
                pd.DataFrame(trace_rows),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info("Press **▶ Run Agent** in the sidebar to generate action plans.")
        st.markdown(
            "The agent will inspect the top N priority turbines, "
            "diagnose the most likely fault and submit a structured maintenance recommendation."
        )