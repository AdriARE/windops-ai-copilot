# src/agent.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import json
import logging
from typing import Annotated, TypedDict

# Third-party
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Local
from src.config import AGENT_MODEL

logger = logging.getLogger(__name__)

# ===============================
# AGENT STATE
# ===============================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ===============================
# SYSTEM PROMPT
# ===============================

SYSTEM_PROMPT = """You are WindOps Copilot, an AI assistant specialised in wind turbine
operations and predictive maintenance.

You have access to fleet analytics computed from SCADA data: risk scores, energy loss
estimates and sensor signals.

Your task:
1. Call get_priority_ranking to identify the turbines requiring attention.
2. For each of the top 3 turbines, call get_turbine_details to understand the subscore pattern.
3. Diagnose the most likely fault based on subscores:
   - High aero_risk, low mech_risk → aerodynamic issue (pitch or yaw misalignment)
   - High mech_risk, elevated gear_oil_temp → gearbox or drivetrain degradation
   - High anomaly_risk with ambiguous subscores → sensor drift or instrumentation fault
   - High across multiple subscores → combined degradation, inspect immediately
4. Call submit_action_plan for each turbine with a concrete, operational recommendation.

Be specific. Recommendations must describe actual maintenance actions, not generic advice.
Reference the specific signals that support your diagnosis.
"""


# ===============================
# TOOL FACTORY
# ===============================

def build_tools(
    priority_df: pd.DataFrame,
    df_risk: pd.DataFrame,
) -> tuple[list, list]:
    """
    Build LangGraph tools with access to current pipeline data via closures.

    Returns (tools_list, action_plans_list).
    action_plans is populated as a side effect of submit_action_plan calls.
    """
    action_plans: list[dict] = []

    @tool
    def get_priority_ranking(top_n: int = 5) -> str:
        """
        Returns the top N turbines sorted by priority score.
        Includes priority_score, risk_score_mean, loss_mwh_total and criticality.
        """
        cols = [
            "turbine_id", "priority_rank", "priority_score",
            "risk_score_mean", "loss_mwh_total", "criticality",
        ]
        available = [c for c in cols if c in priority_df.columns]
        records = priority_df[available].head(top_n).to_dict(orient="records")
        return json.dumps(records, default=float)

    @tool
    def get_turbine_details(turbine_id: str) -> str:
        """
        Returns the latest risk subscores and key sensor values for a specific turbine.
        Subscores: aero_risk, mech_risk, anomaly_risk, risk_score.
        Sensors: power_gap_pct, gear_oil_temp_c, anomaly_score, anomaly_flag.
        """
        subset = df_risk[df_risk["turbine_id"] == turbine_id]
        if subset.empty:
            return json.dumps({"error": f"Turbine {turbine_id} not found."})

        latest = subset.sort_values("timestamp").iloc[-1]

        def safe_float(key: str, decimals: int = 3) -> float:
            val = latest.get(key, 0)
            try:
                return round(float(val), decimals)
            except (TypeError, ValueError):
                return 0.0

        result = {
            "turbine_id": turbine_id,
            "risk_score": safe_float("risk_score"),
            "risk_level": str(latest.get("risk_level", "unknown")),
            "aero_risk": safe_float("aero_risk"),
            "mech_risk": safe_float("mech_risk"),
            "anomaly_risk": safe_float("anomaly_risk"),
            "power_gap_pct": safe_float("power_gap_pct"),
            "gear_oil_temp_c": safe_float("gear_oil_temp_c", 1),
            "anomaly_score": safe_float("anomaly_score"),
            "anomaly_flag": int(latest.get("anomaly_flag", 0)),
        }
        return json.dumps(result, default=float)

    @tool
    def submit_action_plan(
        turbine_id: str,
        urgency: str,
        fault_hypothesis: str,
        recommended_action: str,
        rationale: str,
    ) -> str:
        """
        Submit a structured action plan for a turbine.

        Args:
            turbine_id: Turbine identifier (e.g. WTG-02).
            urgency: One of low, medium, high.
            fault_hypothesis: Suspected fault type based on subscore analysis.
            recommended_action: Specific maintenance action to take.
            rationale: Brief explanation linking data signals to the recommendation.
        """
        plan = {
            "turbine_id": turbine_id,
            "urgency": urgency,
            "fault_hypothesis": fault_hypothesis,
            "recommended_action": recommended_action,
            "rationale": rationale,
        }
        action_plans.append(plan)
        logger.info("Action plan accepted | turbine=%s | urgency=%s", turbine_id, urgency)
        return json.dumps({"status": "accepted", "turbine_id": turbine_id})

    return [get_priority_ranking, get_turbine_details, submit_action_plan], action_plans


# ===============================
# GRAPH BUILDER
# ===============================

def build_graph(tools: list) -> StateGraph:
    """Build the LangGraph ReAct graph with the given tools."""
    llm = ChatAnthropic(model=AGENT_MODEL).bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ===============================
# PIPELINE RUNNER
# ===============================

def run_agent(
    priority_df: pd.DataFrame,
    df_risk: pd.DataFrame,
    top_n: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Run the WindOps agent on the current fleet state.

    Args:
        priority_df: Turbine-level priority ranking from run_prioritization_pipeline().
        df_risk: Row-level risk data from run_risk_pipeline().
        top_n: Number of turbines to analyse.

    Returns:
        Tuple of (action_plans, trace).
        action_plans: List of structured dicts, one per turbine.
        trace: List of step dicts showing tool calls and model responses.
    """
    tools, action_plans = build_tools(priority_df, df_risk)
    graph = build_graph(tools)

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Analyse the current fleet state and generate action plans for the top "
                    f"{top_n} priority turbines. Use the available tools to inspect the data "
                    f"and submit a concrete maintenance recommendation for each turbine."
                )
            ),
        ]
    }

    final_state = graph.invoke(initial_state)

    trace = _build_trace(final_state["messages"])

    logger.info(
        "Agent completed | action_plans=%d | trace_steps=%d",
        len(action_plans),
        len(trace),
    )

    return action_plans, trace


# ===============================
# TRACE BUILDER
# ===============================

def _build_trace(messages: list) -> list[dict]:
    """Extract a structured trace from the final message list."""
    trace = []
    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage":
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    trace.append({
                        "step": "tool_call",
                        "tool": tc.get("name", "unknown"),
                        "input": tc.get("args", {}),
                    })
            content = getattr(msg, "content", "")
            if content:
                trace.append({
                    "step": "ai_message",
                    "content": str(content)[:500],
                })

        elif msg_type == "ToolMessage":
            content = getattr(msg, "content", "")
            trace.append({
                "step": "tool_result",
                "content": str(content)[:500],
            })

    return trace


# ===============================
# MAIN — manual test
# ===============================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.anomaly import run_anomaly_pipeline
    from src.data_generation import load_demo_scenario
    from src.features import build_features
    from src.impact import run_impact_pipeline
    from src.prioritization import run_prioritization_pipeline
    from src.risk import run_risk_pipeline

    print("Loading mixed scenario...")
    df_hourly = load_demo_scenario("mixed")
    df_features = build_features(df_hourly)
    df_anomaly, _ = run_anomaly_pipeline(df_features)
    df_risk, risk_summary = run_risk_pipeline(df_anomaly)
    loss_summary = run_impact_pipeline(df_risk)
    priority = run_prioritization_pipeline(risk_summary, loss_summary)

    print("Running agent...")
    action_plans, trace = run_agent(priority, df_risk, top_n=3)

    print(f"\nTrace ({len(trace)} steps):")
    for i, step in enumerate(trace, 1):
        print(f"  {i}. [{step['step']}]", end=" ")
        if step["step"] == "tool_call":
            print(f"{step['tool']}({step['input']})")
        else:
            print(step.get("content", "")[:120])

    print(f"\nAction plans ({len(action_plans)}):")
    for plan in action_plans:
        print(f"\n  {plan['turbine_id']} [{plan['urgency'].upper()}]")
        print(f"  Fault: {plan['fault_hypothesis']}")
        print(f"  Action: {plan['recommended_action']}")
        print(f"  Rationale: {plan['rationale']}")