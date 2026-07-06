# app/cli.py

# ===============================
# IMPORTS
# ===============================

# Standard library
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Local
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import SCENARIOS
from src.data_generation import load_demo_scenario
from src.features import build_features
from src.anomaly import run_anomaly_pipeline
from src.risk import run_risk_pipeline
from src.impact import run_impact_pipeline
from src.prioritization import run_prioritization_pipeline
from app.agent import run_agent_auto

logging.basicConfig(level=logging.WARNING)


# ===============================
# HELPERS
# ===============================

def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_header(title: str, width: int = 70) -> None:
    print_separator("═", width)
    print(f"  {title}")
    print_separator("═", width)


def print_priority_table(priority, top_n: int, fault_map: dict) -> None:
    cols = [c for c in [
        "priority_rank", "turbine_id", "priority_score",
        "risk_score_mean", "loss_mwh_total",
    ] if c in priority.columns]

    header = (
        f"  {'#':<5} {'Turbine':<10} {'Priority':>10} "
        f"{'Risk':>8} {'Loss MWh':>10}  Fault"
    )
    print(header)
    print_separator()

    for _, row in priority.head(top_n).iterrows():
        tid   = row.get("turbine_id", "?")
        fault = fault_map.get(tid, "none").replace("_", " ")
        print(
            f"  {int(row.get('priority_rank', 0)):<5} "
            f"{tid:<10} "
            f"{row.get('priority_score', 0):>10.3f} "
            f"{row.get('risk_score_mean', 0):>8.3f} "
            f"{row.get('loss_mwh_total', 0):>10.2f}  {fault}"
        )


def print_action_plans(plans: list[dict]) -> None:
    URGENCY_ICONS = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    for plan in plans:
        urgency = plan.get("urgency", "low")
        icon    = URGENCY_ICONS.get(urgency, "⚪")
        tid     = plan.get("turbine_id", "?")

        print_separator()
        print(f"  {icon}  {tid}  [{urgency.upper()}]")
        print(f"  Fault   : {plan.get('fault_hypothesis', '—')}")
        print(f"  Action  : {plan.get('recommended_action', '—')}")
        print(f"  Rationale: {plan.get('rationale', '—')}")

    print_separator()


def export_results(
    plans: list[dict],
    scenario: str,
    mode: str,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = reports_dir / f"action_plans_{scenario}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)

    # CSV
    import pandas as pd
    csv_rows = [
        {
            "turbine_id":         p.get("turbine_id", ""),
            "urgency":            p.get("urgency", ""),
            "fault_hypothesis":   p.get("fault_hypothesis", ""),
            "recommended_action": p.get("recommended_action", ""),
            "rationale":          p.get("rationale", ""),
            "scenario":           scenario,
            "agent_mode":         mode,
        }
        for p in plans
    ]
    csv_path = reports_dir / f"action_plans_{scenario}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\n  Exports saved to {reports_dir.resolve()}/")
    print(f"    JSON : {json_path.name}  ({json_path.stat().st_size} bytes)")
    print(f"    CSV  : {csv_path.name}  ({csv_path.stat().st_size} bytes)")


# ===============================
# MAIN
# ===============================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WindOps AI Copilot — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:<18} {v['description']}"
            for k, v in SCENARIOS.items()
        ),
    )
    parser.add_argument(
        "--scenario", "-s",
        default="mixed",
        choices=list(SCENARIOS.keys()),
        help="Fleet scenario to simulate (default: mixed)",
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int, default=3,
        help="Number of turbines to prioritise and analyse (default: 3)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip agent run, show pipeline output only",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export action plans to reports/ as JSON and CSV",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for exported reports (default: reports/)",
    )

    args = parser.parse_args()

    # ── Pipeline ──────────────────────────────────────────────────────────
    print_header(f"WindOps AI Copilot  |  Scenario: {args.scenario.upper()}")

    print("\n  Running analytics pipeline...")
    t0 = time.time()

    df_hourly   = load_demo_scenario(args.scenario)
    df_features = build_features(df_hourly)
    df_anomaly, _ = run_anomaly_pipeline(df_features)
    df_risk, risk_summary = run_risk_pipeline(df_anomaly)
    loss_summary = run_impact_pipeline(df_risk)
    priority     = run_prioritization_pipeline(risk_summary, loss_summary)
    fault_map    = df_hourly.groupby("turbine_id")["fault_type"].first().to_dict()

    pipeline_time = time.time() - t0

    # Fleet summary
    import pandas as pd
    n_turbines = df_hourly["turbine_id"].nunique()
    mean_avail = df_hourly["availability"].mean()
    total_loss = df_hourly["power_gap_kw"].clip(lower=0).sum() / 1000
    n_faulty   = sum(1 for v in fault_map.values() if v != "none")

    print(f"\n  Turbines      : {n_turbines}")
    print(f"  Faulty        : {n_faulty} / {n_turbines}")
    print(f"  Mean avail.   : {mean_avail:.1%}")
    print(f"  Energy loss   : {total_loss:.1f} MWh")
    print(f"  Pipeline time : {pipeline_time:.2f}s")

    # Priority table
    print(f"\n  Top {args.top_n} turbines by priority:\n")
    print_priority_table(priority, args.top_n, fault_map)

    if args.no_agent:
        print("\n  Agent skipped (--no-agent). Done.")
        return

    # ── Agent ─────────────────────────────────────────────────────────────
    print("\n  Running agent...")
    t1 = time.time()
    plans, trace, mode = run_agent_auto(priority, df_risk, top_n=args.top_n)
    agent_time = time.time() - t1
    tool_calls = sum(1 for s in trace if s["step"] == "tool_call")

    mode_label = "LIVE (Claude API)" if mode == "live" else "DEMO (rule-based fallback)"
    print(f"  Mode          : {mode_label}")
    print(f"  Agent time    : {agent_time:.2f}s")
    print(f"  Tool calls    : {tool_calls}")
    print(f"  Action plans  : {len(plans)}")

    print(f"\n  Action Plans:\n")
    print_action_plans(plans)

    # ── Export ────────────────────────────────────────────────────────────
    if args.export and plans:
        reports_dir = Path(args.reports_dir)
        export_results(plans, args.scenario, mode, reports_dir)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()