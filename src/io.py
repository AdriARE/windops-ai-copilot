# src/io.py

# ===============================
# IMPORTS
# ===============================
# Standard library
import logging
from pathlib import Path

# Third-party
import pandas as pd

# Local
from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# ===============================
# DIRECTORY MANAGEMENT
# ===============================
def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)


# ===============================
# CSV EXPORT
# ===============================
def export_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    Export a DataFrame to CSV inside the reports directory.
    """

    ensure_directory(REPORTS_DIR)

    output_path = REPORTS_DIR / filename

    df.to_csv(output_path, index=False)

    logger.info("CSV exported: %s", output_path)

    return output_path


# ===============================
# MARKDOWN REPORT
# ===============================
def export_markdown_report(
    priority_df: pd.DataFrame,
    filename: str = "sample_report.md",
    top_n: int = 5,
) -> Path:
    """
    Export a simple markdown report with the highest priority turbines.
    """

    ensure_directory(REPORTS_DIR)

    output_path = REPORTS_DIR / filename

    top = priority_df.head(top_n)

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("# WindOps AI Copilot Report\n\n")

        f.write("## Top priority turbines\n\n")

        f.write(
            top[
                [
                    "priority_rank",
                    "turbine_id",
                    "priority_score",
                    "risk_score_mean",
                    "loss_mwh_total",
                    "criticality",
                ]
            ].to_markdown(index=False)
        )

    logger.info("Markdown report exported: %s", output_path)

    return output_path


# ===============================
# PIPELINE WRAPPER
# ===============================
def export_results(
    priority_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Export the default project outputs.
    """

    csv_path = export_csv(
        priority_df,
        "priority_ranking.csv",
    )

    md_path = export_markdown_report(
        priority_df,
    )

    return csv_path, md_path


# ===============================
# MAIN — manual test
# ===============================
if __name__ == "__main__":

    from src.anomaly import run_anomaly_pipeline
    from src.data_generation import load_demo_scenario
    from src.features import build_features
    from src.impact import run_impact_pipeline
    from src.prioritization import run_prioritization_pipeline
    from src.risk import run_risk_pipeline

    print("Loading red scenario...")
    df_hourly = load_demo_scenario("red")

    print("Building features...")
    df_features = build_features(df_hourly)

    print("Running anomaly detection...")
    df_anomaly, _ = run_anomaly_pipeline(df_features)

    print("Computing risk scores...")
    df_risk, risk_summary = run_risk_pipeline(df_anomaly)

    print("Estimating losses...")
    loss_summary = run_impact_pipeline(df_risk)

    print("Computing priority ranking...")
    priority = run_prioritization_pipeline(
        risk_summary,
        loss_summary,
    )

    print("Exporting reports...")
    csv_path, md_path = export_results(priority)

    print(f"\nCSV: {csv_path}")
    print(f"Markdown: {md_path}")