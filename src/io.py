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
# PDF REPORT
# ===============================
from datetime import date
from fpdf import FPDF

URGENCY_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
URGENCY_COLOR = {
    "high":   (220, 50,  50),
    "medium": (230, 140, 30),
    "low":    (50,  160, 80),
}


def export_pdf_report(
    action_plans: list[dict],
    scenario: str,
    report_date: date | None = None,
    filename: str | None = None,
) -> Path:
    """Export agent action plans to a formatted PDF report."""
    ensure_directory(REPORTS_DIR)

    if report_date is None:
        report_date = date.today()
    if filename is None:
        filename = f"action_plans_{report_date.isoformat()}.pdf"

    output_path = REPORTS_DIR / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 20, 20)

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(30)
    pdf.cell(0, 12, "WindOps AI Copilot", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "Maintenance Action Plans", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Scenario: {scenario.upper()}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, f"Date: {report_date.strftime('%d %B %Y')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, f"Turbines analysed: {len(action_plans)}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)

    # One page per turbine
    for plan in action_plans:
        pdf.add_page()

        turbine_id = plan.get("turbine_id", "—")
        urgency    = plan.get("urgency", "low").lower()
        hypothesis = plan.get("fault_hypothesis", "—")
        action     = plan.get("recommended_action", "—")
        rationale  = plan.get("rationale", "—")

        r, g, b = URGENCY_COLOR.get(urgency, (100, 100, 100))

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, turbine_id, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(40, 7, f"  {URGENCY_LABEL.get(urgency, urgency.upper())}",
                 fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        _pdf_section(pdf, "Fault Hypothesis", hypothesis)
        _pdf_section(pdf, "Recommended Action", action)
        _pdf_section(pdf, "Rationale", rationale)

    pdf.output(str(output_path))
    logger.info("PDF report exported: %s", output_path)
    return output_path


def _pdf_section(pdf: FPDF, title: str, body: str) -> None:
    """Render a labeled section block inside a turbine page."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, _sanitize_text(_format_action_steps(body)))
    pdf.ln(6)

def _sanitize_text(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    replacements = {
        "\u2014": "-",    # em-dash —
        "\u2013": "-",    # en-dash –
        "\u2018": "'",    # left single quote '
        "\u2019": "'",    # right single quote '
        "\u201c": '"',    # left double quote "
        "\u201d": '"',    # right double quote "
        "\u2022": "*",    # bullet •
        "\u2026": "...",  # ellipsis …
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text
def _format_action_steps(text: str) -> str:
    """Normalize numbered steps to newline-separated format for PDF rendering."""
    import re
    normalized = re.sub(r'\((\d+)\)', r'\1.', text)
    return re.sub(r'\s+([2-9]\d*)\.\s+(?=[A-Z])', r'\n\1. ', normalized).strip()

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