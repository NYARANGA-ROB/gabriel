"""Generate model comparison PDF reports."""

from __future__ import annotations

import io
from datetime import datetime

import plotly.graph_objects as go
from plotly.io import to_image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _figure_to_png(chart_dict: dict, width: int = 900, height: int = 480) -> bytes:
    figure = go.Figure(data=chart_dict.get("data", []), layout=chart_dict.get("layout", {}))
    return to_image(figure, format="png", width=width, height=height, engine="kaleido")


def generate_comparison_pdf(comparison: dict, app_name: str) -> bytes:
    """Build a PDF report for the model comparison dashboard."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#64748b"),
        spaceAfter=16,
    )

    story = [
        Paragraph("Model Comparison Report", title_style),
        Paragraph(
            f"{app_name} &mdash; generated {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}",
            subtitle_style,
        ),
    ]

    if not comparison.get("has_models"):
        story.append(Paragraph("No completed model evaluations are available for comparison.", styles["Normal"]))
        document.build(story)
        buffer.seek(0)
        return buffer.read()

    table_data = [
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "Train (ms)",
            "Pred (ms)",
        ]
    ]

    best_model_id = comparison.get("best_model_id")
    for row in comparison["models"]:
        prefix = "* " if row["id"] == best_model_id else ""
        table_data.append(
            [
                f"{prefix}{row['label']}",
                f"{row['accuracy'] * 100:.1f}%",
                f"{row['precision'] * 100:.1f}%",
                f"{row['recall'] * 100:.1f}%",
                f"{row['f1_score'] * 100:.1f}%",
                f"{row['roc_auc']:.3f}" if row["roc_auc"] is not None else "—",
                f"{row['training_time_ms']:.1f}",
                f"{row['prediction_time_ms']:.2f}",
            ]
        )

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.2 * inch)])

    if best_model_id:
        best_row = next(row for row in comparison["models"] if row["id"] == best_model_id)
        roc_auc_text = (
            f"{best_row['roc_auc']:.3f}" if best_row["roc_auc"] is not None else "N/A"
        )
        story.append(
            Paragraph(
                f"<b>Best overall model:</b> {best_row['label']} "
                f"(F1 = {best_row['f1_score'] * 100:.1f}%, ROC AUC = {roc_auc_text})",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.15 * inch))

    chart_titles = {
        "roc_comparison": "ROC Curve Comparison",
        "accuracy": "Accuracy Comparison",
        "precision": "Precision Comparison",
        "recall": "Recall Comparison",
        "f1_score": "F1 Score Comparison",
        "training_time": "Training Time Comparison",
        "prediction_time": "Prediction Time Comparison",
    }

    for key, title in chart_titles.items():
        chart = comparison.get("charts", {}).get(key)
        if not chart:
            continue
        story.append(Paragraph(title, styles["Heading3"]))
        story.append(Spacer(1, 0.08 * inch))
        try:
            png_bytes = _figure_to_png(chart)
            story.append(Image(io.BytesIO(png_bytes), width=6.8 * inch, height=3.5 * inch))
        except Exception:
            story.append(Paragraph("Chart could not be rendered in PDF export.", styles["Italic"]))
        story.append(Spacer(1, 0.15 * inch))

    document.build(story)
    buffer.seek(0)
    return buffer.read()
