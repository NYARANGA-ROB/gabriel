"""Generate prediction history PDF exports."""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.models.prediction_record import PredictionRecord


def generate_history_pdf(records: list[PredictionRecord], app_name: str) -> bytes:
    """Build a PDF report for prediction history records."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
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
        Paragraph("Prediction History Report", title_style),
        Paragraph(
            f"{app_name} &mdash; generated {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}",
            subtitle_style,
        ),
        Paragraph(f"<b>Total records:</b> {len(records)}", styles["Normal"]),
        Spacer(1, 0.15 * inch),
    ]

    if not records:
        story.append(Paragraph("No prediction records match the selected filters.", styles["Normal"]))
    else:
        table_data = [
            ["Patient ID", "Prediction", "Probability", "Risk", "Model", "Date (UTC)"]
        ]
        for record in records:
            table_data.append(
                [
                    record.patient_id,
                    record.prediction_label,
                    f"{record.probability * 100:.1f}%",
                    record.risk_level,
                    record.model_name,
                    record.created_at.strftime("%d %b %Y %H:%M"),
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
                    ("ALIGN", (2, 1), (2, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        story.append(table)

    document.build(story)
    buffer.seek(0)
    return buffer.read()
