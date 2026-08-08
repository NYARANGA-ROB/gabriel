"""Generate admin system overview PDF reports."""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_system_report_pdf(stats: dict, user_activity: list[dict], app_name: str) -> bytes:
    """Build a PDF summary of platform usage and system statistics."""
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
    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    story = [
        Paragraph("System Administration Report", title_style),
        Paragraph(
            f"{app_name} &mdash; generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Normal"],
        ),
        Spacer(1, 0.2 * inch),
        Paragraph("Platform statistics", section_style),
    ]

    summary_rows = [
        ["Metric", "Value"],
        ["Total users", str(stats["users"]["total"])],
        ["Active users", str(stats["users"]["active"])],
        ["Admin users", str(stats["users"]["admins"])],
        ["Uploaded datasets", str(stats["datasets"]["uploaded"])],
        ["Processed datasets", str(stats["datasets"]["processed"])],
        ["Trained models", str(stats["models"]["total"])],
        ["Completed models", str(stats["models"]["completed"])],
        ["Prediction logs", str(stats["predictions"]["total"])],
        ["High-risk predictions", str(stats["predictions"]["high_risk"])],
    ]
    summary_table = Table(summary_rows, colWidths=[3.2 * inch, 2.4 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 0.2 * inch), Paragraph("User activity", section_style)])

    activity_rows = [
        ["User", "Role", "Datasets", "Models", "Predictions", "Status"],
    ]
    for row in user_activity:
        activity_rows.append(
            [
                row["email"],
                row["role"].title(),
                str(row["dataset_count"]),
                str(row["model_count"]),
                str(row["prediction_count"]),
                "Active" if row["is_active"] else "Inactive",
            ]
        )

    activity_table = Table(
        activity_rows,
        colWidths=[1.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch, 0.8 * inch],
        repeatRows=1,
    )
    activity_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(activity_table)

    document.build(story)
    return buffer.getvalue()
