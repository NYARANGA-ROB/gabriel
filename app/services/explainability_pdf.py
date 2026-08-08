"""Generate SHAP and LIME explainability PDF reports."""

from __future__ import annotations

import io
from datetime import datetime

import plotly.graph_objects as go
from plotly.io import to_image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


def _figure_to_png(chart_dict: dict, width: int = 900, height: int = 480) -> bytes:
    figure = go.Figure(data=chart_dict.get("data", []), layout=chart_dict.get("layout", {}))
    return to_image(figure, format="png", width=width, height=height, engine="kaleido")


def generate_explainability_pdf(explanation: dict, app_name: str) -> bytes:
    """Build a PDF report for SHAP and LIME explanations."""
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
        Paragraph("Explainability Report (SHAP + LIME)", title_style),
        Paragraph(
            f"{app_name} &mdash; generated {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}",
            subtitle_style,
        ),
        Paragraph(
            f"<b>Model:</b> {explanation.get('model_label', 'N/A')}<br/>"
            f"<b>Dataset:</b> {explanation.get('dataset_name', 'N/A')}<br/>"
            f"<b>Patient:</b> {explanation.get('patient_label', 'Global analysis only')}<br/>"
            f"<b>Background samples:</b> {explanation.get('sample_count', 'N/A')}",
            styles["Normal"],
        ),
        Spacer(1, 0.15 * inch),
    ]

    chart_titles = {
        "summary": "SHAP Summary Plot",
        "feature_importance": "SHAP Feature Importance",
        "dependence": f"SHAP Dependence Plot ({explanation.get('dependence_feature', 'feature')})",
        "local_waterfall": "SHAP Waterfall Plot",
        "local_force": "SHAP Force Plot",
        "lime_weights": "LIME Feature Weights",
        "lime_interactive": "Interactive LIME Explanation",
        "shap_lime_comparison": "SHAP vs LIME Comparison",
    }

    charts = explanation.get("charts", {})
    for key, title in chart_titles.items():
        chart = charts.get(key)
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

    lime = explanation.get("lime")
    if lime:
        story.append(Paragraph("LIME contributing factors", styles["Heading3"]))
        for item in lime.get("positive_factors", []):
            story.append(
                Paragraph(
                    f"<b>+ {item['feature']}</b> ({item['weight']:+.4f})",
                    styles["Normal"],
                )
            )
        for item in lime.get("negative_factors", []):
            story.append(
                Paragraph(
                    f"<b>- {item['feature']}</b> ({item['weight']:+.4f})",
                    styles["Normal"],
                )
            )
        story.append(Spacer(1, 0.1 * inch))

    if explanation.get("has_local") and explanation.get("local", {}).get("top_contributors"):
        story.append(Paragraph("SHAP top contributors", styles["Heading3"]))
        for item in explanation["local"]["top_contributors"]:
            story.append(
                Paragraph(
                    f"<b>{item['feature']}</b> = {item['value']:.3f} "
                    f"({item['shap']:+.4f}, {item['direction']})",
                    styles["Normal"],
                )
            )

    document.build(story)
    buffer.seek(0)
    return buffer.read()
