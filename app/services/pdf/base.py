"""Shared ReportLab PDF building blocks."""

from __future__ import annotations

import io
from datetime import datetime

import plotly.graph_objects as go
from plotly.io import to_image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

from app.core.datetime_utils import utc_now


def figure_to_png(chart_dict: dict, width: int = 900, height: int = 480) -> bytes:
    """Render a Plotly figure dictionary to PNG bytes."""
    figure = go.Figure(data=chart_dict.get("data", []), layout=chart_dict.get("layout", {}))
    return to_image(figure, format="png", width=width, height=height, engine="kaleido")


class PdfReportBuilder:
    """Fluent builder for consistent PDF report documents."""

    def __init__(self) -> None:
        self._buffer = io.BytesIO()
        self._document = SimpleDocTemplate(
            self._buffer,
            pagesize=A4,
            leftMargin=0.6 * inch,
            rightMargin=0.6 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
        )
        self._styles = getSampleStyleSheet()
        self._story: list = []

    @property
    def title_style(self) -> ParagraphStyle:
        return ParagraphStyle(
            "ReportTitle",
            parent=self._styles["Heading1"],
            fontSize=18,
            spaceAfter=12,
        )

    @property
    def subtitle_style(self) -> ParagraphStyle:
        return ParagraphStyle(
            "ReportSubtitle",
            parent=self._styles["Normal"],
            textColor=colors.HexColor("#64748b"),
            spaceAfter=16,
        )

    @property
    def section_style(self) -> ParagraphStyle:
        return ParagraphStyle(
            "SectionTitle",
            parent=self._styles["Heading2"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=8,
        )

    def add_title(self, title: str, *, app_name: str | None = None) -> PdfReportBuilder:
        self._story.append(Paragraph(title, self.title_style))
        subtitle = f"{app_name} — generated {utc_now().strftime('%Y-%m-%d %H:%M UTC')}" if app_name else None
        if subtitle:
            self._story.append(Paragraph(subtitle, self.subtitle_style))
        return self

    def add_paragraph(self, text: str, *, style: ParagraphStyle | None = None) -> PdfReportBuilder:
        self._story.append(Paragraph(text, style or self._styles["Normal"]))
        return self

    def add_section(self, title: str) -> PdfReportBuilder:
        self._story.append(Paragraph(title, self.section_style))
        return self

    def add_elements(self, *elements) -> PdfReportBuilder:
        self._story.extend(elements)
        return self

    @staticmethod
    def styled_table(
        rows: list[list[str]],
        col_widths: list[float],
        *,
        header: bool = True,
        font_size: int = 9,
    ) -> Table:
        table = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
        style_commands = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]
        if header:
            style_commands.extend(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        table.setStyle(TableStyle(style_commands))
        return table

    def build(self) -> bytes:
        self._document.build(self._story)
        return self._buffer.getvalue()
