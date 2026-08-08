"""HTTP response helpers."""

from __future__ import annotations

import io
from typing import Any

from flask import send_file
from flask.wrappers import Response
from werkzeug.datastructures import Headers


def pagination_to_dict(pagination: Any) -> dict[str, int | bool]:
    """Convert a Flask-SQLAlchemy pagination object to a template-friendly dict."""
    return {
        "page": pagination.page,
        "total_pages": pagination.pages,
        "has_prev": pagination.has_prev,
        "has_next": pagination.has_next,
        "total": pagination.total,
    }


def pdf_response(pdf_bytes: bytes, download_name: str) -> Response:
    """Return a PDF file download response."""
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_name,
    )
