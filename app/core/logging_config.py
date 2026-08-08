"""Structured logging configuration for the application."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(app: Flask | None = None, *, level: int | None = None) -> None:
    """Configure root and Flask loggers."""
    log_level = level or (logging.DEBUG if app and app.debug else logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    if app is not None:
        app.logger.setLevel(log_level)

    for noisy_logger in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
