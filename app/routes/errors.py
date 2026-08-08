"""Global error handlers."""

from flask import Blueprint, render_template

from app.core.logging_config import get_logger

errors_bp = Blueprint("errors", __name__)
logger = get_logger(__name__)


@errors_bp.app_errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template("errors/404.html"), 404


@errors_bp.app_errorhandler(401)
def unauthorized(error):
    """Handle unauthorized access."""
    return render_template("errors/404.html"), 401


@errors_bp.app_errorhandler(403)
def forbidden(error):
    """Handle forbidden access."""
    return render_template("errors/404.html"), 403


@errors_bp.app_errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    from app.extensions import db

    logger.exception("Unhandled server error: %s", error)
    db.session.rollback()
    return render_template("errors/500.html"), 500
