"""Main application routes."""

from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user

from app.utils.dashboard import DASHBOARD_MODULES, dashboard_page_context, resolve_module_urls
from app.utils.decorators import login_active_required

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Landing page."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return render_template("main/index.html")


@main_bp.route("/dashboard")
@login_active_required
def dashboard():
    """Authenticated user dashboard."""
    return render_template(
        "main/dashboard.html",
        modules=resolve_module_urls(DASHBOARD_MODULES, url_for),
        **dashboard_page_context("main.dashboard", url_for),
    )
