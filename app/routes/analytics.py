"""Analytics dashboard routes."""

from flask import Blueprint, render_template, url_for
from flask_login import current_user

from app.services.analytics_service import build_analytics_dashboard
from app.utils.dashboard import dashboard_page_context
from app.utils.decorators import login_active_required

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/")
@login_active_required
def index():
    """Population-level analytics dashboard."""
    dashboard = build_analytics_dashboard(current_user.id)
    return render_template(
        "analytics/index.html",
        dashboard=dashboard,
        **dashboard_page_context("analytics.index", url_for),
    )
