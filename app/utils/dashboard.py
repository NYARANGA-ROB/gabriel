"""Dashboard configuration and template context helpers."""

from __future__ import annotations

from typing import Any, Callable

DASHBOARD_MODULES = [
    {
        "id": "datasets",
        "title": "Dataset Management",
        "description": (
            "Upload, validate, and manage secondary healthcare datasets "
            "used for readmission risk modelling."
        ),
        "icon": "database",
        "color": "primary",
        "url_endpoint": "datasets.index",
    },
    {
        "id": "models",
        "title": "Machine Learning Models",
        "description": (
            "Train, evaluate, and deploy classification models for "
            "30-day hospital readmission prediction."
        ),
        "icon": "cpu",
        "color": "success",
        "url_endpoint": "ml.index",
    },
    {
        "id": "comparison",
        "title": "Model Comparison",
        "description": (
            "Compare model performance using accuracy, AUC, F1-score, "
            "and other clinical evaluation metrics."
        ),
        "icon": "bar-chart-steps",
        "color": "info",
        "url_endpoint": "ml.compare",
    },
    {
        "id": "prediction",
        "title": "Patient Prediction",
        "description": (
            "Submit patient features to generate individual readmission "
            "risk scores and probability estimates."
        ),
        "icon": "heart-pulse",
        "color": "danger",
        "url_endpoint": "predictions.index",
    },
    {
        "id": "explainability",
        "title": "Explainable AI",
        "description": (
            "Interpret predictions with SHAP, feature importance, and "
            "local explanations for clinical transparency."
        ),
        "icon": "lightbulb",
        "color": "warning",
        "url_endpoint": "explainability.index",
    },
    {
        "id": "history",
        "title": "Prediction History",
        "description": (
            "Review previously generated risk assessments, outcomes, "
            "and audit trails across patient cohorts."
        ),
        "icon": "clock-history",
        "color": "secondary",
        "url_endpoint": "predictions.history",
    },
    {
        "id": "analytics",
        "title": "Analytics",
        "description": (
            "Explore population-level trends, risk distributions, "
            "and operational insights from prediction outputs."
        ),
        "icon": "graph-up-arrow",
        "color": "primary",
        "url_endpoint": "analytics.index",
    },
    {
        "id": "admin",
        "title": "Admin Panel",
        "description": (
            "Manage users, datasets, models, prediction logs, and "
            "system-wide reports with role-based access control."
        ),
        "icon": "shield-lock",
        "color": "dark",
        "url_endpoint": "admin.index",
        "admin_only": True,
    },
]

SIDEBAR_NAV = [
    {"label": "Dashboard", "icon": "speedometer2", "endpoint": "main.dashboard"},
    {"label": "Datasets", "icon": "database", "endpoint": "datasets.index"},
    {"label": "ML Models", "icon": "cpu", "endpoint": "ml.index"},
    {"label": "Comparison", "icon": "bar-chart-steps", "endpoint": "ml.compare"},
    {"label": "Predictions", "icon": "heart-pulse", "endpoint": "predictions.index"},
    {"label": "Explainability", "icon": "lightbulb", "endpoint": "explainability.index"},
    {"label": "History", "icon": "clock-history", "endpoint": "predictions.history"},
    {"label": "Analytics", "icon": "graph-up-arrow", "endpoint": "analytics.index"},
    {"label": "Admin", "icon": "shield-lock", "endpoint": "admin.index", "admin_only": True},
]

ADMIN_SIDEBAR_NAV = [
    {"label": "Overview", "icon": "speedometer2", "endpoint": "admin.index"},
    {"label": "Users", "icon": "people", "endpoint": "admin.users"},
    {"label": "Datasets", "icon": "database", "endpoint": "admin.datasets"},
    {"label": "Prediction Logs", "icon": "clock-history", "endpoint": "admin.predictions"},
    {"label": "Models", "icon": "cpu", "endpoint": "admin.models"},
    {"label": "Reports", "icon": "file-earmark-bar-graph", "endpoint": "admin.reports"},
]


def resolve_module_urls(modules: list[dict], url_for: Callable[..., str]) -> list[dict]:
    """Attach resolved URLs to dashboard module definitions."""
    resolved = []
    for module in modules:
        item = module.copy()
        endpoint = item.pop("url_endpoint", None)
        item["url"] = url_for(endpoint) if endpoint else "#"
        item["disabled"] = endpoint is None
        resolved.append(item)
    return resolved


def resolve_nav_items(
    nav_items: list[dict],
    active_endpoint: str,
    url_for: Callable[..., str],
) -> list[dict]:
    """Attach resolved URLs and active state to a navigation definition."""
    resolved = []
    for item in nav_items:
        nav = item.copy()
        endpoint = nav.get("endpoint")
        nav["url"] = url_for(endpoint) if endpoint else "#"
        nav["disabled"] = endpoint is None
        nav["is_active"] = endpoint == active_endpoint
        resolved.append(nav)
    return resolved


def resolve_sidebar_nav(
    nav_items: list[dict],
    active_endpoint: str,
    url_for: Callable[..., str],
) -> list[dict]:
    """Backward-compatible wrapper for main sidebar navigation."""
    return resolve_nav_items(nav_items, active_endpoint, url_for)


def resolve_admin_sidebar_nav(
    nav_items: list[dict],
    active_endpoint: str,
    url_for: Callable[..., str],
) -> list[dict]:
    """Backward-compatible wrapper for admin sidebar navigation."""
    return resolve_nav_items(nav_items, active_endpoint, url_for)


def dashboard_page_context(
    active_endpoint: str,
    url_for: Callable[..., str],
    *,
    include_admin_nav: bool = False,
) -> dict[str, Any]:
    """Build shared template context for dashboard layouts."""
    context = {
        "sidebar_nav": resolve_sidebar_nav(SIDEBAR_NAV, active_endpoint, url_for),
        "active_endpoint": active_endpoint,
    }
    if include_admin_nav:
        context["admin_nav"] = resolve_admin_sidebar_nav(
            ADMIN_SIDEBAR_NAV,
            active_endpoint,
            url_for,
        )
    return context
