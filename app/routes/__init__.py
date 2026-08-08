"""Route blueprints package."""

from flask import Flask


def register_blueprints(app: Flask) -> None:
    """Register all application blueprints."""
    from app.routes.auth import auth_bp
    from app.routes.datasets import datasets_bp
    from app.routes.main import main_bp
    from app.routes.ml import ml_bp
    from app.routes.predictions import predictions_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(datasets_bp, url_prefix="/datasets")
    app.register_blueprint(ml_bp, url_prefix="/models")
    app.register_blueprint(predictions_bp, url_prefix="/predictions")
    from app.routes.explainability import explainability_bp

    app.register_blueprint(explainability_bp, url_prefix="/explainability")
    from app.routes.analytics import analytics_bp

    app.register_blueprint(analytics_bp, url_prefix="/analytics")
    from app.routes.admin import admin_bp

    app.register_blueprint(admin_bp, url_prefix="/admin")
