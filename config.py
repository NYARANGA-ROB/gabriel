import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

_DEFAULT_SECRET_KEY = "dev-secret-key-change-in-production"


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", _DEFAULT_SECRET_KEY)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'instance' / 'readmission.db'}",
    )

    # Session / security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_HTTPONLY = True
    WTF_CSRF_ENABLED = True

    # Application metadata
    APP_NAME = "Readmission Risk Framework"
    ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")
    AUTO_PROMOTE_ADMIN = os.environ.get("AUTO_PROMOTE_ADMIN", "false").lower() == "true"

    # Dataset uploads
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB
    ALLOWED_DATASET_EXTENSIONS = {".csv"}
    DATASET_PREVIEW_PAGE_SIZE = 25
    PREDICTION_HISTORY_PAGE_SIZE = 20
    PREDICTION_EXPORT_MAX_ROWS = int(os.environ.get("PREDICTION_EXPORT_MAX_ROWS", "5000"))
    PROCESSED_FOLDER = BASE_DIR / "uploads" / "processed"
    MODELS_FOLDER = BASE_DIR / "app" / "ml" / "models"


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    TESTING = False
    AUTO_PROMOTE_ADMIN = os.environ.get("AUTO_PROMOTE_ADMIN", "true").lower() == "true"


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    AUTO_PROMOTE_ADMIN = False


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    AUTO_PROMOTE_ADMIN = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config() -> type[Config]:
    """Return the configuration class for the current environment."""
    env = os.environ.get("FLASK_ENV", "development")
    return config_by_name.get(env, DevelopmentConfig)


def validate_config(app_config: dict) -> None:
    """Fail fast when production security requirements are not met."""
    if app_config.get("TESTING"):
        return

    env = os.environ.get("FLASK_ENV", "development")
    if env != "production":
        return

    secret_key = app_config.get("SECRET_KEY")
    if not secret_key or secret_key == _DEFAULT_SECRET_KEY:
        raise RuntimeError(
            "SECRET_KEY must be set to a strong random value in production."
        )
