"""Application factory."""

from pathlib import Path

from flask import Flask

from app.core.logging_config import configure_logging, get_logger
from app.extensions import csrf, db, login_manager
from config import Config, validate_config

logger = get_logger(__name__)


def create_app(config_class: type[Config] | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_object(config_class or Config)
    validate_config(app.config)

    configure_logging(app)
    _ensure_instance_folder(app)
    _ensure_upload_folder(app)
    _init_extensions(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_context_processors(app)
    _register_cli_commands(app)
    _init_database(app)

    logger.info("Application started in %s mode", app.config.get("ENV", "unknown"))
    return app


def _ensure_instance_folder(app: Flask) -> None:
    """Ensure the instance folder exists for SQLite and local assets."""
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)


def _ensure_upload_folder(app: Flask) -> None:
    """Ensure upload folders exist."""
    upload_path = Path(app.config["UPLOAD_FOLDER"])
    upload_path.mkdir(parents=True, exist_ok=True)
    processed_path = Path(app.config["PROCESSED_FOLDER"])
    processed_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(app.config["MODELS_FOLDER"])
    models_path.mkdir(parents=True, exist_ok=True)


def _init_extensions(app: Flask) -> None:
    """Initialize Flask extensions."""
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"
    login_manager.session_protection = "strong"

    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        user = db.session.get(User, int(user_id))
        if user is not None and not user.is_active:
            return None
        return user


def _register_blueprints(app: Flask) -> None:
    """Register application blueprints."""
    from app.routes import register_blueprints

    register_blueprints(app)


def _register_error_handlers(app: Flask) -> None:
    """Register global error handlers."""
    from app.routes.errors import errors_bp

    app.register_blueprint(errors_bp)


def _register_context_processors(app: Flask) -> None:
    """Inject shared template variables."""

    @app.context_processor
    def inject_globals():
        return {
            "app_name": app.config["APP_NAME"],
            "config": {
                "APP_NAME": app.config["APP_NAME"],
            },
        }


def _init_database(app: Flask) -> None:
    """Create database tables if they do not exist."""
    with app.app_context():
        db.create_all()
        _migrate_user_role_column()
        _ensure_admin_user(app)


def _migrate_user_role_column() -> None:
    """Add the users.role column on existing SQLite databases."""
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "users" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("users")}
    if "role" in columns:
        return

    with db.engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE users ADD COLUMN role VARCHAR(32) DEFAULT 'user' NOT NULL")
        )
    logger.info("Migrated users table: added role column")


def _ensure_admin_user(app: Flask) -> None:
    """Promote a configured admin when enabled and no admin exists."""
    if not app.config.get("AUTO_PROMOTE_ADMIN"):
        return

    from app.models.user import ROLE_ADMIN, User

    if User.query.filter_by(role=ROLE_ADMIN).first() is not None:
        return

    admin_email = app.config.get("ADMIN_EMAIL")
    if admin_email:
        user = User.query.filter_by(email=admin_email.lower().strip()).first()
        if user is not None:
            user.role = ROLE_ADMIN
            db.session.commit()
            logger.info("Promoted configured admin user: %s", admin_email)
            return

    first_user = User.query.order_by(User.id.asc()).first()
    if first_user is not None:
        first_user.role = ROLE_ADMIN
        db.session.commit()
        logger.info("Promoted first registered user to admin: %s", first_user.email)


def _register_cli_commands(app: Flask) -> None:
    """Register Flask CLI commands."""

    @app.cli.command("init-db")
    def init_db():
        """Initialize the database schema."""
        db.create_all()
        _migrate_user_role_column()
        print("Database initialized.")

    @app.cli.command("promote-admin")
    def promote_admin():
        """Promote a user to admin by email address."""
        import click

        from app.models.user import ROLE_ADMIN, User

        email = click.prompt("Admin email", type=str).lower().strip()
        user = User.query.filter_by(email=email).first()
        if user is None:
            raise click.ClickException(f"No user found with email {email}.")

        user.role = ROLE_ADMIN
        user.is_active = True
        db.session.commit()
        click.echo(f"Promoted {email} to admin.")
