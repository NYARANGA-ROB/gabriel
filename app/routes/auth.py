"""Authentication routes."""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.core.logging_config import get_logger
from app.core.security import is_safe_redirect_url
from app.extensions import db
from app.forms.auth_forms import LoginForm, ProfileForm, RegistrationForm
from app.models.user import User

auth_bp = Blueprint("auth", __name__)
logger = get_logger(__name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Register a new user account."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            email=form.email.data.lower().strip(),
            first_name=form.first_name.data.strip(),
            last_name=form.last_name.data.strip(),
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        flash("Your account has been created. Please sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate a user."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower().strip()).first()

        if user is None or not user.check_password(form.password.data):
            flash("Invalid email or password.", "danger")
            return render_template("auth/login.html", form=form), 401

        if not user.is_active:
            flash("This account has been deactivated. Contact support.", "warning")
            return render_template("auth/login.html", form=form), 403

        login_user(user, remember=form.remember_me.data)
        flash(f"Welcome back, {user.first_name}.", "success")

        next_page = request.args.get("next")
        if is_safe_redirect_url(next_page, request):
            return redirect(next_page)
        return redirect(url_for("main.dashboard"))

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """View and update the current user's profile."""
    form = ProfileForm(
        first_name=current_user.first_name,
        last_name=current_user.last_name,
    )

    if form.validate_on_submit():
        current_user.first_name = form.first_name.data.strip()
        current_user.last_name = form.last_name.data.strip()
        db.session.commit()
        flash("Your profile has been updated.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("auth/profile.html", form=form)
