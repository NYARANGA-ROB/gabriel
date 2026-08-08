"""Route decorators."""

from functools import wraps

from flask import abort
from flask_login import current_user, login_required


def active_user_required(view):
    """Require an authenticated and active user."""

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401)
        if not current_user.is_active:
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    """Require an authenticated admin user."""

    @wraps(view)
    @active_user_required
    def wrapped_view(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view


def login_active_required(view):
    """Require a logged-in, active user account."""

    @wraps(view)
    @login_required
    @active_user_required
    def wrapped_view(*args, **kwargs):
        return view(*args, **kwargs)

    return wrapped_view
