"""Authentication and profile forms."""

import re

from flask_wtf import FlaskForm
from wtforms import BooleanField, PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

from app.models.user import User

PASSWORD_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).+$")


class LoginForm(FlaskForm):
    """User login form."""

    email = StringField(
        "Email address",
        validators=[DataRequired(), Email(message="Enter a valid email address.")],
        render_kw={"placeholder": "you@example.com", "autocomplete": "email"},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter your password", "autocomplete": "current-password"},
    )
    remember_me = BooleanField("Remember me")
    submit = SubmitField("Sign in")


class RegistrationForm(FlaskForm):
    """User registration form."""

    first_name = StringField(
        "First name",
        validators=[DataRequired(), Length(min=2, max=80)],
        render_kw={"placeholder": "Jane", "autocomplete": "given-name"},
    )
    last_name = StringField(
        "Last name",
        validators=[DataRequired(), Length(min=2, max=80)],
        render_kw={"placeholder": "Smith", "autocomplete": "family-name"},
    )
    email = StringField(
        "Email address",
        validators=[DataRequired(), Email(message="Enter a valid email address.")],
        render_kw={"placeholder": "you@example.com", "autocomplete": "email"},
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8, message="Password must be at least 8 characters."),
        ],
        render_kw={"placeholder": "Create a password", "autocomplete": "new-password"},
    )
    confirm_password = PasswordField(
        "Confirm password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match."),
        ],
        render_kw={"placeholder": "Confirm your password", "autocomplete": "new-password"},
    )
    submit = SubmitField("Create account")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data.lower().strip()).first():
            raise ValidationError("An account with this email already exists.")

    def validate_password(self, field):
        if not PASSWORD_PATTERN.match(field.data):
            raise ValidationError(
                "Password must contain at least one letter and one number."
            )


class ProfileForm(FlaskForm):
    """User profile update form."""

    first_name = StringField(
        "First name",
        validators=[DataRequired(), Length(min=2, max=80)],
        render_kw={"autocomplete": "given-name"},
    )
    last_name = StringField(
        "Last name",
        validators=[DataRequired(), Length(min=2, max=80)],
        render_kw={"autocomplete": "family-name"},
    )
    submit = SubmitField("Save changes")
