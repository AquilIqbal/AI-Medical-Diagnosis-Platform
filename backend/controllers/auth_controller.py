"""Thin HTTP controllers for authentication endpoints."""

from flask import flash, redirect, render_template, request, session, url_for
from flask.typing import ResponseReturnValue

from backend.services.auth_service import (
    AuthenticationStatus,
    AuthService,
)
from backend.services.ai_model_catalog_service import AIModelCatalogService
from backend.services.dashboard_service import DashboardService
from backend.services.settings_service import SettingsService, SettingsValidationError


_auth_service = AuthService()
dashboard_service = DashboardService()
ai_model_catalog_service = AIModelCatalogService()
settings_service = SettingsService()


def login() -> ResponseReturnValue:
    """Render the login page or process a submitted credential form."""
    if request.method == "GET":
        return render_template("auth/login.html")

    # ``identifier`` preserves compatibility with the finalized login template.
    username_or_email = request.form.get("username_or_email", "").strip()
    if not username_or_email:
        username_or_email = request.form.get("identifier", "").strip()

    password = request.form.get("password", "")
    result = _auth_service.login(username_or_email, password)

    if result.status is AuthenticationStatus.INVALID_CREDENTIALS:
        flash("Invalid credentials. Please try again.", "error")
        return redirect(url_for("auth.login"))

    if result.status is AuthenticationStatus.INACTIVE_ACCOUNT:
        flash("This account is inactive. Please contact an administrator.", "error")
        return redirect(url_for("auth.login"))

    if result.status is AuthenticationStatus.ERROR or result.user is None:
        flash("Unable to sign in right now. Please try again later.", "error")
        return redirect(url_for("auth.login"))

    # Clear any old state before storing the only authenticated-user values.
    session.clear()
    session.update(
        user_id=result.user.id,
        username=result.user.username,
        role=result.user.role.value,
    )
    return redirect(url_for("auth.dashboard"))


def logout() -> ResponseReturnValue:
    """Clear authenticated session data and return the user to the home page."""
    _auth_service.logout()
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("main.home"))


def dashboard() -> str:
    """Render current platform totals and recent diagnosis activity."""
    return render_template(
        "dashboard/dashboard.html",
        statistics=dashboard_service.get_statistics(),
        recent_activity=dashboard_service.get_recent_diagnosis_activity(),
    )


def ai_models() -> str:
    """Render a read-only catalog of the implemented model artifacts."""
    return render_template(
        "ai_models/list.html",
        models=ai_model_catalog_service.get_models(),
    )


def settings() -> str | ResponseReturnValue:
    """Render the authenticated user's account settings."""
    user = settings_service.get_user_by_id(session["user_id"])
    if user is None:
        session.clear()
        flash("Your account is no longer available. Please log in again.", "error")
        return redirect(url_for("auth.login"))
    return render_template("settings.html", user=user, form_data={})


def update_settings_profile() -> ResponseReturnValue:
    """Persist the authenticated user's permitted profile fields."""
    form_data = request.form.to_dict()
    try:
        user = settings_service.update_profile(session["user_id"], form_data)
    except SettingsValidationError as error:
        user = settings_service.get_user_by_id(session["user_id"])
        if user is None:
            return _settings_account_missing()
        flash(str(error), "error")
        return render_template("settings.html", user=user, form_data=form_data), 400

    if user is None:
        return _settings_account_missing()

    flash("Account information updated successfully.", "success")
    return redirect(url_for("auth.settings"))


def change_settings_password() -> ResponseReturnValue:
    """Change the authenticated user's password after verification."""
    try:
        user = settings_service.change_password(
            session["user_id"],
            request.form.get("current_password", ""),
            request.form.get("new_password", ""),
            request.form.get("confirm_password", ""),
        )
    except SettingsValidationError as error:
        user = settings_service.get_user_by_id(session["user_id"])
        if user is None:
            return _settings_account_missing()
        flash(str(error), "error")
        return render_template("settings.html", user=user, form_data={}), 400

    if user is None:
        return _settings_account_missing()

    flash("Password changed successfully.", "success")
    return redirect(url_for("auth.settings"))


def _settings_account_missing() -> ResponseReturnValue:
    """Clear an invalid authenticated session before redirecting to login."""
    session.clear()
    flash("Your account is no longer available. Please log in again.", "error")
    return redirect(url_for("auth.login"))
