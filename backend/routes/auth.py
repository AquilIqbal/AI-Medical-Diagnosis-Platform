"""Authentication routes backed by the authentication controller layer."""

from flask import Blueprint
from flask.typing import ResponseReturnValue

from backend.controllers.auth_controller import (
    ai_models as ai_models_controller,
    change_settings_password as change_settings_password_controller,
    dashboard as dashboard_controller,
    settings as settings_controller,
    update_settings_profile as update_settings_profile_controller,
)
from backend.controllers.auth_controller import login as login_controller
from backend.controllers.auth_controller import logout as logout_controller
from backend.middleware.auth_middleware import login_required


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login() -> ResponseReturnValue:
    """Delegate login page rendering and form handling to the controller."""
    return login_controller()


@auth_bp.get("/logout")
def logout() -> ResponseReturnValue:
    """Delegate session clearing to the authentication controller."""
    return logout_controller()


@auth_bp.get("/dashboard")
@login_required
def dashboard() -> str:
    """Provide the required temporary post-login destination."""
    return dashboard_controller()


@auth_bp.get("/ai-models")
@login_required
def ai_models() -> str:
    """Show deployed AI model metadata to authenticated users."""
    return ai_models_controller()


@auth_bp.get("/settings", endpoint="settings")
@login_required
def settings() -> ResponseReturnValue:
    """Show account settings for the authenticated user."""
    return settings_controller()


@auth_bp.post("/settings/profile")
@login_required
def update_settings_profile() -> ResponseReturnValue:
    """Update permitted account profile fields."""
    return update_settings_profile_controller()


@auth_bp.post("/settings/password")
@login_required
def change_settings_password() -> ResponseReturnValue:
    """Change the authenticated user's password."""
    return change_settings_password_controller()
