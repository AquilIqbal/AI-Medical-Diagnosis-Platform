"""Public, non-authenticated application routes."""

from flask import Blueprint, render_template


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def home() -> str:
    """Render the platform landing page."""
    return render_template("home.html")
