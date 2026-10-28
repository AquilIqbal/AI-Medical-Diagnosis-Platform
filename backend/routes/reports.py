"""Protected routes for the Reports module."""

from flask import Blueprint
from flask.typing import ResponseReturnValue

from backend.controllers import reports_controller
from backend.middleware.auth_middleware import login_required


reports_bp = Blueprint("reports", __name__)


@reports_bp.get("/reports")
@login_required
def list_reports() -> ResponseReturnValue:
    """Return the protected reports-list page."""
    return reports_controller.list_reports()


@reports_bp.get("/reports/<int:diagnosis_id>")
@login_required
def view_report(diagnosis_id: int) -> ResponseReturnValue:
    """Return the protected report-detail page for one diagnosis."""
    return reports_controller.view_report(diagnosis_id)
