"""Protected HTTP routes for Patient Management presentation pages."""

from flask import Blueprint, Response

from backend.controllers import patient_controller
from backend.middleware.auth_middleware import login_required


patient_bp = Blueprint("patients", __name__)


@patient_bp.get("/patients")
@login_required
def list_patients() -> str:
    """Render the protected patient-list page."""
    return patient_controller.list_patients()


@patient_bp.get("/patients/<int:patient_id>")
@login_required
def view_patient(patient_id: int) -> str | Response:
    """Render the protected detail page for one patient."""
    return patient_controller.view_patient(patient_id)


@patient_bp.get("/patients/new")
@login_required
def new_patient() -> str:
    """Render the protected new-patient page."""
    return patient_controller.new_patient()


@patient_bp.post("/patients")
@login_required
def create_patient() -> Response | tuple[str, int]:
    """Create a protected patient record from the submitted form."""
    return patient_controller.create_patient()


@patient_bp.get("/patients/<int:patient_id>/edit")
@login_required
def edit_patient(patient_id: int) -> str | Response:
    """Render the protected edit form for one patient."""
    return patient_controller.edit_patient(patient_id)


@patient_bp.post("/patients/<int:patient_id>/edit")
@login_required
def update_patient(patient_id: int) -> Response | tuple[str, int]:
    """Update one protected patient record."""
    return patient_controller.update_patient(patient_id)


@patient_bp.post("/patients/<int:patient_id>/delete")
@login_required
def delete_patient(patient_id: int) -> Response:
    """Delete one protected patient record."""
    return patient_controller.delete_patient(patient_id)
