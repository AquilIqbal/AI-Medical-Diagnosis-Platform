"""Protected placeholder routes for the Diagnosis module."""

from flask import Blueprint

from backend.controllers import diagnosis_controller
from backend.middleware.auth_middleware import login_required


diagnosis_bp = Blueprint("diagnosis", __name__)


@diagnosis_bp.get("/diagnosis")
@login_required
def list_diagnoses() -> str:
    """Return the protected diagnosis-list placeholder."""
    return diagnosis_controller.list_diagnoses()


@diagnosis_bp.get("/diagnosis/new")
@login_required
def new_diagnosis() -> str:
    """Return the protected new-diagnosis placeholder."""
    return diagnosis_controller.new_diagnosis()


@diagnosis_bp.post("/diagnosis")
@login_required
def create_diagnosis() -> str:
    """Return the protected diagnosis-creation placeholder."""
    return diagnosis_controller.create_diagnosis()


@diagnosis_bp.get("/diagnosis/<int:diagnosis_id>")
@login_required
def view_diagnosis(diagnosis_id: int) -> str:
    """Return the protected diagnosis-detail placeholder."""
    return diagnosis_controller.view_diagnosis(diagnosis_id)


@diagnosis_bp.get("/diagnosis/<int:diagnosis_id>/edit")
@login_required
def edit_diagnosis(diagnosis_id: int) -> str:
    """Return the protected diagnosis-edit placeholder."""
    return diagnosis_controller.edit_diagnosis(diagnosis_id)


@diagnosis_bp.post("/diagnosis/<int:diagnosis_id>/edit")
@login_required
def update_diagnosis(diagnosis_id: int) -> str:
    """Return the protected diagnosis-update placeholder."""
    return diagnosis_controller.update_diagnosis(diagnosis_id)


@diagnosis_bp.get("/diagnosis/<int:diagnosis_id>/clinical-assessment")
@login_required
def clinical_assessment(diagnosis_id: int) -> str:
    """Return the optional Advanced Clinical Assessment form."""
    return diagnosis_controller.clinical_assessment(diagnosis_id)


@diagnosis_bp.post("/diagnosis/<int:diagnosis_id>/clinical-assessment")
@login_required
def save_clinical_assessment(diagnosis_id: int) -> str:
    """Create or update one Advanced Clinical Assessment."""
    return diagnosis_controller.save_clinical_assessment(diagnosis_id)


@diagnosis_bp.post("/diagnosis/<int:diagnosis_id>/delete")
@login_required
def delete_diagnosis(diagnosis_id: int) -> str:
    """Return the protected diagnosis-deletion placeholder."""
    return diagnosis_controller.delete_diagnosis(diagnosis_id)


@diagnosis_bp.get("/uploads/<path:filename>")
@login_required
def uploaded_file(filename: str):
    """Serve an application-managed uploaded image to an authenticated user."""
    return diagnosis_controller.serve_uploaded_file(filename)
