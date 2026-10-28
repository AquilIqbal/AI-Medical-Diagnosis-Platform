"""Thin request handlers for Patient Management pages."""

from flask import (
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from backend.services.patient_service import PatientService, PatientValidationError


patient_service = PatientService()


def list_patients() -> str:
    """Render patient records, optionally filtered by a search query."""
    search_query = request.args.get("search", "").strip()
    patients = patient_service.get_all_patients(search_query)
    return render_template(
        "patients/list.html",
        patients=patients,
        search_query=search_query,
    )


def view_patient(patient_id: int) -> str | Response:
    """Load one patient and render its detail page."""
    patient = patient_service.get_patient_by_id(patient_id)
    if patient is None:
        flash("Patient record was not found.", "error")
        return redirect(url_for("patients.list_patients"))

    return render_template("patients/detail.html", patient=patient)


def new_patient() -> str:
    """Render the new-patient form UI."""
    return render_template("patients/add.html", form_data={})


def create_patient() -> Response | tuple[str, int]:
    """Create a patient, or re-render the form with submitted values."""
    form_data = request.form.to_dict()

    try:
        patient_service.create_patient(form_data)
    except PatientValidationError as error:
        flash(str(error), "error")
        return render_template("patients/add.html", form_data=form_data), 400
    except Exception:
        current_app.logger.exception("Unable to create patient record.")
        flash("Unable to create the patient. Please try again.", "error")
        return render_template("patients/add.html", form_data=form_data), 500

    flash("Patient created successfully.", "success")
    return redirect(url_for("patients.list_patients"))


def edit_patient(patient_id: int) -> str | Response:
    """Load one patient and render the populated edit form."""
    patient = patient_service.get_patient_by_id(patient_id)
    if patient is None:
        flash("Patient record was not found.", "error")
        return redirect(url_for("patients.list_patients"))

    return render_template("patients/edit.html", patient=patient, form_data={})


def update_patient(patient_id: int) -> Response | tuple[str, int]:
    """Update one patient, or re-render the form with submitted values."""
    form_data = request.form.to_dict()

    try:
        patient = patient_service.update_patient(patient_id, form_data)
    except PatientValidationError as error:
        patient = patient_service.get_patient_by_id(patient_id)
        if patient is None:
            flash("Patient record was not found.", "error")
            return redirect(url_for("patients.list_patients"))
        flash(str(error), "error")
        return render_template(
            "patients/edit.html",
            patient=patient,
            form_data=form_data,
        ), 400
    except Exception:
        current_app.logger.exception("Unable to update patient %s.", patient_id)
        patient = patient_service.get_patient_by_id(patient_id)
        if patient is None:
            flash("Patient record was not found.", "error")
            return redirect(url_for("patients.list_patients"))
        flash("Unable to update the patient. Please try again.", "error")
        return render_template(
            "patients/edit.html",
            patient=patient,
            form_data=form_data,
        ), 500

    if patient is None:
        flash("Patient record was not found.", "error")
        return redirect(url_for("patients.list_patients"))

    flash("Patient updated successfully.", "success")
    return redirect(url_for("patients.list_patients"))


def delete_patient(patient_id: int) -> Response:
    """Delete one patient and return to the patient list."""
    try:
        was_deleted = patient_service.delete_patient(patient_id)
    except Exception:
        current_app.logger.exception("Unable to delete patient %s.", patient_id)
        flash("Unable to delete the patient. Please try again.", "error")
    else:
        if was_deleted:
            flash("Patient deleted successfully.", "success")
        else:
            flash("Patient record was not found.", "error")

    return redirect(url_for("patients.list_patients"))
