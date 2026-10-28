"""Template-rendering controllers for the Diagnosis module UI foundation."""

from pathlib import Path
from uuid import uuid4

from flask import (
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from backend.services.diagnosis_service import (
    DiagnosisService,
    DiagnosisValidationError,
)
from backend.services.nlp_service import NLPPrediction, NLPService, NLPServiceError
from backend.services.image_prediction_service import (
    ImagePrediction,
    ImagePredictionService,
    ImagePredictionServiceError,
    ImageValidationError,
)
from backend.services.clinical_assessment_service import (
    ClinicalAssessmentService,
    ClinicalAssessmentValidationError,
)


diagnosis_service = DiagnosisService()
nlp_service = NLPService()
image_prediction_service = ImagePredictionService()
clinical_assessment_service = ClinicalAssessmentService()

_NLP_MODEL_NAME = "TF-IDF + Logistic Regression"


_PLACEHOLDER_DIAGNOSIS = {
    "diagnosis_code": "DX-0001",
    "patient": "Placeholder Patient",
    "diagnosis_date": "Today",
    "status": "Pending",
    "predicted_disease": "Not available",
    "confidence_score": "--",
    "symptoms": "Placeholder symptoms",
    "medical_history": "Placeholder history",
    "doctor_notes": "Placeholder notes",
    "medical_image": "No image uploaded",
}


def list_diagnoses() -> str:
    """Render diagnosis records, optionally filtered by a search query."""
    search_query = request.args.get("search", "").strip()
    diagnoses = diagnosis_service.get_all_diagnoses(search_query)
    return render_template(
        "diagnosis/list.html",
        diagnoses=diagnoses,
        search_query=search_query,
    )


def view_diagnosis(diagnosis_id: int) -> str:
    """Render the diagnosis detail page with real database data."""
    diagnosis = diagnosis_service.get_diagnosis_by_id(diagnosis_id)
    if diagnosis is None:
        flash("Diagnosis record was not found.", "error")
        return redirect(url_for("diagnosis.list_diagnoses"))

    return render_template("diagnosis/detail.html", diagnosis=diagnosis)


def new_diagnosis() -> str:
    """Render the new-diagnosis form UI with all available patients."""
    patients = diagnosis_service.get_all_patients()
    return render_template("diagnosis/add.html", patients=patients, form_data={})


def create_diagnosis() -> str | tuple[str, int] | object:
    """Create a diagnosis, or re-render the form with submitted values."""
    form_data = request.form.to_dict()
    uploaded_file = request.files.get("medical_image")
    image_prediction: ImagePrediction | None = None
    nlp_prediction: NLPPrediction | None = None
    stored_uploaded_image: str | None = None

    try:
        if uploaded_file and uploaded_file.filename:
            try:
                image_prediction = image_prediction_service.predict(uploaded_file)
            except ImageValidationError:
                raise
            except ImagePredictionServiceError:
                current_app.logger.warning(
                    "Unable to generate a CNN prediction for a new diagnosis.",
                    exc_info=True,
                )
                flash(
                    "The diagnosis will be saved, but a CNN prediction could not be generated.",
                    "warning",
                )

            stored_uploaded_image = _save_uploaded_image(uploaded_file)
            form_data["uploaded_image"] = stored_uploaded_image

        nlp_prediction = _add_nlp_prediction(form_data)
        diagnosis = diagnosis_service.create_diagnosis(form_data)
    except (DiagnosisValidationError, ImageValidationError) as error:
        _cleanup_created_uploads(stored_uploaded_image)
        patients = diagnosis_service.get_all_patients()
        flash(str(error), "error")
        return (
            render_template(
                "diagnosis/add.html",
                patients=patients,
                form_data=form_data,
            ),
            400,
        )
    except Exception:
        _cleanup_created_uploads(stored_uploaded_image)
        current_app.logger.exception("Unable to create diagnosis record.")
        patients = diagnosis_service.get_all_patients()
        flash("Unable to create the diagnosis. Please try again.", "error")
        return (
            render_template(
                "diagnosis/add.html",
                patients=patients,
                form_data=form_data,
            ),
            500,
        )

    if nlp_prediction is not None:
        try:
            diagnosis_service.upsert_ai_prediction(
                diagnosis.id,
                prediction_type="nlp",
                predicted_label=nlp_prediction.label,
                confidence_score=nlp_prediction.confidence,
                model_name=_NLP_MODEL_NAME,
            )
        except Exception:
            current_app.logger.exception(
                "Unable to save the NLP prediction for diagnosis %s.", diagnosis.id
            )
            flash(
                "The diagnosis was saved, but its NLP prediction could not be stored.",
                "warning",
            )

    if image_prediction is not None:
        visualization_path: str | None = None
        try:
            visualization_path = _save_gradcam_visualization(
                diagnosis.diagnosis_code,
                image_prediction.visualization_png,
            )
            diagnosis_service.upsert_ai_prediction(
                diagnosis.id,
                prediction_type=image_prediction.prediction_type,
                predicted_label=image_prediction.predicted_label,
                confidence_score=image_prediction.confidence,
                model_name=image_prediction.model_name,
                visualization_path=visualization_path,
            )
        except Exception:
            _cleanup_created_uploads(visualization_path)
            current_app.logger.exception(
                "Unable to save the CNN prediction for diagnosis %s.",
                diagnosis.id,
            )
            flash(
                "The diagnosis was saved, but its CNN prediction could not be stored.",
                "warning",
            )

    flash("Diagnosis created successfully.", "success")
    return redirect(url_for("diagnosis.list_diagnoses"))


def serve_uploaded_file(filename: str):
    """Serve application-managed uploads to authenticated users only."""
    upload_directory = Path(current_app.root_path).parent / "uploads"
    return send_from_directory(upload_directory, filename)


def _save_uploaded_image(uploaded_file: object) -> str:
    """Store a validated upload under a collision-resistant safe filename."""
    safe_filename = secure_filename(getattr(uploaded_file, "filename", ""))
    if not safe_filename:
        raise ImageValidationError(
            "The uploaded medical image must have a valid filename."
        )

    stored_filename = f"{uuid4().hex}_{safe_filename}"
    upload_directory = Path(current_app.root_path).parent / "uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)
    uploaded_file.save(upload_directory / stored_filename)  # type: ignore[attr-defined]
    return stored_filename


def _save_gradcam_visualization(diagnosis_code: str, visualization_png: bytes) -> str:
    """Store a generated visualization below the managed upload directory."""
    if not visualization_png:
        raise ImagePredictionServiceError("The Grad-CAM visualization is empty.")

    visualization_directory = (
        Path(current_app.root_path).parent / "uploads" / "gradcam"
    )
    visualization_directory.mkdir(parents=True, exist_ok=True)
    filename = f"{diagnosis_code}_{uuid4().hex}.png"
    (visualization_directory / filename).write_bytes(visualization_png)
    return f"gradcam/{filename}"


def _cleanup_created_uploads(*stored_paths: str | None) -> None:
    """Best-effort removal of files created by an unsuccessful new diagnosis."""
    upload_directory = Path(current_app.root_path).parent / "uploads"
    for stored_path in stored_paths:
        if not stored_path:
            continue

        candidate = upload_directory / stored_path
        try:
            if candidate.is_file():
                candidate.unlink()
        except OSError:
            current_app.logger.warning(
                "Unable to remove upload created during failed diagnosis submission: %s",
                stored_path,
                exc_info=True,
            )


def _add_nlp_prediction(form_data: dict[str, str]) -> NLPPrediction | None:
    """Add an NLP prediction when the diagnosis contains symptom narrative.

    Symptoms remain clinician-entered source data.  The prediction is stored
    in the legacy diagnosis fields for backward compatibility and returned so
    the caller can persist the authoritative NLP modality record.
    """
    symptoms = form_data.get("symptoms", "")
    if not isinstance(symptoms, str) or not symptoms.strip():
        return None

    try:
        prediction = nlp_service.predict(symptoms)
    except NLPServiceError:
        current_app.logger.warning(
            "Unable to generate an NLP prediction for a new diagnosis.",
            exc_info=True,
        )
        flash(
            "The diagnosis was saved, but an NLP prediction could not be generated.",
            "warning",
        )
        return None

    form_data.update(
        predicted_disease=prediction.label,
        confidence_score=str(prediction.confidence),
        model_name=_NLP_MODEL_NAME,
    )
    return prediction


def edit_diagnosis(diagnosis_id: int) -> str:
    """Render the diagnosis edit form with a real diagnosis and patient list."""
    diagnosis = diagnosis_service.get_diagnosis_by_id(diagnosis_id)
    if diagnosis is None:
        flash("Diagnosis record was not found.", "error")
        return redirect(url_for("diagnosis.list_diagnoses"))

    patients = diagnosis_service.get_all_patients()
    return render_template(
        "diagnosis/edit.html",
        diagnosis=diagnosis,
        patients=patients,
        form_data={},
    )


def update_diagnosis(diagnosis_id: int) -> str | tuple[str, int] | object:
    """Update a diagnosis, or re-render the form with submitted values."""
    form_data = request.form.to_dict()

    try:
        diagnosis = diagnosis_service.update_diagnosis(diagnosis_id, form_data)
    except DiagnosisValidationError as error:
        diagnosis = diagnosis_service.get_diagnosis_by_id(diagnosis_id)
        if diagnosis is None:
            flash("Diagnosis record was not found.", "error")
            return redirect(url_for("diagnosis.list_diagnoses"))

        patients = diagnosis_service.get_all_patients()
        flash(str(error), "error")
        return (
            render_template(
                "diagnosis/edit.html",
                diagnosis=diagnosis,
                patients=patients,
                form_data=form_data,
            ),
            400,
        )
    except Exception:
        current_app.logger.exception("Unable to update diagnosis %s.", diagnosis_id)
        diagnosis = diagnosis_service.get_diagnosis_by_id(diagnosis_id)
        if diagnosis is None:
            flash("Diagnosis record was not found.", "error")
            return redirect(url_for("diagnosis.list_diagnoses"))

        patients = diagnosis_service.get_all_patients()
        flash("Unable to update the diagnosis. Please try again.", "error")
        return (
            render_template(
                "diagnosis/edit.html",
                diagnosis=diagnosis,
                patients=patients,
                form_data=form_data,
            ),
            500,
        )

    if diagnosis is None:
        flash("Diagnosis record was not found.", "error")
        return redirect(url_for("diagnosis.list_diagnoses"))

    flash("Diagnosis updated successfully.", "success")
    return redirect(url_for("diagnosis.list_diagnoses"))


def clinical_assessment(diagnosis_id: int) -> str | object:
    """Render the optional detailed clinical-assessment form."""
    diagnosis = diagnosis_service.get_diagnosis_by_id(diagnosis_id)
    if diagnosis is None:
        flash("Diagnosis record was not found.", "error")
        return redirect(url_for("diagnosis.list_diagnoses"))

    return render_template(
        "diagnosis/clinical_assessment.html",
        diagnosis=diagnosis,
        assessment=diagnosis.clinical_assessment,
        form_data={},
    )


def save_clinical_assessment(diagnosis_id: int) -> str | tuple[str, int] | object:
    """Create or update an assessment and its independent clinical ML result."""
    form_data = request.form.to_dict()
    diagnosis = diagnosis_service.get_diagnosis_by_id(diagnosis_id)
    if diagnosis is None:
        flash("Diagnosis record was not found.", "error")
        return redirect(url_for("diagnosis.list_diagnoses"))

    try:
        clinical_assessment_service.save_assessment(diagnosis_id, form_data)
    except ClinicalAssessmentValidationError as error:
        flash(str(error), "error")
        return (
            render_template(
                "diagnosis/clinical_assessment.html",
                diagnosis=diagnosis,
                assessment=diagnosis.clinical_assessment,
                form_data=form_data,
            ),
            400,
        )
    except Exception:
        current_app.logger.exception(
            "Unable to save clinical assessment for diagnosis %s.", diagnosis_id
        )
        flash("Unable to save the Advanced Clinical Assessment. Please try again.", "error")
        return (
            render_template(
                "diagnosis/clinical_assessment.html",
                diagnosis=diagnosis,
                assessment=diagnosis.clinical_assessment,
                form_data=form_data,
            ),
            500,
        )

    flash("Advanced Clinical Assessment saved successfully.", "success")
    return redirect(url_for("diagnosis.view_diagnosis", diagnosis_id=diagnosis_id))


def delete_diagnosis(diagnosis_id: int) -> str:
    """Delete one diagnosis and return to the diagnosis list."""
    try:
        was_deleted = diagnosis_service.delete_diagnosis(diagnosis_id)
    except Exception:
        current_app.logger.exception("Unable to delete diagnosis %s.", diagnosis_id)
        flash("Unable to delete the diagnosis. Please try again.", "error")
    else:
        if was_deleted:
            flash("Diagnosis deleted successfully.", "success")
        else:
            flash("Diagnosis record was not found.", "error")

    return redirect(url_for("diagnosis.list_diagnoses"))
