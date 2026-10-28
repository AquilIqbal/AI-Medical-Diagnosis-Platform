"""Regression tests for the diagnosis create/list workflow."""

import io
from unittest.mock import patch

from backend import create_app
from backend.services.diagnosis_service import DiagnosisValidationError
from backend.services.clinical_assessment_service import ClinicalAssessmentValidationError
from backend.services.image_prediction_service import ImagePrediction
from backend.services.nlp_service import NLPServiceError, NLPPrediction
from config import Config


class TestConfig(Config):
    """Use the existing app configuration for route-based tests."""

    TESTING = True


class UploadLimitConfig(TestConfig):
    """Exercise the global request-size guard without a production database."""

    MAX_CONTENT_LENGTH = 64


def test_diagnosis_create_and_list_workflow() -> None:
    """Creating a diagnosis persists a record and shows it in the list view."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "first_name": "Ada",
            "last_name": "Lovelace",
        },
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-12345678",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "predicted_disease": "Migraine",
            "confidence_score": 0.91,
        },
    )()

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as service_mock, patch(
        "backend.controllers.diagnosis_controller.nlp_service"
    ):
        service_mock.get_all_patients.return_value = [patient]
        service_mock.get_all_diagnoses.return_value = [diagnosis]

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis/new")
            assert response.status_code == 200
            assert b"Ada Lovelace" in response.data
            assert b'enctype="multipart/form-data"' in response.data
            assert b'name="medical_image"' in response.data

            service_mock.create_diagnosis.return_value = diagnosis
            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Severe headache",
                    "status": "Pending",
                    "symptoms": "Migraine",
                },
                follow_redirects=False,
            )
            assert response.status_code == 302
            assert response.headers["Location"].endswith("/diagnosis")

            response = client.get("/diagnosis")
            assert response.status_code == 200
            assert b"DX-12345678" in response.data
            assert b"Ada Lovelace" in response.data


def test_diagnosis_list_search_uses_and_retains_the_query() -> None:
    """Diagnosis search follows the existing patient-list GET search pattern."""
    app = create_app(TestConfig)
    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Ada",
            "last_name": "Lovelace",
        },
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-PNEUMONIA",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "In Review",
            "predicted_disease": "Pneumonia",
            "confidence_score": 0.91,
        },
    )()

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.get_all_diagnoses.return_value = [diagnosis]

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis?search=%20Pneumonia%20")

    assert response.status_code == 200
    service_mock.get_all_diagnoses.assert_called_once_with("Pneumonia")
    assert b'name="search"' in response.data
    assert b'value="Pneumonia"' in response.data
    assert b"DX-PNEUMONIA" in response.data


def test_diagnosis_service_searches_diagnosis_and_patient_fields() -> None:
    """The diagnosis query covers displayed diagnosis, patient, and status data."""
    from backend.services.diagnosis_service import DiagnosisService

    with patch("backend.services.diagnosis_service.db.session") as session_mock:
        session_mock.scalars.return_value = []
        DiagnosisService().get_all_diagnoses("pneumonia")

    query_sql = str(session_mock.scalars.call_args.args[0])
    assert "diagnoses.diagnosis_code" in query_sql
    assert "patients.patient_code" in query_sql
    assert "patients.first_name" in query_sql
    assert "diagnoses.predicted_disease" in query_sql
    assert "diagnoses.status" in query_sql


def test_advanced_clinical_assessment_route_is_optional_and_submits_separately() -> None:
    """Quick Assessment stays independent from the optional clinical workflow."""
    app = create_app(TestConfig)
    patient = type("Patient", (), {"first_name": "Ada", "last_name": "Lovelace"})()
    diagnosis = type(
        "Diagnosis",
        (),
        {"id": 7, "patient": patient, "clinical_assessment": None},
    )()

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock, patch(
        "backend.controllers.diagnosis_controller.clinical_assessment_service"
    ) as assessment_service_mock:
        diagnosis_service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis/7/clinical-assessment")
            assert response.status_code == 200
            assert b"Advanced Clinical Assessment" in response.data
            assert b'name="high_blood_pressure"' in response.data

            response = client.post(
                "/diagnosis/7/clinical-assessment",
                data={"bmi": "24.5"},
                follow_redirects=False,
            )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/diagnosis/7")
    assessment_service_mock.save_assessment.assert_called_once_with(7, {"bmi": "24.5"})


def test_advanced_clinical_assessment_validation_preserves_submitted_values() -> None:
    """Invalid advanced input is rejected without affecting the diagnosis record."""
    app = create_app(TestConfig)
    patient = type("Patient", (), {"first_name": "Ada", "last_name": "Lovelace"})()
    diagnosis = type(
        "Diagnosis",
        (),
        {"id": 8, "patient": patient, "clinical_assessment": None},
    )()

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock, patch(
        "backend.controllers.diagnosis_controller.clinical_assessment_service"
    ) as assessment_service_mock:
        diagnosis_service_mock.get_diagnosis_by_id.return_value = diagnosis
        assessment_service_mock.save_assessment.side_effect = ClinicalAssessmentValidationError(
            "Glucose is required."
        )

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis/8/clinical-assessment",
                data={"bmi": "24.5"},
                follow_redirects=False,
            )

    assert response.status_code == 400
    assert b"Glucose is required." in response.data
    assert b'value="24.5"' in response.data


def test_diagnosis_create_reads_and_stores_uploaded_image_filename() -> None:
    """A valid image receives a collision-resistant stored filename."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False
    diagnosis = type("Diagnosis", (), {"id": 5, "diagnosis_code": "DX-IMAGE"})()
    prediction = ImagePrediction(
        predicted_class=1,
        predicted_label="Pneumonia",
        confidence=0.85,
        visualization_png=b"gradcam",
    )

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as service_mock, patch(
        "backend.controllers.diagnosis_controller.image_prediction_service"
    ) as image_service_mock, patch(
        "backend.controllers.diagnosis_controller._save_gradcam_visualization",
        return_value="gradcam/DX-IMAGE.png",
    ), patch("werkzeug.datastructures.FileStorage.save") as save_mock:
        service_mock.create_diagnosis.return_value = diagnosis
        image_service_mock.predict.return_value = prediction

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Severe headache",
                    "status": "Pending",
                    "medical_image": (
                        io.BytesIO(b"valid image content"),
                        "../../chest-scan.png",
                        "image/png",
                    ),
                },
                content_type="multipart/form-data",
                follow_redirects=False,
            )

    assert response.status_code == 302
    submitted_data = service_mock.create_diagnosis.call_args.args[0]
    assert submitted_data["uploaded_image"].endswith("_chest-scan.png")
    assert submitted_data["uploaded_image"] != "chest-scan.png"
    image_service_mock.predict.assert_called_once()
    service_mock.upsert_ai_prediction.assert_called_once_with(
        5,
        prediction_type="cnn",
        predicted_label="Pneumonia",
        confidence_score=0.85,
        model_name="Enhanced Pneumonia CNN",
        visualization_path="gradcam/DX-IMAGE.png",
    )
    save_mock.assert_called_once()


def test_diagnosis_create_rejects_invalid_medical_image() -> None:
    """Invalid images are rejected before diagnosis persistence."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as service_mock:
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Severe headache",
                    "medical_image": (
                        io.BytesIO(b"not an image"),
                        "scan.png",
                        "image/png",
                    ),
                },
                content_type="multipart/form-data",
                follow_redirects=False,
            )

    assert response.status_code == 400
    service_mock.create_diagnosis.assert_not_called()


def test_diagnosis_create_stores_nlp_prediction_from_symptoms() -> None:
    """Symptom narrative should populate the diagnosis AI-result fields."""
    app = create_app(TestConfig)
    prediction = NLPPrediction(
        class_id=14,
        label="Pneumonia",
        confidence=0.86,
        probabilities={"Pneumonia": 0.86},
    )

    diagnosis = type("Diagnosis", (), {"id": 9})()

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock, patch(
        "backend.controllers.diagnosis_controller.nlp_service"
    ) as nlp_service_mock:
        nlp_service_mock.predict.return_value = prediction
        diagnosis_service_mock.create_diagnosis.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Persistent cough",
                    "symptoms": "Persistent cough, fever, and chest pain",
                },
                follow_redirects=False,
            )

    assert response.status_code == 302
    nlp_service_mock.predict.assert_called_once_with(
        "Persistent cough, fever, and chest pain"
    )
    submitted_data = diagnosis_service_mock.create_diagnosis.call_args.args[0]
    assert submitted_data["predicted_disease"] == "Pneumonia"
    assert submitted_data["confidence_score"] == "0.86"
    assert submitted_data["model_name"] == "TF-IDF + Logistic Regression"
    diagnosis_service_mock.upsert_ai_prediction.assert_called_once_with(
        9,
        prediction_type="nlp",
        predicted_label="Pneumonia",
        confidence_score=0.86,
        model_name="TF-IDF + Logistic Regression",
    )


def test_diagnosis_create_skips_nlp_for_empty_symptoms() -> None:
    """A diagnosis without symptom narrative must still be created normally."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock, patch(
        "backend.controllers.diagnosis_controller.nlp_service"
    ) as nlp_service_mock:
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Follow-up appointment",
                    "symptoms": "   ",
                },
                follow_redirects=False,
            )

    assert response.status_code == 302
    nlp_service_mock.predict.assert_not_called()
    submitted_data = diagnosis_service_mock.create_diagnosis.call_args.args[0]
    assert "predicted_disease" not in submitted_data
    assert "confidence_score" not in submitted_data
    assert "model_name" not in submitted_data


def test_diagnosis_create_without_image_skips_cnn() -> None:
    """Existing diagnosis creation remains valid when no image is supplied."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock, patch(
        "backend.controllers.diagnosis_controller.image_prediction_service"
    ) as image_service_mock:
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Follow-up appointment",
                },
                follow_redirects=False,
            )

    assert response.status_code == 302
    image_service_mock.predict.assert_not_called()
    diagnosis_service_mock.upsert_ai_prediction.assert_not_called()


def test_diagnosis_create_continues_when_nlp_is_unavailable() -> None:
    """An NLP failure must not prevent normal diagnosis creation."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock, patch(
        "backend.controllers.diagnosis_controller.nlp_service"
    ) as nlp_service_mock:
        nlp_service_mock.predict.side_effect = NLPServiceError("artifact unavailable")

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "Persistent cough",
                    "symptoms": "Persistent cough and fever",
                },
                follow_redirects=False,
            )

    assert response.status_code == 302
    diagnosis_service_mock.create_diagnosis.assert_called_once()
    submitted_data = diagnosis_service_mock.create_diagnosis.call_args.args[0]
    assert "predicted_disease" not in submitted_data
    diagnosis_service_mock.upsert_ai_prediction.assert_not_called()


def test_diagnosis_creation_cleans_new_upload_when_persistence_fails() -> None:
    """A validation failure after saving an upload removes only that new file."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.diagnosis_controller.image_prediction_service"
    ) as image_service_mock, patch(
        "backend.controllers.diagnosis_controller._save_uploaded_image",
        return_value="new_scan.png",
    ), patch(
        "backend.controllers.diagnosis_controller._cleanup_created_uploads"
    ) as cleanup_mock, patch(
        "backend.controllers.diagnosis_controller.diagnosis_service"
    ) as diagnosis_service_mock:
        image_service_mock.predict.return_value = ImagePrediction(
            predicted_class=0,
            predicted_label="Normal",
            confidence=0.8,
            visualization_png=b"gradcam",
        )
        diagnosis_service_mock.create_diagnosis.side_effect = DiagnosisValidationError(
            "Chief Complaint is required."
        )

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis",
                data={
                    "patient_id": "1",
                    "diagnosis_date": "2026-08-08",
                    "chief_complaint": "",
                    "medical_image": (io.BytesIO(b"image"), "scan.png", "image/png"),
                },
                content_type="multipart/form-data",
            )

    assert response.status_code == 400
    cleanup_mock.assert_called_once_with("new_scan.png")


def test_global_upload_limit_redirects_to_the_referring_form() -> None:
    """Large request bodies are rejected before form/image processing."""
    app = create_app(UploadLimitConfig)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "Test User"

        response = client.post(
            "/diagnosis",
            data={"chief_complaint": "x" * 128},
            headers={"Referer": "http://localhost/diagnosis/new"},
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/diagnosis/new")


def test_diagnosis_view_renders_real_record() -> None:
    """A known diagnosis ID should render its real details."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    patient = type(
        "Patient",
        (),
        {
            "id": 2,
            "first_name": "Grace",
            "last_name": "Hopper",
        },
    )()
    ai_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "cnn",
            "predicted_label": "Pneumonia",
            "confidence_score": 0.86,
            "model_name": "Enhanced Pneumonia CNN",
            "visualization_path": "gradcam/DX-87654321.png",
        },
    )()
    clinical_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Abnormal",
            "confidence_score": 0.255492,
            "model_name": "Clinical Disease Random Forest",
            "visualization_path": None,
        },
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 7,
            "diagnosis_code": "DX-87654321",
            "patient": patient,
            "diagnosis_date": "2026-08-10",
            "status": "In Review",
            "chief_complaint": "Chest pain",
            "symptoms": "Shortness of breath",
            "medical_history": "Asthma",
            "doctor_notes": "Monitor closely",
            "uploaded_image": "DX-87654321_scan.png",
            "predicted_disease": "Pneumonia",
            "confidence_score": 0.86,
            "model_name": "Demo Model",
            "created_at": "2026-08-01 00:00:00",
            "updated_at": "2026-08-02 00:00:00",
            "ai_predictions": [ai_prediction, clinical_prediction],
        },
    )()

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis/7")
            assert response.status_code == 200
            assert b"DX-87654321" in response.data
            assert b"Grace Hopper" in response.data
            assert b"Chest pain" in response.data
            assert b"Enhanced Pneumonia CNN" in response.data
            assert b"/uploads/gradcam/DX-87654321.png" in response.data
            assert b"/uploads/DX-87654321_scan.png" in response.data
            assert b"NLP Prediction" in response.data
            assert b"Clinical ML Prediction" in response.data
            assert b"CNN Image Prediction" in response.data
            assert b"86.0%" in response.data
            assert b"25.5%" in response.data
            assert b"The uploaded medical image is unavailable." in response.data
            assert b"The Grad-CAM visualization is unavailable." in response.data


def test_diagnosis_view_handles_missing_ai_modalities_gracefully() -> None:
    """Absent predictions and images render explicit states instead of broken cards."""
    app = create_app(TestConfig)
    patient = type(
        "Patient", (), {"first_name": "Grace", "last_name": "Hopper", "phone": None, "email": None}
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 8,
            "diagnosis_code": "DX-NONE",
            "patient": patient,
            "diagnosis_date": "2026-08-10",
            "status": "Pending",
            "chief_complaint": "Check-up",
            "symptoms": None,
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "predicted_disease": None,
            "confidence_score": None,
            "model_name": None,
            "created_at": None,
            "updated_at": None,
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis/8")

    assert response.status_code == 200
    assert response.data.count(b"Not performed.") == 3
    assert b"No medical image has been uploaded." in response.data


def test_diagnosis_view_redirects_for_missing_id() -> None:
    """A missing diagnosis ID should redirect back to the list."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.get_diagnosis_by_id.return_value = None

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis/999", follow_redirects=False)
            assert response.status_code == 302
            assert response.headers["Location"].endswith("/diagnosis")


def test_diagnosis_edit_renders_form_with_real_patients() -> None:
    """The edit page should preload diagnosis values and patient choices."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    patient = type(
        "Patient",
        (),
        {
            "id": 3,
            "first_name": "Linus",
            "last_name": "Torvalds",
        },
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 11,
            "diagnosis_code": "DX-AAAA1111",
            "patient": patient,
            "patient_id": 3,
            "diagnosis_date": "2026-08-12",
            "chief_complaint": "Fever",
            "symptoms": "Body aches",
            "medical_history": "No known issues",
            "doctor_notes": "Rest recommended",
            "status": "Pending",
            "predicted_disease": "Flu",
            "confidence_score": 0.75,
            "model_name": "Demo Model",
        },
    )()

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        service_mock.get_all_patients.return_value = [patient]

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/diagnosis/11/edit")
            assert response.status_code == 200
            assert b"Fever" in response.data
            assert b"Linus Torvalds" in response.data
            assert b"DX-AAAA1111" not in response.data


def test_diagnosis_edit_updates_selected_diagnosis() -> None:
    """Submitting valid edit data should call the service and redirect to the list."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    patient = type(
        "Patient",
        (),
        {
            "id": 4,
            "first_name": "Katherine",
            "last_name": "Johnson",
        },
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 12,
            "diagnosis_code": "DX-BBBB2222",
            "patient": patient,
            "patient_id": 4,
            "diagnosis_date": "2026-08-13",
            "chief_complaint": "Headache",
            "symptoms": "Nausea",
            "medical_history": "None",
            "doctor_notes": "Hydrate",
            "status": "Pending",
            "predicted_disease": "Migraine",
            "confidence_score": 0.8,
            "model_name": "Demo Model",
        },
    )()

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.update_diagnosis.return_value = diagnosis
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        service_mock.get_all_patients.return_value = [patient]

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis/12/edit",
                data={
                    "patient_id": "4",
                    "diagnosis_date": "2026-08-14",
                    "chief_complaint": "Migraine",
                    "symptoms": "Light sensitivity",
                    "medical_history": "No history",
                    "doctor_notes": "Watch closely",
                    "status": "In Review",
                },
                follow_redirects=False,
            )
            assert response.status_code == 302
            assert response.headers["Location"].endswith("/diagnosis")
            service_mock.update_diagnosis.assert_called_once()
            assert service_mock.update_diagnosis.call_args.args[0] == 12


def test_diagnosis_delete_redirects_to_list_and_uses_selected_id() -> None:
    """Deleting a diagnosis should call the service with the selected ID and redirect to the list."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.delete_diagnosis.return_value = True

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis/21/delete",
                follow_redirects=False,
            )
            assert response.status_code == 302
            assert response.headers["Location"].endswith("/diagnosis")
            service_mock.delete_diagnosis.assert_called_once_with(21)


def test_diagnosis_delete_handles_missing_diagnosis() -> None:
    """Missing diagnoses should be reported cleanly and redirect to the list."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.delete_diagnosis.return_value = False

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis/999/delete",
                follow_redirects=True,
            )
            assert response.status_code == 200
            assert b"Diagnosis record was not found." in response.data


def test_diagnosis_service_delete_diagnosis_only_targets_selected_record() -> None:
    """The service should delete a single diagnosis and commit once for the selected record."""
    from backend.services.diagnosis_service import DiagnosisService

    service = DiagnosisService()
    diagnosis = object()

    with patch("backend.services.diagnosis_service.db.session") as session_mock:
        session_mock.scalar.return_value = diagnosis

        deleted = service.delete_diagnosis(7)

    assert deleted is True
    session_mock.delete.assert_called_once_with(diagnosis)
    session_mock.commit.assert_called_once()


def test_diagnosis_edit_preserves_values_on_validation_error() -> None:
    """Validation failures should re-render the edit form with submitted values."""
    app = create_app(TestConfig)
    app.config["WTF_CSRF_ENABLED"] = False

    patient = type(
        "Patient",
        (),
        {
            "id": 5,
            "first_name": "Margaret",
            "last_name": "Hamilton",
        },
    )()
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 13,
            "diagnosis_code": "DX-CCCC3333",
            "patient": patient,
            "patient_id": 5,
            "diagnosis_date": "2026-08-15",
            "chief_complaint": "Fatigue",
            "symptoms": "Low energy",
            "medical_history": "None",
            "doctor_notes": "Observe",
            "status": "Pending",
        },
    )()

    with patch("backend.controllers.diagnosis_controller.diagnosis_service") as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        service_mock.get_all_patients.return_value = [patient]
        service_mock.update_diagnosis.side_effect = DiagnosisValidationError(
            "Chief Complaint is required."
        )

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.post(
                "/diagnosis/13/edit",
                data={
                    "patient_id": "5",
                    "diagnosis_date": "2026-08-16",
                    "chief_complaint": "",
                    "symptoms": "Dizziness",
                    "medical_history": "None",
                    "doctor_notes": "Monitor",
                    "status": "Pending",
                },
                follow_redirects=False,
            )
            assert response.status_code == 400
            assert b"Chief Complaint is required." in response.data
            assert b"Dizziness" in response.data
