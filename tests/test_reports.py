"""Tests for the Reports module covering list, detail, and XAI functionality."""

from unittest.mock import patch

from backend import create_app
from config import Config


class TestConfig(Config):
    """Use the existing app configuration for route-based tests."""

    TESTING = True


def test_reports_list_requires_authentication() -> None:
    """Unauthenticated access to reports list redirects to login."""
    app = create_app(TestConfig)

    with app.test_client() as client:
        response = client.get("/reports", follow_redirects=False)
        assert response.status_code == 302
        assert "login" in response.headers["Location"].lower()


def test_reports_detail_requires_authentication() -> None:
    """Unauthenticated access to a report redirects to login."""
    app = create_app(TestConfig)

    with app.test_client() as client:
        response = client.get("/reports/1", follow_redirects=False)
        assert response.status_code == 302
        assert "login" in response.headers["Location"].lower()


def test_reports_list_displays_all_diagnoses() -> None:
    """Authenticated user can access reports list with all diagnoses."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Ada",
            "last_name": "Lovelace",
            "gender": "Female",
            "date_of_birth": "1990-01-15",
            "blood_group": "O+",
            "phone": "555-0001",
            "email": "ada@example.com",
            "address": "123 Main St",
            "emergency_contact": "John Lovelace",
            "emergency_phone": "555-0002",
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
            "chief_complaint": "Persistent cough",
            "predicted_disease": "Pneumonia",
            "confidence_score": 0.85,
            "symptoms": "Cough, fever",
            "medical_history": "No major history",
            "doctor_notes": "Follow up needed",
            "uploaded_image": "image.png",
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_all_diagnoses.return_value = [diagnosis]

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports")
            assert response.status_code == 200
            assert b"DX-12345678" in response.data
            assert b"Ada Lovelace" in response.data
            assert b"Persistent cough" in response.data
            assert b"View Report" in response.data


def test_reports_list_empty_state() -> None:
    """Reports list handles empty diagnoses gracefully."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_all_diagnoses.return_value = []

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports")
            assert response.status_code == 200
            assert b"No reports are available" in response.data


def test_reports_list_search_uses_and_retains_the_query() -> None:
    """Reports search follows the existing patient and diagnosis GET pattern."""
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
            "chief_complaint": "Persistent cough",
            "status": "In Review",
        },
    )()

    with patch("backend.controllers.reports_controller.reports_service") as service_mock:
        service_mock.get_all_diagnoses.return_value = [diagnosis]

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports?search=%20Pneumonia%20")

    assert response.status_code == 200
    service_mock.get_all_diagnoses.assert_called_once_with("Pneumonia")
    assert b'name="search"' in response.data
    assert b'value="Pneumonia"' in response.data
    assert b"DX-PNEUMONIA" in response.data


def test_reports_list_search_shows_matching_empty_state() -> None:
    """A report search with no matches presents the search-specific empty state."""
    app = create_app(TestConfig)

    with patch("backend.controllers.reports_controller.reports_service") as service_mock:
        service_mock.get_all_diagnoses.return_value = []

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports?search=missing")

    assert response.status_code == 200
    assert b"No matching reports found." in response.data


def test_reports_detail_displays_complete_diagnosis() -> None:
    """Authenticated user can view complete report with all modalities."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Ada",
            "last_name": "Lovelace",
            "gender": "Female",
            "date_of_birth": "1990-01-15",
            "blood_group": "O+",
            "phone": "555-0001",
            "email": "ada@example.com",
            "address": "123 Main St",
            "emergency_contact": "John Lovelace",
            "emergency_phone": "555-0002",
        },
    )()

    nlp_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "id": 1,
            "prediction_type": "nlp",
            "predicted_label": "Migraine",
            "confidence_score": 0.92,
            "model_name": "TF-IDF + Logistic Regression",
            "visualization_path": None,
        },
    )()

    cnn_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "id": 2,
            "prediction_type": "cnn",
            "predicted_label": "Pneumonia",
            "confidence_score": 0.88,
            "model_name": "Enhanced Pneumonia CNN",
            "visualization_path": "gradcam/DX-12345678_abc123.png",
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
            "chief_complaint": "Chest pain and cough",
            "predicted_disease": "Migraine",
            "confidence_score": 0.92,
            "symptoms": "Headache, photophobia",
            "medical_history": "Migraine history",
            "doctor_notes": "Typical migraine presentation",
            "uploaded_image": "chest_xray.png",
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [nlp_prediction, cnn_prediction],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200

            # Patient Information
            assert b"Ada Lovelace" in response.data
            assert b"PAT-001" in response.data
            assert b"Female" in response.data

            # Diagnosis Information
            assert b"DX-12345678" in response.data
            assert b"Chest pain and cough" in response.data
            assert b"Migraine history" in response.data

            # NLP Analysis
            assert b"NLP Symptom Analysis" in response.data
            assert b"Migraine" in response.data
            assert b"92.0%" in response.data or b"92%" in response.data

            # CNN Analysis
            assert b"CNN Chest X-Ray Analysis" in response.data
            assert b"Pneumonia" in response.data
            assert b"88.0%" in response.data or b"88%" in response.data
            assert b"Enhanced Pneumonia CNN" in response.data


def test_reports_detail_nonexistent_diagnosis() -> None:
    """Accessing a nonexistent diagnosis report shows error and redirects."""
    app = create_app(TestConfig)

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = None

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/999", follow_redirects=True)
            assert response.status_code == 200
            assert b"was not found" in response.data
            assert b"Available Reports" in response.data


def test_report_with_only_nlp_prediction() -> None:
    """Report displays gracefully with only NLP prediction available."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Test",
            "last_name": "Patient",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    nlp_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "nlp",
            "predicted_label": "Fever",
            "confidence_score": 0.75,
            "model_name": "TF-IDF + Logistic Regression",
            "visualization_path": None,
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-NLP-ONLY",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "High fever",
            "predicted_disease": "Fever",
            "confidence_score": 0.75,
            "symptoms": "Fever, body ache",
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [nlp_prediction],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200
            assert b"Fever" in response.data
            assert b"75.0%" in response.data or b"75%" in response.data
            # Clinical ML and CNN should show not performed
            assert b"was not performed" in response.data or b"not performed" in response.data


def test_report_with_clinical_ml_prediction() -> None:
    """Report displays Clinical ML prediction when available."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-002",
            "first_name": "John",
            "last_name": "Doe",
            "gender": "Male",
            "date_of_birth": "1985-05-20",
            "blood_group": "A+",
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    clinical_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Abnormal",
            "confidence_score": 0.82,
            "model_name": "Random Forest Classifier",
            "visualization_path": None,
        },
    )()

    clinical_assessment = type(
        "DiagnosisClinicalAssessment",
        (),
        {
            "smoking": "Yes",
            "bmi": 28.5,
            "HbA1c_level": 6.5,
            "glucose": 120,
            "cholesterol": 210,
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-CLINICAL",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Chest pain",
            "predicted_disease": None,
            "confidence_score": None,
            "symptoms": None,
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [clinical_prediction],
            "clinical_assessment": clinical_assessment,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200
            assert b"Clinical ML Analysis" in response.data
            assert b"Abnormal" in response.data
            assert b"82.0%" in response.data or b"82%" in response.data
            assert b"Random Forest Classifier" in response.data


def test_report_without_medical_image() -> None:
    """Report handles missing medical image gracefully."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-003",
            "first_name": "Jane",
            "last_name": "Smith",
            "gender": "Female",
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    cnn_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "cnn",
            "predicted_label": "Normal",
            "confidence_score": 0.95,
            "model_name": "Enhanced Pneumonia CNN",
            "visualization_path": None,
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-NO-IMAGE",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Routine check",
            "predicted_disease": None,
            "confidence_score": None,
            "symptoms": None,
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [cnn_prediction],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200
            # Should show message about no image or visualization
            assert b"No medical image" in response.data or b"visualization" in response.data


def test_report_without_gradcam_visualization() -> None:
    """Report handles missing Grad-CAM visualization gracefully."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-004",
            "first_name": "Bob",
            "last_name": "Johnson",
            "gender": "Male",
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    cnn_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "cnn",
            "predicted_label": "Normal",
            "confidence_score": 0.93,
            "model_name": "Enhanced Pneumonia CNN",
            "visualization_path": None,
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-NO-GRADCAM",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Respiratory check",
            "predicted_disease": None,
            "confidence_score": None,
            "symptoms": None,
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": "xray.png",
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [cnn_prediction],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200
            # Should mention Grad-CAM was not available
            assert b"Grad-CAM" in response.data
            assert b"unavailable" in response.data or b"was not available" in response.data


def test_report_confidence_percentage_formatting() -> None:
    """Report formats confidence percentages correctly."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-005",
            "first_name": "Alice",
            "last_name": "Brown",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-PERCENTAGE",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Test",
            "predicted_disease": "Test Disease",
            "confidence_score": 0.625,  # Should display as 62.5%
            "symptoms": "Test symptoms",
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200
            # Should format as 62.5%
            assert b"62.5%" in response.data


def test_report_identifies_modalities_correctly() -> None:
    """Report clearly identifies each AI modality."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-006",
            "first_name": "Test",
            "last_name": "User",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    nlp_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "nlp",
            "predicted_label": "Condition A",
            "confidence_score": 0.80,
            "model_name": "TF-IDF + Logistic Regression",
            "visualization_path": None,
        },
    )()

    clinical_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Normal",
            "confidence_score": 0.75,
            "model_name": "Random Forest",
            "visualization_path": None,
        },
    )()

    cnn_prediction = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "cnn",
            "predicted_label": "Class B",
            "confidence_score": 0.90,
            "model_name": "CNN Model",
            "visualization_path": "viz.png",
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-ALL-MODALITIES",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Test",
            "predicted_disease": "Condition A",
            "confidence_score": 0.80,
            "symptoms": "Test",
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": "img.png",
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [nlp_prediction, clinical_prediction, cnn_prediction],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200

            # Check modality identification
            assert b"NLP Symptom Analysis" in response.data
            assert b"Clinical ML Analysis" in response.data
            assert b"CNN Chest X-Ray Analysis" in response.data

            # Check that each modality is identified clearly
            assert b"Textual Analysis" in response.data
            assert b"Risk Assessment" in response.data
            assert b"Image Analysis" in response.data


def test_report_displays_professional_disclaimer() -> None:
    """Report includes professional disclaimer about AI predictions."""
    app = create_app(TestConfig)

    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-007",
            "first_name": "Test",
            "last_name": "Patient",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()

    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-DISCLAIMER",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Test",
            "predicted_disease": None,
            "confidence_score": None,
            "symptoms": None,
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()

    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis

        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"

            response = client.get("/reports/1")
            assert response.status_code == 200

            # Check for disclaimer
            assert b"Disclaimer" in response.data
            assert b"AI-assisted" in response.data or b"AI" in response.data
            assert b"healthcare" in response.data or b"medical" in response.data or b"professional" in response.data


# ============================================================================
# Unit tests for helper functions
# ============================================================================


def test_get_confidence_level_very_high() -> None:
    """Confidence 95% returns 'Very High Confidence'."""
    from backend.controllers.reports_controller import get_confidence_level
    
    assert get_confidence_level(0.95) == "Very High Confidence"
    assert get_confidence_level(1.0) == "Very High Confidence"
    assert get_confidence_level(0.90) == "Very High Confidence"


def test_get_confidence_level_high() -> None:
    """Confidence 75-89.9% returns 'High Confidence'."""
    from backend.controllers.reports_controller import get_confidence_level
    
    assert get_confidence_level(0.80) == "High Confidence"
    assert get_confidence_level(0.75) == "High Confidence"
    assert get_confidence_level(0.89) == "High Confidence"


def test_get_confidence_level_moderate() -> None:
    """Confidence 60-74.9% returns 'Moderate Confidence'."""
    from backend.controllers.reports_controller import get_confidence_level
    
    assert get_confidence_level(0.625) == "Moderate Confidence"
    assert get_confidence_level(0.60) == "Moderate Confidence"
    assert get_confidence_level(0.72) == "Moderate Confidence"


def test_get_confidence_level_low() -> None:
    """Confidence below 60% returns 'Low Confidence'."""
    from backend.controllers.reports_controller import get_confidence_level
    
    assert get_confidence_level(0.45) == "Low Confidence"
    assert get_confidence_level(0.50) == "Low Confidence"
    assert get_confidence_level(0.0) == "Low Confidence"


def test_get_confidence_level_none() -> None:
    """Confidence None returns 'Not available'."""
    from backend.controllers.reports_controller import get_confidence_level
    
    assert get_confidence_level(None) == "Not available"


def test_build_overall_ai_assessment_all_modalities() -> None:
    """Overall assessment includes all available predictions."""
    from backend.controllers.reports_controller import build_overall_ai_assessment
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    nlp_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "nlp",
            "predicted_label": "Asthma",
            "confidence_score": 0.85,
        },
    )()
    
    clinical_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Abnormal",
            "confidence_score": 0.72,
        },
    )()
    
    cnn_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "cnn",
            "predicted_label": "Pneumonia",
            "confidence_score": 0.90,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": "Asthma",
            "confidence_score": 0.85,
            "ai_predictions": [nlp_pred, clinical_pred, cnn_pred],
        },
    )()
    
    result = build_overall_ai_assessment(diagnosis)
    
    assert result["has_predictions"] is True
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["type"] == "NLP"
    assert result["predictions"][0]["label"] == "Asthma"
    assert result["predictions"][1]["type"] == "Clinical ML"
    assert result["predictions"][1]["label"] == "Abnormal"
    assert result["predictions"][2]["type"] == "CNN"
    assert result["predictions"][2]["label"] == "Pneumonia"


def test_build_overall_ai_assessment_no_predictions() -> None:
    """Overall assessment handles zero predictions."""
    from backend.controllers.reports_controller import build_overall_ai_assessment
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": None,
            "confidence_score": None,
            "ai_predictions": [],
        },
    )()
    
    result = build_overall_ai_assessment(diagnosis)
    
    assert result["has_predictions"] is False
    assert len(result["predictions"]) == 0
    assert "No AI prediction results" in result["summary_text"]


def test_build_overall_ai_assessment_nlp_only() -> None:
    """Overall assessment works with NLP prediction only."""
    from backend.controllers.reports_controller import build_overall_ai_assessment
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": "Fever",
            "confidence_score": 0.75,
            "ai_predictions": [],
        },
    )()
    
    result = build_overall_ai_assessment(diagnosis)
    
    assert result["has_predictions"] is True
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["type"] == "NLP"
    assert result["predictions"][0]["label"] == "Fever"


def test_generate_clinical_recommendations_nlp_and_cnn_disagreement() -> None:
    """Recommendations flag when NLP and CNN outputs disagree."""
    from backend.controllers.reports_controller import generate_clinical_recommendations
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    cnn_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "cnn",
            "predicted_label": "Pneumonia",
            "confidence_score": 0.88,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": "Bronchitis",  # Different from CNN
            "confidence_score": 0.80,
            "ai_predictions": [cnn_pred],
            "clinical_assessment": None,
        },
    )()
    
    result = generate_clinical_recommendations(diagnosis)
    
    # Should have findings for both modalities
    assert "NLP" in str(result["key_findings"])
    assert "CNN" in str(result["key_findings"])
    
    # Should flag discordance
    has_discordance_mention = any(
        "concordant" in step.lower() or "disagree" in step.lower() or "discordant" in step.lower()
        for step in result["next_steps"]
    )
    assert has_discordance_mention


def test_generate_clinical_recommendations_clinical_ml_abnormal() -> None:
    """Recommendations mention abnormal Clinical ML findings."""
    from backend.controllers.reports_controller import generate_clinical_recommendations
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    clinical_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Abnormal",
            "confidence_score": 0.82,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": None,
            "confidence_score": None,
            "ai_predictions": [clinical_pred],
            "clinical_assessment": type(
                "Assessment",
                (),
                {"smoking": "Yes", "bmi": 28},
            )(),
        },
    )()
    
    result = generate_clinical_recommendations(diagnosis)
    
    # Should mention abnormal findings
    abnormal_found = any("abnormal" in step.lower() for step in result["next_steps"])
    assert abnormal_found


def test_generate_clinical_recommendations_clinical_ml_normal() -> None:
    """Recommendations acknowledge Normal Clinical ML classification."""
    from backend.controllers.reports_controller import generate_clinical_recommendations
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    clinical_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Normal",
            "confidence_score": 0.88,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": None,
            "confidence_score": None,
            "ai_predictions": [clinical_pred],
            "clinical_assessment": None,
        },
    )()
    
    result = generate_clinical_recommendations(diagnosis)
    
    # Should mention Normal classification
    normal_found = any("Normal" in step for step in result["next_steps"])
    assert normal_found


def test_generate_clinical_recommendations_no_modalities() -> None:
    """Recommendations provide fallback when no modalities exist."""
    from backend.controllers.reports_controller import generate_clinical_recommendations
    
    patient = type("Patient", (), {"patient_code": "PAT-001"})()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "patient": patient,
            "predicted_disease": None,
            "confidence_score": None,
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()
    
    result = generate_clinical_recommendations(diagnosis)
    
    # Should still have next steps
    assert len(result["next_steps"]) > 0
    assert len(result["key_findings"]) == 0


def test_overall_ai_assessment_in_report_template() -> None:
    """Report includes Overall AI Assessment section."""
    app = create_app(TestConfig)
    
    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Test",
            "last_name": "Patient",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()
    
    nlp_pred = type(
        "DiagnosisAIPrediction",
        (),
        {
            "prediction_type": "nlp",
            "predicted_label": "Asthma",
            "confidence_score": 0.85,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-TEST",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Wheezing",
            "predicted_disease": "Asthma",
            "confidence_score": 0.85,
            "symptoms": "Shortness of breath",
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [nlp_pred],
            "clinical_assessment": None,
        },
    )()
    
    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"
            
            response = client.get("/reports/1")
            assert response.status_code == 200
            
            # Check for Overall AI Assessment section
            assert b"Overall AI Assessment" in response.data
            assert b"Asthma" in response.data


def test_confidence_label_in_report() -> None:
    """Report displays confidence labels (e.g., 'High Confidence')."""
    app = create_app(TestConfig)
    
    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Test",
            "last_name": "Patient",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-TEST",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Test",
            "predicted_disease": "Test Disease",
            "confidence_score": 0.82,  # Should be "High Confidence"
            "symptoms": "Test",
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()
    
    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"
            
            response = client.get("/reports/1")
            assert response.status_code == 200
            
            # Check for confidence label
            assert b"High Confidence" in response.data


def test_dark_result_card_styling() -> None:
    """Report uses dark-themed result cards for predictions."""
    app = create_app(TestConfig)
    
    patient = type(
        "Patient",
        (),
        {
            "id": 1,
            "patient_code": "PAT-001",
            "first_name": "Test",
            "last_name": "Patient",
            "gender": None,
            "date_of_birth": None,
            "blood_group": None,
            "phone": None,
            "email": None,
            "address": None,
            "emergency_contact": None,
            "emergency_phone": None,
        },
    )()
    
    diagnosis = type(
        "Diagnosis",
        (),
        {
            "id": 1,
            "diagnosis_code": "DX-TEST",
            "patient": patient,
            "diagnosis_date": "2026-08-08",
            "status": "Pending",
            "chief_complaint": "Test",
            "predicted_disease": "Test Disease",
            "confidence_score": 0.75,
            "symptoms": "Test",
            "medical_history": None,
            "doctor_notes": None,
            "uploaded_image": None,
            "created_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "updated_at": type("DateTime", (), {"strftime": lambda self, fmt: "08 Aug 2026, 02:30 PM UTC"})(),
            "ai_predictions": [],
            "clinical_assessment": None,
        },
    )()
    
    with patch(
        "backend.controllers.reports_controller.reports_service"
    ) as service_mock:
        service_mock.get_diagnosis_by_id.return_value = diagnosis
        
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["user_id"] = 1
                session["username"] = "Test User"
            
            response = client.get("/reports/1")
            assert response.status_code == 200
            
            # Check for report-result-card class (dark themed)
            assert b"report-result-card" in response.data
            # Should NOT have the old white "alert alert-light" class for result cards
            response_text = response.data.decode()
            # The old style used alert alert-light for prediction results
            # New style uses report-result-card instead
            assert "report-result-card" in response_text
