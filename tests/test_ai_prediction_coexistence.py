"""Integration tests for independent AI modality persistence."""

from datetime import date

from backend import create_app
from backend.extensions import db
from backend.models.diagnosis import Diagnosis
from backend.models.patient import Patient
from backend.services.diagnosis_service import DiagnosisService
from config import Config


class SqliteTestConfig(Config):
    """Use an isolated in-memory database instead of the configured MySQL DB."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_ENGINE_OPTIONS = {}


def test_nlp_cnn_and_clinical_ml_predictions_coexist_without_overwrites() -> None:
    """One diagnosis retains independent rows for every implemented modality."""
    app = create_app(SqliteTestConfig)

    with app.app_context():
        db.create_all()
        patient = Patient(patient_code="PT-TEST", first_name="Ada", last_name="Lovelace")
        diagnosis = Diagnosis(
            patient=patient,
            diagnosis_code="DX-TEST",
            diagnosis_date=date(2026, 8, 28),
            chief_complaint="Cough",
            predicted_disease="Pneumonia",
            confidence_score=0.86,
            model_name="TF-IDF + Logistic Regression",
            status="Pending",
        )
        db.session.add(diagnosis)
        db.session.commit()

        service = DiagnosisService()
        service.upsert_ai_prediction(
            diagnosis.id,
            prediction_type="nlp",
            predicted_label="Pneumonia",
            confidence_score=0.86,
            model_name="TF-IDF + Logistic Regression",
        )
        service.upsert_ai_prediction(
            diagnosis.id,
            prediction_type="cnn",
            predicted_label="Pneumonia",
            confidence_score=0.91,
            model_name="Enhanced Pneumonia CNN",
            visualization_path="gradcam/DX-TEST.png",
        )
        service.upsert_ai_prediction(
            diagnosis.id,
            prediction_type="clinical_ml",
            predicted_label="Abnormal",
            confidence_score=0.78,
            model_name="Clinical Disease Random Forest",
        )
        service.upsert_ai_prediction(
            diagnosis.id,
            prediction_type="nlp",
            predicted_label="Influenza",
            confidence_score=0.88,
            model_name="TF-IDF + Logistic Regression",
        )

        refreshed = service.get_diagnosis_by_id(diagnosis.id)
        assert refreshed is not None
        predictions = {prediction.prediction_type: prediction for prediction in refreshed.ai_predictions}
        assert set(predictions) == {"nlp", "cnn", "clinical_ml"}
        assert predictions["nlp"].predicted_label == "Influenza"
        assert predictions["cnn"].predicted_label == "Pneumonia"
        assert predictions["cnn"].visualization_path == "gradcam/DX-TEST.png"
        assert predictions["clinical_ml"].predicted_label == "Abnormal"
        assert refreshed.predicted_disease == "Pneumonia"
        db.session.remove()
        db.drop_all()
