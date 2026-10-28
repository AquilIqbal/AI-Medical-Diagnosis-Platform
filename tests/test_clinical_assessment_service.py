"""Tests for the optional Advanced Clinical Assessment workflow."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import inspect

from backend.models.diagnosis import Diagnosis
from backend.models.diagnosis_clinical_assessment import DiagnosisClinicalAssessment
from backend.services.clinical_assessment_service import (
    CLINICAL_PREDICTION_TYPE,
    ClinicalAssessmentService,
    ClinicalAssessmentValidationError,
    DIRECT_ASSESSMENT_FIELDS,
)
from backend.services.clinical_ml_service import CLINICAL_FEATURES, ClinicalPrediction


def _assessment_data() -> dict[str, str]:
    return {
        "smoking": "Never",
        "bmi": "24.5",
        "HbA1c_level": "5.8",
        "glucose": "100",
        "cholesterol": "190",
        "sleep_hours": "7",
        "triglycerides": "140",
        "physical_activity": "High",
        "family_history": "No",
        "stress_level": "Low",
        "low_hdl_cholesterol": "No",
        "high_ldl_cholesterol": "No",
        "blood_pressure": "120",
        "high_blood_pressure": "No",
        "sugar_consumption": "Low",
        "crp_level": "2",
        "homocysteine_level": "8",
        "systolic_bp": "120",
        "diastolic_bp": "80",
        "alcohol_intake": "1",
        "salt_intake": "5",
        "heart_rate": "72",
        "hdl": "55",
        "ldl": "110",
        "education_level": "Primary",
        "employment_status": "Employed",
    }


def _diagnosis(*, gender: str | None = "Female", dob: date | None = date(1981, 8, 10)):
    patient = type("Patient", (), {"gender": gender, "date_of_birth": dob})()
    return type(
        "Diagnosis",
        (),
        {
            "id": 7,
            "patient": patient,
            "diagnosis_date": date(2026, 8, 10),
            "clinical_assessment": None,
        },
    )()


def test_clinical_assessment_schema_and_relationship() -> None:
    table = DiagnosisClinicalAssessment.__table__

    assert set(table.columns.keys()) == {"id", "diagnosis_id", *DIRECT_ASSESSMENT_FIELDS, "created_at", "updated_at"}
    assert table.c.diagnosis_id.unique is True
    assert all(table.c[field].nullable is False for field in DIRECT_ASSESSMENT_FIELDS)
    foreign_key = next(iter(table.foreign_keys))
    assert foreign_key.target_fullname == "diagnoses.id"
    assert foreign_key.ondelete == "CASCADE"
    relationship = inspect(Diagnosis).relationships["clinical_assessment"]
    assert relationship.uselist is False
    assert relationship.back_populates == "diagnosis"


def test_feature_payload_has_exact_model_order_and_derived_values() -> None:
    service = ClinicalAssessmentService()
    cleaned = service._clean_assessment_data(_assessment_data())

    payload = service.build_feature_payload(_diagnosis(), cleaned)

    assert tuple(payload) == CLINICAL_FEATURES
    assert len(payload) == 30
    assert payload["age"] == 45
    assert payload["age_level"] == "Middle"
    assert payload["gender"] == "Female"
    assert payload["bmi_level"] == "Normal"
    assert payload["high_blood_pressure"] == "No"


@pytest.mark.parametrize(
    ("age", "expected"),
    [(2, "Young"), (23, "Young"), (24, "Middle"), (45, "Middle"), (46, "Mature"), (67, "Mature"), (68, "Senior"), (89, "Senior")],
)
def test_age_level_boundaries(age: int, expected: str) -> None:
    assert ClinicalAssessmentService._derive_age_level(age) == expected


@pytest.mark.parametrize(
    ("bmi", "expected"),
    [(18.49, "Underweight"), (18.5, "Normal"), (24.99, "Normal"), (25, "Overweight"), (29.99, "Overweight"), (30, "Obese")],
)
def test_bmi_level_boundaries(bmi: float, expected: str) -> None:
    assert ClinicalAssessmentService._derive_bmi_level(bmi) == expected


def test_missing_direct_field_is_rejected_before_inference() -> None:
    data = _assessment_data()
    data.pop("glucose")

    with pytest.raises(ClinicalAssessmentValidationError, match="Glucose is required"):
        ClinicalAssessmentService()._clean_assessment_data(data)


@pytest.mark.parametrize(
    ("gender", "dob", "message"),
    [
        ("Other", date(1981, 8, 10), "Female or Male"),
        ("Female", None, "date of birth"),
    ],
)
def test_unsupported_patient_demographics_are_rejected(gender, dob, message) -> None:
    service = ClinicalAssessmentService()
    cleaned = service._clean_assessment_data(_assessment_data())

    with pytest.raises(ClinicalAssessmentValidationError, match=message):
        service.build_feature_payload(_diagnosis(gender=gender, dob=dob), cleaned)


def test_save_assessment_invokes_model_and_creates_only_clinical_prediction() -> None:
    model = MagicMock()
    model.predict.return_value = ClinicalPrediction(1, "Abnormal", 0.82, {"Abnormal": 0.82})
    service = ClinicalAssessmentService(model)
    diagnosis = _diagnosis()

    with patch.object(service, "get_diagnosis", return_value=diagnosis), patch(
        "backend.services.clinical_assessment_service.db.session"
    ) as session_mock:
        session_mock.scalar.return_value = None
        assessment = service.save_assessment(7, _assessment_data())

    assert assessment.diagnosis_id == 7
    model.predict.assert_called_once()
    assert tuple(model.predict.call_args.args[0]) == CLINICAL_FEATURES
    persisted_prediction = next(
        call.args[0]
        for call in session_mock.add.call_args_list
        if getattr(call.args[0], "prediction_type", None) == CLINICAL_PREDICTION_TYPE
    )
    assert persisted_prediction.predicted_label == "Abnormal"
    assert persisted_prediction.model_name == "Clinical Disease Random Forest"
    session_mock.commit.assert_called_once()


def test_repeated_submission_updates_only_existing_clinical_prediction() -> None:
    model = MagicMock()
    model.predict.return_value = ClinicalPrediction(0, "Normal", 0.77, {"Normal": 0.77})
    service = ClinicalAssessmentService(model)
    assessment = DiagnosisClinicalAssessment(diagnosis_id=7, **service._clean_assessment_data(_assessment_data()))
    diagnosis = _diagnosis()
    diagnosis.clinical_assessment = assessment
    clinical_prediction = type(
        "Prediction",
        (),
        {
            "prediction_type": "clinical_ml",
            "predicted_label": "Abnormal",
            "confidence_score": 0.82,
            "model_name": "Clinical Disease Random Forest",
            "visualization_path": None,
        },
    )()
    legacy_nlp = {"predicted_disease": "Pneumonia", "confidence_score": 0.86, "model_name": "TF-IDF + Logistic Regression"}
    cnn_prediction = {"prediction_type": "cnn", "predicted_label": "Pneumonia", "confidence_score": 0.9}

    with patch.object(service, "get_diagnosis", return_value=diagnosis), patch(
        "backend.services.clinical_assessment_service.db.session"
    ) as session_mock:
        session_mock.scalar.return_value = clinical_prediction
        returned = service.save_assessment(7, {**_assessment_data(), "bmi": "31"})

    assert returned is assessment
    assert assessment.bmi == 31
    assert clinical_prediction.predicted_label == "Normal"
    assert clinical_prediction.confidence_score == pytest.approx(0.77)
    assert legacy_nlp == {"predicted_disease": "Pneumonia", "confidence_score": 0.86, "model_name": "TF-IDF + Logistic Regression"}
    assert cnn_prediction == {"prediction_type": "cnn", "predicted_label": "Pneumonia", "confidence_score": 0.9}
    session_mock.add.assert_not_called()
