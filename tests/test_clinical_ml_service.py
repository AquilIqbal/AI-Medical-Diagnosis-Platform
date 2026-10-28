"""Tests for the isolated clinical ML prediction service."""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services.clinical_ml_service import (
    CLINICAL_FEATURES,
    ClinicalMLService,
    ClinicalMLServiceError,
    ClinicalPrediction,
    _load_clinical_model,
)


def _features() -> dict[str, object]:
    """Return one complete record using the notebook feature contract."""
    numeric_features = {
        "age": 45,
        "bmi": 24.5,
        "HbA1c_level": 5.8,
        "glucose": 100.0,
        "cholesterol": 190.0,
        "sleep_hours": 7.0,
        "triglycerides": 140.0,
        "blood_pressure": 120.0,
        "crp_level": 2.0,
        "homocysteine_level": 8.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "alcohol_intake": 1.0,
        "salt_intake": 5.0,
        "heart_rate": 72.0,
        "hdl": 55.0,
        "ldl": 110.0,
    }
    categorical_features = {
        "age_level": "Middle",
        "gender": "Female",
        "bmi_level": "Normal",
        "smoking": "Never",
        "physical_activity": "High",
        "family_history": "No",
        "stress_level": "Low",
        "low_hdl_cholesterol": "No",
        "high_ldl_cholesterol": "No",
        "high_blood_pressure": "No",
        "sugar_consumption": "Low",
        "education_level": "Primary",
        "employment_status": "Employed",
    }
    return {**numeric_features, **categorical_features}


def test_predict_uses_notebook_features_and_maps_binary_output() -> None:
    """The service sends the ordered feature frame and returns named output."""
    model = type(
        "Model",
        (),
        {
            "predict": lambda self, frame: [1],
            "predict_proba": lambda self, frame: [[0.2, 0.8]],
            "classes_": [0, 1],
        },
    )()

    with patch(
        "backend.services.clinical_ml_service._load_clinical_model",
        return_value=model,
    ):
        result = ClinicalMLService().predict(_features())

    assert isinstance(result, ClinicalPrediction)
    assert result.class_id == 1
    assert result.label == "Abnormal"
    assert result.confidence == pytest.approx(0.8)
    assert result.probabilities == {"Normal": 0.2, "Abnormal": 0.8}


def test_predict_rejects_missing_required_feature() -> None:
    """Incomplete clinical data fails before model execution."""
    features = _features()
    features.pop(CLINICAL_FEATURES[0])

    with pytest.raises(ClinicalMLServiceError, match="age_level"):
        ClinicalMLService().predict(features)


def test_model_loader_reports_missing_artifact() -> None:
    """An unavailable artifact produces a user-safe service error."""
    with patch(
        "backend.services.clinical_ml_service.CLINICAL_MODEL_PATH",
        Path("missing-clinical-model.joblib"),
    ):
        _load_clinical_model.cache_clear()
        with pytest.raises(ClinicalMLServiceError, match="artifact is unavailable"):
            _load_clinical_model()
        _load_clinical_model.cache_clear()