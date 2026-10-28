"""Advanced clinical-assessment validation, feature assembly, and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from math import isfinite
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from backend.extensions import db
from backend.models.diagnosis import Diagnosis
from backend.models.diagnosis_ai_prediction import DiagnosisAIPrediction
from backend.models.diagnosis_clinical_assessment import DiagnosisClinicalAssessment
from backend.services.clinical_ml_service import CLINICAL_FEATURES, ClinicalMLService


class ClinicalAssessmentValidationError(ValueError):
    """Raised when an advanced clinical assessment is incomplete or invalid."""


CLINICAL_PREDICTION_TYPE = "clinical_ml"
CLINICAL_MODEL_NAME = "Clinical Disease Random Forest"

_CATEGORICAL_VALUES = {
    "smoking": frozenset({"Current", "Former", "Never"}),
    "physical_activity": frozenset({"High", "Moderate", "Low"}),
    "family_history": frozenset({"Yes", "No"}),
    "stress_level": frozenset({"High", "Medium", "Low"}),
    "low_hdl_cholesterol": frozenset({"Yes", "No"}),
    "high_ldl_cholesterol": frozenset({"Yes", "No"}),
    "high_blood_pressure": frozenset({"Yes", "No"}),
    "sugar_consumption": frozenset({"High", "Medium", "Low"}),
    "education_level": frozenset({"Primary", "Secondary", "Tertiary"}),
    "employment_status": frozenset({"Employed", "Retired", "Unemployed"}),
}
_NUMERIC_FIELDS = frozenset(
    {
        "bmi",
        "HbA1c_level",
        "glucose",
        "cholesterol",
        "sleep_hours",
        "triglycerides",
        "blood_pressure",
        "crp_level",
        "homocysteine_level",
        "systolic_bp",
        "diastolic_bp",
        "alcohol_intake",
        "salt_intake",
        "heart_rate",
        "hdl",
        "ldl",
    }
)
DIRECT_ASSESSMENT_FIELDS = (
    "smoking",
    "bmi",
    "HbA1c_level",
    "glucose",
    "cholesterol",
    "sleep_hours",
    "triglycerides",
    "physical_activity",
    "family_history",
    "stress_level",
    "low_hdl_cholesterol",
    "high_ldl_cholesterol",
    "blood_pressure",
    "high_blood_pressure",
    "sugar_consumption",
    "crp_level",
    "homocysteine_level",
    "systolic_bp",
    "diastolic_bp",
    "alcohol_intake",
    "salt_intake",
    "heart_rate",
    "hdl",
    "ldl",
    "education_level",
    "employment_status",
)


class ClinicalAssessmentService:
    """Manages optional diagnosis-specific clinical ML assessments."""

    def __init__(self, clinical_ml_service: ClinicalMLService | None = None) -> None:
        self._clinical_ml_service = clinical_ml_service or ClinicalMLService()

    def get_diagnosis(self, diagnosis_id: int) -> Diagnosis | None:
        """Load a diagnosis with the patient and optional assessment."""
        statement = (
            select(Diagnosis)
            .options(joinedload(Diagnosis.patient), joinedload(Diagnosis.clinical_assessment))
            .where(Diagnosis.id == diagnosis_id)
        )
        return db.session.scalar(statement)

    def save_assessment(
        self,
        diagnosis_id: int,
        submitted_data: Mapping[str, Any],
    ) -> DiagnosisClinicalAssessment:
        """Validate, predict, and atomically create or update an assessment."""
        diagnosis = self.get_diagnosis(diagnosis_id)
        if diagnosis is None:
            raise ClinicalAssessmentValidationError("Diagnosis record was not found.")

        cleaned_data = self._clean_assessment_data(submitted_data)
        features = self.build_feature_payload(diagnosis, cleaned_data)
        prediction = self._clinical_ml_service.predict(features)

        assessment = diagnosis.clinical_assessment
        if assessment is None:
            assessment = DiagnosisClinicalAssessment(
                diagnosis_id=diagnosis.id,
                **cleaned_data,
            )
            db.session.add(assessment)
        else:
            for field, value in cleaned_data.items():
                setattr(assessment, field, value)

        self._upsert_clinical_prediction(diagnosis.id, prediction.label, prediction.confidence)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            raise
        return assessment

    def build_feature_payload(
        self,
        diagnosis: Diagnosis,
        assessment_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the exact saved-model feature mapping in its required order."""
        patient = diagnosis.patient
        if patient is None or patient.date_of_birth is None:
            raise ClinicalAssessmentValidationError(
                "The patient date of birth is required for Advanced Clinical Assessment."
            )
        if diagnosis.diagnosis_date is None:
            raise ClinicalAssessmentValidationError(
                "Diagnosis date is required for Advanced Clinical Assessment."
            )
        if patient.gender not in {"Female", "Male"}:
            raise ClinicalAssessmentValidationError(
                "The patient gender must be Female or Male for the trained clinical model."
            )

        age = self._calculate_age(patient.date_of_birth, diagnosis.diagnosis_date)
        if not 2 <= age <= 89:
            raise ClinicalAssessmentValidationError(
                "The patient age at diagnosis must be between 2 and 89 years for the trained clinical model."
            )

        bmi = assessment_data["bmi"]
        feature_values = {
            "age_level": self._derive_age_level(age),
            "gender": patient.gender,
            "bmi_level": self._derive_bmi_level(bmi),
            "age": age,
            **assessment_data,
        }
        return {feature: feature_values[feature] for feature in CLINICAL_FEATURES}

    @staticmethod
    def _calculate_age(date_of_birth: date, diagnosis_date: date) -> int:
        """Return the completed years of age on the assessment date."""
        return diagnosis_date.year - date_of_birth.year - (
            (diagnosis_date.month, diagnosis_date.day)
            < (date_of_birth.month, date_of_birth.day)
        )

    @staticmethod
    def _derive_age_level(age: int) -> str:
        if 2 <= age <= 23:
            return "Young"
        if age <= 45:
            return "Middle"
        if age <= 67:
            return "Mature"
        if age <= 89:
            return "Senior"
        raise ClinicalAssessmentValidationError(
            "The patient age is outside the trained clinical model range."
        )

    @staticmethod
    def _derive_bmi_level(bmi: float) -> str:
        if bmi < 18.5:
            return "Underweight"
        if bmi < 25:
            return "Normal"
        if bmi < 30:
            return "Overweight"
        return "Obese"

    @staticmethod
    def _display_name(field: str) -> str:
        return field.replace("_", " ").replace("HbA1c", "HbA1c").title()

    def _clean_assessment_data(self, submitted_data: Mapping[str, Any]) -> dict[str, Any]:
        """Require every direct field and enforce the trained-data categories."""
        cleaned_data: dict[str, Any] = {}
        for field in DIRECT_ASSESSMENT_FIELDS:
            value = submitted_data.get(field)
            if isinstance(value, str):
                value = value.strip()
            if value in (None, ""):
                raise ClinicalAssessmentValidationError(
                    f"{self._display_name(field)} is required."
                )

            if field in _NUMERIC_FIELDS:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as error:
                    raise ClinicalAssessmentValidationError(
                        f"{self._display_name(field)} must be a numeric value."
                    ) from error
                if not isfinite(numeric_value):
                    raise ClinicalAssessmentValidationError(
                        f"{self._display_name(field)} must be a finite numeric value."
                    )
                if field == "bmi" and numeric_value <= 0:
                    raise ClinicalAssessmentValidationError("BMI must be greater than zero.")
                cleaned_data[field] = numeric_value
            else:
                allowed_values = _CATEGORICAL_VALUES[field]
                if value not in allowed_values:
                    allowed = ", ".join(sorted(allowed_values))
                    raise ClinicalAssessmentValidationError(
                        f"{self._display_name(field)} must be one of: {allowed}."
                    )
                cleaned_data[field] = value
        return cleaned_data

    @staticmethod
    def _upsert_clinical_prediction(
        diagnosis_id: int,
        predicted_label: str,
        confidence_score: float,
    ) -> None:
        """Upsert only this modality, leaving NLP and CNN data independent."""
        prediction = db.session.scalar(
            select(DiagnosisAIPrediction).where(
                DiagnosisAIPrediction.diagnosis_id == diagnosis_id,
                DiagnosisAIPrediction.prediction_type == CLINICAL_PREDICTION_TYPE,
            )
        )
        if prediction is None:
            db.session.add(
                DiagnosisAIPrediction(
                    diagnosis_id=diagnosis_id,
                    prediction_type=CLINICAL_PREDICTION_TYPE,
                    predicted_label=predicted_label,
                    confidence_score=confidence_score,
                    model_name=CLINICAL_MODEL_NAME,
                )
            )
            return

        prediction.predicted_label = predicted_label
        prediction.confidence_score = confidence_score
        prediction.model_name = CLINICAL_MODEL_NAME
        prediction.visualization_path = None
