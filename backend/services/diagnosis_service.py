"""Business operations for diagnosis persistence."""

from collections.abc import Mapping
from datetime import date
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload, selectinload

from backend.extensions import db
from backend.models.diagnosis import Diagnosis
from backend.models.diagnosis_ai_prediction import DiagnosisAIPrediction
from backend.models.diagnosis_clinical_assessment import DiagnosisClinicalAssessment
from backend.models.patient import Patient


class DiagnosisValidationError(ValueError):
    """Raised when diagnosis data cannot be safely persisted."""


class DiagnosisService:
    """Provides reusable CRUD operations for diagnosis records."""

    _CREATE_FIELDS = frozenset(
        {
            "patient_id",
            "diagnosis_date",
            "chief_complaint",
            "symptoms",
            "medical_history",
            "doctor_notes",
            "uploaded_image",
            "predicted_disease",
            "confidence_score",
            "model_name",
            "status",
        }
    )
    _UPDATE_FIELDS = frozenset(
        {
            "patient_id",
            "diagnosis_date",
            "chief_complaint",
            "symptoms",
            "medical_history",
            "doctor_notes",
            "status",
        }
    )

    def create_diagnosis(self, diagnosis_data: Mapping[str, Any]) -> Diagnosis:
        """Validate, create, and persist a diagnosis with a generated code."""
        cleaned_data = self._clean_diagnosis_data(
            diagnosis_data,
            allowed_fields=self._CREATE_FIELDS,
        )
        diagnosis = Diagnosis(
            **cleaned_data,
            diagnosis_code=self._generate_diagnosis_code(),
        )
        db.session.add(diagnosis)
        self._commit_or_raise("Unable to create the diagnosis record.")
        return diagnosis

    def get_diagnosis_by_id(self, diagnosis_id: int) -> Diagnosis | None:
        """Return one diagnosis and its selected patient, when present."""
        statement = (
            select(Diagnosis)
            .options(
                joinedload(Diagnosis.patient),
                selectinload(Diagnosis.ai_predictions),
                selectinload(Diagnosis.clinical_assessment),
            )
            .where(Diagnosis.id == diagnosis_id)
        )
        return db.session.scalar(statement)

    def get_all_diagnoses(self, search_query: str | None = None) -> list[Diagnosis]:
        """Return diagnoses, optionally filtered by a case-insensitive query."""
        statement = select(Diagnosis).options(joinedload(Diagnosis.patient))
        normalized_query = (search_query or "").strip()

        if normalized_query:
            search_pattern = f"%{normalized_query}%"
            patient_name = Patient.first_name + " " + Patient.last_name
            statement = statement.join(Diagnosis.patient).where(
                or_(
                    Diagnosis.diagnosis_code.ilike(search_pattern),
                    Patient.patient_code.ilike(search_pattern),
                    Patient.first_name.ilike(search_pattern),
                    Patient.last_name.ilike(search_pattern),
                    patient_name.ilike(search_pattern),
                    Diagnosis.predicted_disease.ilike(search_pattern),
                    Diagnosis.status.ilike(search_pattern),
                )
            )

        statement = statement.order_by(Diagnosis.created_at.desc())
        return list(db.session.scalars(statement))

    def get_all_patients(self) -> list[Patient]:
        """Return patients for Diagnosis form selection, ordered by name."""
        statement = select(Patient).order_by(Patient.first_name, Patient.last_name)
        return list(db.session.scalars(statement))

    def update_diagnosis(
        self,
        diagnosis_id: int,
        diagnosis_data: Mapping[str, Any],
    ) -> Diagnosis | None:
        """Validate and persist editable fields for one diagnosis record."""
        diagnosis = self.get_diagnosis_by_id(diagnosis_id)
        if diagnosis is None:
            return None

        cleaned_data = self._clean_diagnosis_data(
            diagnosis_data,
            allowed_fields=self._UPDATE_FIELDS,
        )
        for field, value in cleaned_data.items():
            setattr(diagnosis, field, value)

        self._commit_or_raise("Unable to update the diagnosis record.")
        return diagnosis

    def delete_diagnosis(self, diagnosis_id: int) -> bool:
        """Delete one diagnosis record and return whether it existed."""
        diagnosis = self.get_diagnosis_by_id(diagnosis_id)
        if diagnosis is None:
            return False

        db.session.delete(diagnosis)
        self._commit_or_raise("Unable to delete the diagnosis record.")
        return True

    def upsert_ai_prediction(
        self,
        diagnosis_id: int,
        *,
        prediction_type: str,
        predicted_label: str,
        confidence_score: float,
        model_name: str,
        visualization_path: str | None = None,
    ) -> DiagnosisAIPrediction:
        """Create or refresh one authoritative model modality result."""
        prediction = db.session.scalar(
            select(DiagnosisAIPrediction).where(
                DiagnosisAIPrediction.diagnosis_id == diagnosis_id,
                DiagnosisAIPrediction.prediction_type == prediction_type,
            )
        )
        if prediction is None:
            prediction = DiagnosisAIPrediction(
                diagnosis_id=diagnosis_id,
                prediction_type=prediction_type,
                predicted_label=predicted_label,
                confidence_score=confidence_score,
                model_name=model_name,
                visualization_path=visualization_path,
            )
            db.session.add(prediction)
        else:
            prediction.predicted_label = predicted_label
            prediction.confidence_score = confidence_score
            prediction.model_name = model_name
            prediction.visualization_path = visualization_path

        self._commit_or_raise("Unable to save the AI prediction.")
        return prediction

    def delete_ai_prediction(self, diagnosis_id: int, prediction_type: str) -> bool:
        """Remove one stale modality result after its input is cleared or changed."""
        prediction = db.session.scalar(
            select(DiagnosisAIPrediction).where(
                DiagnosisAIPrediction.diagnosis_id == diagnosis_id,
                DiagnosisAIPrediction.prediction_type == prediction_type,
            )
        )
        if prediction is None:
            return False

        db.session.delete(prediction)
        self._commit_or_raise("Unable to remove the AI prediction.")
        return True

    def _clean_diagnosis_data(
        self,
        diagnosis_data: Mapping[str, Any],
        *,
        allowed_fields: frozenset[str],
    ) -> dict[str, Any]:
        """Normalize submitted fields and enforce diagnosis requirements."""
        cleaned_data = {
            field: self._normalize_value(value)
            for field, value in diagnosis_data.items()
            if field in allowed_fields
        }

        required_fields = ("patient_id", "diagnosis_date", "chief_complaint")
        missing_fields = [
            field.replace("_", " ").title()
            for field in required_fields
            if not cleaned_data.get(field)
        ]
        if missing_fields:
            raise DiagnosisValidationError(
                f"{', '.join(missing_fields)} is required."
            )

        if not cleaned_data.get("status"):
            cleaned_data["status"] = "Pending"

        try:
            patient_id = int(cleaned_data["patient_id"])
        except (TypeError, ValueError) as error:
            raise DiagnosisValidationError("Please select a valid patient.") from error

        if db.session.get(Patient, patient_id) is None:
            raise DiagnosisValidationError("The selected patient does not exist.")
        cleaned_data["patient_id"] = patient_id

        try:
            cleaned_data["diagnosis_date"] = date.fromisoformat(
                cleaned_data["diagnosis_date"]
            )
        except (TypeError, ValueError) as error:
            raise DiagnosisValidationError(
                "Diagnosis Date must be a valid date."
            ) from error

        if cleaned_data.get("confidence_score") is not None:
            try:
                cleaned_data["confidence_score"] = float(
                    cleaned_data["confidence_score"]
                )
            except (TypeError, ValueError) as error:
                raise DiagnosisValidationError(
                    "Confidence Score must be a numeric value."
                ) from error

        return cleaned_data

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Trim submitted strings and convert blank strings to ``None``."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    def _generate_diagnosis_code(self) -> str:
        """Generate a unique, human-readable code for a new diagnosis."""
        while True:
            diagnosis_code = f"DX-{uuid4().hex[:8].upper()}"
            statement = select(Diagnosis.id).where(
                Diagnosis.diagnosis_code == diagnosis_code
            )
            if db.session.scalar(statement) is None:
                return diagnosis_code

    @staticmethod
    def _commit_or_raise(error_message: str) -> None:
        """Commit the active transaction or roll it back with a safe error."""
        try:
            db.session.commit()
        except IntegrityError as error:
            db.session.rollback()
            raise DiagnosisValidationError(error_message) from error
        except Exception:
            db.session.rollback()
            raise
