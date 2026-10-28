"""Business operations for patient persistence."""

from collections.abc import Mapping
from datetime import date
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from backend.extensions import db
from backend.models.patient import Patient


class PatientValidationError(ValueError):
    """Raised when patient data cannot be safely persisted."""


class PatientService:
    """Provides reusable CRUD operations for patient records."""

    _MUTABLE_FIELDS = frozenset(
        {
            "patient_code",
            "first_name",
            "last_name",
            "gender",
            "date_of_birth",
            "blood_group",
            "phone",
            "email",
            "address",
            "emergency_contact",
            "emergency_phone",
        }
    )

    def create_patient(self, patient_data: Mapping[str, Any]) -> Patient:
        """Validate, create, and persist a patient record."""
        cleaned_data = self._clean_patient_data(patient_data)
        patient = Patient(
            **{
                field: value
                for field, value in cleaned_data.items()
                if field in self._MUTABLE_FIELDS
            }
        )
        db.session.add(patient)

        try:
            db.session.commit()
        except IntegrityError as error:
            db.session.rollback()
            raise PatientValidationError(
                "A patient with this patient code already exists."
            ) from error

        return patient

    def get_patient_by_id(self, patient_id: int) -> Patient | None:
        """Return a patient by primary key, or ``None`` when absent."""
        return db.session.get(Patient, patient_id)

    def get_patient_by_code(self, patient_code: str) -> Patient | None:
        """Return a patient by unique patient code, or ``None`` when absent."""
        return db.session.scalar(
            select(Patient).where(Patient.patient_code == patient_code)
        )

    def get_all_patients(self, search_query: str | None = None) -> list[Patient]:
        """Return patients, optionally filtered by a case-insensitive query."""
        statement = select(Patient)
        normalized_query = (search_query or "").strip()

        if normalized_query:
            search_pattern = f"%{normalized_query}%"
            full_name = Patient.first_name + " " + Patient.last_name
            statement = statement.where(
                or_(
                    Patient.patient_code.ilike(search_pattern),
                    Patient.first_name.ilike(search_pattern),
                    Patient.last_name.ilike(search_pattern),
                    full_name.ilike(search_pattern),
                    Patient.phone.ilike(search_pattern),
                    Patient.email.ilike(search_pattern),
                )
            )

        statement = statement.order_by(Patient.created_at.desc())
        return list(db.session.scalars(statement))

    def update_patient(
        self,
        patient_id: int,
        patient_data: Mapping[str, Any],
    ) -> Patient | None:
        """Validate and persist changes to one existing patient record."""
        patient = self.get_patient_by_id(patient_id)
        if patient is None:
            return None

        mutable_data = {
            field: value
            for field, value in patient_data.items()
            if field != "patient_code"
        }
        cleaned_data = self._clean_patient_data(
            mutable_data,
            generate_patient_code=False,
        )

        for field, value in cleaned_data.items():
            if field in self._MUTABLE_FIELDS:
                setattr(patient, field, value)

        try:
            db.session.commit()
        except IntegrityError as error:
            db.session.rollback()
            raise PatientValidationError(
                "Unable to update the patient record."
            ) from error

        return patient

    def delete_patient(self, patient_id: int) -> bool:
        """Delete a patient and return whether a record was removed."""
        patient = self.get_patient_by_id(patient_id)
        if patient is None:
            return False

        db.session.delete(patient)
        db.session.commit()
        return True

    def _clean_patient_data(
        self,
        patient_data: Mapping[str, Any],
        *,
        generate_patient_code: bool = True,
    ) -> dict[str, Any]:
        """Normalize submitted values and enforce create-patient requirements."""
        cleaned_data = {
            field: self._normalize_value(value)
            for field, value in patient_data.items()
            if field in self._MUTABLE_FIELDS
        }

        required_fields = ("first_name", "last_name")
        missing_fields = [
            field.replace("_", " ").title()
            for field in required_fields
            if not cleaned_data.get(field)
        ]
        if missing_fields:
            raise PatientValidationError(
                f"{', '.join(missing_fields)} is required."
            )

        if cleaned_data.get("gender") == "Select gender":
            cleaned_data["gender"] = None
        if cleaned_data.get("blood_group") == "Select blood group":
            cleaned_data["blood_group"] = None

        date_of_birth = cleaned_data.get("date_of_birth")
        if date_of_birth:
            try:
                cleaned_data["date_of_birth"] = date.fromisoformat(date_of_birth)
            except (TypeError, ValueError) as error:
                raise PatientValidationError(
                    "Date of birth must be a valid date."
                ) from error

        patient_code = cleaned_data.get("patient_code")
        if patient_code:
            if self.get_patient_by_code(patient_code):
                raise PatientValidationError(
                    "A patient with this patient code already exists."
                )
        elif generate_patient_code:
            cleaned_data["patient_code"] = self._generate_patient_code()

        return cleaned_data

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Trim submitted strings and convert blank strings to ``None``."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    def _generate_patient_code(self) -> str:
        """Generate a unique, human-readable code for a new patient."""
        while True:
            patient_code = f"PT-{uuid4().hex[:8].upper()}"
            if self.get_patient_by_code(patient_code) is None:
                return patient_code
