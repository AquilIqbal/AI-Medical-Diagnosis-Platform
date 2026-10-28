"""Diagnosis persistence model for future clinical decision workflows."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import BaseModel

if TYPE_CHECKING:
    from backend.models.diagnosis_ai_prediction import DiagnosisAIPrediction
    from backend.models.diagnosis_clinical_assessment import DiagnosisClinicalAssessment
    from backend.models.patient import Patient


class Diagnosis(BaseModel):
    """Represents a clinical diagnosis and its future AI prediction metadata."""

    __tablename__ = "diagnoses"

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        index=True,
        nullable=False,
    )
    diagnosis_code: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
        nullable=False,
    )
    diagnosis_date: Mapped[date] = mapped_column(Date, nullable=False)
    chief_complaint: Mapped[str | None] = mapped_column(Text, nullable=True)
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)
    medical_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    doctor_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    predicted_disease: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)

    patient: Mapped[Patient] = relationship("Patient", backref="diagnoses")
    ai_predictions: Mapped[list[DiagnosisAIPrediction]] = relationship(
        "DiagnosisAIPrediction",
        back_populates="diagnosis",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    clinical_assessment: Mapped[DiagnosisClinicalAssessment | None] = relationship(
        "DiagnosisClinicalAssessment",
        back_populates="diagnosis",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
