"""Persisted detailed clinical inputs for an optional diagnosis assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import BaseModel

if TYPE_CHECKING:
    from backend.models.diagnosis import Diagnosis


class DiagnosisClinicalAssessment(BaseModel):
    """Stores the direct inputs required by the clinical ML model.

    Demographic and derived model features are deliberately excluded.  They are
    assembled from the associated diagnosis and patient at prediction time.
    """

    __tablename__ = "diagnosis_clinical_assessments"

    diagnosis_id: Mapped[int] = mapped_column(
        ForeignKey("diagnoses.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    smoking: Mapped[str] = mapped_column(String(16), nullable=False)
    bmi: Mapped[float] = mapped_column(Float, nullable=False)
    HbA1c_level: Mapped[float] = mapped_column(Float, nullable=False)
    glucose: Mapped[float] = mapped_column(Float, nullable=False)
    cholesterol: Mapped[float] = mapped_column(Float, nullable=False)
    sleep_hours: Mapped[float] = mapped_column(Float, nullable=False)
    triglycerides: Mapped[float] = mapped_column(Float, nullable=False)
    physical_activity: Mapped[str] = mapped_column(String(16), nullable=False)
    family_history: Mapped[str] = mapped_column(String(8), nullable=False)
    stress_level: Mapped[str] = mapped_column(String(16), nullable=False)
    low_hdl_cholesterol: Mapped[str] = mapped_column(String(8), nullable=False)
    high_ldl_cholesterol: Mapped[str] = mapped_column(String(8), nullable=False)
    blood_pressure: Mapped[float] = mapped_column(Float, nullable=False)
    high_blood_pressure: Mapped[str] = mapped_column(String(8), nullable=False)
    sugar_consumption: Mapped[str] = mapped_column(String(16), nullable=False)
    crp_level: Mapped[float] = mapped_column(Float, nullable=False)
    homocysteine_level: Mapped[float] = mapped_column(Float, nullable=False)
    systolic_bp: Mapped[float] = mapped_column(Float, nullable=False)
    diastolic_bp: Mapped[float] = mapped_column(Float, nullable=False)
    alcohol_intake: Mapped[float] = mapped_column(Float, nullable=False)
    salt_intake: Mapped[float] = mapped_column(Float, nullable=False)
    heart_rate: Mapped[float] = mapped_column(Float, nullable=False)
    hdl: Mapped[float] = mapped_column(Float, nullable=False)
    ldl: Mapped[float] = mapped_column(Float, nullable=False)
    education_level: Mapped[str] = mapped_column(String(16), nullable=False)
    employment_status: Mapped[str] = mapped_column(String(16), nullable=False)

    diagnosis: Mapped[Diagnosis] = relationship(
        "Diagnosis",
        back_populates="clinical_assessment",
    )
