"""Persisted AI predictions associated with a diagnosis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import BaseModel

if TYPE_CHECKING:
    from backend.models.diagnosis import Diagnosis


class DiagnosisAIPrediction(BaseModel):
    """Stores one model modality's result for a diagnosis.

    A diagnosis can retain independent clinical ML, NLP, and CNN results.
    Visualization paths are relevant only to explainable image predictions.
    """

    __tablename__ = "diagnosis_ai_predictions"
    __table_args__ = (
        UniqueConstraint(
            "diagnosis_id",
            "prediction_type",
            name="uq_diagnosis_ai_predictions_diagnosis_prediction_type",
        ),
    )

    diagnosis_id: Mapped[int] = mapped_column(
        ForeignKey("diagnoses.id", ondelete="CASCADE"),
        nullable=False,
    )
    prediction_type: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False,
    )
    predicted_label: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    visualization_path: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    diagnosis: Mapped[Diagnosis] = relationship(
        "Diagnosis",
        back_populates="ai_predictions",
    )
