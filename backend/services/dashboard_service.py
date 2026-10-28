"""Read-only summary queries for the authenticated dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload, selectinload

from backend.extensions import db
from backend.models.diagnosis import Diagnosis
from backend.models.diagnosis_ai_prediction import DiagnosisAIPrediction
from backend.models.patient import Patient


@dataclass(frozen=True)
class DashboardStatistics:
    """Small set of truthful dashboard totals."""

    patient_count: int
    diagnosis_count: int
    ai_prediction_count: int
    report_count: int


@dataclass(frozen=True)
class RecentDiagnosisActivity:
    """A diagnosis and its most recently stored model result, when present."""

    diagnosis: Diagnosis
    latest_prediction: DiagnosisAIPrediction | None


class DashboardService:
    """Provides lightweight dashboard data using existing persisted records."""

    def get_statistics(self) -> DashboardStatistics:
        diagnosis_count = int(
            db.session.scalar(select(func.count(Diagnosis.id))) or 0
        )
        return DashboardStatistics(
            patient_count=int(db.session.scalar(select(func.count(Patient.id))) or 0),
            diagnosis_count=diagnosis_count,
            ai_prediction_count=int(
                db.session.scalar(select(func.count(DiagnosisAIPrediction.id))) or 0
            ),
            # Reports are available one-to-one with persisted diagnoses.
            report_count=diagnosis_count,
        )

    def get_recent_diagnosis_activity(
        self,
        limit: int = 5,
    ) -> list[RecentDiagnosisActivity]:
        """Return recent diagnoses with patient and latest available model result."""
        statement = (
            select(Diagnosis)
            .options(joinedload(Diagnosis.patient), selectinload(Diagnosis.ai_predictions))
            .order_by(Diagnosis.created_at.desc())
            .limit(limit)
        )
        diagnoses = list(db.session.scalars(statement))
        return [
            RecentDiagnosisActivity(
                diagnosis=diagnosis,
                latest_prediction=max(
                    diagnosis.ai_predictions,
                    key=lambda prediction: prediction.updated_at,
                    default=None,
                ),
            )
            for diagnosis in diagnoses
        ]
