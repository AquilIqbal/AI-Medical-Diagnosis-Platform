"""Database model imports required for SQLAlchemy migration discovery."""

from backend.models.base import BaseModel
from backend.models.diagnosis import Diagnosis
from backend.models.diagnosis_ai_prediction import DiagnosisAIPrediction
from backend.models.diagnosis_clinical_assessment import DiagnosisClinicalAssessment
from backend.models.patient import Patient
from backend.models.user import User, UserRole


__all__ = [
    "BaseModel",
    "Diagnosis",
    "DiagnosisAIPrediction",
    "DiagnosisClinicalAssessment",
    "Patient",
    "User",
    "UserRole",
]
