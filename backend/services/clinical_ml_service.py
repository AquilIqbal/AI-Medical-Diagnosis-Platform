"""Clinical tabular model loading and prediction services."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


class ClinicalMLServiceError(RuntimeError):
    """Raised when the clinical model cannot load or predict safely."""


CLINICAL_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "ml"
    / "clinical_disease_random_forest.joblib"
)

CLINICAL_FEATURES = (
    "age_level",
    "gender",
    "bmi_level",
    "smoking",
    "age",
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

CLINICAL_CLASS_NAMES = {0: "Normal", 1: "Abnormal"}


@dataclass(frozen=True)
class ClinicalPrediction:
    """A clinical model prediction and its class probabilities."""

    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def _load_clinical_model() -> Any:
    """Load the serialized pipeline once per process."""
    if not CLINICAL_MODEL_PATH.is_file():
        raise ClinicalMLServiceError(
            "The clinical model artifact is unavailable."
        )

    try:
        return joblib.load(CLINICAL_MODEL_PATH)
    except Exception as error:
        raise ClinicalMLServiceError(
            "The clinical model could not be loaded."
        ) from error


class ClinicalMLService:
    """Predict the binary clinical status from notebook-compatible features."""

    def predict(self, features: Mapping[str, Any]) -> ClinicalPrediction:
        """Run the saved pipeline against one validated feature record."""
        missing_features = [
            feature for feature in CLINICAL_FEATURES if feature not in features
        ]
        if missing_features:
            missing = ", ".join(missing_features)
            raise ClinicalMLServiceError(
                f"Clinical prediction is missing required features: {missing}."
            )

        feature_frame = pd.DataFrame(
            [[features[feature] for feature in CLINICAL_FEATURES]],
            columns=CLINICAL_FEATURES,
        )

        try:
            model = _load_clinical_model()
            class_id = int(model.predict(feature_frame)[0])
            probabilities = model.predict_proba(feature_frame)[0]
            model_classes = [int(value) for value in model.classes_]
        except ClinicalMLServiceError:
            raise
        except Exception as error:
            raise ClinicalMLServiceError(
                "The clinical prediction could not be completed."
            ) from error

        label = CLINICAL_CLASS_NAMES.get(class_id)
        if label is None:
            raise ClinicalMLServiceError(
                "The clinical model returned an unsupported class."
            )

        probability_by_label = {
            CLINICAL_CLASS_NAMES[model_class]: float(probability)
            for model_class, probability in zip(model_classes, probabilities)
            if model_class in CLINICAL_CLASS_NAMES
        }
        confidence = probability_by_label.get(label)
        if confidence is None:
            raise ClinicalMLServiceError(
                "The clinical model returned an invalid probability output."
            )

        return ClinicalPrediction(
            class_id=class_id,
            label=label,
            confidence=confidence,
            probabilities=probability_by_label,
        )