"""Read-only metadata describing the application's deployed model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.services.clinical_ml_service import CLINICAL_MODEL_PATH
from backend.services.image_prediction_service import (
    ENHANCED_CNN_MODEL_PATH,
    IMAGE_SIZE,
    PNEUMONIA_THRESHOLD,
)
from backend.services.nlp_service import NLP_MODEL_PATH, NLP_VECTORIZER_PATH


@dataclass(frozen=True)
class AIModelCatalogEntry:
    """Verified static metadata for one implemented AI modality."""

    title: str
    model_name: str
    artifacts: tuple[str, ...]
    artifact_paths: tuple[Path, ...]
    input_description: str
    output_description: str
    details: tuple[tuple[str, str], ...]

    @property
    def is_available(self) -> bool:
        return all(path.is_file() for path in self.artifact_paths)


class AIModelCatalogService:
    """Expose deployed-model details without loading or invoking any model."""

    def get_models(self) -> tuple[AIModelCatalogEntry, ...]:
        return (
            AIModelCatalogEntry(
                title="Clinical ML",
                model_name="Random Forest",
                artifacts=("models/ml/clinical_disease_random_forest.joblib",),
                artifact_paths=(CLINICAL_MODEL_PATH,),
                input_description="30 clinical features",
                output_description="Normal / Abnormal",
                details=(("Preprocessing", "Included in the serialized pipeline"),),
            ),
            AIModelCatalogEntry(
                title="NLP",
                model_name="TF-IDF + Logistic Regression",
                artifacts=(
                    "models/nlp/clinical_disease_tfidf_vectorizer.joblib",
                    "models/nlp/clinical_disease_nlp_logistic_regression.joblib",
                ),
                artifact_paths=(NLP_VECTORIZER_PATH, NLP_MODEL_PATH),
                input_description="Symptom or clinical narrative",
                output_description="Disease classification",
                details=(),
            ),
            AIModelCatalogEntry(
                title="CNN Image Prediction",
                model_name="Enhanced Pneumonia CNN",
                artifacts=("models/image_pneumonia_cnn_enhanced_best.keras",),
                artifact_paths=(ENHANCED_CNN_MODEL_PATH,),
                input_description=f"Uploaded image adapted to {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} grayscale",
                output_description="Normal / Pneumonia",
                details=(
                    ("Threshold", f"{PNEUMONIA_THRESHOLD:.2f}"),
                    ("Explainability", "Grad-CAM"),
                    ("Final convolutional layer", "conv2d_5"),
                ),
            ),
        )
