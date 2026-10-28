"""NLP model loading and symptom-text prediction services."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


class NLPServiceError(RuntimeError):
    """Raised when the NLP artifacts cannot load or predict safely."""


NLP_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "nlp"
    / "clinical_disease_nlp_logistic_regression.joblib"
)
NLP_VECTORIZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "nlp"
    / "clinical_disease_tfidf_vectorizer.joblib"
)


@dataclass(frozen=True)
class NLPPrediction:
    """An NLP prediction and probabilities for every model class."""

    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def _load_nlp_model() -> Any:
    """Load the serialized Logistic Regression model once per process."""
    if not NLP_MODEL_PATH.is_file():
        raise NLPServiceError("The NLP model artifact is unavailable.")

    try:
        return joblib.load(NLP_MODEL_PATH)
    except Exception as error:
        raise NLPServiceError("The NLP model could not be loaded.") from error


@lru_cache(maxsize=1)
def _load_nlp_vectorizer() -> Any:
    """Load the serialized TF-IDF vectorizer once per process."""
    if not NLP_VECTORIZER_PATH.is_file():
        raise NLPServiceError("The NLP vectorizer artifact is unavailable.")

    try:
        return joblib.load(NLP_VECTORIZER_PATH)
    except Exception as error:
        raise NLPServiceError("The NLP vectorizer could not be loaded.") from error


class NLPService:
    """Predict a disease label from one symptom narrative."""

    def predict(self, text: str) -> NLPPrediction:
        """Vectorize raw symptom text and run the saved classifier."""
        if not isinstance(text, str) or not text.strip():
            raise NLPServiceError("Symptom text is required for NLP prediction.")

        try:
            vectorizer = _load_nlp_vectorizer()
            model = _load_nlp_model()
            text_vector = vectorizer.transform([text])
            predicted_label = str(model.predict(text_vector)[0])
            probabilities = model.predict_proba(text_vector)[0]
            model_classes = [str(value) for value in model.classes_]
        except NLPServiceError:
            raise
        except Exception as error:
            raise NLPServiceError(
                "The NLP prediction could not be completed."
            ) from error

        if predicted_label not in model_classes:
            raise NLPServiceError("The NLP model returned an unsupported class.")

        class_id = model_classes.index(predicted_label)
        probability_by_label = {
            label: float(probability)
            for label, probability in zip(model_classes, probabilities)
        }
        confidence = probability_by_label.get(predicted_label)
        if confidence is None:
            raise NLPServiceError(
                "The NLP model returned an invalid probability output."
            )

        return NLPPrediction(
            class_id=class_id,
            label=predicted_label,
            confidence=confidence,
            probabilities=probability_by_label,
        )