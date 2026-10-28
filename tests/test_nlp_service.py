"""Tests for the isolated NLP prediction service."""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services.nlp_service import (
    NLPService,
    NLPServiceError,
    NLPPrediction,
    _load_nlp_model,
)


def test_predict_vectorizes_raw_text_and_maps_probabilities() -> None:
    """The service sends raw text to TF-IDF and maps classifier output."""
    vectorizer = type(
        "Vectorizer",
        (),
        {"transform": lambda self, texts: texts},
    )()
    model = type(
        "Model",
        (),
        {
            "predict": lambda self, values: ["Pneumonia"],
            "predict_proba": lambda self, values: [[0.25, 0.75]],
            "classes_": ["Common Cold", "Pneumonia"],
        },
    )()

    with patch(
        "backend.services.nlp_service._load_nlp_vectorizer",
        return_value=vectorizer,
    ) as vectorizer_loader, patch(
        "backend.services.nlp_service._load_nlp_model",
        return_value=model,
    ):
        result = NLPService().predict("persistent cough and fever")

    assert isinstance(result, NLPPrediction)
    assert result.class_id == 1
    assert result.label == "Pneumonia"
    assert result.confidence == pytest.approx(0.75)
    assert result.probabilities == {"Common Cold": 0.25, "Pneumonia": 0.75}
    vectorizer_loader.assert_called_once_with()


@pytest.mark.parametrize("invalid_text", ["", "   ", None, 123])
def test_predict_rejects_empty_or_invalid_text(invalid_text: object) -> None:
    """Invalid input fails before loading either artifact."""
    with patch("backend.services.nlp_service._load_nlp_model") as model_loader:
        with pytest.raises(NLPServiceError, match="Symptom text is required"):
            NLPService().predict(invalid_text)  # type: ignore[arg-type]

    model_loader.assert_not_called()


def test_model_loader_reports_missing_artifact() -> None:
    """An unavailable model artifact produces a safe service error."""
    with patch(
        "backend.services.nlp_service.NLP_MODEL_PATH",
        Path("missing-nlp-model.joblib"),
    ):
        _load_nlp_model.cache_clear()
        with pytest.raises(NLPServiceError, match="model artifact is unavailable"):
            _load_nlp_model()
        _load_nlp_model.cache_clear()