"""Tests for enhanced CNN image prediction and Grad-CAM services."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from backend.services.image_prediction_service import (
    IMAGE_SIZE,
    MAX_IMAGE_FILE_SIZE,
    PNEUMONIA_THRESHOLD,
    ImagePredictionService,
    ImagePredictionServiceError,
    ImageValidationError,
    _load_enhanced_cnn_model,
)


def _upload_image(
    *,
    mode: str = "RGB",
    image_format: str = "PNG",
    filename: str = "chest.png",
) -> FileStorage:
    image = Image.new(mode, (52, 36), color=128)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return FileStorage(stream=buffer, filename=filename)


def test_prepare_image_adapts_arbitrary_image_to_enhanced_cnn_input() -> None:
    """An uploaded RGB image becomes one 28x28 enhanced grayscale input."""
    prepared = ImagePredictionService().prepare_image(_upload_image())

    assert prepared.shape == (1, IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
    assert prepared.dtype == np.float32
    assert float(prepared.min()) >= 0.0
    assert float(prepared.max()) <= 1.0


@pytest.mark.parametrize(
    ("probability", "expected_class", "expected_label", "expected_confidence"),
    [
        (PNEUMONIA_THRESHOLD - 0.01, 0, "Normal", 0.81),
        (PNEUMONIA_THRESHOLD, 1, "Pneumonia", PNEUMONIA_THRESHOLD),
    ],
)
def test_predict_uses_notebook_class_mapping_and_threshold(
    probability: float,
    expected_class: int,
    expected_label: str,
    expected_confidence: float,
) -> None:
    """The selected class changes precisely at Notebook 03's 0.20 threshold."""
    model = type(
        "Model",
        (),
        {"predict": lambda self, values, verbose=0: np.array([[probability]])},
    )()
    service = ImagePredictionService()

    with patch(
        "backend.services.image_prediction_service._load_enhanced_cnn_model",
        return_value=model,
    ), patch.object(
        service,
        "_last_convolutional_layer_name",
        return_value="conv2d_5",
    ), patch.object(
        service,
        "generate_gradcam_overlay",
        return_value=b"png",
    ):
        result = service.predict(_upload_image())

    assert result.predicted_class == expected_class
    assert result.predicted_label == expected_label
    assert result.confidence == pytest.approx(expected_confidence)
    assert result.prediction_type == "cnn"
    assert result.model_name == "Enhanced Pneumonia CNN"


def test_prepare_image_rejects_non_images() -> None:
    """File extensions and MIME declarations cannot bypass image validation."""
    upload = FileStorage(stream=BytesIO(b"not an image"), filename="scan.png")

    with pytest.raises(ImageValidationError, match="not a valid image"):
        ImagePredictionService().prepare_image(upload)


def test_prepare_image_rejects_oversized_upload() -> None:
    """Image uploads are bounded before decoding or model loading."""
    upload = FileStorage(
        stream=BytesIO(b"0" * (MAX_IMAGE_FILE_SIZE + 1)),
        filename="scan.png",
    )

    with pytest.raises(ImageValidationError, match="5 MB or smaller"):
        ImagePredictionService().prepare_image(upload)


def test_model_loader_reports_missing_artifact() -> None:
    """A missing CNN artifact produces a safe service error."""
    with patch(
        "backend.services.image_prediction_service.ENHANCED_CNN_MODEL_PATH",
        Path("missing-enhanced-cnn.keras"),
    ):
        _load_enhanced_cnn_model.cache_clear()
        with pytest.raises(ImagePredictionServiceError, match="unavailable"):
            _load_enhanced_cnn_model()
        _load_enhanced_cnn_model.cache_clear()


def test_predict_wraps_model_failure() -> None:
    """CNN loading or inference failures are not exposed as raw exceptions."""
    service = ImagePredictionService()

    with patch(
        "backend.services.image_prediction_service._load_enhanced_cnn_model",
        side_effect=ImagePredictionServiceError("model unavailable"),
    ):
        with pytest.raises(ImagePredictionServiceError, match="model unavailable"):
            service.predict(_upload_image())


def test_real_enhanced_cnn_prediction_generates_gradcam() -> None:
    """The real trained artifact accepts an adapted image and returns a PNG."""
    service = ImagePredictionService()
    model = _load_enhanced_cnn_model()
    assert service._last_convolutional_layer_name(model) == "conv2d_5"

    result = service.predict(_upload_image(mode="L"))

    assert result.predicted_label in {"Normal", "Pneumonia"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.visualization_png.startswith(b"\x89PNG\r\n\x1a\n")
