"""Enhanced PneumoniaMNIST CNN prediction and Grad-CAM services."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.filters import gaussian
from skimage.transform import resize


class ImagePredictionServiceError(RuntimeError):
    """Raised when the CNN artifact, preprocessing, or inference fails."""


class ImageValidationError(ImagePredictionServiceError):
    """Raised when an uploaded image is missing, unsupported, or unsafe."""


ENHANCED_CNN_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "image_pneumonia_cnn_enhanced_best.keras"
)
PREDICTION_TYPE = "cnn"
MODEL_NAME = "Enhanced Pneumonia CNN"
IMAGE_SIZE = (28, 28)
PNEUMONIA_THRESHOLD = 0.20
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}
ALLOWED_IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG", "WEBP"})
MAX_IMAGE_FILE_SIZE = 5 * 1024 * 1024


@dataclass(frozen=True)
class ImagePrediction:
    """CNN prediction details and a PNG Grad-CAM visualization."""

    predicted_class: int
    predicted_label: str
    confidence: float
    model_name: str = MODEL_NAME
    prediction_type: str = PREDICTION_TYPE
    visualization_png: bytes = field(repr=False, default=b"")


@lru_cache(maxsize=1)
def _load_enhanced_cnn_model() -> Any:
    """Load the trained enhanced CNN once per application process."""
    if not ENHANCED_CNN_MODEL_PATH.is_file():
        raise ImagePredictionServiceError("The enhanced CNN model is unavailable.")

    try:
        import tensorflow as tf

        return tf.keras.models.load_model(ENHANCED_CNN_MODEL_PATH)
    except Exception as error:
        raise ImagePredictionServiceError(
            "The enhanced CNN model could not be loaded."
        ) from error


class ImagePredictionService:
    """Adapt an uploaded image for the enhanced CNN and explain its result."""

    def predict(self, uploaded_file: Any) -> ImagePrediction:
        """Validate an uploaded image and return its CNN and Grad-CAM result."""
        image_input = self.prepare_image(uploaded_file)

        try:
            model = _load_enhanced_cnn_model()
            probability = float(model.predict(image_input, verbose=0).ravel()[0])
            predicted_class = int(probability >= PNEUMONIA_THRESHOLD)
            label = CLASS_LABELS[predicted_class]
            confidence = probability if predicted_class else 1.0 - probability
            last_conv_layer_name = self._last_convolutional_layer_name(model)
            visualization_png = self.generate_gradcam_overlay(
                image_input,
                model,
                last_conv_layer_name,
            )
        except ImagePredictionServiceError:
            raise
        except Exception as error:
            raise ImagePredictionServiceError(
                "The CNN prediction could not be completed."
            ) from error

        return ImagePrediction(
            predicted_class=predicted_class,
            predicted_label=label,
            confidence=confidence,
            visualization_png=visualization_png,
        )

    def prepare_image(self, uploaded_file: Any) -> np.ndarray:
        """Decode, adapt, and apply Notebook 03 Experiment 2 preprocessing."""
        image_bytes = self._read_and_validate_upload(uploaded_file)
        try:
            with Image.open(BytesIO(image_bytes)) as source_image:
                grayscale_image = source_image.convert("L")
                resized_image = grayscale_image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS,
                )
                image_array = np.asarray(resized_image, dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ImageValidationError("The uploaded file is not a valid image.") from error

        enhanced_image = self._enhance_image(image_array)
        return enhanced_image[np.newaxis, ..., np.newaxis].astype(np.float32)

    @staticmethod
    def _read_and_validate_upload(uploaded_file: Any) -> bytes:
        """Read a bounded upload and verify its format with Pillow."""
        if uploaded_file is None or not getattr(uploaded_file, "filename", ""):
            raise ImageValidationError("Please select a medical image to analyze.")

        try:
            uploaded_file.stream.seek(0)
            image_bytes = uploaded_file.stream.read(MAX_IMAGE_FILE_SIZE + 1)
            uploaded_file.stream.seek(0)
        except (AttributeError, OSError) as error:
            raise ImageValidationError("The uploaded image could not be read.") from error

        if not image_bytes:
            raise ImageValidationError("The uploaded image is empty.")
        if len(image_bytes) > MAX_IMAGE_FILE_SIZE:
            raise ImageValidationError("The uploaded image must be 5 MB or smaller.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ImageValidationError("The uploaded file is not a valid image.") from error

        if image_format not in ALLOWED_IMAGE_FORMATS:
            supported_formats = ", ".join(sorted(ALLOWED_IMAGE_FORMATS))
            raise ImageValidationError(
                f"Unsupported image format. Use one of: {supported_formats}."
            )

        return image_bytes

    @staticmethod
    def _enhance_image(image: np.ndarray) -> np.ndarray:
        """Apply the exact Experiment 2 enhancement sequence from Notebook 03."""
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        contrast_image = exposure.rescale_intensity(
            image,
            in_range="image",
            out_range=(0.0, 1.0),
        )
        clahe_image = exposure.equalize_adapthist(
            contrast_image,
            clip_limit=0.02,
        )
        denoised_image = gaussian_filter(clahe_image, sigma=0.5)
        return np.clip(denoised_image, 0.0, 1.0)

    @staticmethod
    def _last_convolutional_layer_name(model: Any) -> str:
        """Determine the final convolutional layer from the loaded artifact."""
        try:
            import tensorflow as tf

            convolutional_layers = [
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.layers.Conv2D)
            ]
        except Exception as error:
            raise ImagePredictionServiceError(
                "The enhanced CNN architecture could not be inspected."
            ) from error

        if not convolutional_layers:
            raise ImagePredictionServiceError(
                "The enhanced CNN does not contain a convolutional layer."
            )
        return convolutional_layers[-1].name

    def generate_gradcam_overlay(
        self,
        image_input: np.ndarray,
        model: Any,
        last_conv_layer_name: str,
    ) -> bytes:
        """Create Notebook 03's display-only enhanced Grad-CAM PNG overlay."""
        heatmap = self._make_gradcam_heatmap(
            image_input,
            model,
            last_conv_layer_name,
        )
        display_image, smooth_heatmap = self._create_gradcam_overlay(
            image_input[0],
            heatmap,
        )

        try:
            from matplotlib import colormaps

            heatmap_rgb = colormaps["jet"](smooth_heatmap)[..., :3]
            base_rgb = np.repeat(display_image[..., np.newaxis], 3, axis=-1)
            overlay = (0.60 * base_rgb) + (0.40 * heatmap_rgb)
            output = Image.fromarray(
                np.uint8(np.clip(overlay, 0.0, 1.0) * 255),
                mode="RGB",
            )
            buffer = BytesIO()
            output.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as error:
            raise ImagePredictionServiceError(
                "The Grad-CAM visualization could not be generated."
            ) from error

    @staticmethod
    def _make_gradcam_heatmap(
        image_input: np.ndarray,
        model: Any,
        last_conv_layer_name: str,
    ) -> np.ndarray:
        """Use Notebook 03's Keras 3-compatible Grad-CAM calculation."""
        try:
            import tensorflow as tf

            image_batch = tf.cast(image_input, tf.float32)
            inputs = tf.keras.Input(shape=tuple(image_batch.shape[1:]))
            output = inputs
            convolutional_output = None

            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                output = layer(output)
                if layer.name == last_conv_layer_name:
                    convolutional_output = output

            if convolutional_output is None:
                raise ImagePredictionServiceError(
                    "The CNN Grad-CAM layer could not be found."
                )

            grad_model = tf.keras.Model(
                inputs=inputs,
                outputs=[convolutional_output, output],
            )
            with tf.GradientTape() as tape:
                convolutional_outputs, predictions = grad_model(image_batch)
                pneumonia_probability = predictions[:, 0]
                if float(pneumonia_probability.numpy()[0]) >= PNEUMONIA_THRESHOLD:
                    target_score = pneumonia_probability
                else:
                    target_score = 1.0 - pneumonia_probability

            gradients = tape.gradient(target_score, convolutional_outputs)
            if gradients is None:
                raise ImagePredictionServiceError(
                    "Grad-CAM gradients could not be calculated."
                )

            pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))[0]
            heatmap = tf.reduce_sum(
                convolutional_outputs[0] * pooled_gradients,
                axis=-1,
            )
            heatmap = tf.maximum(heatmap, 0)
            max_value = tf.reduce_max(heatmap)
            heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
            return heatmap.numpy()
        except ImagePredictionServiceError:
            raise
        except Exception as error:
            raise ImagePredictionServiceError(
                "The Grad-CAM heatmap could not be generated."
            ) from error

    @staticmethod
    def _create_gradcam_overlay(
        image: np.ndarray,
        heatmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Notebook 03's display-only heatmap enhancement."""
        base_image = np.squeeze(image).astype(np.float32)
        base_min, base_max = base_image.min(), base_image.max()
        if base_max > base_min:
            base_image = (base_image - base_min) / (base_max - base_min)
        else:
            base_image = np.zeros_like(base_image)

        resized_heatmap = resize(
            heatmap,
            base_image.shape,
            order=3,
            mode="reflect",
            anti_aliasing=True,
        )
        smooth_heatmap = gaussian(resized_heatmap, sigma=1.0)
        heatmap_min, heatmap_max = smooth_heatmap.min(), smooth_heatmap.max()
        if heatmap_max > heatmap_min:
            smooth_heatmap = (smooth_heatmap - heatmap_min) / (
                heatmap_max - heatmap_min
            )
        else:
            smooth_heatmap = np.zeros_like(smooth_heatmap)

        return base_image, smooth_heatmap
