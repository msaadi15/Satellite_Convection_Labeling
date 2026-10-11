"""U-Net loading and inference (TensorFlow/Keras imported lazily)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import postprocess_mask, preprocess_image


def register_custom_objects() -> dict:
    """Losses/metrics the checkpoint was compiled with (needed to deserialize it)."""
    import keras
    from keras import ops
    from keras.losses import binary_crossentropy

    @keras.saving.register_keras_serializable()
    def dice_coef(y_true, y_pred, smooth=1):
        inter = ops.sum(y_true * y_pred, axis=[1, 2, 3])
        union = ops.sum(y_true, axis=[1, 2, 3]) + ops.sum(y_pred, axis=[1, 2, 3])
        return ops.mean((2.0 * inter + smooth) / (union + smooth), axis=0)

    @keras.saving.register_keras_serializable()
    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    @keras.saving.register_keras_serializable()
    def combined_loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return {"dice_coef": dice_coef, "dice_loss": dice_loss, "combined_loss": combined_loss}


class Segmenter:
    """Wraps the trained U-Net: RGB convection image in, binary mask out."""

    def __init__(self, model, input_size=(160, 160), threshold=0.5):
        self.model = model
        self.input_size = tuple(input_size)
        self.threshold = threshold

    @classmethod
    def from_config(cls, cfg: dict) -> Segmenter:
        import keras

        path = Path(cfg["model"]["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. See README ('Model weights')."
            )
        model = keras.models.load_model(
            path, custom_objects=register_custom_objects(), compile=False
        )
        return cls(model, cfg["model"]["input_size"], cfg["model"]["threshold"])

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """Probability map at the model resolution, shape (H, W)."""
        x = preprocess_image(image, self.input_size)[None]
        return np.asarray(self.model.predict(x, verbose=0))[0, ..., 0]

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary uint8 mask 0/255 at original size, probability map)."""
        proba = self.predict_proba(image)
        return postprocess_mask(proba, image.size, self.threshold), proba
