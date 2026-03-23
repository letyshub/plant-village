"""EfficientNetB0 classifier for plant disease detection."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_model(num_classes: int, image_size: int = 224, dropout: float = 0.3) -> tf.keras.Model:
    """Build EfficientNetB0 transfer learning model.

    Args:
        num_classes: Number of output classes.
        image_size: Input image size (square).
        dropout: Dropout rate before the output layer.

    Returns:
        Compiled-ready Keras model with frozen base.
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


def unfreeze_top_layers(model: tf.keras.Model, n: int = 20) -> None:
    """Unfreeze the last *n* layers of the EfficientNetB0 backbone for fine-tuning.

    Args:
        model: The full classifier model (backbone is model.layers[1]).
        n: Number of layers from the end of the backbone to unfreeze.
    """
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-n]:
        layer.trainable = False
