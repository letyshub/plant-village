"""Architecture tests for PlantDiseaseClassifier (TensorFlow/Keras)."""

import numpy as np
import pytest
import tensorflow as tf

from src.models.classifier import build_model, unfreeze_top_layers


NUM_CLASSES = 38
IMAGE_SIZE = 224


@pytest.fixture(scope="module")
def model() -> tf.keras.Model:
    return build_model(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)


def test_model_output_shape(model: tf.keras.Model) -> None:
    dummy = np.zeros((4, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    output = model(dummy, training=False)
    assert output.shape == (4, NUM_CLASSES), f"Expected (4, {NUM_CLASSES}), got {output.shape}"


def test_model_no_nan_output(model: tf.keras.Model) -> None:
    dummy = np.random.rand(2, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    output = model(dummy, training=False)
    assert not tf.reduce_any(tf.math.is_nan(output)).numpy(), "Model output contains NaN"


def test_model_output_is_probability_distribution(model: tf.keras.Model) -> None:
    dummy = np.random.rand(3, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    output = model(dummy, training=False).numpy()
    sums = output.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)


def test_base_frozen_by_default(model: tf.keras.Model) -> None:
    base = model.layers[1]
    assert not base.trainable, "EfficientNetB0 backbone should be frozen by default"


def test_unfreeze_top_layers() -> None:
    m = build_model(num_classes=NUM_CLASSES)
    unfreeze_top_layers(m, n=20)
    base = m.layers[1]
    assert base.trainable, "Backbone should be trainable after unfreeze_top_layers()"
    trainable_count = sum(1 for l in base.layers if l.trainable)
    assert trainable_count > 0, "At least some backbone layers should be trainable after unfreezing"
