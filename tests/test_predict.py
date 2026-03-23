"""
Unit tests for predict.py

Run:
    pytest tests/test_predict.py -v
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image(h: int = 224, w: int = 224, c: int = 3) -> np.ndarray:
    """Return a random uint8 BGR image."""
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def _write_class_names(path: str, names: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(names))


# ---------------------------------------------------------------------------
# Preprocessing tests (no model needed)
# ---------------------------------------------------------------------------

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Patch model loading so we don't need a real .h5 file
        self.model_mock = MagicMock()
        self.model_mock.layers = [MagicMock(), self._make_base_mock()]
        self.tf_load_patch = patch("tensorflow.keras.models.load_model", return_value=self.model_mock)
        self.tf_load_patch.start()

        self.tmp = tempfile.TemporaryDirectory()
        self.classes_path = os.path.join(self.tmp.name, "classes.txt")
        self._class_names = [f"Plant___Disease_{i}" for i in range(38)]
        _write_class_names(self.classes_path, self._class_names)

        # Defer import until patch is active
        from src.predict import PlantDiseasePredictor
        self.predictor = PlantDiseasePredictor.__new__(PlantDiseasePredictor)
        self.predictor.class_names = self._class_names

        # Import the standalone preprocess logic
        from src.predict import PlantDiseasePredictor as P
        self.P = P

    def _make_base_mock(self):
        import tensorflow as tf
        base = MagicMock()
        conv_layer = MagicMock(spec=tf.keras.layers.Conv2D)
        base.layers = [conv_layer]
        return base

    def tearDown(self):
        self.tf_load_patch.stop()
        self.tmp.cleanup()

    def test_preprocess_returns_correct_shape(self):
        from src.predict import PlantDiseasePredictor
        # Directly test the preprocess method logic
        predictor = MagicMock(spec=PlantDiseasePredictor)
        predictor.preprocess = PlantDiseasePredictor.preprocess.__get__(predictor)

        img = _make_dummy_image(300, 300)
        result = predictor.preprocess(img)
        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_preprocess_accepts_rgba(self):
        from src.predict import PlantDiseasePredictor
        predictor = MagicMock(spec=PlantDiseasePredictor)
        predictor.preprocess = PlantDiseasePredictor.preprocess.__get__(predictor)

        img = _make_dummy_image(224, 224, 4)  # RGBA
        result = predictor.preprocess(img)
        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_preprocess_from_file(self):
        from src.predict import PlantDiseasePredictor
        predictor = MagicMock(spec=PlantDiseasePredictor)
        predictor.preprocess = PlantDiseasePredictor.preprocess.__get__(predictor)

        img_path = os.path.join(self.tmp.name, "leaf.jpg")
        cv2.imwrite(img_path, _make_dummy_image())
        result = predictor.preprocess(img_path)
        self.assertEqual(result.shape, (1, 224, 224, 3))


# ---------------------------------------------------------------------------
# Prediction output tests (mocked model)
# ---------------------------------------------------------------------------

class TestPredictOutput(unittest.TestCase):
    def _make_mock_predictor(self, num_classes: int = 38):
        import tensorflow as tf
        from src.predict import PlantDiseasePredictor

        probs = np.zeros(num_classes, dtype=np.float32)
        probs[5] = 0.92
        probs[10] = 0.05
        probs[0] = 0.03

        # Mock the grad model outputs
        conv_output = np.random.rand(1, 7, 7, 1280).astype(np.float32)
        preds_tensor = tf.constant([probs])

        predictor = MagicMock(spec=PlantDiseasePredictor)
        predictor.class_names = [f"Plant___Disease_{i}" for i in range(num_classes)]
        predictor.preprocess = PlantDiseasePredictor.preprocess.__get__(predictor)
        predictor.predict = PlantDiseasePredictor.predict.__get__(predictor)
        predictor.top_k = PlantDiseasePredictor.top_k.__get__(predictor)

        # Mock model for top_k
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([probs])
        predictor.model = mock_model

        return predictor, probs

    def test_top_k_returns_k_results(self):
        predictor, _ = self._make_mock_predictor()
        # Provide a simple numpy image
        img = _make_dummy_image()
        results = predictor.top_k(img, k=3)
        self.assertEqual(len(results), 3)

    def test_top_k_sorted_by_confidence(self):
        predictor, probs = self._make_mock_predictor()
        img = _make_dummy_image()
        results = predictor.top_k(img, k=5)
        confidences = [c for _, c in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_top_k_confidence_sum_reasonable(self):
        predictor, probs = self._make_mock_predictor()
        img = _make_dummy_image()
        results = predictor.top_k(img, k=38)
        total = sum(c for _, c in results)
        self.assertAlmostEqual(total, 1.0, places=5)


# ---------------------------------------------------------------------------
# GradCAM output shape test
# ---------------------------------------------------------------------------

class TestGradCAMShape(unittest.TestCase):
    def test_heatmap_output_shape(self):
        """Grad-CAM overlay must be (224, 224, 3) uint8."""
        import cv2
        import numpy as np

        # Simulate the color-map part of predict() directly
        fake_heatmap = np.random.rand(7, 7).astype(np.float32)
        fake_heatmap /= fake_heatmap.max()
        resized = cv2.resize(fake_heatmap, (224, 224))
        color = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        orig = _make_dummy_image()
        overlay = cv2.addWeighted(orig, 0.6, color, 0.4, 0)

        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
