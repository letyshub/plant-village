"""
Prediction pipeline with Grad-CAM visualization.

Usage:
    from predict import PlantDiseasePredictor
    predictor = PlantDiseasePredictor("model/plant_disease_model.h5", "model/plant_disease_model_classes.txt")
    label, confidence, heatmap = predictor.predict("leaf.jpg")
"""

from __future__ import annotations

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)


class PlantDiseasePredictor:
    def __init__(self, model_path: str, class_names_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(class_names_path) as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        # Grad-CAM: target the last conv layer of EfficientNetB0
        self._grad_model = self._build_grad_model()

    def _build_grad_model(self) -> tf.keras.Model:
        base = self.model.layers[1]  # EfficientNetB0
        last_conv = next(
            l for l in reversed(base.layers) if isinstance(l, tf.keras.layers.Conv2D)
        )
        # In Keras 3 we must build from the backbone's own inputs/outputs
        return tf.keras.Model(
            inputs=base.inputs,
            outputs=[last_conv.output, base.output],
        )

    def preprocess(self, image: np.ndarray | str) -> np.ndarray:
        """Accept file path or numpy array (H×W×3, uint8)."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB").resize(IMAGE_SIZE)
            image = np.array(img)
        else:
            image = cv2.resize(image, IMAGE_SIZE)
            if image.shape[2] == 4:
                image = image[:, :, :3]
        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )
        return np.expand_dims(image, 0)

    def predict(self, image: np.ndarray | str) -> tuple[str, float, np.ndarray]:
        """
        Returns:
            label       — predicted class name
            confidence  — probability in [0, 1]
            heatmap     — uint8 BGR Grad-CAM overlay (224×224×3)
        """
        tensor = self.preprocess(image)
        with tf.GradientTape() as tape:
            conv_outputs, backbone_out = self._grad_model(tensor, training=False)
            tape.watch(conv_outputs)
            # Apply layers after the backbone (GAP, Dropout, Dense)
            x = backbone_out
            for layer in self.model.layers[2:]:
                x = layer(x, training=False)
            predictions = x
            class_idx = tf.argmax(predictions[0])
            score = predictions[:, class_idx]

        grads = tape.gradient(score, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_map = conv_outputs[0].numpy()
        for i, w in enumerate(pooled_grads):
            conv_map[:, :, i] *= w.numpy()

        heatmap = np.mean(conv_map, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Reconstruct original image for overlay
        if isinstance(image, str):
            orig = np.array(Image.open(image).convert("RGB").resize(IMAGE_SIZE))
            orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        else:
            orig_bgr = cv2.resize(image, IMAGE_SIZE)
            if orig_bgr.shape[2] == 4:
                orig_bgr = orig_bgr[:, :, :3]

        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        label = self.class_names[int(class_idx)]
        confidence = float(predictions[0][class_idx])
        return label, confidence, overlay

    def top_k(self, image: np.ndarray | str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (label, confidence) pairs."""
        tensor = self.preprocess(image)
        preds = self.model.predict(tensor, verbose=0)[0]
        indices = np.argsort(preds)[::-1][:k]
        return [(self.class_names[i], float(preds[i])) for i in indices]
