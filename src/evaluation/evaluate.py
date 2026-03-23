"""Model evaluation with full metrics report and confusion matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
    figures_dir: Path | None = None,
) -> dict:
    """Run full evaluation on a dataset.

    Reports accuracy, precision, recall, F1-score per class, and plots
    the confusion matrix when *figures_dir* is provided.

    Args:
        model: Trained Keras model.
        dataset: tf.data.Dataset yielding (images, one-hot labels).
        class_names: Ordered list of class names.
        figures_dir: Optional directory to save confusion matrix plot.

    Returns:
        Dict with keys ``preds``, ``targets``, and ``report``.
    """
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in dataset:
        outputs = model(images, training=False)
        preds = tf.argmax(outputs, axis=1).numpy()
        targets = tf.argmax(labels, axis=1).numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

    labels = list(range(len(class_names)))
    report = classification_report(
        all_targets, all_preds, labels=labels, target_names=class_names, output_dict=True
    )
    logger.info(
        "\n%s",
        classification_report(all_targets, all_preds, labels=labels, target_names=class_names),
    )

    if figures_dir is not None:
        _plot_confusion_matrix(all_targets, all_preds, class_names, Path(figures_dir))

    return {"preds": all_preds, "targets": all_targets, "report": report}


def _plot_confusion_matrix(
    targets: list[int],
    preds: list[int],
    class_names: list[str],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cmap="Blues",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    out = figures_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out)
