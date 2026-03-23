"""Training loop with two-phase transfer learning."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from src.models.classifier import build_model, unfreeze_top_layers

logger = logging.getLogger(__name__)


class TimingCallback(tf.keras.callbacks.Callback):
    """Logs epoch duration and ETA for the current phase."""

    def __init__(self, total_epochs: int, phase: str) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.phase = phase
        self._phase_start = 0.0
        self._epoch_start = 0.0
        self._epoch_times: list[float] = []

    def on_train_begin(self, logs=None) -> None:
        self._phase_start = time.time()
        logger.info("[%s] started", self.phase)

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None) -> None:
        elapsed = time.time() - self._epoch_start
        self._epoch_times.append(elapsed)
        avg = sum(self._epoch_times) / len(self._epoch_times)
        remaining = avg * (self.total_epochs - epoch - 1)
        val_acc = logs.get("val_accuracy", 0)
        logger.info(
            "[%s] epoch %d/%d — %.0fs — val_acc: %.4f — ETA: %s",
            self.phase,
            epoch + 1,
            self.total_epochs,
            elapsed,
            val_acc,
            _fmt_duration(remaining),
        )

    def on_train_end(self, logs=None) -> None:
        total = time.time() - self._phase_start
        logger.info("[%s] finished in %s", self.phase, _fmt_duration(total))


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "train.log"),
            logging.StreamHandler(),
        ],
    )


def train(
    cfg: DictConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> tf.keras.Model:
    """Two-phase training: head-only then fine-tuning.

    Args:
        cfg: Full OmegaConf config.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        class_names: List of class names from the dataset.
        output_dir: Directory to save model and class names.

    Returns:
        Trained Keras model.
    """
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU detected: %s", [g.name for g in gpus])
    else:
        logger.warning("No GPU detected — training on CPU (will be slow)")

    training_start = time.time()

    model = build_model(
        num_classes=len(class_names),
        image_size=cfg.model.image_size,
        dropout=cfg.model.dropout,
    )

    p1 = cfg.training.phase1
    phase1_ckpt = output_dir / "best_phase1.h5"
    logger.info("Phase 1 — training head only (%d epochs)", p1.epochs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(p1.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=p1.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=p1.early_stopping_patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(str(phase1_ckpt), save_best_only=True),
            TimingCallback(p1.epochs, "Phase 1"),
        ],
    )

    p2 = cfg.training.phase2
    output_path = output_dir / cfg.outputs.model_filename
    logger.info(
        "Phase 2 — fine-tuning last %d layers (%d epochs)",
        p2.unfreeze_last_n_layers,
        p2.epochs,
    )
    unfreeze_top_layers(model, n=p2.unfreeze_last_n_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(p2.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=p2.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=p2.early_stopping_patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(str(output_path), save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=p2.reduce_lr_factor, patience=p2.reduce_lr_patience
            ),
            TimingCallback(p2.epochs, "Phase 2"),
        ],
    )

    classes_path = output_dir / cfg.outputs.model_filename.replace(".h5", "_classes.txt")
    classes_path.write_text("\n".join(class_names))

    total_time = time.time() - training_start
    best_val_acc = max(history2.history.get("val_accuracy", [0]))
    logger.info("=" * 50)
    logger.info("Training complete in %s", _fmt_duration(total_time))
    logger.info("Best val accuracy: %.4f (%.1f%%)", best_val_acc, best_val_acc * 100)
    logger.info("Model saved to %s", output_path)
    logger.info("Class names saved to %s", classes_path)
    logger.info("=" * 50)

    return model
