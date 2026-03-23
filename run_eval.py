"""Run evaluation on a saved model without retraining."""

from pathlib import Path

import tensorflow as tf
from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders
from src.evaluation.evaluate import evaluate
from src.training.trainer import setup_logging

cfg = OmegaConf.load("configs/train.yaml")
setup_logging(cfg.outputs.log_dir)

class_names = Path("outputs/models/plant_disease_model_classes.txt").read_text().splitlines()
model = tf.keras.models.load_model("outputs/models/plant_disease_model.h5")

_, val_ds, _ = build_dataloaders(cfg)

evaluate(model, val_ds, class_names, figures_dir=Path("outputs/figures"))
