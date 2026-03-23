"""Data loading and augmentation pipeline for PlantVillage dataset."""

from __future__ import annotations

import tensorflow as tf
from omegaconf import DictConfig


def build_dataloaders(
    cfg: DictConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build train and validation tf.data pipelines from a directory of images.

    Args:
        cfg: OmegaConf config with `data`, `training`, and `model` sections.

    Returns:
        (train_ds, val_ds, class_names)
    """
    image_size = (cfg.model.image_size, cfg.model.image_size)
    batch_size = cfg.training.batch_size
    val_split = cfg.data.validation_split
    data_dir = cfg.data.train_dir

    shared_kwargs = dict(
        validation_split=val_split,
        seed=cfg.training.seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="training", **shared_kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="validation", **shared_kwargs
    )
    class_names: list[str] = train_ds.class_names

    augment = _build_augmentation(cfg.data.augmentation)

    def preprocess(images: tf.Tensor, labels: tf.Tensor):
        images = tf.cast(images, tf.float32)
        images = tf.keras.applications.efficientnet.preprocess_input(images)
        return images, labels

    train_ds = (
        train_ds
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names


def _build_augmentation(aug_cfg: DictConfig) -> tf.keras.Sequential:
    aug_layers = []
    if aug_cfg.get("horizontal_and_vertical_flip"):
        aug_layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
    if aug_cfg.get("rotation"):
        aug_layers.append(tf.keras.layers.RandomRotation(aug_cfg.rotation))
    if aug_cfg.get("zoom"):
        aug_layers.append(tf.keras.layers.RandomZoom(aug_cfg.zoom))
    if aug_cfg.get("brightness"):
        aug_layers.append(tf.keras.layers.RandomBrightness(aug_cfg.brightness))
    return tf.keras.Sequential(aug_layers, name="augmentation")
