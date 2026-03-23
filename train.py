"""
Plant disease classification — EfficientNetB0 transfer learning on PlantVillage.

Usage:
    python train.py
    python train.py configs/train.yaml
    python train.py configs/train.yaml training.phase1.epochs=5
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders
from src.evaluation.evaluate import evaluate
from src.training.trainer import set_seed, setup_logging, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/train.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dot-notation, e.g. training.phase1.epochs=5",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    output_dir = Path(cfg.outputs.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.outputs.log_dir)
    set_seed(cfg.training.seed)

    train_ds, val_ds, class_names = build_dataloaders(cfg)
    model = train(cfg, train_ds, val_ds, class_names, output_dir)

    evaluate(
        model,
        val_ds,
        class_names,
        figures_dir=Path("outputs/figures"),
    )


if __name__ == "__main__":
    main()
