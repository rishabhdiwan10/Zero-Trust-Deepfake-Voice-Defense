"""
scripts/train.py
=================
Training script for the CNN deepfake audio detector.

Usage::

    python scripts/train.py --config configs/model_config.yaml \\
                            --data-dir data/asvspoof2019 \\
                            --dataset asvspoof2019

Run ``python scripts/train.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import DatasetLoader, DatasetType, Split
from src.data.audio_preprocessor import AudioPreprocessor
from src.data.feature_extractor import FeatureExtractor, FeatureType
from src.models.cnn_detector import CNNDetector
from src.models.model_utils import save_checkpoint, get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.timer import timeit

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNN deepfake audio detector."
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to model config YAML (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="custom",
        choices=["asvspoof2019", "asvspoof5", "in_the_wild", "custom"],
        help="Dataset type (default: custom)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/",
        help="Directory to save checkpoints (default: models/)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu / cuda / mps). Auto-detected if not set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or get_device()

    logger.info("Training CNN detector on %s using device: %s", args.dataset, device)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    train_loader = DatasetLoader(
        dataset_type=DatasetType(args.dataset),
        root_dir=args.data_dir,
        split=Split.TRAIN,
    ).load()
    logger.info("Training set: %s", train_loader.class_distribution())

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    detector = CNNDetector(
        backbone=model_cfg.get("backbone", "resnet34"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        device=device,
    ).build()

    logger.info("Model built: %s backbone", model_cfg.get("backbone", "resnet34"))

    # ------------------------------------------------------------------
    # Training loop (placeholder — integrate PyTorch DataLoader here)
    # ------------------------------------------------------------------
    epochs = train_cfg.get("epochs", 50)
    logger.info("Starting training for %d epochs...", epochs)
    logger.info(
        "TODO: Implement full DataLoader + training loop. "
        "See notebooks/02_model_training.ipynb for reference."
    )

    # Placeholder: save a dummy checkpoint at epoch 0
    import torch
    import torch.optim as optim

    optimizer = optim.Adam(
        detector._model.parameters(),
        lr=train_cfg.get("learning_rate", 0.0001),
    )
    save_checkpoint(
        model=detector._model,
        optimizer=optimizer,
        epoch=0,
        metrics={"val_auc": 0.0},
        checkpoint_dir=args.output_dir,
        filename="checkpoint_epoch_0000.pt",
    )
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
