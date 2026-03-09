"""
scripts/train.py
=================
Training script for the CNN deepfake audio detector.

Usage::

    python scripts/train.py --config configs/model_config.yaml \\
                            --data-dir data/asvspoof2019 \\
                            --dataset asvspoof2019

    # With a separate validation directory:
    python scripts/train.py --config configs/model_config.yaml \\
                            --data-dir data/asvspoof2019/train \\
                            --val-dir data/asvspoof2019/dev \\
                            --dataset asvspoof2019

Run ``python scripts/train.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import DatasetLoader, DatasetType, Split
from src.data.audio_preprocessor import AudioPreprocessor
from src.data.feature_extractor import FeatureExtractor, FeatureType
from src.models.cnn_detector import CNNDetector
from src.models.model_utils import save_checkpoint, get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class DeepfakeAudioDataset:
    """
    PyTorch-compatible Dataset that wraps DatasetLoader samples through
    AudioPreprocessor and FeatureExtractor to produce (feature_tensor, label)
    pairs.

    Parameters
    ----------
    samples : list
        List of AudioSample objects from DatasetLoader.
    preprocessor : AudioPreprocessor
        Audio preprocessor instance.
    feature_extractor : FeatureExtractor
        Feature extractor instance.
    cache_features : bool
        If ``True``, extract all features once and cache in memory.
        Use only when the dataset fits in RAM.
    """

    def __init__(
        self,
        samples: list,
        preprocessor: AudioPreprocessor,
        feature_extractor: FeatureExtractor,
        cache_features: bool = False,
    ) -> None:
        import torch
        from torch.utils.data import Dataset as TorchDataset

        self._samples = samples
        self._preprocessor = preprocessor
        self._feature_extractor = feature_extractor
        self._cache: dict = {}
        self._torch = torch

        if cache_features:
            logger.info("Pre-caching %d samples …", len(samples))
            for idx in range(len(samples)):
                result = self._load_sample(idx)
                if result is not None:
                    self._cache[idx] = result
            logger.info("Cached %d / %d samples.", len(self._cache), len(samples))

    # torch.utils.data.Dataset protocol
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        result = self._load_sample(idx)
        if result is None:
            # Return a zero tensor as a safe fallback for corrupted files
            dummy = self._torch.zeros(1, 128, 128)
            return dummy, -1
        return result

    def _load_sample(self, idx: int):
        """Load, preprocess, and extract features for sample at *idx*."""
        import soundfile as sf

        sample = self._samples[idx]
        try:
            waveform, sr = sf.read(
                str(sample.file_path), dtype="float32", always_2d=False
            )
            waveform, sr = self._preprocessor.process(waveform, sr)
            features = self._feature_extractor.extract(waveform)  # (C, H, W)
            tensor = self._torch.from_numpy(features).float()
            label = self._torch.tensor(sample.label, dtype=self._torch.long)
            return tensor, label
        except Exception as exc:
            warnings.warn(
                f"Skipping corrupted file {sample.file_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


def _collate_skip_none(batch):
    """Collate function that silently drops None / invalid samples."""
    import torch
    valid = [(f, l) for f, l in batch if l is not None and hasattr(l, 'item') and l.item() >= 0]
    if not valid:
        return None, None
    features = torch.stack([f for f, _ in valid])
    labels = torch.stack([l for _, l in valid])
    return features, labels


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    all_labels: List[int],
    all_probs: List[float],
    all_preds: List[int],
) -> Tuple[float, float]:
    """Return (accuracy, AUC) from accumulated predictions."""
    from sklearn.metrics import roc_auc_score

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return accuracy, auc


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

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
        help="Root directory of the training dataset.",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help=(
            "Root directory of the validation dataset. "
            "If omitted, an 80/20 split of --data-dir is used."
        ),
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


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset

    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or get_device()

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpointing", {})

    # ── Reproducibility ──────────────────────────────────────────────────
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    logger.info("Training CNN detector on %s using device: %s", args.dataset, device)

    # ── Preprocessing & feature extraction ───────────────────────────────
    preprocessor = AudioPreprocessor(
        target_sr=16_000,
        normalize=True,
        trim_silence=True,
    )
    feature_type = FeatureType(model_cfg.get("input_feature", "mel_spectrogram"))
    feature_extractor = FeatureExtractor(feature_type=feature_type)

    # ── Data loading ──────────────────────────────────────────────────────
    full_loader = DatasetLoader(
        dataset_type=DatasetType(args.dataset),
        root_dir=args.data_dir,
        split=Split.TRAIN,
    ).load()
    logger.info("Full training set: %s", full_loader.class_distribution())

    all_samples = full_loader.samples  # list of AudioSample

    if args.val_dir:
        # Separate validation directory provided
        val_loader_ds = DatasetLoader(
            dataset_type=DatasetType(args.dataset),
            root_dir=args.val_dir,
            split=Split.DEV,
        ).load()
        val_samples = val_loader_ds.samples
        train_samples = all_samples
        logger.info("Validation set (separate dir): %d samples", len(val_samples))
    else:
        # 80/20 random split
        val_split = train_cfg.get("val_split", 0.2)
        indices = list(range(len(all_samples)))
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]
        logger.info(
            "Split: %d train / %d val (%.0f%% held out)",
            len(train_samples),
            len(val_samples),
            val_split * 100,
        )

    train_dataset = DeepfakeAudioDataset(
        samples=train_samples,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
    )
    val_dataset = DeepfakeAudioDataset(
        samples=val_samples,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
    )

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = min(4, os.cpu_count() or 1)
    pin_mem = device == "cuda"

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=_collate_skip_none,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=_collate_skip_none,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    detector = CNNDetector(
        backbone=model_cfg.get("backbone", "resnet34"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        device=device,
    ).build()
    model = detector._model
    logger.info("Model built: %s backbone", model_cfg.get("backbone", "resnet34"))

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    epochs = train_cfg.get("epochs", 50)
    scheduler_name = train_cfg.get("lr_scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("lr_step_size", 10),
            gamma=train_cfg.get("lr_gamma", 0.5),
        )
    else:
        scheduler = None

    # ── Mixed-precision scaler ────────────────────────────────────────────
    use_amp = train_cfg.get("mixed_precision", True) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    patience = train_cfg.get("early_stopping_patience", 10)
    save_every = ckpt_cfg.get("save_every", 5)
    checkpoint_dir = args.output_dir

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_path: Optional[Path] = None

    logger.info("Starting training for %d epochs…", epochs)

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # ── Training phase ────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for features, labels in train_dl:
            if features is None:
                continue
            # 3-channel input expected by ImageNet backbones
            if features.shape[1] == 1:
                features = features.repeat(1, 3, 1, 1)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # ── Validation phase ───────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_labels: List[int] = []
        all_probs: List[float] = []
        all_preds: List[int] = []

        with torch.no_grad():
            for features, labels in val_dl:
                if features is None:
                    continue
                if features.shape[1] == 1:
                    features = features.repeat(1, 3, 1, 1)
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(features)
                    loss = criterion(logits, labels)

                import torch.nn.functional as F
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1).tolist()
                synthetic_probs = probs[:, 1].tolist()  # P(synthetic)

                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(synthetic_probs)
                all_preds.extend(preds)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_accuracy, val_auc = _compute_metrics(all_labels, all_probs, all_preds)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.4f | val_auc=%.4f",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_accuracy,
            val_auc,
        )

        if scheduler is not None:
            scheduler.step()

        # ── Checkpoint (periodic) ─────────────────────────────────────────
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_auc": val_auc,
                },
                checkpoint_dir=checkpoint_dir,
            )

        # ── Best model & early stopping ────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_ckpt_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_auc": val_auc,
                },
                checkpoint_dir=checkpoint_dir,
                filename="best_checkpoint.pt",
            )
            logger.info("New best model saved → %s", best_ckpt_path)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    epochs_no_improve,
                )
                break

    logger.info(
        "Training complete. Best val_loss=%.4f (checkpoint: %s)",
        best_val_loss,
        best_ckpt_path,
    )


if __name__ == "__main__":
    main()
