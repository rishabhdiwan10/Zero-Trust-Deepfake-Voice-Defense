"""
scripts/evaluate.py
====================
Evaluation script for the CNN deepfake audio detector.

Computes:
  - Accuracy
  - Equal Error Rate (EER)
  - Confusion matrix
  - Per-class precision / recall / F1

Usage::

    python scripts/evaluate.py --checkpoint models/checkpoint_epoch_0050.pt \\
                                --data-dir data/asvspoof2019 \\
                                --dataset asvspoof2019
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import DatasetLoader, DatasetType, Split
from src.data.audio_preprocessor import AudioPreprocessor
from src.data.feature_extractor import FeatureExtractor, FeatureType
from src.models.cnn_detector import CNNDetector
from src.models.model_utils import load_checkpoint, get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN deepfake detector.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--dataset",
        default="custom",
        choices=["asvspoof2019", "asvspoof5", "in_the_wild", "custom"],
    )
    parser.add_argument(
        "--split",
        default="eval",
        choices=["train", "dev", "eval"],
    )
    parser.add_argument("--output-json", default=None, help="Save metrics to JSON.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def compute_eer(y_true: list, y_scores: list) -> float:
    """
    Compute the Equal Error Rate (EER).

    EER is the point where False Accept Rate equals False Reject Rate.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels (0=genuine, 1=synthetic).
    y_scores : list[float]
        Synthetic probability scores.

    Returns
    -------
    float
        EER value in [0, 1].
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find the point where FPR ≈ FNR
    eer_idx = (fpr - fnr).abs().argmin() if hasattr(fpr, "abs") else \
        min(range(len(fpr)), key=lambda i: abs(fpr[i] - fnr[i]))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return float(eer)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or get_device()
    model_cfg = cfg.get("model", {})

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    eval_set = DatasetLoader(
        dataset_type=DatasetType(args.dataset),
        root_dir=args.data_dir,
        split=Split(args.split),
    ).load()
    logger.info("Eval set: %s", eval_set.class_distribution())

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    detector = CNNDetector(
        backbone=model_cfg.get("backbone", "resnet34"),
        num_classes=2,
        pretrained=False,
        device=device,
    ).build()
    load_checkpoint(args.checkpoint, detector._model, device=device)

    preprocessor = AudioPreprocessor(target_sr=16_000, normalize=True)
    extractor = FeatureExtractor(
        feature_type=FeatureType(model_cfg.get("input_feature", "mel_spectrogram"))
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    y_true, y_pred, y_scores = [], [], []

    import soundfile as sf

    for sample in eval_set:
        if not sample.file_path.exists():
            continue
        try:
            waveform, sr = sf.read(str(sample.file_path), dtype="float32", always_2d=False)
            waveform, sr = preprocessor.process(waveform, sr)
            features = extractor.extract(waveform)
            result = detector.predict(features)
            y_true.append(sample.label)
            y_pred.append(result["prediction"])
            y_scores.append(result["synthetic_prob"])
        except Exception as exc:
            logger.warning("Skipping %s: %s", sample.file_path.name, exc)

    if not y_true:
        logger.error("No samples evaluated. Check data path and file existence.")
        return

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    eer = compute_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["genuine", "synthetic"])

    metrics = {
        "accuracy": round(accuracy, 4),
        "eer": round(eer, 4),
        "confusion_matrix": cm,
        "n_samples": len(y_true),
    }

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("EER:      %.4f", eer)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s", report)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Metrics saved to %s", args.output_json)


if __name__ == "__main__":
    main()
