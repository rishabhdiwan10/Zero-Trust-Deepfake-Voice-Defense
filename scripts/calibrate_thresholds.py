"""
scripts/calibrate_thresholds.py
=================================
Threshold calibration and confidence score analysis script.

Addresses professor feedback: "How would you decide threshold value and
confidence score?"

This script:
  1. Runs the trained CNN detector on the validation/eval dataset
  2. Computes ROC curve, DET curve, and EER
  3. Finds the optimal pass/challenge/reject thresholds based on:
     - EER operating point
     - Target False Accept Rate (FAR)
     - Target False Reject Rate (FRR)
  4. Generates calibration plots and a justification report

Usage::

    python scripts/calibrate_thresholds.py \\
        --checkpoint models/best_checkpoint.pt \\
        --data-dir data/asvspoof2019 \\
        --dataset asvspoof2019 \\
        --output-dir reports/threshold_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate decision thresholds from model scores."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--dataset",
        default="custom",
        choices=["asvspoof2019", "asvspoof5", "in_the_wild", "custom"],
    )
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument(
        "--output-dir",
        default="reports/threshold_calibration",
        help="Directory for calibration outputs.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="Target false accept rate for the reject threshold (default: 1%%).",
    )
    parser.add_argument(
        "--target-frr",
        type=float,
        default=0.05,
        help="Target false reject rate for the pass threshold (default: 5%%).",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def collect_scores(args) -> tuple:
    """Run inference on the eval set and collect genuine_prob scores + labels."""
    import soundfile as sf

    from src.data.dataset_loader import DatasetLoader, DatasetType, Split
    from src.data.audio_preprocessor import AudioPreprocessor
    from src.data.feature_extractor import FeatureExtractor, FeatureType
    from src.models.cnn_detector import CNNDetector
    from src.models.model_utils import load_checkpoint, get_device
    from src.utils.config_loader import load_config

    cfg = load_config(args.config)
    device = args.device or get_device()
    m_cfg = cfg.get("model", {})

    eval_set = DatasetLoader(
        dataset_type=DatasetType(args.dataset),
        root_dir=args.data_dir,
        split=Split(args.split),
    ).load()
    logger.info("Eval set: %s", eval_set.class_distribution())

    detector = CNNDetector(
        backbone=m_cfg.get("backbone", "resnet34"),
        num_classes=2,
        pretrained=False,
        device=device,
    ).build()
    load_checkpoint(args.checkpoint, detector._model, device=device)

    preprocessor = AudioPreprocessor(target_sr=16_000, normalize=True)
    extractor = FeatureExtractor(
        feature_type=FeatureType(m_cfg.get("input_feature", "mel_spectrogram"))
    )

    y_true, y_genuine_prob = [], []

    for sample in eval_set:
        if not sample.file_path.exists():
            continue
        try:
            waveform, sr = sf.read(str(sample.file_path), dtype="float32", always_2d=False)
            waveform, sr = preprocessor.process(waveform, sr)
            features = extractor.extract(waveform)
            result = detector.predict(features)
            y_true.append(sample.label)
            y_genuine_prob.append(result["genuine_prob"])
        except Exception as exc:
            logger.warning("Skipping %s: %s", sample.file_path.name, exc)

    return np.array(y_true), np.array(y_genuine_prob)


def compute_eer(fpr, tpr) -> tuple:
    """Find Equal Error Rate from ROC curve."""
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return float(eer), eer_idx


def find_threshold_at_far(fpr, thresholds, target_far: float) -> float:
    """Find the threshold that achieves a target False Accept Rate."""
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return float(thresholds[-1])
    return float(thresholds[valid[-1]])


def find_threshold_at_frr(tpr, thresholds, target_frr: float) -> float:
    """Find the threshold that achieves a target False Reject Rate."""
    fnr = 1 - tpr
    valid = np.where(fnr <= target_frr)[0]
    if len(valid) == 0:
        return float(thresholds[0])
    return float(thresholds[valid[0]])


def generate_plots(
    y_true, y_scores, fpr, tpr, thresholds, eer, eer_idx,
    pass_threshold, challenge_threshold, output_dir: Path
):
    """Generate ROC curve, DET curve, and score distribution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import det_curve
    except ImportError:
        logger.warning("matplotlib not available — skipping plot generation.")
        return

    # 1. ROC Curve with thresholds marked
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (EER = {eer:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.plot(fpr[eer_idx], tpr[eer_idx], "ro", markersize=10, label=f"EER point")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (1 - FRR)")
    ax.set_title("ROC Curve — Threshold Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve → %s", output_dir / "roc_curve.png")

    # 2. DET Curve
    fpr_det, fnr_det, _ = det_curve(y_true, 1 - y_scores)  # synthetic prob
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr_det * 100, fnr_det * 100, "b-", linewidth=2)
    ax.set_xlabel("False Accept Rate (%)")
    ax.set_ylabel("False Reject Rate (%)")
    ax.set_title("DET Curve")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "det_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved DET curve → %s", output_dir / "det_curve.png")

    # 3. Score distribution with threshold lines
    genuine_scores = y_scores[y_true == 0]
    synthetic_scores = y_scores[y_true == 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(genuine_scores, bins=50, alpha=0.6, color="green", label="Genuine", density=True)
    ax.hist(synthetic_scores, bins=50, alpha=0.6, color="red", label="Synthetic", density=True)
    ax.axvline(x=pass_threshold, color="blue", linestyle="--", linewidth=2,
               label=f"Pass threshold = {pass_threshold:.3f}")
    ax.axvline(x=challenge_threshold, color="orange", linestyle="--", linewidth=2,
               label=f"Challenge threshold = {challenge_threshold:.3f}")
    ax.set_xlabel("Genuine Probability Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution with Decision Thresholds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved score distribution → %s", output_dir / "score_distribution.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect scores
    logger.info("Collecting model scores on %s/%s...", args.dataset, args.split)
    y_true, y_genuine_prob = collect_scores(args)

    if len(y_true) == 0:
        logger.error("No samples evaluated. Check data path.")
        return

    logger.info("Collected %d scores (genuine=%d, synthetic=%d)",
                len(y_true), sum(y_true == 0), sum(y_true == 1))

    # Compute ROC curve
    from sklearn.metrics import roc_curve, roc_auc_score

    # y_genuine_prob is P(genuine), so for ROC we need P(genuine) with pos_label=0
    # Alternatively: use 1-y_genuine_prob as synthetic score
    fpr, tpr, thresholds = roc_curve(y_true, 1 - y_genuine_prob, pos_label=1)
    auc = roc_auc_score(y_true, 1 - y_genuine_prob)

    # EER
    eer, eer_idx = compute_eer(fpr, tpr)
    eer_threshold = float(1 - thresholds[eer_idx])  # convert back to genuine_prob
    logger.info("EER = %.4f at genuine_prob threshold = %.4f", eer, eer_threshold)
    logger.info("AUC = %.4f", auc)

    # Find optimal thresholds
    # Pass threshold: at target FRR (false reject rate for genuine speakers)
    # We want genuine speakers to pass, so we set pass_threshold where FRR <= target
    pass_threshold_raw = find_threshold_at_frr(tpr, thresholds, args.target_frr)
    pass_threshold = float(1 - pass_threshold_raw)  # genuine_prob space

    # Challenge threshold: at target FAR (false accept rate for synthetic)
    challenge_threshold_raw = find_threshold_at_far(fpr, thresholds, args.target_far)
    challenge_threshold = float(1 - challenge_threshold_raw)  # genuine_prob space

    # Ensure pass > challenge
    if pass_threshold <= challenge_threshold:
        pass_threshold = max(challenge_threshold + 0.1, eer_threshold)

    logger.info("Calibrated thresholds:")
    logger.info("  PASS threshold:      %.4f (genuine_prob >= this → PASS)", pass_threshold)
    logger.info("  CHALLENGE threshold:  %.4f (genuine_prob >= this → CHALLENGE)", challenge_threshold)
    logger.info("  REJECT:               genuine_prob < %.4f", challenge_threshold)

    # Generate plots
    generate_plots(
        y_true, y_genuine_prob, fpr, tpr, thresholds, eer, eer_idx,
        pass_threshold, challenge_threshold, output_dir
    )

    # Build calibration report
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "n_samples": int(len(y_true)),
        "n_genuine": int(sum(y_true == 0)),
        "n_synthetic": int(sum(y_true == 1)),
        "metrics": {
            "auc": round(auc, 4),
            "eer": round(eer, 4),
            "eer_threshold": round(eer_threshold, 4),
        },
        "calibrated_thresholds": {
            "pass_threshold": round(pass_threshold, 4),
            "challenge_threshold": round(challenge_threshold, 4),
            "justification": {
                "pass": (
                    f"Set at genuine_prob >= {pass_threshold:.4f} to achieve "
                    f"<= {args.target_frr*100:.1f}% false reject rate for genuine speakers. "
                    f"This means legitimate users are incorrectly rejected at most "
                    f"{args.target_frr*100:.1f}% of the time."
                ),
                "challenge": (
                    f"Set at genuine_prob >= {challenge_threshold:.4f} to achieve "
                    f"<= {args.target_far*100:.1f}% false accept rate for synthetic audio. "
                    f"Samples scoring between challenge and pass thresholds undergo "
                    f"a liveness challenge for additional verification."
                ),
                "reject": (
                    f"Samples with genuine_prob < {challenge_threshold:.4f} are rejected "
                    f"outright. At this threshold, the false accept rate is below "
                    f"{args.target_far*100:.1f}%, ensuring high-confidence synthetic "
                    f"audio is blocked."
                ),
            },
        },
        "target_rates": {
            "target_far": args.target_far,
            "target_frr": args.target_frr,
        },
        "genuine_score_stats": {
            "mean": round(float(np.mean(y_genuine_prob[y_true == 0])), 4),
            "std": round(float(np.std(y_genuine_prob[y_true == 0])), 4),
            "min": round(float(np.min(y_genuine_prob[y_true == 0])), 4),
            "max": round(float(np.max(y_genuine_prob[y_true == 0])), 4),
        },
        "synthetic_score_stats": {
            "mean": round(float(np.mean(y_genuine_prob[y_true == 1])), 4),
            "std": round(float(np.std(y_genuine_prob[y_true == 1])), 4),
            "min": round(float(np.min(y_genuine_prob[y_true == 1])), 4),
            "max": round(float(np.max(y_genuine_prob[y_true == 1])), 4),
        },
    }

    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Calibration report saved to %s", report_path)

    # Print summary for report inclusion
    print("\n" + "=" * 70)
    print("THRESHOLD CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Dataset:            {args.dataset} ({args.split})")
    print(f"Samples:            {len(y_true)} (genuine={sum(y_true==0)}, synthetic={sum(y_true==1)})")
    print(f"AUC:                {auc:.4f}")
    print(f"EER:                {eer:.4f}")
    print(f"")
    print(f"Calibrated Thresholds:")
    print(f"  PASS      >= {pass_threshold:.4f}  (FRR <= {args.target_frr*100:.1f}%)")
    print(f"  CHALLENGE >= {challenge_threshold:.4f}  (FAR <= {args.target_far*100:.1f}%)")
    print(f"  REJECT    <  {challenge_threshold:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
