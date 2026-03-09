"""
scripts/benchmark_latency.py
==============================
Latency benchmarking script for the Zero-Trust Deepfake Voice Defense
pipeline layers.

Measures and reports:
  - Per-layer latency (preprocessing, CNN, Whisper, decision, liveness)
  - End-to-end pipeline latency
  - P50 / P95 / P99 percentiles across N runs

Usage::

    python scripts/benchmark_latency.py --n-runs 50 \\
                                         --audio-dir data/benchmark_samples

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker, Timer
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline latency across layers."
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=20,
        help="Number of benchmark runs per layer (default: 20).",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory containing .wav files for benchmarking. "
             "Uses synthetic sine waves if not provided.",
    )
    parser.add_argument(
        "--config",
        default="configs/latency_config.yaml",
        help="Latency config YAML path.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Save latency report to JSON file.",
    )
    return parser.parse_args()


def _make_synthetic_audio(duration: float = 2.0, sr: int = 16_000):
    """Generate a numpy sine-wave waveform for benchmarking."""
    import numpy as np

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def benchmark_preprocessing(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark audio preprocessing + feature extraction."""
    from src.data.audio_preprocessor import AudioPreprocessor
    from src.data.feature_extractor import FeatureExtractor

    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()
    waveform, sr = _make_synthetic_audio()

    for _ in range(n_runs):
        with Timer("preprocessing") as t:
            proc_wav, proc_sr = preprocessor.process(waveform.copy(), sr)
            _ = extractor.extract(proc_wav)
        tracker.record("preprocessing_ms", t.elapsed_ms)


def benchmark_cnn(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark CNN detector inference."""
    import numpy as np
    from src.models.cnn_detector import CNNDetector
    from src.data.audio_preprocessor import AudioPreprocessor
    from src.data.feature_extractor import FeatureExtractor

    detector = CNNDetector(backbone="resnet18", pretrained=False, device="cpu").build()
    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()
    waveform, sr = _make_synthetic_audio()
    proc_wav, _ = preprocessor.process(waveform, sr)
    features = extractor.extract(proc_wav)

    for _ in range(n_runs):
        with Timer("cnn_inference") as t:
            _ = detector.predict(features)
        tracker.record("cnn_inference_ms", t.elapsed_ms)


def benchmark_decision(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark threshold engine + trust scorer."""
    from src.decision.threshold_engine import ThresholdEngine
    from src.decision.trust_scorer import TrustScorer

    scorer = TrustScorer()
    engine = ThresholdEngine()

    for _ in range(n_runs):
        with Timer("decision") as t:
            trust = scorer.score(cnn_score=0.75, whisper_score=0.80)
            _ = engine.evaluate(trust)
        tracker.record("decision_ms", t.elapsed_ms)


def benchmark_challenge_gen(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark dynamic challenge generation."""
    from src.liveness.challenge_generator import ChallengeGenerator

    gen = ChallengeGenerator()
    for _ in range(n_runs):
        with Timer("challenge_gen") as t:
            _ = gen.generate()
        tracker.record("challenge_generation_ms", t.elapsed_ms)


def main() -> None:
    args = parse_args()
    tracker = LatencyTracker()

    try:
        latency_cfg = load_config(args.config)
        budgets = latency_cfg.get("budgets", {})
    except FileNotFoundError:
        logger.warning("Latency config not found — using defaults.")
        budgets = {}

    logger.info("Starting latency benchmarks (%d runs each)...", args.n_runs)

    benchmarks = [
        ("Preprocessing + Feature Extraction", benchmark_preprocessing),
        ("CNN Inference", benchmark_cnn),
        ("Decision Engine", benchmark_decision),
        ("Challenge Generator", benchmark_challenge_gen),
    ]

    for name, fn in benchmarks:
        logger.info("Benchmarking: %s", name)
        fn(args.n_runs, tracker)

    summary = tracker.summary()

    logger.info("\n=== Latency Benchmark Results ===")
    for stage, stats in summary.items():
        budget_key = stage.replace("_ms", "") + "_ms"
        budget = budgets.get(budget_key, None)
        budget_str = f" (budget: {budget} ms)" if budget else ""
        logger.info(
            "%-30s | mean=%.1f ms | p95=%.1f ms | p99=%.1f ms%s",
            stage,
            stats["mean_ms"],
            stats["p95_ms"],
            stats["p99_ms"],
            budget_str,
        )

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Latency report saved to %s", args.output_json)


if __name__ == "__main__":
    main()
