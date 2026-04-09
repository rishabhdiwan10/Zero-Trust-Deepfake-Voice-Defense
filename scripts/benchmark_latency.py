"""
scripts/benchmark_latency.py
==============================
Formal latency profiling script for the Zero-Trust Deepfake Voice Defense
pipeline.

Measures and reports per-layer latency for **every stage** including:
  - Audio preprocessing + feature extraction
  - CNN deepfake detector inference
  - Whisper transcription + artifact analysis
  - Prosody / rhythm analysis
  - Trust scoring + threshold decision (LLM decision-making time)
  - Liveness challenge generation
  - End-to-end pipeline latency

Reports P50 / P95 / P99 percentiles, mean, max across N runs.
Outputs a formal JSON report and a formatted table suitable for inclusion
in a capstone report.

Usage::

    python scripts/benchmark_latency.py --n-runs 50 \\
                                         --audio-dir data/benchmark_samples

    # Save formal report to JSON
    python scripts/benchmark_latency.py --n-runs 50 --output-json reports/latency_profile.json

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
        description="Formal latency profiling for all pipeline layers."
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
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu / cuda / mps). Auto-detected if not set.",
    )
    return parser.parse_args()


def _make_synthetic_audio(duration: float = 2.0, sr: int = 16_000):
    """Generate a numpy sine-wave waveform for benchmarking."""
    import numpy as np

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def _save_temp_wav(waveform, sr: int) -> str:
    """Save waveform to a temporary WAV file and return the path."""
    import tempfile
    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, waveform, sr)
    return tmp.name


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


def benchmark_cnn(n_runs: int, tracker: LatencyTracker, device: str = "cpu") -> None:
    """Benchmark CNN detector inference."""
    from src.models.cnn_detector import CNNDetector
    from src.data.audio_preprocessor import AudioPreprocessor
    from src.data.feature_extractor import FeatureExtractor

    detector = CNNDetector(backbone="resnet34", pretrained=False, device=device).build()
    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()
    waveform, sr = _make_synthetic_audio()
    proc_wav, _ = preprocessor.process(waveform, sr)
    features = extractor.extract(proc_wav)

    # Warm-up run
    _ = detector.predict(features)

    for _ in range(n_runs):
        with Timer("cnn_inference") as t:
            _ = detector.predict(features)
        tracker.record("cnn_inference_ms", t.elapsed_ms)


def benchmark_whisper(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark Whisper transcription + artifact analysis."""
    from src.models.whisper_analyzer import WhisperAnalyzer

    waveform, sr = _make_synthetic_audio()
    audio_path = _save_temp_wav(waveform, sr)

    analyzer = WhisperAnalyzer(model_size="base", device="cpu")
    analyzer.load()

    # Warm-up run
    _ = analyzer.analyze(audio_path)

    for _ in range(n_runs):
        with Timer("whisper_transcription") as t:
            result = analyzer.analyze(audio_path)
        tracker.record("whisper_transcription_ms", t.elapsed_ms)


def benchmark_prosody(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark prosody / rhythm analysis."""
    from src.models.prosody_analyzer import ProsodyAnalyzer

    analyzer = ProsodyAnalyzer()
    waveform, sr = _make_synthetic_audio()

    for _ in range(n_runs):
        with Timer("prosody_analysis") as t:
            _ = analyzer.analyze(waveform, sr)
        tracker.record("prosody_analysis_ms", t.elapsed_ms)


def benchmark_decision(n_runs: int, tracker: LatencyTracker) -> None:
    """Benchmark threshold engine + trust scorer (LLM decision-making time)."""
    from src.decision.threshold_engine import ThresholdEngine
    from src.decision.trust_scorer import TrustScorer

    scorer = TrustScorer()
    engine = ThresholdEngine()

    for _ in range(n_runs):
        with Timer("decision") as t:
            trust = scorer.score(cnn_score=0.75, whisper_score=0.80, prosody_score=0.65)
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


def _format_report_table(summary: dict, budgets: dict) -> str:
    """Format the latency summary as a markdown-style table for reports."""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"{'Stage':<35} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8} {'Budget':>8}")
    lines.append("-" * 90)

    for stage, stats in summary.items():
        budget_key = stage.replace("_ms", "") + "_ms"
        budget = budgets.get(budget_key, None)
        budget_str = f"{budget}" if budget else "—"

        # Check if budget exceeded
        exceeded = ""
        if budget and stats["p95_ms"] > budget:
            exceeded = " !"

        lines.append(
            f"{stage:<35} {stats['mean_ms']:>7.1f}ms {stats['p50_ms']:>7.1f}ms "
            f"{stats['p95_ms']:>7.1f}ms {stats['p99_ms']:>7.1f}ms "
            f"{stats['max_ms']:>7.1f}ms {budget_str:>7}ms{exceeded}"
        )
    lines.append("=" * 90)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tracker = LatencyTracker()
    device = args.device or "cpu"

    try:
        latency_cfg = load_config(args.config)
        budgets = latency_cfg.get("budgets", {})
    except FileNotFoundError:
        logger.warning("Latency config not found — using defaults.")
        budgets = {}

    logger.info("Starting formal latency profiling (%d runs each)...", args.n_runs)
    logger.info("Device: %s", device)

    benchmarks = [
        ("Preprocessing + Feature Extraction", lambda: benchmark_preprocessing(args.n_runs, tracker)),
        ("CNN Inference", lambda: benchmark_cnn(args.n_runs, tracker, device)),
        ("Whisper Transcription + Analysis", lambda: benchmark_whisper(args.n_runs, tracker)),
        ("Prosody / Rhythm Analysis", lambda: benchmark_prosody(args.n_runs, tracker)),
        ("Decision Engine (Trust + Threshold)", lambda: benchmark_decision(args.n_runs, tracker)),
        ("Challenge Generator", lambda: benchmark_challenge_gen(args.n_runs, tracker)),
    ]

    for name, fn in benchmarks:
        logger.info("Benchmarking: %s", name)
        try:
            fn()
        except Exception as exc:
            logger.warning("Skipped %s: %s", name, exc)

    summary = tracker.summary()

    # Compute estimated end-to-end latency
    total_mean = sum(s["mean_ms"] for s in summary.values())
    total_p95 = sum(s["p95_ms"] for s in summary.values())

    # Print formatted report
    table = _format_report_table(summary, budgets)
    logger.info(table)
    logger.info(
        "Estimated end-to-end (sum of means): %.1f ms | sum of P95s: %.1f ms",
        total_mean, total_p95,
    )

    # Build full report dict
    report = {
        "n_runs": args.n_runs,
        "device": device,
        "per_stage": summary,
        "estimated_total_mean_ms": round(total_mean, 2),
        "estimated_total_p95_ms": round(total_p95, 2),
        "budgets": budgets,
        "budget_violations": {
            stage: {
                "p95_ms": stats["p95_ms"],
                "budget_ms": budgets.get(stage.replace("_ms", "") + "_ms"),
            }
            for stage, stats in summary.items()
            if budgets.get(stage.replace("_ms", "") + "_ms")
            and stats["p95_ms"] > budgets[stage.replace("_ms", "") + "_ms"]
        },
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Formal latency report saved to %s", output_path)


if __name__ == "__main__":
    main()
