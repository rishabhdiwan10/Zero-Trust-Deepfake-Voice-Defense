"""
scripts/generate_synthetic.py
===============================
Script to generate synthetic voice samples for adversarial testing using
modern TTS / voice-cloning backends (ElevenLabs, Bark, XTTS, gTTS).

Usage::

    # Using gTTS (no API key required — baseline quality)
    python scripts/generate_synthetic.py --backend gtts \\
                                          --texts-file data/prompts.txt \\
                                          --output-dir data/synthetic/gtts

    # Using ElevenLabs (requires ELEVENLABS_API_KEY env var)
    python scripts/generate_synthetic.py --backend elevenlabs \\
                                          --text "Hello, this is a test." \\
                                          --output-dir data/synthetic/elevenlabs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import SyntheticGenerator, TTSBackend
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please verify your identity by speaking this phrase.",
    "Zero trust security requires continuous authentication.",
    "Artificial intelligence is transforming voice security.",
    "Dynamic liveness challenges prevent replay attacks effectively.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic voice samples for adversarial testing."
    )
    parser.add_argument(
        "--backend",
        default="gtts",
        choices=["gtts", "bark", "xtts", "elevenlabs"],
        help="TTS backend to use (default: gtts).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic",
        help="Output directory for generated audio files.",
    )
    parser.add_argument(
        "--texts-file",
        default=None,
        help="Path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--text",
        default=None,
        nargs="+",
        help="One or more text prompts to synthesise.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Target sample rate for generated audio (default: 16000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine text prompts
    if args.text:
        texts = args.text
    elif args.texts_file:
        texts = Path(args.texts_file).read_text(encoding="utf-8").splitlines()
        texts = [t.strip() for t in texts if t.strip()]
    else:
        logger.info("No texts provided — using default prompts.")
        texts = DEFAULT_TEXTS

    logger.info(
        "Generating %d synthetic samples using backend: %s",
        len(texts),
        args.backend,
    )

    generator = SyntheticGenerator(
        backend=TTSBackend(args.backend),
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
    )

    results = generator.generate(texts)

    success = sum(1 for r in results if r.success)
    failed = len(results) - success

    logger.info(
        "Generation complete: %d succeeded, %d failed. Output: %s",
        success,
        failed,
        args.output_dir,
    )

    for r in results:
        if not r.success:
            logger.warning("Failed: '%s...' — %s", r.text[:40], r.error)


if __name__ == "__main__":
    main()
