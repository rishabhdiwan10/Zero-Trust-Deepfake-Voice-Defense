"""
src.data.synthetic_generator
=============================
Utility for generating synthetic (AI-cloned) voice samples to supplement
training and evaluation datasets.

Supports multiple modern TTS / voice-cloning backends:
  - ElevenLabs API  (elevenlabs)
  - Bark             (suno-ai/bark)
  - XTTS v2          (coqui-ai/TTS)
  - gTTS             (Google Text-to-Speech — baseline, low quality)

Generated samples are saved as .wav files with a structured naming convention
and accompanied by a metadata CSV for easy dataset integration.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class TTSBackend(str, Enum):
    """Supported TTS / voice-cloning backends."""

    ELEVENLABS = "elevenlabs"
    BARK = "bark"
    XTTS = "xtts"
    GTTS = "gtts"


@dataclass
class GenerationResult:
    """Outcome of a single synthetic sample generation."""

    text: str
    backend: TTSBackend
    output_path: Path
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class SyntheticGenerator:
    """
    Generate synthetic voice samples for adversarial dataset augmentation.

    Parameters
    ----------
    backend : TTSBackend | str
        Which TTS / voice-cloning backend to use.
    output_dir : str | Path
        Directory where generated audio files will be saved.
    sample_rate : int
        Target sample rate for saved audio.
    api_key : str | None
        API key for cloud-based backends (e.g., ElevenLabs).
    """

    def __init__(
        self,
        backend: TTSBackend | str = TTSBackend.GTTS,
        output_dir: str | Path = "data/synthetic",
        sample_rate: int = 16_000,
        api_key: Optional[str] = None,
    ) -> None:
        self.backend = TTSBackend(backend)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.api_key = api_key or os.environ.get("TTS_API_KEY")
        self._results: List[GenerationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, texts: List[str]) -> List[GenerationResult]:
        """
        Generate synthetic audio for each text prompt.

        Parameters
        ----------
        texts : List[str]
            List of text strings to synthesise.

        Returns
        -------
        List[GenerationResult]
            One result per input text.
        """
        self._results = []
        for i, text in enumerate(texts):
            filename = f"{self.backend.value}_{i:04d}_{int(time.time())}.wav"
            output_path = self.output_dir / filename
            result = self._generate_single(text, output_path)
            self._results.append(result)
            logger.info(
                "Generated %s [%s] → %s",
                self.backend.value,
                "OK" if result.success else "FAILED",
                output_path,
            )
        self._save_metadata()
        return self._results

    def generate_from_file(self, text_file: str | Path) -> List[GenerationResult]:
        """Read text prompts from a file (one per line) and generate audio."""
        lines = Path(text_file).read_text(encoding="utf-8").splitlines()
        texts = [l.strip() for l in lines if l.strip()]
        return self.generate(texts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_single(self, text: str, output_path: Path) -> GenerationResult:
        """Dispatch generation to the appropriate backend."""
        dispatch = {
            TTSBackend.ELEVENLABS: self._generate_elevenlabs,
            TTSBackend.BARK: self._generate_bark,
            TTSBackend.XTTS: self._generate_xtts,
            TTSBackend.GTTS: self._generate_gtts,
        }
        try:
            duration = dispatch[self.backend](text, output_path)
            return GenerationResult(
                text=text,
                backend=self.backend,
                output_path=output_path,
                duration_seconds=duration,
                success=True,
            )
        except Exception as exc:
            logger.error("Generation failed for '%s': %s", text[:50], exc)
            return GenerationResult(
                text=text,
                backend=self.backend,
                output_path=output_path,
                duration_seconds=0.0,
                success=False,
                error=str(exc),
            )

    def _generate_elevenlabs(self, text: str, output_path: Path) -> float:
        """
        Generate audio via ElevenLabs API.

        Requires the ``elevenlabs`` Python package and a valid API key
        set via the ``ELEVENLABS_API_KEY`` environment variable or
        the ``api_key`` constructor parameter.
        """
        try:
            from elevenlabs import generate, save  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ElevenLabs package not installed. Run: pip install elevenlabs"
            ) from exc

        api_key = self.api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY env var."
            )

        audio = generate(text=text, api_key=api_key)
        save(audio, str(output_path))
        return self._get_duration(output_path)

    def _generate_bark(self, text: str, output_path: Path) -> float:
        """
        Generate audio via Suno Bark (local inference).

        Requires: ``pip install suno-bark``
        """
        try:
            import numpy as np
            from bark import generate_audio, SAMPLE_RATE  # type: ignore
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "Bark package not installed. Run: pip install suno-bark"
            ) from exc

        audio_array = generate_audio(text)
        sf.write(str(output_path), audio_array, SAMPLE_RATE)
        return len(audio_array) / SAMPLE_RATE

    def _generate_xtts(self, text: str, output_path: Path) -> float:
        """
        Generate audio via Coqui XTTS v2 (local inference).

        Requires: ``pip install TTS``
        """
        try:
            from TTS.api import TTS  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Coqui TTS not installed. Run: pip install TTS"
            ) from exc

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        tts.tts_to_file(text=text, file_path=str(output_path), language="en")
        return self._get_duration(output_path)

    def _generate_gtts(self, text: str, output_path: Path) -> float:
        """
        Generate audio via gTTS (Google Text-to-Speech — baseline quality).

        Requires: ``pip install gTTS``
        """
        try:
            from gtts import gTTS  # type: ignore
            import io
            import soundfile as sf
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "gTTS not installed. Run: pip install gTTS"
            ) from exc

        # gTTS outputs mp3; save directly and note the format
        mp3_path = output_path.with_suffix(".mp3")
        tts = gTTS(text=text, lang="en")
        tts.save(str(mp3_path))

        # Convert mp3 → wav using soundfile/pydub if available
        try:
            from pydub import AudioSegment  # type: ignore

            audio = AudioSegment.from_mp3(str(mp3_path))
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            audio.export(str(output_path), format="wav")
            mp3_path.unlink(missing_ok=True)
        except ImportError:
            # Keep mp3 if pydub not available
            logger.warning(
                "pydub not installed — saving as .mp3 instead of .wav"
            )

        return self._get_duration(output_path if output_path.exists() else mp3_path)

    @staticmethod
    def _get_duration(audio_path: Path) -> float:
        """Return duration of an audio file in seconds."""
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            return info.duration
        except Exception:
            return 0.0

    def _save_metadata(self) -> None:
        """Write a metadata CSV alongside the generated samples."""
        meta_path = self.output_dir / "metadata.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["filename", "text", "backend", "duration", "success", "error"],
            )
            writer.writeheader()
            for r in self._results:
                writer.writerow(
                    {
                        "filename": r.output_path.name,
                        "text": r.text,
                        "backend": r.backend.value,
                        "duration": round(r.duration_seconds, 3),
                        "success": r.success,
                        "error": r.error or "",
                    }
                )
        logger.info("Metadata saved to %s", meta_path)
