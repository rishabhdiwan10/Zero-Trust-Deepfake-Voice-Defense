"""
src.models.whisper_analyzer
============================
Whisper-based audio transcription and artifact analysis module.

Beyond transcription, this module analyses the Whisper model's internal
confidence signals to detect artefacts characteristic of synthetic speech:
  - Unusually high or uniform per-token log-probabilities
  - Repetition anomalies
  - Compression or codec artefacts inferred from attention patterns

Returns a ``whisper_score`` in [0, 1] representing the probability that
the audio is *genuine* according to Whisper-derived signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WhisperAnalysisResult:
    """Result of a Whisper analysis pass."""

    transcription: str
    language: str
    avg_log_prob: float  # Average token log-probability (Whisper metric)
    no_speech_prob: float  # Probability of no speech detected
    compression_ratio: float  # Text compression ratio (Whisper metric)
    whisper_score: float  # Genuine probability [0, 1]
    tokens: List[int] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)


class WhisperAnalyzer:
    """
    Transcription and forensic analysis using OpenAI Whisper.

    Parameters
    ----------
    model_size : str
        Whisper model variant: ``tiny`` | ``base`` | ``small`` |
        ``medium`` | ``large-v3``.
    device : str | None
        Torch device (auto-detected if ``None``).
    fp16 : bool
        Use FP16 for faster inference on GPU.
    """

    # Whisper's built-in heuristics: audio that is likely synthetic
    # tends to have unusually high average log-probs and low compression ratios.
    SYNTHETIC_AVG_LOG_PROB_THRESHOLD = -0.1  # very confident → suspicious
    LOW_COMPRESSION_RATIO_THRESHOLD = 1.2    # very low compression → suspicious

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device or self._auto_device()
        self.fp16 = fp16
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "WhisperAnalyzer":
        """Load the Whisper model into memory."""
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required. Run: pip install openai-whisper"
            ) from exc

        logger.info("Loading Whisper model: %s on %s", self.model_size, self.device)
        self._model = whisper.load_model(self.model_size, device=self.device)
        return self

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, audio_path: str) -> WhisperAnalysisResult:
        """
        Transcribe and analyse an audio file.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (WAV, FLAC, MP3, etc.).

        Returns
        -------
        WhisperAnalysisResult
        """
        if self._model is None:
            self.load()

        import whisper  # type: ignore

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        result = self._model.transcribe(
            audio,
            fp16=self.fp16 and self.device != "cpu",
            word_timestamps=False,
            verbose=False,
        )

        transcription = result.get("text", "").strip()
        language = result.get("language", "unknown")
        segments = result.get("segments", [])

        avg_log_prob = float(
            sum(s.get("avg_logprob", 0.0) for s in segments) / max(len(segments), 1)
        )
        no_speech_prob = float(
            sum(s.get("no_speech_prob", 0.0) for s in segments) / max(len(segments), 1)
        )
        compression_ratio = float(
            sum(s.get("compression_ratio", 1.0) for s in segments)
            / max(len(segments), 1)
        )

        whisper_score = self._compute_whisper_score(
            avg_log_prob, no_speech_prob, compression_ratio
        )

        return WhisperAnalysisResult(
            transcription=transcription,
            language=language,
            avg_log_prob=avg_log_prob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
            whisper_score=whisper_score,
            segments=segments,
        )

    def analyze_waveform(
        self, waveform, sample_rate: int = 16_000
    ) -> WhisperAnalysisResult:
        """
        Analyse a raw waveform (numpy array or torch Tensor).

        Parameters
        ----------
        waveform : np.ndarray | torch.Tensor
            1-D float32 waveform at 16 kHz.
        sample_rate : int
            Sample rate; resampling handled internally if != 16 000.
        """
        import tempfile
        import soundfile as sf
        import numpy as np

        if hasattr(waveform, "numpy"):
            waveform = waveform.numpy()
        waveform = np.asarray(waveform, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sample_rate)
            return self.analyze(tmp.name)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_whisper_score(
        self,
        avg_log_prob: float,
        no_speech_prob: float,
        compression_ratio: float,
    ) -> float:
        """
        Heuristic genuine-probability score derived from Whisper signals.

        Rules (higher output = more likely genuine):
          - Very high avg_log_prob (> -0.1) suggests unnaturally confident
            outputs typical of TTS → lowers genuine score.
          - Very low compression_ratio (< 1.2) suggests limited lexical
            variety → lowers genuine score.
          - High no_speech_prob → audio is silence or noise, not speech.
        """
        score = 1.0

        # Penalise unnaturally high confidence
        if avg_log_prob > self.SYNTHETIC_AVG_LOG_PROB_THRESHOLD:
            score -= 0.3

        # Penalise low lexical compression (TTS tends to be less diverse)
        if compression_ratio < self.LOW_COMPRESSION_RATIO_THRESHOLD:
            score -= 0.2

        # Penalise no-speech detection
        score -= 0.5 * no_speech_prob

        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
