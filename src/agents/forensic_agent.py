"""
src.agents.forensic_agent
==========================
LangGraph agent node responsible for CNN-based and Whisper-based forensic
analysis of an input audio sample.

Reads ``audio_path`` (or ``waveform``) from the pipeline state, runs the CNN
detector and Whisper analyser, and writes back ``cnn_score``,
``whisper_score``, ``transcription``, and ``forensic_metadata``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .state import PipelineState

logger = logging.getLogger(__name__)


class ForensicAgent:
    """
    Forensic analysis agent combining CNN deepfake detection and Whisper
    artifact analysis.

    Parameters
    ----------
    cnn_detector : CNNDetector
        Pre-built and loaded CNN detector instance.
    whisper_analyzer : WhisperAnalyzer
        Pre-built and loaded Whisper analyzer instance.
    feature_extractor : FeatureExtractor
        Feature extractor for converting waveforms to CNN input.
    preprocessor : AudioPreprocessor
        Audio preprocessor.
    run_parallel : bool
        If ``True``, run CNN and Whisper inference concurrently using
        ``asyncio`` to reduce total latency.
    """

    def __init__(
        self,
        cnn_detector,
        whisper_analyzer,
        feature_extractor,
        preprocessor,
        run_parallel: bool = True,
    ) -> None:
        self.cnn_detector = cnn_detector
        self.whisper_analyzer = whisper_analyzer
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.run_parallel = run_parallel

    # ------------------------------------------------------------------
    # LangGraph node entry point
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute forensic analysis and update state.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state (must contain ``audio_path``).

        Returns
        -------
        PipelineState
            Updated state with forensic results.
        """
        t_start = time.perf_counter()
        audio_path = state.get("audio_path", "")

        if not audio_path:
            logger.error("ForensicAgent: no audio_path in state.")
            state["error"] = "Missing audio_path in state."
            return state

        try:
            cnn_score, whisper_result = self._run_analysis(audio_path)
        except Exception as exc:
            logger.exception("ForensicAgent analysis failed: %s", exc)
            state["error"] = f"ForensicAgent error: {exc}"
            state["cnn_score"] = 0.5
            state["whisper_score"] = 0.5
            state["transcription"] = ""
            state["forensic_metadata"] = {}
            return state

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        state["cnn_score"] = cnn_score
        state["whisper_score"] = whisper_result.whisper_score
        state["transcription"] = whisper_result.transcription
        state["forensic_metadata"] = {
            "avg_log_prob": whisper_result.avg_log_prob,
            "no_speech_prob": whisper_result.no_speech_prob,
            "compression_ratio": whisper_result.compression_ratio,
            "language": whisper_result.language,
        }

        latencies = state.get("stage_latencies", {})
        latencies["forensic_ms"] = round(elapsed_ms, 2)
        state["stage_latencies"] = latencies

        logger.info(
            "ForensicAgent done — cnn=%.3f whisper=%.3f (%.1f ms)",
            cnn_score,
            whisper_result.whisper_score,
            elapsed_ms,
        )
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_analysis(self, audio_path: str):
        """Run CNN and Whisper analysis (sequentially or in parallel)."""
        import soundfile as sf

        waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        waveform, sr = self.preprocessor.process(waveform, sr)
        features = self.feature_extractor.extract(waveform)

        cnn_result = self.cnn_detector.predict(features)
        cnn_score = cnn_result["genuine_prob"]

        whisper_result = self.whisper_analyzer.analyze(audio_path)

        return cnn_score, whisper_result
