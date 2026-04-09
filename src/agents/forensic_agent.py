"""
src.agents.forensic_agent
==========================
LangGraph agent node responsible for CNN-based and Whisper-based forensic
analysis of an input audio sample.

Reads ``audio_path`` (or ``waveform``) from the pipeline state, runs the CNN
detector and Whisper analyser, and writes back ``cnn_score``,
``whisper_score``, ``transcription``, and ``forensic_metadata``.

When ``run_parallel=True`` (the default), CNN and Whisper inference are
executed concurrently using a ``ThreadPoolExecutor``, reducing total latency
because the two models are independent of each other.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

from .state import PipelineState

logger = logging.getLogger(__name__)


class ForensicAgent:
    """
    Forensic analysis agent combining CNN deepfake detection, Whisper
    artifact analysis, and prosody/rhythm analysis.

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
    prosody_analyzer : ProsodyAnalyzer | None
        Optional prosody/rhythm analyzer. If ``None``, prosody analysis
        is skipped and a neutral score of 0.5 is used.
    run_parallel : bool
        If ``True``, run CNN, Whisper, and prosody inference concurrently
        using a ``ThreadPoolExecutor`` to reduce total latency.
    """

    def __init__(
        self,
        cnn_detector,
        whisper_analyzer,
        feature_extractor,
        preprocessor,
        prosody_analyzer=None,
        run_parallel: bool = True,
    ) -> None:
        self.cnn_detector = cnn_detector
        self.whisper_analyzer = whisper_analyzer
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.prosody_analyzer = prosody_analyzer
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
            cnn_score, whisper_result, prosody_result, timing = self._run_analysis(audio_path)
        except Exception as exc:
            logger.exception("ForensicAgent analysis failed: %s", exc)
            state["error"] = f"ForensicAgent error: {exc}"
            state["cnn_score"] = 0.5
            state["whisper_score"] = 0.5
            state["prosody_score"] = 0.5
            state["transcription"] = ""
            state["forensic_metadata"] = {}
            return state

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        state["cnn_score"] = cnn_score
        state["whisper_score"] = whisper_result.whisper_score
        state["prosody_score"] = prosody_result.prosody_score if prosody_result else 0.5
        state["transcription"] = whisper_result.transcription
        state["forensic_metadata"] = {
            "avg_log_prob": whisper_result.avg_log_prob,
            "no_speech_prob": whisper_result.no_speech_prob,
            "compression_ratio": whisper_result.compression_ratio,
            "language": whisper_result.language,
            "pitch_cv": prosody_result.pitch_cv if prosody_result else 0.0,
            "jitter_percent": prosody_result.jitter_percent if prosody_result else 0.0,
            "shimmer_percent": prosody_result.shimmer_percent if prosody_result else 0.0,
            "pitch_transition_rate": prosody_result.pitch_transition_rate if prosody_result else 0.0,
            **timing,
        }

        latencies = state.get("stage_latencies", {})
        latencies["forensic_ms"] = round(elapsed_ms, 2)
        state["stage_latencies"] = latencies

        logger.info(
            "ForensicAgent done — cnn=%.3f whisper=%.3f prosody=%.3f (%.1f ms)",
            cnn_score,
            whisper_result.whisper_score,
            prosody_result.prosody_score if prosody_result else 0.5,
            elapsed_ms,
        )
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_analysis(self, audio_path: str) -> Tuple[float, Any, Any, Dict[str, float]]:
        """Run CNN, Whisper, and prosody analysis — sequentially or in parallel.

        When ``self.run_parallel`` is ``True``, all three analyses are
        submitted to a :class:`~concurrent.futures.ThreadPoolExecutor`
        so they run concurrently (all are I/O-heavy / GIL-releasing).

        Returns
        -------
        Tuple of (cnn_score, whisper_result, prosody_result, timing_dict).
        """
        import soundfile as sf

        waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        waveform, sr = self.preprocessor.process(waveform, sr)
        features = self.feature_extractor.extract(waveform)

        if self.run_parallel:
            return self._run_parallel(audio_path, features, waveform, sr)
        else:
            return self._run_sequential(audio_path, features, waveform, sr)

    def _run_parallel(
        self, audio_path: str, features, waveform, sr: int
    ) -> Tuple[float, Any, Any, Dict[str, float]]:
        """Execute CNN, Whisper, and prosody concurrently via ThreadPoolExecutor."""

        def _cnn_task():
            t0 = time.perf_counter()
            result = self.cnn_detector.predict(features)
            return result, (time.perf_counter() - t0) * 1000

        def _whisper_task():
            t0 = time.perf_counter()
            result = self.whisper_analyzer.analyze(audio_path)
            return result, (time.perf_counter() - t0) * 1000

        def _prosody_task():
            t0 = time.perf_counter()
            if self.prosody_analyzer is not None:
                result = self.prosody_analyzer.analyze(waveform, sr)
            else:
                result = None
            return result, (time.perf_counter() - t0) * 1000

        max_workers = 3 if self.prosody_analyzer else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_cnn = executor.submit(_cnn_task)
            future_whisper = executor.submit(_whisper_task)
            future_prosody = executor.submit(_prosody_task)

            cnn_raw, cnn_ms = future_cnn.result()
            whisper_raw, whisper_ms = future_whisper.result()
            prosody_raw, prosody_ms = future_prosody.result()

        logger.debug(
            "Parallel forensics — CNN: %.1f ms, Whisper: %.1f ms, Prosody: %.1f ms",
            cnn_ms, whisper_ms, prosody_ms,
        )
        timing = {
            "cnn_inference_ms": round(cnn_ms, 2),
            "whisper_inference_ms": round(whisper_ms, 2),
            "prosody_analysis_ms": round(prosody_ms, 2),
        }
        return cnn_raw["genuine_prob"], whisper_raw, prosody_raw, timing

    def _run_sequential(
        self, audio_path: str, features, waveform, sr: int
    ) -> Tuple[float, Any, Any, Dict[str, float]]:
        """Execute CNN, Whisper, and prosody sequentially."""
        t0 = time.perf_counter()
        cnn_result = self.cnn_detector.predict(features)
        cnn_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        whisper_result = self.whisper_analyzer.analyze(audio_path)
        whisper_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.prosody_analyzer is not None:
            prosody_result = self.prosody_analyzer.analyze(waveform, sr)
        else:
            prosody_result = None
        prosody_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Sequential forensics — CNN: %.1f ms, Whisper: %.1f ms, Prosody: %.1f ms",
            cnn_ms, whisper_ms, prosody_ms,
        )
        timing = {
            "cnn_inference_ms": round(cnn_ms, 2),
            "whisper_inference_ms": round(whisper_ms, 2),
            "prosody_analysis_ms": round(prosody_ms, 2),
        }
        return cnn_result["genuine_prob"], whisper_result, prosody_result, timing
