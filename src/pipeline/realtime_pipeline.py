"""
src.pipeline.realtime_pipeline
================================
Real-time, async pipeline that orchestrates all layers of the Zero-Trust
Deepfake Voice Defense System for low-latency, per-request audio analysis.

The pipeline supports both file-based and streaming (chunk-by-chunk) input
and is designed to meet the latency budgets defined in
``configs/latency_config.yaml``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..agents.state import PipelineState
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RealtimePipeline:
    """
    Real-time, asynchronous deepfake voice defence pipeline.

    Parameters
    ----------
    orchestrator : Orchestrator
        Pre-built LangGraph orchestrator with all agents wired up.
    pipeline_timeout : float
        Maximum seconds to wait for a complete pipeline result.
    """

    def __init__(
        self,
        orchestrator,
        pipeline_timeout: float = 5.0,
    ) -> None:
        self.orchestrator = orchestrator
        self.pipeline_timeout = pipeline_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Asynchronously process a single audio file through the full pipeline.

        Parameters
        ----------
        audio_path : str
            Absolute or relative path to the input audio file.

        Returns
        -------
        dict
            Result dictionary containing:
            ``decision``, ``trust_score``, ``cnn_score``, ``whisper_score``,
            ``transcription``, ``stage_latencies``, and optional ``error``.
        """
        if not Path(audio_path).exists():
            return self._error_result(f"Audio file not found: {audio_path}")

        initial_state: PipelineState = {
            "audio_path": audio_path,
            "liveness_retry_count": 0,
            "stage_latencies": {},
        }

        t_start = time.perf_counter()

        try:
            final_state = await asyncio.wait_for(
                self.orchestrator.run_async(initial_state),
                timeout=self.pipeline_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Pipeline timed out after %.1f s for %s", self.pipeline_timeout, audio_path
            )
            return self._error_result(
                f"Pipeline timeout after {self.pipeline_timeout}s"
            )
        except Exception as exc:
            logger.exception("Pipeline error for %s: %s", audio_path, exc)
            return self._error_result(str(exc))

        total_ms = (time.perf_counter() - t_start) * 1000
        latencies = final_state.get("stage_latencies", {})
        latencies["total_ms"] = round(total_ms, 2)

        return {
            "decision": final_state.get("decision", "reject"),
            "action": final_state.get("action", ""),
            "trust_score": round(final_state.get("trust_score", 0.0), 4),
            "cnn_score": round(final_state.get("cnn_score", 0.0), 4),
            "whisper_score": round(final_state.get("whisper_score", 0.0), 4),
            "transcription": final_state.get("transcription", ""),
            "liveness_passed": final_state.get("liveness_passed"),
            "stage_latencies": latencies,
            "error": final_state.get("error"),
        }

    def process_sync(self, audio_path: str) -> Dict[str, Any]:
        """Synchronous wrapper for ``process``."""
        return asyncio.run(self.process(audio_path))

    # ------------------------------------------------------------------
    # Streaming support
    # ------------------------------------------------------------------

    async def process_stream(
        self,
        audio_chunks,
        sample_rate: int = 16_000,
    ) -> Dict[str, Any]:
        """
        Process an async generator of audio chunks (streaming mode).

        Parameters
        ----------
        audio_chunks : AsyncGenerator[np.ndarray, None]
            Async generator yielding numpy float32 audio chunks.
        sample_rate : int
            Sample rate of the chunks.

        Returns
        -------
        dict
            Same structure as ``process``.
        """
        import numpy as np
        import tempfile
        import soundfile as sf

        chunks = []
        async for chunk in audio_chunks:
            chunks.append(chunk)

        if not chunks:
            return self._error_result("No audio chunks received.")

        waveform = np.concatenate(chunks)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sample_rate)
            return await self.process(tmp.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(message: str) -> Dict[str, Any]:
        return {
            "decision": "reject",
            "action": "Access denied — pipeline error.",
            "trust_score": 0.0,
            "cnn_score": 0.0,
            "whisper_score": 0.0,
            "transcription": "",
            "liveness_passed": None,
            "stage_latencies": {},
            "error": message,
        }
