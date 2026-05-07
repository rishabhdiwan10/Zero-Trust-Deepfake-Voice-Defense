"""
tests/test_integration.py
==========================
End-to-end integration tests for the full Zero-Trust pipeline.

All ML models (CNN, Whisper, Prosody) are mocked so these tests run
without GPU, trained weights, or internet access.

Flow tested:
    audio file → ForensicAgent → DecisionAgent → (LivenessAgent) → result
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.decision_agent import DecisionAgent
from src.agents.forensic_agent import ForensicAgent
from src.agents.liveness_agent import LivenessAgent
from src.agents.orchestrator import Orchestrator
from src.agents.state import PipelineState
from src.decision.action_router import ActionRouter
from src.decision.threshold_engine import ThresholdEngine
from src.decision.trust_scorer import TrustScorer
from src.liveness.challenge_generator import ChallengeGenerator
from src.liveness.response_validator import ResponseValidator
from src.pipeline.realtime_pipeline import RealtimePipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_forensic_agent(cnn_mock, whisper_mock, prosody_mock):
    preprocessor = MagicMock()
    preprocessor.process.side_effect = lambda w, sr: (w, sr)
    extractor = MagicMock()
    extractor.extract.return_value = np.zeros((1, 128, 128), dtype=np.float32)
    return ForensicAgent(
        cnn_detector=cnn_mock,
        whisper_analyzer=whisper_mock,
        feature_extractor=extractor,
        preprocessor=preprocessor,
        prosody_analyzer=prosody_mock,
        run_parallel=False,
    )


def _make_decision_agent():
    return DecisionAgent(
        trust_scorer=TrustScorer(),
        threshold_engine=ThresholdEngine(),
        action_router=ActionRouter(),
    )


def _make_liveness_agent():
    challenge_gen = ChallengeGenerator(seed=42)
    validator = MagicMock(spec=ResponseValidator)
    validator.validate.return_value = True
    return LivenessAgent(challenge_gen, validator)


def _build_pipeline(forensic_agent, timeout=10.0) -> RealtimePipeline:
    orchestrator = Orchestrator(
        forensic_agent=forensic_agent,
        decision_agent=_make_decision_agent(),
        liveness_agent=_make_liveness_agent(),
    ).build()
    return RealtimePipeline(orchestrator, pipeline_timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — genuine audio path (should PASS)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenuineAudioPipeline:
    def test_genuine_audio_gives_pass(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_genuine, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)

        assert result["decision"] == "pass"
        assert result["trust_score"] > 0.7
        assert result["error"] is None

    def test_genuine_result_has_all_keys(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_genuine, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)

        for key in ["decision", "trust_score", "cnn_score", "whisper_score",
                    "prosody_score", "transcription", "forensic_metadata",
                    "stage_latencies", "liveness_challenge"]:
            assert key in result, f"Missing key: {key}"

    def test_genuine_cnn_score_in_result(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_genuine, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)
        assert abs(result["cnn_score"] - 0.92) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Tests — synthetic audio path (should REJECT)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticAudioPipeline:
    def test_synthetic_audio_gives_reject(
        self,
        mock_cnn_synthetic,
        mock_whisper_result_synthetic,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_synthetic
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_synthetic, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)

        assert result["decision"] == "reject"
        assert result["trust_score"] < 0.4

    def test_synthetic_result_no_error(
        self,
        mock_cnn_synthetic,
        mock_whisper_result_synthetic,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_synthetic
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_synthetic, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)
        assert result["error"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests — uncertain audio path (should CHALLENGE then re-evaluate)
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertainAudioPipeline:
    def test_uncertain_audio_triggers_liveness_challenge(
        self,
        mock_cnn_uncertain,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_uncertain, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)

        # After liveness (mocked to pass), decision should not be reject
        assert result["decision"] in ("pass", "challenge")
        # Liveness challenge phrase should be present
        assert isinstance(result.get("liveness_challenge"), str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineErrorHandling:
    def test_missing_audio_file_returns_reject(self) -> None:
        pipeline = _build_pipeline(MagicMock())
        result = pipeline.process_sync("/nonexistent/path/audio.wav")
        assert result["decision"] == "reject"
        assert result["error"] is not None

    def test_pipeline_timeout_returns_reject(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        import time

        def slow_analyze(*args, **kwargs):
            time.sleep(5)
            return mock_whisper_result_genuine

        whisper = MagicMock()
        whisper.analyze.side_effect = slow_analyze
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_genuine, whisper, prosody),
            timeout=0.5,  # very short timeout
        )
        result = pipeline.process_sync(tmp_wav_file)
        assert result["decision"] == "reject"
        assert result["error"] is not None

    def test_stage_latencies_populated(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_genuine, whisper, prosody)
        )
        result = pipeline.process_sync(tmp_wav_file)
        latencies = result["stage_latencies"]
        assert "total_ms" in latencies
        assert latencies["total_ms"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — async interface
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncPipeline:
    def test_async_process_returns_same_result(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        pipeline = _build_pipeline(
            _make_forensic_agent(mock_cnn_genuine, whisper, prosody)
        )

        async def run():
            return await pipeline.process(tmp_wav_file)

        result = asyncio.run(run())
        assert result["decision"] == "pass"
        assert "trust_score" in result