"""
Tests for src.agents.forensic_agent — CNN + Whisper + Prosody analysis.

All ML models are mocked so these tests run without GPU or trained weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.forensic_agent import ForensicAgent
from src.agents.state import PipelineState


@pytest.fixture()
def forensic_agent(
    mock_cnn_genuine,
    mock_whisper_result_genuine,
    mock_prosody_result,
):
    """ForensicAgent with all ML models mocked."""
    whisper = MagicMock()
    whisper.analyze.return_value = mock_whisper_result_genuine

    preprocessor = MagicMock()
    preprocessor.process.side_effect = lambda w, sr: (w, sr)

    extractor = MagicMock()
    extractor.extract.return_value = np.zeros((1, 128, 128), dtype=np.float32)

    prosody = MagicMock()
    prosody.analyze.return_value = mock_prosody_result

    return ForensicAgent(
        cnn_detector=mock_cnn_genuine,
        whisper_analyzer=whisper,
        feature_extractor=extractor,
        preprocessor=preprocessor,
        prosody_analyzer=prosody,
        run_parallel=False,
    )


class TestForensicAgent:
    def test_run_populates_cnn_score(self, forensic_agent: ForensicAgent, tmp_wav_file: str) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "cnn_score" in result
        assert 0.0 <= result["cnn_score"] <= 1.0

    def test_run_populates_whisper_score(self, forensic_agent: ForensicAgent, tmp_wav_file: str) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "whisper_score" in result
        assert 0.0 <= result["whisper_score"] <= 1.0

    def test_run_populates_prosody_score(self, forensic_agent: ForensicAgent, tmp_wav_file: str) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "prosody_score" in result

    def test_run_populates_transcription(self, forensic_agent: ForensicAgent, tmp_wav_file: str) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "transcription" in result
        assert isinstance(result["transcription"], str)

    def test_run_populates_forensic_metadata(self, forensic_agent: ForensicAgent, tmp_wav_file: str) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "forensic_metadata" in result
        assert isinstance(result["forensic_metadata"], dict)

    def test_run_records_latency(self, forensic_agent: ForensicAgent, tmp_wav_file: str) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "forensic_ms" in result["stage_latencies"]

    def test_missing_audio_path_sets_error(self, forensic_agent: ForensicAgent) -> None:
        state: PipelineState = {"audio_path": "", "stage_latencies": {}}
        result = forensic_agent.run(state)
        assert "error" in result
        assert result["error"]

    def test_run_parallel_mode(
        self,
        mock_cnn_genuine,
        mock_whisper_result_genuine,
        mock_prosody_result,
        tmp_wav_file: str,
    ) -> None:
        whisper = MagicMock()
        whisper.analyze.return_value = mock_whisper_result_genuine
        preprocessor = MagicMock()
        preprocessor.process.side_effect = lambda w, sr: (w, sr)
        extractor = MagicMock()
        extractor.extract.return_value = np.zeros((1, 128, 128), dtype=np.float32)
        prosody = MagicMock()
        prosody.analyze.return_value = mock_prosody_result

        agent = ForensicAgent(
            cnn_detector=mock_cnn_genuine,
            whisper_analyzer=whisper,
            feature_extractor=extractor,
            preprocessor=preprocessor,
            prosody_analyzer=prosody,
            run_parallel=True,
        )
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = agent.run(state)
        assert "cnn_score" in result
        assert "whisper_score" in result

    def test_genuine_scores_propagate_correctly(
        self, forensic_agent: ForensicAgent, tmp_wav_file: str
    ) -> None:
        state: PipelineState = {"audio_path": tmp_wav_file, "stage_latencies": {}}
        result = forensic_agent.run(state)
        # Mock CNN returns genuine_prob=0.92
        assert abs(result["cnn_score"] - 0.92) < 1e-3
        # Mock Whisper returns whisper_score=0.88
        assert abs(result["whisper_score"] - 0.88) < 1e-3