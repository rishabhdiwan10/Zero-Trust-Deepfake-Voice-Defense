"""
tests/conftest.py
==================
Shared pytest fixtures for all test modules.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf


# ─────────────────────────────────────────────────────────────────────────────
# Audio fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_rate() -> int:
    return 16_000


@pytest.fixture(scope="session")
def sine_waveform(sample_rate: int) -> np.ndarray:
    """1-second 440 Hz sine wave at 16 kHz."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture()
def tmp_wav_file(sine_waveform: np.ndarray, sample_rate: int) -> Generator[str, None, None]:
    """Write a sine wave to a temp .wav file and yield its path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, sine_waveform, sample_rate)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def mel_feature() -> np.ndarray:
    """Random mel-spectrogram feature (1, 128, 128)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, 128, 128)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Mock model fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_cnn_genuine():
    """CNN detector that always returns genuine."""
    m = MagicMock()
    m.predict.return_value = {"genuine_prob": 0.92, "synthetic_prob": 0.08, "prediction": 0}
    return m


@pytest.fixture()
def mock_cnn_synthetic():
    """CNN detector that always returns synthetic."""
    m = MagicMock()
    m.predict.return_value = {"genuine_prob": 0.08, "synthetic_prob": 0.92, "prediction": 1}
    return m


@pytest.fixture()
def mock_cnn_uncertain():
    """CNN detector that returns uncertain score."""
    m = MagicMock()
    m.predict.return_value = {"genuine_prob": 0.55, "synthetic_prob": 0.45, "prediction": 0}
    return m


@pytest.fixture()
def mock_whisper_result_genuine():
    """Whisper result object for genuine audio."""
    r = MagicMock()
    r.whisper_score    = 0.88
    r.transcription    = "hello this is a test"
    r.avg_log_prob     = -0.3
    r.no_speech_prob   = 0.02
    r.compression_ratio = 1.1
    r.language         = "en"
    return r


@pytest.fixture()
def mock_whisper_result_synthetic():
    """Whisper result object for synthetic audio."""
    r = MagicMock()
    r.whisper_score    = 0.12
    r.transcription    = "synthesized speech"
    r.avg_log_prob     = -0.9
    r.no_speech_prob   = 0.25
    r.compression_ratio = 0.7
    r.language         = "en"
    return r


@pytest.fixture()
def mock_prosody_result():
    """Prosody result object."""
    r = MagicMock()
    r.prosody_score        = 0.75
    r.pitch_cv             = 0.15
    r.jitter_percent       = 0.8
    r.shimmer_percent      = 2.1
    r.pitch_transition_rate = 0.4
    return r