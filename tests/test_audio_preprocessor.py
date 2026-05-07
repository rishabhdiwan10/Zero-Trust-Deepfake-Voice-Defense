"""
Tests for src.data.audio_preprocessor — waveform preprocessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.audio_preprocessor import AudioPreprocessor


@pytest.fixture()
def preprocessor() -> AudioPreprocessor:
    return AudioPreprocessor(target_sr=16_000, normalize=True, trim_silence=True)


@pytest.fixture()
def stereo_waveform() -> np.ndarray:
    """2-channel (stereo) waveform."""
    return np.random.randn(2, 16_000).astype(np.float32)


@pytest.fixture()
def silent_waveform() -> np.ndarray:
    return np.zeros(16_000, dtype=np.float32)


class TestAudioPreprocessor:
    def test_output_is_mono(self, preprocessor: AudioPreprocessor, stereo_waveform: np.ndarray) -> None:
        out, _ = preprocessor.process(stereo_waveform, 16_000)
        assert out.ndim == 1

    def test_resamples_to_target_sr(self, preprocessor: AudioPreprocessor, sine_waveform: np.ndarray) -> None:
        # Feed 8 kHz audio — should resample to 16 kHz
        out, sr = preprocessor.process(sine_waveform[:8_000], 8_000)
        assert sr == 16_000

    def test_normalize_peak_is_one(self) -> None:
        proc = AudioPreprocessor(normalize=True, trim_silence=False)
        loud = np.full(1_000, 5.0, dtype=np.float32)
        out, _ = proc.process(loud, 16_000)
        assert abs(np.max(np.abs(out)) - 1.0) < 1e-5

    def test_silent_waveform_does_not_crash(
        self, preprocessor: AudioPreprocessor, silent_waveform: np.ndarray
    ) -> None:
        out, _ = preprocessor.process(silent_waveform, 16_000)
        assert out is not None

    def test_output_dtype_is_float32(
        self, preprocessor: AudioPreprocessor, sine_waveform: np.ndarray
    ) -> None:
        out, _ = preprocessor.process(sine_waveform, 16_000)
        assert out.dtype == np.float32

    def test_target_duration_pads_short_audio(self) -> None:
        proc = AudioPreprocessor(target_duration=2.0, trim_silence=False, normalize=False)
        short = np.zeros(8_000, dtype=np.float32)  # 0.5s at 16kHz
        out, _ = proc.process(short, 16_000)
        assert len(out) == 32_000  # 2s × 16kHz

    def test_target_duration_truncates_long_audio(self) -> None:
        proc = AudioPreprocessor(target_duration=1.0, trim_silence=False, normalize=False)
        long_wav = np.zeros(32_000, dtype=np.float32)  # 2s at 16kHz
        out, _ = proc.process(long_wav, 16_000)
        assert len(out) == 16_000

    def test_no_resampling_when_sr_matches(
        self, preprocessor: AudioPreprocessor, sine_waveform: np.ndarray
    ) -> None:
        original_len = len(sine_waveform)
        out, sr = preprocessor.process(sine_waveform, 16_000)
        assert sr == 16_000
        # Length may differ due to silence trimming but should be ≤ original
        assert len(out) <= original_len