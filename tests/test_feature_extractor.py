"""
Tests for src.data.feature_extractor — audio feature extraction.
"""

import numpy as np
import pytest

from src.data.feature_extractor import FeatureExtractor, FeatureType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_waveform() -> np.ndarray:
    """Generate a 1-second synthetic sine wave at 16 kHz."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    def test_mel_spectrogram_shape(self, sample_waveform: np.ndarray) -> None:
        extractor = FeatureExtractor(
            feature_type=FeatureType.MEL_SPECTROGRAM,
            sample_rate=16_000,
            n_mels=128,
        )
        feature = extractor.extract(sample_waveform)
        assert feature.ndim == 3, "Expected (channels, bins, frames)"
        assert feature.shape[1] == 128, "Expected 128 mel bins"

    def test_mfcc_shape(self, sample_waveform: np.ndarray) -> None:
        extractor = FeatureExtractor(
            feature_type=FeatureType.MFCC,
            sample_rate=16_000,
            n_mfcc=40,
        )
        feature = extractor.extract(sample_waveform)
        assert feature.ndim == 3
        assert feature.shape[1] == 40

    def test_lfcc_shape(self, sample_waveform: np.ndarray) -> None:
        extractor = FeatureExtractor(
            feature_type=FeatureType.LFCC,
            sample_rate=16_000,
            n_lfcc=40,
        )
        feature = extractor.extract(sample_waveform)
        assert feature.ndim == 3
        assert feature.shape[1] == 40

    def test_combined_shape(self, sample_waveform: np.ndarray) -> None:
        extractor = FeatureExtractor(
            feature_type=FeatureType.COMBINED,
            sample_rate=16_000,
            n_mels=128,
            n_mfcc=40,
            n_lfcc=40,
        )
        feature = extractor.extract(sample_waveform)
        assert feature.ndim == 3
        # Combined = mel(128) + mfcc(40) + lfcc(40) = 208
        assert feature.shape[1] == 208

    def test_output_dtype(self, sample_waveform: np.ndarray) -> None:
        extractor = FeatureExtractor(feature_type=FeatureType.MEL_SPECTROGRAM)
        feature = extractor.extract(sample_waveform)
        assert feature.dtype == np.float32

    def test_string_feature_type(self, sample_waveform: np.ndarray) -> None:
        extractor = FeatureExtractor(feature_type="mel_spectrogram")
        feature = extractor.extract(sample_waveform)
        assert feature.ndim == 3

    def test_invalid_feature_type_raises(self) -> None:
        with pytest.raises(ValueError):
            FeatureExtractor(feature_type="invalid_type")
