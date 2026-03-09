"""
src.data.feature_extractor
===========================
Audio feature extraction utilities for CNN-based deepfake detection.

Supported feature types:
  - Mel-spectrogram (log-magnitude)
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - LFCC (Linear-Frequency Cepstral Coefficients)
  - Combined stack (mel + mfcc + lfcc)

All features are returned as NumPy arrays with shape
(n_channels, n_bins, n_frames) suitable for direct use as CNN input.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np


class FeatureType(str, Enum):
    """Supported feature extraction modes."""

    MEL_SPECTROGRAM = "mel_spectrogram"
    MFCC = "mfcc"
    LFCC = "lfcc"
    COMBINED = "combined"


class FeatureExtractor:
    """
    Extract audio features from a pre-processed waveform.

    Parameters
    ----------
    feature_type : FeatureType | str
        Which feature(s) to extract.
    sample_rate : int
        Sample rate of the input waveform (Hz).
    n_mels : int
        Number of mel filter-bank bins.
    n_mfcc : int
        Number of MFCC coefficients.
    n_lfcc : int
        Number of LFCC coefficients.
    n_fft : int
        FFT window size (samples).
    hop_length : int
        Hop length between frames (samples).
    win_length : int | None
        Window length (samples); defaults to n_fft.
    f_min : float
        Minimum frequency for mel filter-bank (Hz).
    f_max : float | None
        Maximum frequency for mel filter-bank (Hz); defaults to sr/2.
    """

    def __init__(
        self,
        feature_type: FeatureType | str = FeatureType.MEL_SPECTROGRAM,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_lfcc: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        self.feature_type = FeatureType(feature_type)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract the configured feature from a 1-D waveform.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 waveform array.

        Returns
        -------
        np.ndarray
            Feature tensor with shape (1, n_bins, n_frames).
        """
        dispatch = {
            FeatureType.MEL_SPECTROGRAM: self._mel_spectrogram,
            FeatureType.MFCC: self._mfcc,
            FeatureType.LFCC: self._lfcc,
            FeatureType.COMBINED: self._combined,
        }
        feature = dispatch[self.feature_type](waveform)
        # Ensure shape is (channels, bins, frames)
        if feature.ndim == 2:
            feature = feature[np.newaxis, ...]
        return feature.astype(np.float32)

    # ------------------------------------------------------------------
    # Private feature implementations
    # ------------------------------------------------------------------

    def _mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute log-magnitude mel-spectrogram."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel  # shape: (n_mels, n_frames)

    def _mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Compute MFCC features."""
        import librosa

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        return mfcc  # shape: (n_mfcc, n_frames)

    def _lfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute LFCC (Linear-Frequency Cepstral Coefficients).

        Uses a linear filter-bank instead of the mel scale.
        """
        import librosa

        # Linear-frequency spectrogram
        stft = np.abs(
            librosa.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )
        )
        # Sum across linear-spaced frequency bins
        n_linear_filters = self.n_lfcc * 2
        filter_bank = np.linspace(0, stft.shape[0], n_linear_filters + 2).astype(int)
        linear_spec = np.stack(
            [
                stft[filter_bank[i] : filter_bank[i + 2]].mean(axis=0)
                for i in range(n_linear_filters)
            ]
        )
        log_linear = np.log1p(linear_spec)
        # Apply DCT to get cepstral coefficients
        from scipy.fft import dct

        lfcc = dct(log_linear, type=2, axis=0, norm="ortho")[: self.n_lfcc]
        return lfcc  # shape: (n_lfcc, n_frames)

    def _combined(self, waveform: np.ndarray) -> np.ndarray:
        """
        Concatenate mel-spectrogram, MFCC, and LFCC along the frequency axis.
        """
        mel = self._mel_spectrogram(waveform)  # (n_mels, T)
        mfcc = self._mfcc(waveform)  # (n_mfcc, T)
        lfcc = self._lfcc(waveform)  # (n_lfcc, T)

        # Align time dimension (truncate to shortest)
        min_t = min(mel.shape[1], mfcc.shape[1], lfcc.shape[1])
        combined = np.concatenate(
            [mel[:, :min_t], mfcc[:, :min_t], lfcc[:, :min_t]], axis=0
        )
        return combined  # shape: (n_mels + n_mfcc + n_lfcc, T)
