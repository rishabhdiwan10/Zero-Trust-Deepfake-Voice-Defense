"""
src.data.audio_preprocessor
============================
Audio preprocessing utilities for the deepfake detection pipeline.

Responsibilities:
  - Resampling to a target sample rate
  - Mono conversion
  - Amplitude normalisation
  - Silence / leading-trailing noise trimming
  - Zero-padding or truncation to a fixed duration
  - Optional data augmentation (additive noise, time-stretch, pitch-shift)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class AudioPreprocessor:
    """
    Stateless audio preprocessing helper.

    Parameters
    ----------
    target_sr : int
        Target sample rate in Hz (default: 16 000).
    target_duration : float | None
        If set, clips are padded or truncated to this duration (seconds).
    normalize : bool
        Whether to peak-normalise the waveform to [-1, 1].
    trim_silence : bool
        Whether to trim leading/trailing silence.
    trim_top_db : float
        Silence threshold in dB for trimming (default: 30).
    augment : bool
        Whether to apply random augmentation during training.
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        target_duration: Optional[float] = None,
        normalize: bool = True,
        trim_silence: bool = True,
        trim_top_db: float = 30.0,
        augment: bool = False,
    ) -> None:
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.augment = augment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Apply the full preprocessing chain to a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            Raw audio waveform, shape (samples,) or (channels, samples).
        sr : int
            Original sample rate of the waveform.

        Returns
        -------
        Tuple[np.ndarray, int]
            Preprocessed waveform and the target sample rate.
        """
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr, self.target_sr)

        if self.trim_silence:
            waveform = self._trim_silence(waveform, self.trim_top_db)

        if self.normalize:
            waveform = self._normalize(waveform)

        if self.target_duration is not None:
            waveform = self._fix_length(waveform, self.target_sr, self.target_duration)

        if self.augment:
            waveform = self._augment(waveform, self.target_sr)

        return waveform, self.target_sr

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging channels."""
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        return waveform

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample waveform from orig_sr to target_sr using scipy."""
        if orig_sr == target_sr:
            return waveform
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(orig_sr, target_sr)
            waveform = resample_poly(waveform, target_sr // g, orig_sr // g)
        except ImportError:
            # Fallback: linear interpolation (lower quality)
            duration = len(waveform) / orig_sr
            new_length = int(duration * target_sr)
            waveform = np.interp(
                np.linspace(0, len(waveform) - 1, new_length),
                np.arange(len(waveform)),
                waveform,
            )
        return waveform.astype(np.float32)

    @staticmethod
    def _trim_silence(waveform: np.ndarray, top_db: float) -> np.ndarray:
        """Trim leading and trailing silence."""
        try:
            import librosa

            waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
        except ImportError:
            # Fallback: simple energy-based trim
            threshold = 10 ** (-top_db / 20.0)
            above = np.where(np.abs(waveform) > threshold)[0]
            if len(above):
                waveform = waveform[above[0] : above[-1] + 1]
        return waveform

    @staticmethod
    def _normalize(waveform: np.ndarray) -> np.ndarray:
        """Peak-normalise waveform to [-1, 1]."""
        peak = np.max(np.abs(waveform))
        if peak > 1e-8:
            waveform = waveform / peak
        return waveform.astype(np.float32)

    @staticmethod
    def _fix_length(
        waveform: np.ndarray, sr: int, duration: float
    ) -> np.ndarray:
        """Pad or truncate waveform to exactly `duration` seconds."""
        target_len = int(sr * duration)
        current_len = len(waveform)
        if current_len < target_len:
            pad = target_len - current_len
            waveform = np.pad(waveform, (0, pad), mode="constant")
        elif current_len > target_len:
            waveform = waveform[:target_len]
        return waveform

    @staticmethod
    def _augment(waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply random training-time augmentation.

        Augmentations applied with random probability:
          - Additive Gaussian noise
          - Random gain variation
        """
        rng = np.random.default_rng()

        # Additive noise (SNR ~20 dB)
        if rng.random() < 0.5:
            noise = rng.normal(0, 0.005, size=waveform.shape).astype(np.float32)
            waveform = waveform + noise

        # Random gain [0.8, 1.2]
        if rng.random() < 0.5:
            gain = rng.uniform(0.8, 1.2)
            waveform = waveform * gain

        # Clip to [-1, 1]
        waveform = np.clip(waveform, -1.0, 1.0)
        return waveform
