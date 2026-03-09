"""
src.utils.audio_utils
======================
Audio I/O helper functions for loading, saving, and converting audio files
in the Zero-Trust Deepfake Voice Defense System.

Wraps ``soundfile`` and ``librosa`` to provide a consistent interface
regardless of file format (WAV, FLAC, MP3, OGG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Supported input formats
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio(
    path: str | Path,
    target_sr: int = 16_000,
    mono: bool = True,
    dtype: str = "float32",
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return a normalised waveform.

    Parameters
    ----------
    path : str | Path
        Path to the audio file.
    target_sr : int
        Target sample rate. Audio is resampled if the file SR differs.
    mono : bool
        Convert to mono if ``True``.
    dtype : str
        Output numpy dtype (``"float32"`` or ``"float64"``).

    Returns
    -------
    Tuple[np.ndarray, int]
        (waveform, sample_rate) — waveform shape is (n_samples,) for mono.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    try:
        import soundfile as sf

        waveform, sr = sf.read(str(path), dtype=dtype, always_2d=True)
        # soundfile returns (samples, channels) — transpose to (channels, samples)
        waveform = waveform.T
    except Exception:
        # Fallback to librosa (handles mp3, m4a via ffmpeg)
        import librosa

        waveform, sr = librosa.load(str(path), sr=None, mono=False, dtype=dtype)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
    elif mono:
        waveform = waveform.squeeze(0)

    if sr != target_sr:
        waveform = _resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform.astype(dtype), sr


def save_audio(
    waveform: np.ndarray,
    path: str | Path,
    sample_rate: int = 16_000,
    subtype: Optional[str] = None,
) -> Path:
    """
    Save a waveform to an audio file.

    Parameters
    ----------
    waveform : np.ndarray
        1-D or 2-D (channels, samples) waveform array.
    path : str | Path
        Output file path. Format is inferred from the extension.
    sample_rate : int
        Sample rate of the waveform.
    subtype : str | None
        soundfile subtype (e.g. ``"PCM_16"``). Auto-selected if ``None``.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.ndim == 2:
        # (channels, samples) → (samples, channels)
        data = waveform.T
    else:
        data = waveform

    kwargs = {}
    if subtype:
        kwargs["subtype"] = subtype

    sf.write(str(path), data, sample_rate, **kwargs)
    logger.debug("Audio saved → %s (sr=%d)", path, sample_rate)
    return path.resolve()


def get_duration(path: str | Path) -> float:
    """Return the duration of an audio file in seconds."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.duration
    except Exception as exc:
        logger.warning("Could not get duration for %s: %s", path, exc)
        return 0.0


def compute_rms(waveform: np.ndarray) -> float:
    """Compute the root-mean-square energy of a waveform."""
    return float(np.sqrt(np.mean(waveform ** 2)))


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample using scipy or fall back to linear interpolation."""
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)
        return resample_poly(waveform, target_sr // g, orig_sr // g).astype(
            waveform.dtype
        )
    except ImportError:
        dur = len(waveform) / orig_sr
        new_length = int(dur * target_sr)
        return np.interp(
            np.linspace(0, len(waveform) - 1, new_length),
            np.arange(len(waveform)),
            waveform,
        ).astype(waveform.dtype)
