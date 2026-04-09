"""
src.models.prosody_analyzer
=============================
Prosody and rhythm analysis module for deepfake voice detection.

Analyses temporal characteristics of speech that distinguish human speakers
from TTS/voice-cloning systems:

  - **Pitch (F0) variability**: Humans exhibit natural micro-variations in
    fundamental frequency; synthetic speech tends toward smoother, more
    uniform pitch contours.
  - **Speaking rate variability**: Natural speech has variable phone durations
    and pauses; TTS systems produce more uniform timing.
  - **Energy contour dynamics**: Human speech has irregular energy envelopes;
    synthetic speech is often more compressed and uniform.
  - **Pause patterns**: Natural speech contains hesitations, breathing pauses,
    and variable silence durations; synthetic speech has mechanical pause
    placement.
  - **Jitter and shimmer**: Cycle-to-cycle variations in pitch period (jitter)
    and amplitude (shimmer) are present in natural speech but often absent
    or artificially regular in synthetic speech.

Returns a ``prosody_score`` in [0, 1] representing the probability that the
audio exhibits *natural human* prosodic patterns (higher = more likely genuine).

References
----------
Professor feedback: "Beyond phonetic matching, consider how the rhythm of
the voice can detect deepfakes. Does the response have the natural frequency
transitions of a human or the uniformity of an API?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProsodyResult:
    """Result of prosody / rhythm analysis."""

    # Pitch (F0) features
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    pitch_range_hz: float = 0.0
    pitch_cv: float = 0.0  # coefficient of variation

    # Speaking rate features
    syllable_rate: float = 0.0  # estimated syllables/sec
    pause_rate: float = 0.0  # pauses per second
    mean_pause_duration_ms: float = 0.0
    pause_duration_std_ms: float = 0.0

    # Energy dynamics
    energy_std: float = 0.0
    energy_cv: float = 0.0
    energy_dynamic_range_db: float = 0.0

    # Jitter and shimmer (voice quality)
    jitter_percent: float = 0.0  # pitch period perturbation
    shimmer_percent: float = 0.0  # amplitude perturbation

    # Rhythm metrics
    pitch_transition_rate: float = 0.0  # rate of F0 direction changes
    spectral_flux_mean: float = 0.0
    spectral_flux_std: float = 0.0

    # Final score
    prosody_score: float = 0.5  # genuine probability [0, 1]

    # Feature vector for optional ML integration
    feature_vector: List[float] = field(default_factory=list)


class ProsodyAnalyzer:
    """
    Analyse prosodic and rhythmic characteristics of speech to detect
    synthetic audio.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of input audio (Hz).
    frame_length_ms : float
        Analysis frame length in milliseconds.
    hop_length_ms : float
        Hop between frames in milliseconds.
    f0_min : float
        Minimum expected fundamental frequency (Hz).
    f0_max : float
        Maximum expected fundamental frequency (Hz).
    silence_threshold_db : float
        Energy threshold (dB below peak) for silence detection.
    """

    # Empirical thresholds derived from speech analysis literature.
    # Human speech typically has higher variability in these measures
    # compared to TTS systems.
    NATURAL_PITCH_CV_MIN = 0.10  # humans usually > 0.10
    NATURAL_ENERGY_CV_MIN = 0.15  # humans usually > 0.15
    NATURAL_JITTER_MIN = 0.2  # percent; humans ~0.2-1.0%
    NATURAL_JITTER_MAX = 3.0  # very high jitter = pathological
    NATURAL_SHIMMER_MIN = 1.0  # percent; humans ~1-3%
    NATURAL_SHIMMER_MAX = 8.0  # very high shimmer = pathological
    NATURAL_PITCH_TRANSITION_MIN = 0.15  # direction changes per frame
    NATURAL_PAUSE_STD_MIN_MS = 30.0  # humans have variable pause lengths

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        f0_min: float = 75.0,
        f0_max: float = 600.0,
        silence_threshold_db: float = -40.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.silence_threshold_db = silence_threshold_db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, waveform: np.ndarray, sr: int = 16_000) -> ProsodyResult:
        """
        Perform full prosody analysis on a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 waveform.
        sr : int
            Sample rate.

        Returns
        -------
        ProsodyResult
        """
        if len(waveform) < self.frame_length * 2:
            logger.warning("Audio too short for prosody analysis.")
            return ProsodyResult(prosody_score=0.5)

        # Extract features
        pitch_features = self._analyze_pitch(waveform, sr)
        energy_features = self._analyze_energy(waveform)
        pause_features = self._analyze_pauses(waveform, sr)
        voice_quality = self._analyze_jitter_shimmer(waveform, sr)
        rhythm_features = self._analyze_rhythm(waveform, sr)

        # Build result
        result = ProsodyResult(
            **pitch_features,
            **energy_features,
            **pause_features,
            **voice_quality,
            **rhythm_features,
        )

        # Compute prosody score
        result.prosody_score = self._compute_prosody_score(result)

        # Build feature vector for potential ML use
        result.feature_vector = self._build_feature_vector(result)

        logger.debug(
            "ProsodyAnalyzer: pitch_cv=%.3f energy_cv=%.3f jitter=%.2f%% "
            "shimmer=%.2f%% → score=%.3f",
            result.pitch_cv,
            result.energy_cv,
            result.jitter_percent,
            result.shimmer_percent,
            result.prosody_score,
        )
        return result

    # ------------------------------------------------------------------
    # Pitch analysis
    # ------------------------------------------------------------------

    def _analyze_pitch(self, waveform: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract F0 contour and compute pitch statistics."""
        f0 = self._extract_f0(waveform, sr)
        voiced = f0[f0 > 0]  # only voiced frames

        if len(voiced) < 5:
            return {
                "pitch_mean_hz": 0.0,
                "pitch_std_hz": 0.0,
                "pitch_range_hz": 0.0,
                "pitch_cv": 0.0,
            }

        pitch_mean = float(np.mean(voiced))
        pitch_std = float(np.std(voiced))
        pitch_range = float(np.max(voiced) - np.min(voiced))
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0.0

        return {
            "pitch_mean_hz": round(pitch_mean, 2),
            "pitch_std_hz": round(pitch_std, 2),
            "pitch_range_hz": round(pitch_range, 2),
            "pitch_cv": round(pitch_cv, 4),
        }

    def _extract_f0(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract fundamental frequency contour using autocorrelation.

        Returns an array of F0 values per frame (0 = unvoiced).
        """
        n_frames = 1 + (len(waveform) - self.frame_length) // self.hop_length
        f0 = np.zeros(n_frames)

        min_lag = int(sr / self.f0_max)
        max_lag = int(sr / self.f0_min)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]

            if len(frame) < self.frame_length:
                continue

            # Check if frame has enough energy to be voiced
            frame_energy = np.sum(frame ** 2)
            if frame_energy < 1e-6:
                continue

            # Autocorrelation-based pitch detection
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]  # positive lags only

            if max_lag >= len(corr):
                max_lag_safe = len(corr) - 1
            else:
                max_lag_safe = max_lag

            if min_lag >= max_lag_safe:
                continue

            search_region = corr[min_lag : max_lag_safe + 1]
            if len(search_region) == 0:
                continue

            peak_idx = np.argmax(search_region) + min_lag

            # Voiced/unvoiced decision based on normalized autocorrelation
            if corr[0] > 0 and corr[peak_idx] / corr[0] > 0.3:
                f0[i] = sr / peak_idx

        return f0

    # ------------------------------------------------------------------
    # Energy analysis
    # ------------------------------------------------------------------

    def _analyze_energy(self, waveform: np.ndarray) -> Dict[str, float]:
        """Compute frame-level energy statistics."""
        n_frames = 1 + (len(waveform) - self.frame_length) // self.hop_length
        energies = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]
            energies[i] = np.sqrt(np.mean(frame ** 2))  # RMS

        # Filter out silent frames
        voiced_energy = energies[energies > 1e-5]
        if len(voiced_energy) < 3:
            return {
                "energy_std": 0.0,
                "energy_cv": 0.0,
                "energy_dynamic_range_db": 0.0,
            }

        energy_mean = float(np.mean(voiced_energy))
        energy_std = float(np.std(voiced_energy))
        energy_cv = energy_std / energy_mean if energy_mean > 0 else 0.0

        # Dynamic range in dB
        e_max = np.max(voiced_energy)
        e_min = np.min(voiced_energy[voiced_energy > 0])
        dynamic_range_db = 20 * np.log10(e_max / e_min) if e_min > 0 else 0.0

        return {
            "energy_std": round(energy_std, 6),
            "energy_cv": round(energy_cv, 4),
            "energy_dynamic_range_db": round(float(dynamic_range_db), 2),
        }

    # ------------------------------------------------------------------
    # Pause analysis
    # ------------------------------------------------------------------

    def _analyze_pauses(
        self, waveform: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """Detect and characterise silence/pause patterns."""
        n_frames = 1 + (len(waveform) - self.frame_length) // self.hop_length
        frame_energies = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            frame_energies[i] = 20 * np.log10(rms + 1e-10)

        peak_db = np.max(frame_energies)
        silence_mask = frame_energies < (peak_db + self.silence_threshold_db)

        # Find pause segments (consecutive silent frames)
        pause_durations_ms: List[float] = []
        in_pause = False
        pause_start = 0
        frame_duration_ms = self.hop_length / sr * 1000

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                duration = (i - pause_start) * frame_duration_ms
                if duration > 20:  # ignore very short gaps
                    pause_durations_ms.append(duration)

        total_duration_s = len(waveform) / sr
        pause_rate = len(pause_durations_ms) / total_duration_s if total_duration_s > 0 else 0.0

        if pause_durations_ms:
            mean_pause = float(np.mean(pause_durations_ms))
            std_pause = float(np.std(pause_durations_ms))
        else:
            mean_pause = 0.0
            std_pause = 0.0

        return {
            "pause_rate": round(pause_rate, 3),
            "mean_pause_duration_ms": round(mean_pause, 2),
            "pause_duration_std_ms": round(std_pause, 2),
        }

    # ------------------------------------------------------------------
    # Jitter and shimmer (voice quality measures)
    # ------------------------------------------------------------------

    def _analyze_jitter_shimmer(
        self, waveform: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """
        Compute jitter (pitch period perturbation) and shimmer (amplitude
        perturbation) — key indicators of natural vs synthetic voice.

        Human voices naturally exhibit small cycle-to-cycle variations
        (jitter ~0.2-1%, shimmer ~1-3%). TTS systems produce unnaturally
        stable signals with very low jitter/shimmer.
        """
        f0 = self._extract_f0(waveform, sr)
        voiced_f0 = f0[f0 > 0]

        if len(voiced_f0) < 3:
            return {"jitter_percent": 0.0, "shimmer_percent": 0.0}

        # Jitter: relative average perturbation of pitch periods
        periods = 1.0 / voiced_f0  # convert F0 to period
        period_diffs = np.abs(np.diff(periods))
        jitter = float(np.mean(period_diffs) / np.mean(periods) * 100)

        # Shimmer: compute from frame amplitudes at voiced positions
        voiced_indices = np.where(f0 > 0)[0]
        amplitudes = []
        for idx in voiced_indices:
            start = idx * self.hop_length
            end = start + self.frame_length
            if end <= len(waveform):
                amplitudes.append(np.max(np.abs(waveform[start:end])))

        if len(amplitudes) < 3:
            return {
                "jitter_percent": round(jitter, 4),
                "shimmer_percent": 0.0,
            }

        amplitudes = np.array(amplitudes)
        amp_diffs = np.abs(np.diff(amplitudes))
        shimmer = float(np.mean(amp_diffs) / np.mean(amplitudes) * 100)

        return {
            "jitter_percent": round(jitter, 4),
            "shimmer_percent": round(shimmer, 4),
        }

    # ------------------------------------------------------------------
    # Rhythm / spectral flux analysis
    # ------------------------------------------------------------------

    def _analyze_rhythm(
        self, waveform: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """
        Analyse spectral flux and pitch transition patterns.

        Spectral flux measures frame-to-frame spectral change. Natural speech
        has higher and more variable spectral flux compared to TTS output.
        """
        f0 = self._extract_f0(waveform, sr)
        voiced_f0 = f0[f0 > 0]

        # Pitch transition rate: how often F0 changes direction
        if len(voiced_f0) >= 3:
            diffs = np.diff(voiced_f0)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            transition_rate = sign_changes / len(diffs) if len(diffs) > 0 else 0.0
        else:
            transition_rate = 0.0

        # Spectral flux
        n_frames = 1 + (len(waveform) - self.frame_length) // self.hop_length
        spectra = []
        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.frame_length]
            if len(frame) < self.frame_length:
                continue
            spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            spectra.append(spec)

        if len(spectra) < 2:
            return {
                "pitch_transition_rate": round(float(transition_rate), 4),
                "spectral_flux_mean": 0.0,
                "spectral_flux_std": 0.0,
                "syllable_rate": 0.0,
            }

        spectra = np.array(spectra)
        # Normalize each spectrum
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms[norms == 0] = 1
        spectra_norm = spectra / norms

        flux = np.sqrt(np.sum(np.diff(spectra_norm, axis=0) ** 2, axis=1))
        flux_mean = float(np.mean(flux))
        flux_std = float(np.std(flux))

        # Estimate syllable rate from energy envelope peaks
        total_duration_s = len(waveform) / sr
        frame_energies = np.array(
            [np.sqrt(np.mean(spectra[i] ** 2)) for i in range(len(spectra))]
        )
        if len(frame_energies) > 5:
            # Smooth and find peaks
            from scipy.ndimage import uniform_filter1d

            smoothed = uniform_filter1d(frame_energies, size=5)
            # Simple peak detection
            peaks = []
            for j in range(1, len(smoothed) - 1):
                if smoothed[j] > smoothed[j - 1] and smoothed[j] > smoothed[j + 1]:
                    if smoothed[j] > np.mean(smoothed) * 0.5:
                        peaks.append(j)
            syllable_rate = len(peaks) / total_duration_s if total_duration_s > 0 else 0.0
        else:
            syllable_rate = 0.0

        return {
            "pitch_transition_rate": round(float(transition_rate), 4),
            "spectral_flux_mean": round(flux_mean, 6),
            "spectral_flux_std": round(flux_std, 6),
            "syllable_rate": round(syllable_rate, 2),
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_prosody_score(self, result: ProsodyResult) -> float:
        """
        Compute a heuristic genuine-probability score from prosodic features.

        Higher output = more likely genuine (natural human speech).

        The score is based on whether the prosodic features fall within
        ranges typical of natural human speech. TTS systems tend to produce:
          - Lower pitch CV (more uniform F0)
          - Lower energy CV (more compressed dynamics)
          - Lower jitter/shimmer (unnaturally stable voice)
          - Lower pitch transition rate (smoother contours)
          - Lower pause duration variability (mechanical timing)
        """
        score = 0.5  # neutral starting point
        n_signals = 0

        # --- Pitch variability ---
        if result.pitch_cv > 0:
            n_signals += 1
            if result.pitch_cv >= self.NATURAL_PITCH_CV_MIN:
                score += 0.1  # natural range
            else:
                score -= 0.15  # too uniform → likely synthetic

        # --- Energy variability ---
        if result.energy_cv > 0:
            n_signals += 1
            if result.energy_cv >= self.NATURAL_ENERGY_CV_MIN:
                score += 0.08
            else:
                score -= 0.12  # too uniform → likely synthetic

        # --- Jitter ---
        if result.jitter_percent > 0:
            n_signals += 1
            if self.NATURAL_JITTER_MIN <= result.jitter_percent <= self.NATURAL_JITTER_MAX:
                score += 0.1  # healthy human range
            elif result.jitter_percent < self.NATURAL_JITTER_MIN:
                score -= 0.15  # too stable → synthetic
            else:
                score -= 0.05  # excessive jitter → possibly degraded

        # --- Shimmer ---
        if result.shimmer_percent > 0:
            n_signals += 1
            if self.NATURAL_SHIMMER_MIN <= result.shimmer_percent <= self.NATURAL_SHIMMER_MAX:
                score += 0.1  # healthy human range
            elif result.shimmer_percent < self.NATURAL_SHIMMER_MIN:
                score -= 0.12  # too stable → synthetic
            else:
                score -= 0.05  # excessive shimmer

        # --- Pitch transitions ---
        if result.pitch_transition_rate > 0:
            n_signals += 1
            if result.pitch_transition_rate >= self.NATURAL_PITCH_TRANSITION_MIN:
                score += 0.08  # natural frequency transitions
            else:
                score -= 0.1  # too smooth → "uniformity of an API"

        # --- Pause variability ---
        if result.pause_duration_std_ms > 0:
            n_signals += 1
            if result.pause_duration_std_ms >= self.NATURAL_PAUSE_STD_MIN_MS:
                score += 0.06  # variable pauses = natural
            else:
                score -= 0.08  # mechanical pause timing

        # --- Spectral flux variability ---
        if result.spectral_flux_std > 0:
            n_signals += 1
            # Higher flux variability = more natural speech dynamics
            if result.spectral_flux_std > 0.05:
                score += 0.05
            else:
                score -= 0.05

        return float(max(0.0, min(1.0, score)))

    def _build_feature_vector(self, result: ProsodyResult) -> List[float]:
        """Build a flat feature vector for potential ML-based scoring."""
        return [
            result.pitch_mean_hz,
            result.pitch_std_hz,
            result.pitch_range_hz,
            result.pitch_cv,
            result.syllable_rate,
            result.pause_rate,
            result.mean_pause_duration_ms,
            result.pause_duration_std_ms,
            result.energy_std,
            result.energy_cv,
            result.energy_dynamic_range_db,
            result.jitter_percent,
            result.shimmer_percent,
            result.pitch_transition_rate,
            result.spectral_flux_mean,
            result.spectral_flux_std,
        ]
