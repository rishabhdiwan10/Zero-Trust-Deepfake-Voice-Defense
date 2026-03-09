"""
src.decision.threshold_engine
==============================
Confidence threshold evaluation engine for the Zero-Trust Deepfake Voice
Defense System.

Decision logic
--------------
Given a ``trust_score`` in [0, 1]:

  trust_score ≥ pass_threshold                        → PASS
  challenge_threshold ≤ trust_score < pass_threshold  → CHALLENGE
  trust_score < challenge_threshold                   → REJECT (default)

Quality differentiation
-----------------------
When audio quality indicators are poor (low SNR, low energy) AND the CNN
score is below a soft ceiling, the system escalates to CHALLENGE instead of
REJECT.  This prevents false rejections of genuine speakers in noisy
environments.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    """Possible pipeline decisions."""

    PASS = "pass"
    CHALLENGE = "challenge"
    REJECT = "reject"


class ThresholdEngine:
    """
    Evaluate the aggregated trust score against configured thresholds.

    Parameters
    ----------
    pass_threshold : float
        Minimum trust score to unconditionally pass (default: 0.80).
    challenge_threshold : float
        Minimum trust score to trigger a liveness challenge (default: 0.40).
    enable_quality_differentiation : bool
        Whether to apply poor-quality vs. synthetic differentiation logic.
    low_snr_threshold_db : float
        SNR (dB) below which audio is considered poor quality.
    low_energy_threshold : float
        RMS energy below which audio is considered low quality.
    poor_quality_cnn_score_max : float
        If quality is poor and CNN score is below this, escalate to CHALLENGE
        instead of REJECT.
    """

    def __init__(
        self,
        pass_threshold: float = 0.80,
        challenge_threshold: float = 0.40,
        enable_quality_differentiation: bool = True,
        low_snr_threshold_db: float = 10.0,
        low_energy_threshold: float = 0.01,
        poor_quality_cnn_score_max: float = 0.60,
    ) -> None:
        if not (0.0 <= challenge_threshold < pass_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: "
                "0 ≤ challenge_threshold < pass_threshold ≤ 1"
            )
        self.pass_threshold = pass_threshold
        self.challenge_threshold = challenge_threshold
        self.enable_quality_differentiation = enable_quality_differentiation
        self.low_snr_threshold_db = low_snr_threshold_db
        self.low_energy_threshold = low_energy_threshold
        self.poor_quality_cnn_score_max = poor_quality_cnn_score_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trust_score: float,
        cnn_score: Optional[float] = None,
        audio_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Evaluate a trust score and return a decision string.

        Parameters
        ----------
        trust_score : float
            Aggregated trust score in [0, 1].
        cnn_score : float | None
            Raw CNN genuine probability (used for quality differentiation).
        audio_metadata : dict | None
            Dict that may contain ``snr_db`` and ``rms_energy`` fields.

        Returns
        -------
        str
            ``"pass"`` | ``"challenge"`` | ``"reject"``
        """
        audio_metadata = audio_metadata or {}

        if trust_score >= self.pass_threshold:
            decision = Decision.PASS
        elif trust_score >= self.challenge_threshold:
            decision = Decision.CHALLENGE
        else:
            # Below reject threshold — but check quality differentiation
            decision = self._apply_quality_differentiation(
                trust_score=trust_score,
                cnn_score=cnn_score,
                audio_metadata=audio_metadata,
            )

        logger.debug(
            "ThresholdEngine: trust=%.3f → %s (pass≥%.2f, challenge≥%.2f)",
            trust_score,
            decision.value,
            self.pass_threshold,
            self.challenge_threshold,
        )
        return decision.value

    def get_thresholds(self) -> Dict[str, float]:
        """Return the configured threshold values."""
        return {
            "pass_threshold": self.pass_threshold,
            "challenge_threshold": self.challenge_threshold,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_quality_differentiation(
        self,
        trust_score: float,
        cnn_score: Optional[float],
        audio_metadata: Dict[str, Any],
    ) -> Decision:
        """
        Differentiate between poor-quality genuine audio and synthetic speech.

        If audio quality indicators suggest the signal is degraded (low SNR,
        low energy) rather than synthetic, escalate to CHALLENGE instead of
        hard REJECT.
        """
        if not self.enable_quality_differentiation:
            return Decision.REJECT

        snr_db = audio_metadata.get("snr_db")
        rms_energy = audio_metadata.get("rms_energy")
        is_poor_quality = False

        if snr_db is not None and snr_db < self.low_snr_threshold_db:
            is_poor_quality = True
            logger.debug("Poor audio quality detected: SNR=%.1f dB", snr_db)

        if rms_energy is not None and rms_energy < self.low_energy_threshold:
            is_poor_quality = True
            logger.debug("Poor audio quality detected: RMS energy=%.4f", rms_energy)

        if (
            is_poor_quality
            and cnn_score is not None
            and cnn_score < self.poor_quality_cnn_score_max
        ):
            logger.info(
                "Quality differentiation: escalating REJECT → CHALLENGE "
                "(poor quality audio, cnn_score=%.3f)",
                cnn_score,
            )
            return Decision.CHALLENGE

        return Decision.REJECT
