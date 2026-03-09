"""
src.decision.trust_scorer
==========================
Aggregates per-layer forensic scores into a unified trust score.

The trust score is a weighted linear combination of:
  - ``cnn_score``      — CNN genuine probability
  - ``whisper_score``  — Whisper-derived genuine probability
  - ``liveness_score`` — bonus/penalty based on liveness challenge outcome

Weights are configurable (see ``configs/thresholds_config.yaml``).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TrustScorer:
    """
    Multi-layer trust score aggregator.

    Parameters
    ----------
    cnn_weight : float
        Weight for the CNN score (default: 0.50).
    whisper_weight : float
        Weight for the Whisper score (default: 0.30).
    liveness_weight : float
        Weight applied to the liveness bonus/penalty (default: 0.20).
    liveness_pass_bonus : float
        Score bonus when liveness challenge is passed [0, 1].
    liveness_fail_penalty : float
        Score penalty when liveness challenge is failed [0, 1].
    """

    def __init__(
        self,
        cnn_weight: float = 0.50,
        whisper_weight: float = 0.30,
        liveness_weight: float = 0.20,
        liveness_pass_bonus: float = 1.0,
        liveness_fail_penalty: float = 0.0,
    ) -> None:
        total_weight = cnn_weight + whisper_weight + liveness_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.4f}"
            )
        self.cnn_weight = cnn_weight
        self.whisper_weight = whisper_weight
        self.liveness_weight = liveness_weight
        self.liveness_pass_bonus = liveness_pass_bonus
        self.liveness_fail_penalty = liveness_fail_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        cnn_score: float,
        whisper_score: float,
        liveness_passed: Optional[bool] = None,
    ) -> float:
        """
        Compute the unified trust score.

        Parameters
        ----------
        cnn_score : float
            CNN genuine probability in [0, 1].
        whisper_score : float
            Whisper genuine probability in [0, 1].
        liveness_passed : bool | None
            ``True`` = passed, ``False`` = failed, ``None`` = not attempted.

        Returns
        -------
        float
            Aggregated trust score in [0, 1].
        """
        cnn_score = float(max(0.0, min(1.0, cnn_score)))
        whisper_score = float(max(0.0, min(1.0, whisper_score)))

        # Liveness component
        if liveness_passed is True:
            liveness_score = self.liveness_pass_bonus
        elif liveness_passed is False:
            liveness_score = self.liveness_fail_penalty
        else:
            # Not yet attempted — use average of forensic scores as neutral estimate
            liveness_score = (cnn_score + whisper_score) / 2.0

        trust = (
            self.cnn_weight * cnn_score
            + self.whisper_weight * whisper_score
            + self.liveness_weight * liveness_score
        )
        trust = float(max(0.0, min(1.0, trust)))

        logger.debug(
            "TrustScorer: cnn=%.3f whisper=%.3f liveness=%.3f → trust=%.3f",
            cnn_score,
            whisper_score,
            liveness_score,
            trust,
        )
        return trust

    def breakdown(
        self,
        cnn_score: float,
        whisper_score: float,
        liveness_passed: Optional[bool] = None,
    ) -> dict:
        """Return a detailed score breakdown alongside the final trust score."""
        return {
            "cnn_contribution": round(self.cnn_weight * cnn_score, 4),
            "whisper_contribution": round(self.whisper_weight * whisper_score, 4),
            "liveness_contribution": round(
                self.liveness_weight
                * (
                    self.liveness_pass_bonus
                    if liveness_passed is True
                    else self.liveness_fail_penalty
                    if liveness_passed is False
                    else (cnn_score + whisper_score) / 2.0
                ),
                4,
            ),
            "trust_score": round(
                self.score(cnn_score, whisper_score, liveness_passed), 4
            ),
        }
