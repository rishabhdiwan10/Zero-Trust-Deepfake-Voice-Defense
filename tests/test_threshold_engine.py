"""
Tests for src.decision.threshold_engine — decision thresholds and
poor-quality vs. synthetic audio differentiation logic.
"""

import pytest

from src.decision.threshold_engine import Decision, ThresholdEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> ThresholdEngine:
    return ThresholdEngine(
        pass_threshold=0.80,
        challenge_threshold=0.40,
        enable_quality_differentiation=True,
    )


# ---------------------------------------------------------------------------
# Tests — basic threshold logic
# ---------------------------------------------------------------------------


class TestThresholdEngine:
    def test_pass_above_pass_threshold(self, engine: ThresholdEngine) -> None:
        assert engine.evaluate(0.90) == "pass"

    def test_pass_at_exact_pass_threshold(self, engine: ThresholdEngine) -> None:
        assert engine.evaluate(0.80) == "pass"

    def test_challenge_in_band(self, engine: ThresholdEngine) -> None:
        assert engine.evaluate(0.60) == "challenge"

    def test_challenge_at_exact_lower_boundary(
        self, engine: ThresholdEngine
    ) -> None:
        assert engine.evaluate(0.40) == "challenge"

    def test_reject_below_challenge_threshold(
        self, engine: ThresholdEngine
    ) -> None:
        assert engine.evaluate(0.20) == "reject"

    def test_reject_at_zero(self, engine: ThresholdEngine) -> None:
        assert engine.evaluate(0.0) == "reject"

    def test_pass_at_one(self, engine: ThresholdEngine) -> None:
        assert engine.evaluate(1.0) == "pass"


# ---------------------------------------------------------------------------
# Tests — quality differentiation
# ---------------------------------------------------------------------------


class TestQualityDifferentiation:
    def test_poor_snr_escalates_reject_to_challenge(self) -> None:
        engine = ThresholdEngine(
            pass_threshold=0.80,
            challenge_threshold=0.40,
            enable_quality_differentiation=True,
            low_snr_threshold_db=10.0,
            poor_quality_cnn_score_max=0.60,
        )
        # Score below challenge threshold, but audio has poor SNR and low CNN score
        result = engine.evaluate(
            trust_score=0.20,
            cnn_score=0.45,
            audio_metadata={"snr_db": 5.0},  # poor SNR
        )
        assert result == "challenge"

    def test_high_snr_does_not_escalate(self) -> None:
        engine = ThresholdEngine(
            pass_threshold=0.80,
            challenge_threshold=0.40,
            enable_quality_differentiation=True,
            low_snr_threshold_db=10.0,
            poor_quality_cnn_score_max=0.60,
        )
        result = engine.evaluate(
            trust_score=0.20,
            cnn_score=0.45,
            audio_metadata={"snr_db": 25.0},  # good SNR
        )
        assert result == "reject"

    def test_poor_energy_escalates_reject(self) -> None:
        engine = ThresholdEngine(
            enable_quality_differentiation=True,
            low_energy_threshold=0.01,
            poor_quality_cnn_score_max=0.60,
        )
        result = engine.evaluate(
            trust_score=0.15,
            cnn_score=0.50,
            audio_metadata={"rms_energy": 0.005},  # very low energy
        )
        assert result == "challenge"

    def test_quality_diff_disabled_keeps_reject(self) -> None:
        engine = ThresholdEngine(
            enable_quality_differentiation=False,
        )
        result = engine.evaluate(
            trust_score=0.10,
            cnn_score=0.45,
            audio_metadata={"snr_db": 5.0},
        )
        assert result == "reject"

    def test_high_cnn_score_not_escalated(self) -> None:
        """If CNN score is too high (not poor quality indicator), don't escalate."""
        engine = ThresholdEngine(
            enable_quality_differentiation=True,
            poor_quality_cnn_score_max=0.60,
        )
        result = engine.evaluate(
            trust_score=0.20,
            cnn_score=0.85,  # high CNN score — synthetic, not poor quality
            audio_metadata={"snr_db": 5.0},
        )
        assert result == "reject"


# ---------------------------------------------------------------------------
# Tests — configuration validation
# ---------------------------------------------------------------------------


class TestThresholdEngineConfig:
    def test_invalid_thresholds_raises(self) -> None:
        with pytest.raises(ValueError):
            ThresholdEngine(pass_threshold=0.30, challenge_threshold=0.80)

    def test_get_thresholds(self, engine: ThresholdEngine) -> None:
        t = engine.get_thresholds()
        assert t["pass_threshold"] == 0.80
        assert t["challenge_threshold"] == 0.40
