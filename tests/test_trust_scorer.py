"""
Tests for src.decision.trust_scorer — multi-layer trust score aggregation.
"""

import pytest

from src.decision.trust_scorer import TrustScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scorer() -> TrustScorer:
    return TrustScorer(
        cnn_weight=0.50,
        whisper_weight=0.30,
        liveness_weight=0.20,
        liveness_pass_bonus=1.0,
        liveness_fail_penalty=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrustScorer:
    def test_score_with_liveness_passed(self, scorer: TrustScorer) -> None:
        trust = scorer.score(cnn_score=0.9, whisper_score=0.8, liveness_passed=True)
        expected = 0.50 * 0.9 + 0.30 * 0.8 + 0.20 * 1.0
        assert abs(trust - expected) < 1e-5

    def test_score_with_liveness_failed(self, scorer: TrustScorer) -> None:
        trust = scorer.score(cnn_score=0.9, whisper_score=0.8, liveness_passed=False)
        expected = 0.50 * 0.9 + 0.30 * 0.8 + 0.20 * 0.0
        assert abs(trust - expected) < 1e-5

    def test_score_without_liveness(self, scorer: TrustScorer) -> None:
        trust = scorer.score(cnn_score=0.8, whisper_score=0.6, liveness_passed=None)
        liveness_neutral = (0.8 + 0.6) / 2.0
        expected = 0.50 * 0.8 + 0.30 * 0.6 + 0.20 * liveness_neutral
        assert abs(trust - expected) < 1e-5

    def test_score_clipped_to_one(self, scorer: TrustScorer) -> None:
        trust = scorer.score(cnn_score=1.0, whisper_score=1.0, liveness_passed=True)
        assert trust == 1.0

    def test_score_clipped_to_zero(self, scorer: TrustScorer) -> None:
        trust = scorer.score(cnn_score=0.0, whisper_score=0.0, liveness_passed=False)
        assert trust == 0.0

    def test_score_clamped_over_range(self, scorer: TrustScorer) -> None:
        trust = scorer.score(cnn_score=1.5, whisper_score=1.5, liveness_passed=True)
        assert trust == 1.0

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            TrustScorer(cnn_weight=0.5, whisper_weight=0.5, liveness_weight=0.5)

    def test_breakdown_contains_expected_keys(self, scorer: TrustScorer) -> None:
        bd = scorer.breakdown(cnn_score=0.8, whisper_score=0.7, liveness_passed=True)
        assert "cnn_contribution" in bd
        assert "whisper_contribution" in bd
        assert "liveness_contribution" in bd
        assert "trust_score" in bd

    def test_breakdown_trust_matches_score(self, scorer: TrustScorer) -> None:
        trust = scorer.score(0.75, 0.65, True)
        bd = scorer.breakdown(0.75, 0.65, True)
        assert abs(bd["trust_score"] - trust) < 1e-4

    def test_contributions_sum_approximately_to_trust(
        self, scorer: TrustScorer
    ) -> None:
        bd = scorer.breakdown(0.8, 0.7, True)
        total = (
            bd["cnn_contribution"]
            + bd["whisper_contribution"]
            + bd["liveness_contribution"]
        )
        assert abs(total - bd["trust_score"]) < 1e-4
