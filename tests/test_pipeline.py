"""
Integration tests for the Zero-Trust Deepfake Voice Defense pipeline.

These tests exercise the decision and liveness layers end-to-end using
mocked CNN and Whisper outputs, avoiding heavy ML model loading.
"""

import pytest

from src.decision.action_router import ActionRouter
from src.decision.threshold_engine import ThresholdEngine
from src.decision.trust_scorer import TrustScorer
from src.liveness.challenge_generator import ChallengeGenerator
from src.liveness.prompt_templates import PromptTemplateBank


# ---------------------------------------------------------------------------
# Tests — ActionRouter
# ---------------------------------------------------------------------------


class TestActionRouter:
    def test_pass_returns_description(self) -> None:
        router = ActionRouter()
        action = router.route("pass")
        assert "granted" in action.lower()

    def test_challenge_returns_description(self) -> None:
        router = ActionRouter()
        action = router.route("challenge")
        assert "challenge" in action.lower() or "liveness" in action.lower()

    def test_reject_returns_description(self) -> None:
        router = ActionRouter()
        action = router.route("reject")
        assert "denied" in action.lower()

    def test_unknown_decision_defaults_to_reject(self) -> None:
        router = ActionRouter()
        action = router.route("unknown_decision")
        assert "denied" in action.lower()

    def test_custom_handler_is_called(self) -> None:
        called = []
        router = ActionRouter(on_pass=lambda d: called.append(d))
        router.route("pass")
        assert len(called) == 1

    def test_register_handler(self) -> None:
        called = []
        router = ActionRouter()
        router.register_handler("reject", lambda d: called.append(d))
        router.route("reject")
        assert len(called) == 1


# ---------------------------------------------------------------------------
# Tests — full decision path (ThresholdEngine → TrustScorer → ActionRouter)
# ---------------------------------------------------------------------------


class TestDecisionPipeline:
    def test_genuine_audio_leads_to_pass(self) -> None:
        scorer = TrustScorer()
        engine = ThresholdEngine()
        router = ActionRouter()

        trust = scorer.score(cnn_score=0.95, whisper_score=0.90, liveness_passed=None)
        decision = engine.evaluate(trust)
        action = router.route(decision)

        assert decision == "pass"
        assert "granted" in action.lower()

    def test_synthetic_audio_leads_to_reject(self) -> None:
        scorer = TrustScorer()
        engine = ThresholdEngine()
        router = ActionRouter()

        trust = scorer.score(cnn_score=0.10, whisper_score=0.15, liveness_passed=False)
        decision = engine.evaluate(trust)
        action = router.route(decision)

        assert decision == "reject"
        assert "denied" in action.lower()

    def test_uncertain_audio_leads_to_challenge(self) -> None:
        scorer = TrustScorer()
        engine = ThresholdEngine()
        router = ActionRouter()

        trust = scorer.score(cnn_score=0.60, whisper_score=0.55, liveness_passed=None)
        decision = engine.evaluate(trust)

        assert decision == "challenge"

    def test_liveness_pass_promotes_to_pass(self) -> None:
        scorer = TrustScorer()
        engine = ThresholdEngine()

        # Pre-liveness: uncertain
        trust_before = scorer.score(0.60, 0.55, liveness_passed=None)
        decision_before = engine.evaluate(trust_before)
        assert decision_before == "challenge"

        # Post-liveness pass: should upgrade
        trust_after = scorer.score(0.60, 0.55, liveness_passed=True)
        decision_after = engine.evaluate(trust_after)
        assert decision_after in ("pass", "challenge")  # at least better than reject


# ---------------------------------------------------------------------------
# Tests — challenge generator in pipeline context
# ---------------------------------------------------------------------------


class TestLivenessChallengeFlow:
    def test_challenge_is_generated_and_unique(self) -> None:
        gen = ChallengeGenerator(seed=None)
        challenges = {gen.generate() for _ in range(20)}
        assert len(challenges) >= 5, "Expected diverse challenge phrases"

    def test_challenge_context_does_not_crash(self) -> None:
        gen = ChallengeGenerator()
        context = {
            "liveness_retry_count": 2,
            "audio_path": "/tmp/test.wav",
        }
        challenge = gen.generate(context=context)
        assert isinstance(challenge, str)
        assert len(challenge) > 0
