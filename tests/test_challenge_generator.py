"""
Tests for src.liveness.challenge_generator — dynamic liveness challenge
generation.
"""

import pytest

from src.liveness.challenge_generator import ChallengeGenerator
from src.liveness.prompt_templates import PromptTemplateBank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> ChallengeGenerator:
    """ChallengeGenerator with a fixed seed for reproducible tests."""
    return ChallengeGenerator(difficulty="medium", seed=42)


@pytest.fixture()
def generator_hard() -> ChallengeGenerator:
    return ChallengeGenerator(difficulty="hard", seed=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChallengeGenerator:
    def test_generate_returns_non_empty_string(
        self, generator: ChallengeGenerator
    ) -> None:
        challenge = generator.generate()
        assert isinstance(challenge, str)
        assert len(challenge) > 0

    def test_generate_is_non_deterministic_across_calls(
        self, generator: ChallengeGenerator
    ) -> None:
        """Challenges should differ across calls (with very high probability)."""
        challenges = {generator.generate() for _ in range(20)}
        # Expect at least a handful of distinct challenges
        assert len(challenges) >= 3

    def test_generate_batch_length(self, generator: ChallengeGenerator) -> None:
        batch = generator.generate_batch(n=5)
        assert len(batch) == 5
        assert all(isinstance(c, str) and len(c) > 0 for c in batch)

    def test_seeded_generator_is_reproducible(self) -> None:
        g1 = ChallengeGenerator(seed=99)
        g2 = ChallengeGenerator(seed=99)
        assert g1.generate() == g2.generate()

    def test_different_seeds_produce_different_challenges(self) -> None:
        g1 = ChallengeGenerator(seed=1)
        g2 = ChallengeGenerator(seed=2)
        # With different seeds, should differ (very likely over many samples)
        challenges_1 = {g1.generate() for _ in range(10)}
        challenges_2 = {g2.generate() for _ in range(10)}
        assert challenges_1 != challenges_2

    def test_hard_difficulty_uses_longer_digits(
        self, generator_hard: ChallengeGenerator
    ) -> None:
        # Hard difficulty: 8 digits → 8 single digits separated by spaces = 15 chars
        digits = generator_hard._random_digits()
        digit_count = len(digits.split())
        assert digit_count == 8

    def test_medium_difficulty_digit_length(
        self, generator: ChallengeGenerator
    ) -> None:
        digits = generator._random_digits()
        assert len(digits.split()) == 6  # medium = 6 digits

    def test_context_passed_to_challenge(self, generator: ChallengeGenerator) -> None:
        context = {"liveness_retry_count": 3}
        challenge = generator.generate(context=context)
        assert isinstance(challenge, str)


class TestPromptTemplateBank:
    def test_default_enabled_categories(self) -> None:
        bank = PromptTemplateBank()
        assert "phoneme_rich" in bank.enabled_categories
        assert "date_time" in bank.enabled_categories

    def test_get_templates_returns_list(self) -> None:
        bank = PromptTemplateBank()
        templates = bank.get_templates("phoneme_rich")
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_get_unknown_category_returns_empty(self) -> None:
        bank = PromptTemplateBank()
        assert bank.get_templates("nonexistent") == []

    def test_custom_enabled_categories(self) -> None:
        bank = PromptTemplateBank(enabled_categories=["date_time"])
        assert bank.enabled_categories == ["date_time"]

    def test_all_enabled_templates_keys(self) -> None:
        bank = PromptTemplateBank(
            enabled_categories=["phoneme_rich", "date_time"]
        )
        all_tpl = bank.all_enabled_templates()
        assert set(all_tpl.keys()) == {"phoneme_rich", "date_time"}
