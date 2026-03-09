"""
src.liveness.challenge_generator
==================================
Dynamic, context-aware liveness challenge prompt generator.

Generates unique challenge phrases on every invocation using:
  - Current date/time components
  - Random digit sequences
  - Random session tokens
  - Template randomisation from ``PromptTemplateBank``

This design makes replay attacks impractical: a recorded response to one
challenge is valid only for that exact session and cannot satisfy a future
challenge.
"""

from __future__ import annotations

import logging
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from .prompt_templates import PromptTemplateBank

logger = logging.getLogger(__name__)


class ChallengeGenerator:
    """
    Generate context-aware, one-time liveness challenge phrases.

    Parameters
    ----------
    template_bank : PromptTemplateBank | None
        Template bank to draw from. Created with defaults if ``None``.
    difficulty : str
        ``"easy"`` | ``"medium"`` | ``"hard"`` — controls phrase complexity.
    include_session_token : bool
        Whether to embed a short random token in challenges to prevent replay.
    seed : int | None
        Random seed. If ``None``, truly random behaviour each call.
    """

    DIGIT_LENGTHS = {"easy": 4, "medium": 6, "hard": 8}
    TOKEN_LENGTHS = {"easy": 4, "medium": 6, "hard": 8}

    def __init__(
        self,
        template_bank: Optional[PromptTemplateBank] = None,
        difficulty: str = "medium",
        include_session_token: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.template_bank = template_bank or PromptTemplateBank()
        self.difficulty = difficulty
        self.include_session_token = include_session_token
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a single dynamic challenge phrase.

        Parameters
        ----------
        context : dict | None
            Optional pipeline state or session context. May contain
            ``attempt_number`` or other contextual data to embed.

        Returns
        -------
        str
            A unique challenge string to be read aloud by the user.
        """
        context = context or {}
        category = self._pick_category()
        templates = self.template_bank.get_templates(category)

        if not templates:
            return self._fallback_challenge()

        template = self._rng.choice(templates)
        filled = self._fill_template(template, category, context)

        logger.debug("Generated liveness challenge [%s]: %s", category, filled)
        return filled

    def generate_batch(
        self,
        n: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate ``n`` distinct challenge phrases."""
        return [self.generate(context) for _ in range(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_category(self) -> str:
        """Randomly select an enabled template category."""
        enabled = list(self.template_bank.enabled_categories)
        return self._rng.choice(enabled)

    def _fill_template(
        self,
        template: str,
        category: str,
        context: Dict[str, Any],
    ) -> str:
        """Fill template placeholders with dynamic values."""
        now = datetime.now()

        fill_values: Dict[str, str] = {
            "date": now.strftime("%B %d %Y"),
            "time": now.strftime("%I %M %p"),
            "hour": now.strftime("%I"),
            "minute": now.strftime("%M"),
            "day_of_week": now.strftime("%A"),
            "token": self._random_token(),
            "digits": self._random_digits(),
            "phrase": self._random_phrase(category),
            "attempt_number": str(context.get("liveness_retry_count", 1)),
        }

        try:
            return template.format(**fill_values)
        except KeyError as exc:
            logger.warning("Template fill failed for key %s — using fallback.", exc)
            return self._fallback_challenge()

    def _random_token(self) -> str:
        """Generate a random alphanumeric session token."""
        length = self.TOKEN_LENGTHS.get(self.difficulty, 6)
        chars = string.ascii_uppercase + string.digits
        return "".join(self._rng.choices(chars, k=length))

    def _random_digits(self) -> str:
        """Generate a random digit string with spaces for natural speech."""
        length = self.DIGIT_LENGTHS.get(self.difficulty, 6)
        return " ".join(str(self._rng.randint(0, 9)) for _ in range(length))

    def _random_phrase(self, category: str) -> str:
        """Pick a random filler phrase for the given category."""
        phrases = self.template_bank.get_phrases(category)
        if phrases:
            return self._rng.choice(phrases)
        return "the verification is now complete"

    def _fallback_challenge(self) -> str:
        """Return a minimal fallback challenge if template filling fails."""
        token = self._random_token()
        digits = self._random_digits()
        return f"Please say: session code {token}, digits {digits}"
