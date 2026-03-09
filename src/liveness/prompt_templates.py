"""
src.liveness.prompt_templates
==============================
Template bank for dynamic liveness challenge generation.

Templates use Python ``str.format`` placeholders that are filled with
session-specific values (timestamps, random tokens, digits, etc.) at
generation time, ensuring every challenge is unique and cannot be replayed.

Template categories:
  - ``phoneme_rich``       — high phoneme diversity phrases
  - ``date_time``          — include current date/time component
  - ``random_digits``      — random digit sequences
  - ``contextual``         — context-aware fill-in sentences
  - ``emotional_tone``     — prosody challenges (optional)
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Template definitions
# Each template is a string with ``{placeholder}`` slots.
# Placeholders are filled by ChallengeGenerator at runtime.
# ---------------------------------------------------------------------------

PHONEME_RICH_TEMPLATES: List[str] = [
    "Please say: {phrase}",
    "Read aloud the following: {phrase}",
    "Repeat clearly: {phrase}",
    "Speak this phrase: {phrase}",
    "Say each word clearly: {phrase}",
]

# Phoneme-rich phrases (high consonant and vowel diversity)
PHONEME_RICH_PHRASES: List[str] = [
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore swiftly",
    "how much wood would a woodchuck chuck",
    "unique New York, unique New York, you know you need unique New York",
    "red lorry yellow lorry red lorry yellow lorry",
    "whether the weather is warm whether the weather is hot",
    "six slippery snails slid slowly seaward",
    "fresh french fried fish",
    "around the rugged rocks the ragged rascal ran",
    "peter piper picked a peck of pickled peppers",
    "truly rural, truly rural, truly rural",
    "which witch watched which watch",
]

DATE_TIME_TEMPLATES: List[str] = [
    "Please say today's date: {date}",
    "State the current time: {time}",
    "Say the date and time: {date} at {time}",
    "Confirm the session date: {date}",
    "Speak the current hour and minute: {hour} {minute}",
]

RANDOM_DIGIT_TEMPLATES: List[str] = [
    "Read aloud the following digits: {digits}",
    "Say each digit: {digits}",
    "Recite these numbers in order: {digits}",
    "Confirm the code: {digits}",
    "Speak the sequence: {digits}",
]

CONTEXTUAL_TEMPLATES: List[str] = [
    "Complete this sentence aloud: Today is {day_of_week} and the session token is {token}.",
    "Speak the following verification phrase: My session started at {time} on {date}.",
    "Say aloud: I confirm this request at {time} with code {token}.",
    "Read this statement: Verification requested on {date}, token {token}.",
    "Recite: Access attempt number {attempt_number} at {time}.",
]

EMOTIONAL_TONE_TEMPLATES: List[str] = [
    "Say the following in a calm voice: {phrase}",
    "Repeat with confidence: {phrase}",
    "Say this clearly and slowly: {phrase}",
]


class PromptTemplateBank:
    """
    Repository of challenge prompt templates organised by category.

    Parameters
    ----------
    enabled_categories : list[str] | None
        Which categories to include. Defaults to all except
        ``emotional_tone``.
    """

    CATEGORY_TEMPLATES: Dict[str, List[str]] = {
        "phoneme_rich": PHONEME_RICH_TEMPLATES,
        "date_time": DATE_TIME_TEMPLATES,
        "random_digits": RANDOM_DIGIT_TEMPLATES,
        "contextual": CONTEXTUAL_TEMPLATES,
        "emotional_tone": EMOTIONAL_TONE_TEMPLATES,
    }

    CATEGORY_PHRASES: Dict[str, List[str]] = {
        "phoneme_rich": PHONEME_RICH_PHRASES,
    }

    DEFAULT_ENABLED = ["phoneme_rich", "date_time", "random_digits", "contextual"]

    def __init__(
        self,
        enabled_categories: List[str] | None = None,
    ) -> None:
        self.enabled_categories = enabled_categories or self.DEFAULT_ENABLED

    def get_templates(self, category: str) -> List[str]:
        """Return templates for a given category."""
        return self.CATEGORY_TEMPLATES.get(category, [])

    def get_phrases(self, category: str) -> List[str]:
        """Return filler phrases for categories that use them."""
        return self.CATEGORY_PHRASES.get(category, [])

    def all_enabled_templates(self) -> Dict[str, List[str]]:
        """Return templates for all enabled categories."""
        return {
            cat: self.CATEGORY_TEMPLATES[cat]
            for cat in self.enabled_categories
            if cat in self.CATEGORY_TEMPLATES
        }
