"""
src.liveness.response_validator
================================
Validates a user's spoken response to a liveness challenge.

Validation strategy:
  1. Transcribe the response audio using Whisper.
  2. Normalise both the challenge text and the transcription.
  3. Compute token-level similarity (Jaccard) and Word Error Rate (WER).
  4. Pass if similarity ≥ ``min_token_similarity`` AND
             WER ≤ ``max_allowed_wer``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseValidator:
    """
    Validate a user's audio response to a liveness challenge.

    Parameters
    ----------
    min_token_similarity : float
        Minimum Jaccard token similarity required [0, 1].
    max_allowed_wer : float
        Maximum Word Error Rate allowed [0, 1].
    whisper_model_size : str
        Whisper model size for transcription.
    device : str | None
        Torch device for Whisper (auto-detected if ``None``).
    """

    def __init__(
        self,
        min_token_similarity: float = 0.70,
        max_allowed_wer: float = 0.20,
        whisper_model_size: str = "base",
        device: Optional[str] = None,
    ) -> None:
        self.min_token_similarity = min_token_similarity
        self.max_allowed_wer = max_allowed_wer
        self.whisper_model_size = whisper_model_size
        self.device = device
        self._whisper = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        challenge: str,
        response_audio_path: str,
    ) -> bool:
        """
        Determine whether the audio response satisfies the challenge.

        Parameters
        ----------
        challenge : str
            The original challenge text presented to the user.
        response_audio_path : str
            Path to the recorded audio response.

        Returns
        -------
        bool
            ``True`` if the response is valid; ``False`` otherwise.
        """
        transcription = self._transcribe(response_audio_path)
        similarity, wer = self._compare(challenge, transcription)

        passed = (
            similarity >= self.min_token_similarity and wer <= self.max_allowed_wer
        )
        logger.info(
            "Liveness validation — similarity=%.3f WER=%.3f → %s",
            similarity,
            wer,
            "PASS" if passed else "FAIL",
        )
        return passed

    def validate_with_details(
        self,
        challenge: str,
        response_audio_path: str,
    ) -> dict:
        """
        Validate and return a detailed result dictionary.

        Returns
        -------
        dict with keys: ``passed``, ``transcription``, ``similarity``, ``wer``.
        """
        transcription = self._transcribe(response_audio_path)
        similarity, wer = self._compare(challenge, transcription)
        passed = (
            similarity >= self.min_token_similarity and wer <= self.max_allowed_wer
        )
        return {
            "passed": passed,
            "transcription": transcription,
            "similarity": round(similarity, 4),
            "wer": round(wer, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> str:
        """Transcribe the response audio with Whisper."""
        if self._whisper is None:
            self._load_whisper()
        try:
            import whisper  # type: ignore

            result = self._whisper.transcribe(audio_path, fp16=False, verbose=False)
            return result.get("text", "").strip()
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return ""

    def _load_whisper(self) -> None:
        """Lazily load the Whisper model."""
        try:
            import whisper  # type: ignore

            self._whisper = whisper.load_model(
                self.whisper_model_size, device=self.device
            )
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required. Run: pip install openai-whisper"
            ) from exc

    @staticmethod
    def _normalise(text: str) -> list[str]:
        """Lower-case, strip punctuation, and tokenise."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def _compare(
        self, challenge: str, transcription: str
    ) -> Tuple[float, float]:
        """Compute Jaccard similarity and WER between challenge and response."""
        ref_tokens = self._normalise(challenge)
        hyp_tokens = self._normalise(transcription)

        # Jaccard similarity on token sets
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)
        intersection = ref_set & hyp_set
        union = ref_set | hyp_set
        similarity = len(intersection) / len(union) if union else 0.0

        # Word Error Rate (simplified: edit distance / len(ref))
        wer = self._word_error_rate(ref_tokens, hyp_tokens)
        return similarity, wer

    @staticmethod
    def _word_error_rate(ref: list[str], hyp: list[str]) -> float:
        """Compute WER using dynamic-programming edit distance."""
        if not ref:
            return 1.0 if hyp else 0.0
        r, h = len(ref), len(hyp)
        # DP table
        dp = list(range(h + 1))
        for i in range(1, r + 1):
            new_dp = [i] + [0] * h
            for j in range(1, h + 1):
                if ref[i - 1] == hyp[j - 1]:
                    new_dp[j] = dp[j - 1]
                else:
                    new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
            dp = new_dp
        return dp[h] / r
