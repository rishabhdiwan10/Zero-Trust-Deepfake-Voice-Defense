"""
src.agents.liveness_agent
==========================
LangGraph agent node responsible for managing the dynamic liveness
challenge / response flow.

On entry, the agent generates a fresh challenge phrase and returns it
in the state. On re-entry (after the user has responded), it validates
the response and updates ``liveness_passed`` and ``liveness_retry_count``.
"""

from __future__ import annotations

import logging
import time

from .state import PipelineState

logger = logging.getLogger(__name__)


class LivenessAgent:
    """
    Liveness challenge management agent.

    Parameters
    ----------
    challenge_generator : ChallengeGenerator
        Pre-built challenge generator instance.
    response_validator : ResponseValidator
        Pre-built response validator instance.
    """

    def __init__(self, challenge_generator, response_validator) -> None:
        self.challenge_generator = challenge_generator
        self.response_validator = response_validator

    # ------------------------------------------------------------------
    # LangGraph node entry point
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the liveness challenge step.

        Behaviour:
          - If no ``liveness_challenge`` exists in state → generate one.
          - If a challenge exists and ``liveness_response_path`` is set
            → validate the response.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state.

        Returns
        -------
        PipelineState
            Updated state.
        """
        t_start = time.perf_counter()

        existing_challenge = state.get("liveness_challenge", "")
        response_path = state.get("liveness_response_path")

        if not existing_challenge:
            # First visit: generate a challenge
            challenge = self.challenge_generator.generate(context=state)
            state["liveness_challenge"] = challenge
            state["liveness_passed"] = None
            state["liveness_retry_count"] = state.get("liveness_retry_count", 0)
            logger.info("Liveness challenge generated: '%s'", challenge)
        elif response_path:
            # Second visit: validate the response
            passed = self.response_validator.validate(
                challenge=existing_challenge,
                response_audio_path=response_path,
            )
            state["liveness_passed"] = passed
            state["liveness_retry_count"] = state.get("liveness_retry_count", 0) + 1

            if passed:
                # Promote trust score if liveness passed
                current_trust = state.get("trust_score", 0.5)
                state["trust_score"] = min(1.0, current_trust + 0.2)
                logger.info("Liveness challenge PASSED.")
            else:
                # Reduce trust score on failure
                current_trust = state.get("trust_score", 0.5)
                state["trust_score"] = max(0.0, current_trust - 0.1)
                logger.warning("Liveness challenge FAILED.")
        else:
            logger.warning(
                "LivenessAgent called with existing challenge but no response_path."
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies = state.get("stage_latencies", {})
        latencies["liveness_ms"] = round(elapsed_ms, 2)
        state["stage_latencies"] = latencies

        return state
