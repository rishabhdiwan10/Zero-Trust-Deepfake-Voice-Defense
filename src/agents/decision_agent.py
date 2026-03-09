"""
src.agents.decision_agent
==========================
LangGraph agent node responsible for aggregating per-layer forensic scores
into a unified trust score and making the final access control decision.

Decision outcomes:
  - ``"pass"``      — trust score ≥ pass_threshold
  - ``"challenge"`` — challenge_threshold ≤ trust_score < pass_threshold
  - ``"reject"``    — trust_score < challenge_threshold (or max retries exceeded)
"""

from __future__ import annotations

import logging
import time

from .state import PipelineState

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Trust aggregation and decision-making agent.

    Parameters
    ----------
    trust_scorer : TrustScorer
        Pre-built trust score aggregator.
    threshold_engine : ThresholdEngine
        Pre-built threshold evaluator.
    action_router : ActionRouter
        Pre-built action router.
    """

    def __init__(
        self,
        trust_scorer,
        threshold_engine,
        action_router,
    ) -> None:
        self.trust_scorer = trust_scorer
        self.threshold_engine = threshold_engine
        self.action_router = action_router

    # ------------------------------------------------------------------
    # LangGraph node entry point
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """
        Aggregate scores and determine the pipeline decision.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state (must contain ``cnn_score``,
            ``whisper_score``, and optionally ``liveness_passed``).

        Returns
        -------
        PipelineState
            Updated state with ``trust_score``, ``decision``, and ``action``.
        """
        t_start = time.perf_counter()

        cnn_score = state.get("cnn_score", 0.5)
        whisper_score = state.get("whisper_score", 0.5)
        liveness_passed = state.get("liveness_passed")
        forensic_metadata = state.get("forensic_metadata", {})

        # Aggregate trust score
        trust_score = self.trust_scorer.score(
            cnn_score=cnn_score,
            whisper_score=whisper_score,
            liveness_passed=liveness_passed,
        )
        state["trust_score"] = trust_score

        # Evaluate threshold — with quality differentiation
        decision = self.threshold_engine.evaluate(
            trust_score=trust_score,
            cnn_score=cnn_score,
            audio_metadata=forensic_metadata,
        )
        state["decision"] = decision

        # Route to action
        action = self.action_router.route(decision)
        state["action"] = action

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies = state.get("stage_latencies", {})
        latencies["decision_ms"] = round(elapsed_ms, 2)
        state["stage_latencies"] = latencies

        logger.info(
            "DecisionAgent — trust=%.3f decision=%s action=%s (%.1f ms)",
            trust_score,
            decision,
            action,
            elapsed_ms,
        )
        return state
