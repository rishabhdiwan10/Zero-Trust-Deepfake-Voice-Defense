"""
src.agents.orchestrator
========================
LangGraph main orchestrator for the Zero-Trust Deepfake Voice Defense System.

Defines the agent graph and state machine that routes audio samples through:
  1. ForensicAgent  — CNN + Whisper analysis
  2. DecisionAgent  — trust score aggregation and threshold evaluation
  3. LivenessAgent  — dynamic challenge/response (only if score is uncertain)

The graph uses conditional edges so that the liveness branch is only triggered
when the trust score falls in the challenge band.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .state import PipelineState

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    LangGraph-based orchestrator for the deepfake defence pipeline.

    Parameters
    ----------
    forensic_agent : ForensicAgent
        Pre-built forensic analysis agent.
    decision_agent : DecisionAgent
        Pre-built decision / threshold agent.
    liveness_agent : LivenessAgent
        Pre-built liveness challenge agent.
    """

    def __init__(
        self,
        forensic_agent,
        decision_agent,
        liveness_agent,
    ) -> None:
        self.forensic_agent = forensic_agent
        self.decision_agent = decision_agent
        self.liveness_agent = liveness_agent
        self._graph = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(self) -> "Orchestrator":
        """
        Construct the LangGraph state machine.

        Graph topology::

            START
              │
              ▼
            forensic  ──────────────────────────────────────────►  decision
                                                                      │
                                              ┌───────────────────────┤
                                              │                       │
                                         challenge?               pass/reject
                                              │                       │
                                              ▼                       ▼
                                          liveness                  END
                                              │
                                              ▼
                                           decision  ──► END
        """
        try:
            from langgraph.graph import StateGraph, END  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langgraph is required. Run: pip install langgraph"
            ) from exc

        graph = StateGraph(PipelineState)

        # Register nodes
        graph.add_node("forensic", self.forensic_agent.run)
        graph.add_node("decision", self.decision_agent.run)
        graph.add_node("liveness", self.liveness_agent.run)

        # Linear edges
        graph.set_entry_point("forensic")
        graph.add_edge("forensic", "decision")

        # Conditional edge: challenge → liveness, otherwise END
        graph.add_conditional_edges(
            "decision",
            self._route_after_decision,
            {"liveness": "liveness", "end": END},
        )
        graph.add_edge("liveness", "decision")

        self._graph = graph.compile()
        logger.info("Orchestrator graph compiled successfully.")
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_async(self, initial_state: PipelineState) -> PipelineState:
        """
        Execute the pipeline asynchronously.

        Parameters
        ----------
        initial_state : PipelineState
            Seed state containing at least ``audio_path``.

        Returns
        -------
        PipelineState
            Final state after all agents have run.
        """
        if self._graph is None:
            self.build()
        final_state = await self._graph.ainvoke(initial_state)
        return final_state

    def run(self, initial_state: PipelineState) -> PipelineState:
        """Synchronous wrapper around ``run_async``."""
        import asyncio

        return asyncio.run(self.run_async(initial_state))

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_decision(state: PipelineState) -> str:
        """
        Conditional edge router after the decision node.

        Returns ``"liveness"`` if the decision is ``"challenge"`` and
        the retry count has not been exhausted; otherwise ``"end"``.
        """
        decision = state.get("decision", "reject")
        retry_count = state.get("liveness_retry_count", 0)
        max_retries = 2  # configurable — see thresholds_config.yaml

        if decision == "challenge" and retry_count < max_retries:
            return "liveness"
        return "end"
