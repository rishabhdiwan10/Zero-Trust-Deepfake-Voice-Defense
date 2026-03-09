"""
src.decision.action_router
===========================
Routes a pipeline decision string to a concrete action or event.

The router maps the three decision outcomes to system-level actions that
downstream consumers (API handlers, access control systems, audit logs)
can act upon.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ActionRouter:
    """
    Map pipeline decisions to concrete system actions.

    Parameters
    ----------
    on_pass : callable | None
        Callback invoked when decision is ``"pass"``.
    on_challenge : callable | None
        Callback invoked when decision is ``"challenge"``.
    on_reject : callable | None
        Callback invoked when decision is ``"reject"``.
    """

    # Human-readable action descriptions
    ACTION_DESCRIPTIONS: Dict[str, str] = {
        "pass": "Access granted — audio verified as genuine.",
        "challenge": "Liveness challenge required — audio is uncertain.",
        "reject": "Access denied — audio classified as synthetic or adversarial.",
    }

    def __init__(
        self,
        on_pass: Optional[Callable] = None,
        on_challenge: Optional[Callable] = None,
        on_reject: Optional[Callable] = None,
    ) -> None:
        self._handlers: Dict[str, Optional[Callable]] = {
            "pass": on_pass,
            "challenge": on_challenge,
            "reject": on_reject,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, decision: str) -> str:
        """
        Execute the action associated with a decision and return an action label.

        Parameters
        ----------
        decision : str
            One of ``"pass"``, ``"challenge"``, ``"reject"``.

        Returns
        -------
        str
            Human-readable action description string.
        """
        decision = decision.lower().strip()
        if decision not in self.ACTION_DESCRIPTIONS:
            logger.error("Unknown decision '%s' — defaulting to reject.", decision)
            decision = "reject"

        handler = self._handlers.get(decision)
        if handler is not None:
            try:
                handler(decision)
            except Exception as exc:
                logger.exception("Action handler for '%s' raised: %s", decision, exc)

        action = self.ACTION_DESCRIPTIONS[decision]
        logger.info("Action routed: [%s] %s", decision.upper(), action)
        return action

    def register_handler(self, decision: str, handler: Callable) -> None:
        """Register or replace the handler for a given decision."""
        if decision not in self._handlers:
            raise ValueError(
                f"Unknown decision '{decision}'. "
                f"Valid options: {list(self._handlers.keys())}"
            )
        self._handlers[decision] = handler

    def get_description(self, decision: str) -> str:
        """Return the human-readable description for a decision."""
        return self.ACTION_DESCRIPTIONS.get(decision, "Unknown decision.")
