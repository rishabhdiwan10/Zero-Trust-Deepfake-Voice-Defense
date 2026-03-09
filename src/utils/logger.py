"""
src.utils.logger
=================
Structured logging setup for the Zero-Trust Deepfake Voice Defense System.

Provides a ``get_logger`` factory function that returns a consistently
configured logger with an optional structured (JSON) formatter for production
and a human-readable formatter for development.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED_LOGGERS: set = set()
_ROOT_CONFIGURED = False


def get_logger(
    name: str,
    level: int = logging.INFO,
    json_format: bool = False,
) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__``).
    level : int
        Logging level (default: ``logging.INFO``).
    json_format : bool
        If ``True``, emit logs as JSON lines (for production / log aggregators).

    Returns
    -------
    logging.Logger
    """
    global _ROOT_CONFIGURED

    if not _ROOT_CONFIGURED:
        _configure_root(level, json_format)
        _ROOT_CONFIGURED = True

    logger = logging.getLogger(name)
    return logger


def _configure_root(level: int, json_format: bool) -> None:
    """Configure the root logger once."""
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return  # Already configured (e.g., by pytest or FastAPI)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_format:
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_obj = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)
