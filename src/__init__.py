"""
Zero-Trust Deepfake Voice Defense System — top-level package.

Provides unified access to all submodules: data, models, agents,
liveness, decision, pipeline, and utils.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zero-trust-deepfake-voice-defense")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
