"""
src.agents.state
=================
Shared state schema for the LangGraph multi-agent pipeline.

The ``PipelineState`` TypedDict is passed between all agents in the graph.
Each agent reads from and writes to specific fields, ensuring a clean
separation of concerns and full auditability of each decision step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore


class PipelineState(TypedDict, total=False):
    """
    Shared mutable state threaded through the LangGraph agent pipeline.

    Fields
    ------
    audio_path : str
        Absolute path to the input audio file being analysed.
    waveform : Any
        Raw numpy waveform array (populated by preprocessing).
    sample_rate : int
        Sample rate of the waveform.
    features : Any
        Extracted feature tensor (numpy or torch).

    cnn_score : float
        Genuine probability from the CNN detector [0, 1].
    whisper_score : float
        Genuine probability from Whisper analysis [0, 1].
    transcription : str
        Whisper transcription of the audio.
    forensic_metadata : dict
        Additional metadata from the forensic agent.

    trust_score : float
        Aggregated trust score from the TrustScorer [0, 1].
    decision : str
        Final decision: ``"pass"`` | ``"challenge"`` | ``"reject"``.
    action : str
        Routed action string (mirrors decision with additional context).

    liveness_challenge : str
        The dynamic challenge phrase presented to the user.
    liveness_response_path : str | None
        Path to the user's audio response to the liveness challenge.
    liveness_passed : bool | None
        Whether the liveness challenge was passed.
    liveness_retry_count : int
        Number of liveness challenge attempts so far.

    error : str | None
        Error message if any pipeline stage failed.
    stage_latencies : Dict[str, float]
        Per-stage latency measurements in milliseconds.
    """

    audio_path: str
    waveform: Any
    sample_rate: int
    features: Any

    # Forensic analysis outputs
    cnn_score: float
    whisper_score: float
    transcription: str
    forensic_metadata: Dict[str, Any]

    # Decision outputs
    trust_score: float
    decision: str
    action: str

    # Liveness challenge
    liveness_challenge: str
    liveness_response_path: Optional[str]
    liveness_passed: Optional[bool]
    liveness_retry_count: int

    # Meta
    error: Optional[str]
    stage_latencies: Dict[str, float]
