"""
app.py
=======
Streamlit web UI for the Zero-Trust Deepfake Voice Defense System.

Run with::

    streamlit run app.py

Features
--------
* Upload or record live audio for deepfake analysis
* Real-time CNN + Whisper forensic scoring
* Trust score visualisation with pass / challenge / reject verdict
* Dynamic liveness challenge when the trust score is uncertain
* Batch analysis of multiple uploaded files
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zero-Trust Deepfake Voice Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = ["wav", "flac", "mp3", "ogg", "m4a"]
MODEL_CHECKPOINT_PATH = Path("models/best_checkpoint.pt")


@st.cache_resource(show_spinner="Loading AI models…")
def _load_pipeline():
    """
    Build and cache the full inference pipeline.

    Returns
    -------
    RealtimePipeline | None
        The loaded pipeline, or ``None`` if required models are unavailable.
    str | None
        An error message if the pipeline could not be built, else ``None``.
    """
    try:
        from src.models.cnn_detector import CNNDetector
        from src.models.whisper_analyzer import WhisperAnalyzer
        from src.data.audio_preprocessor import AudioPreprocessor
        from src.data.feature_extractor import FeatureExtractor, FeatureType
        from src.agents.forensic_agent import ForensicAgent
        from src.agents.decision_agent import DecisionAgent
        from src.agents.liveness_agent import LivenessAgent
        from src.agents.orchestrator import Orchestrator
        from src.decision.trust_scorer import TrustScorer
        from src.decision.threshold_engine import ThresholdEngine
        from src.decision.action_router import ActionRouter
        from src.liveness.challenge_generator import ChallengeGenerator
        from src.liveness.response_validator import ResponseValidator
        from src.pipeline.realtime_pipeline import RealtimePipeline
        from src.models.model_utils import get_device, load_checkpoint
        from src.utils.config_loader import load_config

        device = os.environ.get("ZTDVD_DEVICE") or get_device()

        # Load configs
        model_cfg = load_config("configs/model_config.yaml")
        threshold_cfg = load_config("configs/thresholds_config.yaml")
        liveness_cfg = load_config("configs/liveness_config.yaml")
        m_cfg = model_cfg.get("model", {})

        # CNN detector
        cnn = CNNDetector(
            backbone=m_cfg.get("backbone", "resnet34"),
            num_classes=m_cfg.get("num_classes", 2),
            pretrained=not MODEL_CHECKPOINT_PATH.exists(),
            dropout=m_cfg.get("dropout", 0.3),
            device=device,
        ).build()

        if MODEL_CHECKPOINT_PATH.exists():
            try:
                load_checkpoint(MODEL_CHECKPOINT_PATH, cnn._model, device=device)
            except Exception as exc:
                st.warning(
                    f"Could not load checkpoint ({MODEL_CHECKPOINT_PATH}): {exc}. "
                    "Running with random (untrained) weights."
                )

        # Whisper analyzer
        whisper = WhisperAnalyzer()

        # Preprocessing & features
        preprocessor = AudioPreprocessor(target_sr=16_000, normalize=True)
        feature_extractor = FeatureExtractor(
            feature_type=FeatureType(m_cfg.get("input_feature", "mel_spectrogram"))
        )

        # Agents
        forensic_agent = ForensicAgent(
            cnn_detector=cnn,
            whisper_analyzer=whisper,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            run_parallel=True,
        )

        trust_scorer = TrustScorer(threshold_cfg)
        threshold_engine = ThresholdEngine(threshold_cfg)
        action_router = ActionRouter(threshold_cfg)
        decision_agent = DecisionAgent(trust_scorer, threshold_engine, action_router)

        challenge_gen = ChallengeGenerator(liveness_cfg)
        response_validator = ResponseValidator(liveness_cfg)
        liveness_agent = LivenessAgent(challenge_gen, response_validator)

        orchestrator = Orchestrator(forensic_agent, decision_agent, liveness_agent).build()
        pipeline = RealtimePipeline(orchestrator, pipeline_timeout=30.0)
        return pipeline, None

    except Exception as exc:
        return None, str(exc)


def _analyse_file(audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
    """Write *audio_bytes* to a temp file and run the pipeline."""
    pipeline, err = _load_pipeline()
    if pipeline is None:
        return {
            "error": err or "Pipeline unavailable.",
            "decision": "reject",
            "trust_score": 0.0,
            "cnn_score": 0.0,
            "whisper_score": 0.0,
            "transcription": "",
            "stage_latencies": {},
        }

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = pipeline.process_sync(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


def _verdict_ui(result: Dict[str, Any]) -> None:
    """Render the analysis results for a single audio file."""
    decision = result.get("decision", "reject")
    trust_score = result.get("trust_score", 0.0)
    cnn_score = result.get("cnn_score", 0.0)
    whisper_score = result.get("whisper_score", 0.0)
    transcription = result.get("transcription", "")
    metadata = result.get("forensic_metadata", {})
    latencies = result.get("stage_latencies", {})

    # ── Verdict badge ─────────────────────────────────────────────────────
    if decision == "pass":
        st.success("✅ VERDICT: **PASS** — Audio appears genuine.")
    elif decision == "challenge":
        st.warning("⚠️ VERDICT: **CHALLENGE** — Trust score is uncertain.")
    else:
        st.error("❌ VERDICT: **REJECT** — Audio flagged as synthetic / deepfake.")

    # ── Trust score ───────────────────────────────────────────────────────
    st.subheader("Trust Score")
    col_score, col_bar = st.columns([1, 3])
    col_score.metric("Trust Score", f"{trust_score:.2%}")
    col_bar.progress(min(max(trust_score, 0.0), 1.0))

    # ── CNN / Whisper scores ──────────────────────────────────────────────
    st.subheader("Score Breakdown")
    c1, c2 = st.columns(2)
    c1.metric(
        "CNN Detector Score",
        f"{cnn_score:.2%}",
        help="Probability the audio is genuine according to the CNN model.",
    )
    c2.metric(
        "Whisper Analyzer Score",
        f"{whisper_score:.2%}",
        help="Genuine probability derived from Whisper log-probabilities.",
    )

    # ── Transcription ─────────────────────────────────────────────────────
    with st.expander("📝 Transcription"):
        st.write(transcription or "*(no speech detected)*")

    # ── Forensic metadata ─────────────────────────────────────────────────
    if metadata:
        with st.expander("🔬 Forensic Metadata"):
            meta_display = {
                "Avg Log Probability": round(metadata.get("avg_log_prob", 0.0), 4),
                "Compression Ratio": round(metadata.get("compression_ratio", 0.0), 4),
                "No-Speech Probability": round(metadata.get("no_speech_prob", 0.0), 4),
                "Language": metadata.get("language", "—"),
                "CNN Inference (ms)": metadata.get("cnn_inference_ms", "—"),
                "Whisper Inference (ms)": metadata.get("whisper_inference_ms", "—"),
            }
            for k, v in meta_display.items():
                st.write(f"**{k}:** {v}")

    # ── Latency breakdown ─────────────────────────────────────────────────
    if latencies:
        with st.expander("⏱️ Latency Breakdown"):
            for stage, ms in latencies.items():
                st.write(f"**{stage}:** {ms} ms")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    with st.sidebar:
        st.title("🛡️ Zero-Trust DVS")

        # Model status
        st.subheader("Model Status")
        if MODEL_CHECKPOINT_PATH.exists():
            st.success("✅ Trained checkpoint loaded")
        else:
            st.warning(
                "⚠️ No trained checkpoint found.\n\n"
                "Run `python scripts/train.py` first, then restart this app.\n\n"
                "The system will still run with untrained (random) weights — "
                "results will not be meaningful until the model is trained."
            )

        # Config summary
        st.subheader("Configuration")
        try:
            from src.utils.config_loader import load_config

            t_cfg = load_config("configs/thresholds_config.yaml")
            thresholds = t_cfg.get("thresholds", {})
            st.json(
                {
                    "pass_threshold": thresholds.get("pass", "—"),
                    "challenge_threshold": thresholds.get("challenge", "—"),
                    "reject_threshold": thresholds.get("reject", "—"),
                },
            )
        except Exception:
            st.info("Config not available.")

        st.subheader("About")
        st.markdown(
            """
**Zero-Trust Deepfake Voice Defense System**

Multi-layered pipeline combining:
- 🤖 CNN deepfake audio detector (ResNet / EfficientNet)
- 🎙️ Whisper artifact analysis
- 🔐 Dynamic liveness challenges
- 📊 Trust scoring & threshold engine

Built with LangGraph agentic orchestration.
"""
        )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🛡️ Zero-Trust Deepfake Voice Defense System")
    st.caption(
        "Upload or record audio to detect AI-generated deepfake speech. "
        "Every voice sample is treated as potentially adversarial."
    )

    _sidebar()

    tab_upload, tab_record, tab_batch = st.tabs(
        ["📂 Upload Audio", "🎙️ Record Audio", "📋 Batch Analysis"]
    )

    # ────────────────────────────────────────────────────────────────────────
    # Tab 1: Upload a single audio file
    # ────────────────────────────────────────────────────────────────────────
    with tab_upload:
        st.subheader("Upload an Audio File")
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(f'.{f}' for f in SUPPORTED_FORMATS)}",
        )

        if uploaded is not None:
            st.audio(uploaded)
            if st.button("🔍 Analyze Audio", key="btn_upload"):
                suffix = f".{uploaded.name.rsplit('.', 1)[-1]}"
                with st.spinner("Running forensic analysis…"):
                    result = _analyse_file(uploaded.read(), suffix=suffix)

                if result.get("error") and not result.get("trust_score"):
                    st.error(f"Pipeline error: {result['error']}")
                else:
                    _verdict_ui(result)

                    # ── Liveness challenge (when CHALLENGE verdict) ────────
                    if result.get("decision") == "challenge":
                        st.divider()
                        st.subheader("🔐 Liveness Challenge Required")
                        challenge_phrase = result.get("liveness_challenge", "")
                        if challenge_phrase:
                            st.info(
                                "Please read aloud the following phrase:\n\n"
                                f'**\u201c{challenge_phrase}\u201d**'  # " and " curly quotes
                            )
                        else:
                            st.info("Please record your response to complete verification.")

                        response_audio = st.audio_input(
                            "Record your liveness response",
                            key="liveness_response_upload",
                        )
                        if response_audio is not None and st.button(
                            "✅ Submit Response", key="submit_liveness_upload"
                        ):
                            with st.spinner("Validating liveness response…"):
                                liveness_result = _analyse_file(
                                    response_audio.read(), suffix=".wav"
                                )
                            if liveness_result.get("liveness_passed"):
                                st.success("🎉 Liveness challenge passed! Access granted.")
                            else:
                                st.error(
                                    "❌ Liveness challenge failed. "
                                    f"Updated trust score: {liveness_result.get('trust_score', 0):.2%}"
                                )

    # ────────────────────────────────────────────────────────────────────────
    # Tab 2: Record live audio
    # ────────────────────────────────────────────────────────────────────────
    with tab_record:
        st.subheader("Record Live Audio")
        st.caption(
            "Use the recorder below to capture audio directly from your microphone."
        )

        recorded_audio = st.audio_input(
            "Click to start recording",
            key="live_recorder",
        )

        if recorded_audio is not None:
            st.audio(recorded_audio)
            if st.button("🔍 Analyze Recording", key="btn_record"):
                with st.spinner("Running forensic analysis…"):
                    result = _analyse_file(recorded_audio.read(), suffix=".wav")

                if result.get("error") and not result.get("trust_score"):
                    st.error(f"Pipeline error: {result['error']}")
                else:
                    _verdict_ui(result)

                    # ── Liveness challenge ─────────────────────────────────
                    if result.get("decision") == "challenge":
                        st.divider()
                        st.subheader("🔐 Liveness Challenge Required")
                        challenge_phrase = result.get("liveness_challenge", "")
                        if challenge_phrase:
                            st.info(
                                "Please read aloud the following phrase:\n\n"
                                f'**\u201c{challenge_phrase}\u201d**'  # " and " curly quotes
                            )

                        response_audio = st.audio_input(
                            "Record your liveness response",
                            key="liveness_response_record",
                        )
                        if response_audio is not None and st.button(
                            "✅ Submit Response", key="submit_liveness_record"
                        ):
                            with st.spinner("Validating liveness response…"):
                                liveness_result = _analyse_file(
                                    response_audio.read(), suffix=".wav"
                                )
                            if liveness_result.get("liveness_passed"):
                                st.success("🎉 Liveness challenge passed! Access granted.")
                            else:
                                st.error(
                                    "❌ Liveness challenge failed. "
                                    f"Updated trust score: {liveness_result.get('trust_score', 0):.2%}"
                                )

    # ────────────────────────────────────────────────────────────────────────
    # Tab 3: Batch analysis
    # ────────────────────────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Audio Analysis")
        st.caption("Upload multiple audio files to analyse them all at once.")

        batch_files = st.file_uploader(
            "Choose audio files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key="batch_uploader",
        )

        if batch_files and st.button("🔍 Analyze All", key="btn_batch"):
            import pandas as pd

            rows = []
            progress = st.progress(0.0, text="Analysing…")

            for i, f in enumerate(batch_files):
                suffix = f".{f.name.rsplit('.', 1)[-1]}"
                result = _analyse_file(f.read(), suffix=suffix)
                decision = result.get("decision", "error")
                rows.append(
                    {
                        "File": f.name,
                        "Decision": decision.upper(),
                        "Trust Score": f"{result.get('trust_score', 0):.2%}",
                        "CNN Score": f"{result.get('cnn_score', 0):.2%}",
                        "Whisper Score": f"{result.get('whisper_score', 0):.2%}",
                        "Error": result.get("error") or "—",
                    }
                )
                progress.progress((i + 1) / len(batch_files), text=f"Processed {f.name}")

            progress.empty()

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Summary statistics
            st.subheader("Summary")
            counts = df["Decision"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ PASS", counts.get("PASS", 0))
            c2.metric("⚠️ CHALLENGE", counts.get("CHALLENGE", 0))
            c3.metric("❌ REJECT", counts.get("REJECT", 0))


if __name__ == "__main__":
    main()
