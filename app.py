"""
app.py
=======
Streamlit web UI for the Zero-Trust Deepfake Voice Defense System.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="VoiceGuard — Zero-Trust Voice Authentication",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — bank / security authentication theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0f1e;
    color: #e8eaf0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0d1428;
    border-right: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8892b0;
    background-color: transparent;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    color: #f0c040 !important;
    background-color: #1a2640 !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #1a2640);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    color: #f0c040 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #8892b0 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #0e2244);
    color: #f0c040;
    border: 1px solid #2a5298;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 10px 28px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a5298, #1a3a6b);
    border-color: #f0c040;
    box-shadow: 0 0 12px rgba(240,192,64,0.3);
}
[data-testid="stExpander"] {
    background-color: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #1a3a6b, #f0c040);
    border-radius: 4px;
}
.stSuccess { background-color: #0d2e1f; border-left: 4px solid #22c55e; }
.stWarning { background-color: #2d1f00; border-left: 4px solid #f0c040; }
.stError   { background-color: #2d0f0f; border-left: 4px solid #ef4444; }
.stInfo    { background-color: #0d1e2d; border-left: 4px solid #3b82f6; }
hr { border-color: #1e3a5f; }
.verdict-pass {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
}
.verdict-challenge {
    background: linear-gradient(135deg, #1a1000, #2d1f00);
    border: 1px solid #f0c040;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
}
.verdict-reject {
    background: linear-gradient(135deg, #1a0000, #2d0f0f);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
}
.verdict-title {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.verdict-subtitle { font-size: 0.9rem; color: #8892b0; margin-top: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = ["wav", "flac", "mp3", "ogg", "m4a"]
MODEL_CHECKPOINT_PATH = Path("models/best_checkpoint.pt")


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising VoiceGuard AI models…")
def _load_pipeline():
    try:
        from src.models.cnn_detector import CNNDetector
        from src.models.whisper_analyzer import WhisperAnalyzer
        from src.models.prosody_analyzer import ProsodyAnalyzer
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
        model_cfg = load_config("configs/model_config.yaml")
        m_cfg = model_cfg.get("model", {})

        cnn = CNNDetector(
            backbone=m_cfg.get("backbone", "resnet18"),
            num_classes=m_cfg.get("num_classes", 2),
            pretrained=not MODEL_CHECKPOINT_PATH.exists(),
            dropout=m_cfg.get("dropout", 0.3),
            device=device,
        ).build()

        if MODEL_CHECKPOINT_PATH.exists():
            try:
                load_checkpoint(MODEL_CHECKPOINT_PATH, cnn._model, device=device)
            except Exception as exc:
                st.warning(f"Checkpoint load warning: {exc}. Using untrained weights.")

        whisper_az = WhisperAnalyzer()
        preprocessor = AudioPreprocessor(target_sr=16_000, normalize=True)
        feature_extractor = FeatureExtractor(
            feature_type=FeatureType(m_cfg.get("input_feature", "mel_spectrogram"))
        )
        prosody = ProsodyAnalyzer(sample_rate=16_000)

        forensic_agent = ForensicAgent(
            cnn_detector=cnn,
            whisper_analyzer=whisper_az,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            prosody_analyzer=prosody,
            run_parallel=True,
        )

        trust_scorer = TrustScorer()
        threshold_engine = ThresholdEngine(pass_threshold=0.65, challenge_threshold=0.40)
        action_router = ActionRouter()
        decision_agent = DecisionAgent(trust_scorer, threshold_engine, action_router)

        challenge_gen = ChallengeGenerator()
        response_validator = ResponseValidator(
            min_token_similarity=0.40,
            max_allowed_wer=0.50,
            whisper_model_size="small",
        )
        liveness_agent = LivenessAgent(challenge_gen, response_validator)

        orchestrator = Orchestrator(forensic_agent, decision_agent, liveness_agent).build()
        pipeline = RealtimePipeline(orchestrator, pipeline_timeout=30.0)
        return pipeline, response_validator, None

    except Exception as exc:
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _analyse_file(audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
    pipeline, _, err = _load_pipeline()
    if pipeline is None:
        return {
            "error": err or "Pipeline unavailable.",
            "decision": "reject",
            "trust_score": 0.0,
            "cnn_score": 0.0,
            "whisper_score": 0.0,
            "prosody_score": 0.0,
            "transcription": "",
            "stage_latencies": {},
            "forensic_metadata": {},
            "liveness_challenge": "",
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

    result.setdefault("forensic_metadata", {})
    result.setdefault("liveness_challenge", "")
    return result


def _validate_liveness(audio_bytes: bytes, challenge_phrase: str) -> Dict[str, Any]:
    _, response_validator, err = _load_pipeline()
    if response_validator is None:
        return {"passed": False, "transcription": "", "similarity": 0.0, "wer": 1.0,
                "error": err or "Pipeline unavailable."}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        details = response_validator.validate_with_details(
            challenge=challenge_phrase,
            response_audio_path=tmp_path,
        )
    except Exception as exc:
        details = {"passed": False, "transcription": "", "similarity": 0.0, "wer": 1.0,
                   "error": str(exc)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return details


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 0.65:
        return "#22c55e"
    if score >= 0.40:
        return "#f0c040"
    return "#ef4444"


def _render_score_bar(label: str, score: float) -> None:
    color = _score_color(score)
    pct = int(score * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#8892b0;">{label}</span>
                <span style="font-size:1rem;font-weight:700;color:{color};">{pct}%</span>
            </div>
            <div style="background:#1a2640;border-radius:6px;height:10px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:6px;
                            box-shadow:0 0 8px {color}55;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_banner(decision: str, trust_score: float) -> None:
    if decision == "pass":
        css_class, icon, label = "verdict-pass", "✅", "ACCESS GRANTED"
        subtitle = "Voice authentication successful — identity verified."
        color = "#22c55e"
    elif decision == "challenge":
        css_class, icon, label = "verdict-challenge", "⚠️", "IDENTITY CHALLENGE"
        subtitle = "Trust score inconclusive — liveness verification required."
        color = "#f0c040"
    else:
        css_class, icon, label = "verdict-reject", "🚫", "ACCESS DENIED"
        subtitle = "Voice flagged as synthetic or AI-generated."
        color = "#ef4444"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="verdict-title" style="color:{color};">{icon} &nbsp; {label}</div>
            <div class="verdict-subtitle">{subtitle}</div>
            <div style="margin-top:10px;font-size:1.1rem;font-weight:600;color:{color};">
                Trust Score: {trust_score:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")


def _render_pipeline_steps(result: Dict[str, Any]) -> None:
    decision = result.get("decision", "reject")
    liveness_passed = result.get("liveness_passed")

    steps = [
        ("1", "Audio Ingestion", True),
        ("2", "CNN Forensic Analysis", result.get("cnn_score", 0) > 0),
        ("3", "Whisper Artifact Scan", result.get("whisper_score", 0) > 0),
        ("4", "Prosody Analysis", result.get("prosody_score", 0) > 0),
        ("5", "Trust Scoring", True),
        ("6", "Threshold Decision", True),
        ("7", "Liveness Challenge", decision == "challenge" or liveness_passed is not None),
    ]

    badges = ""
    for num, name, active in steps:
        color = "#f0c040" if active else "#2a3a5a"
        text_color = "#0a0f1e" if active else "#4a5568"
        badges += (
            f'<span style="display:inline-block;background:{color};color:{text_color};'
            f'border-radius:20px;padding:4px 14px;font-size:0.72rem;font-weight:700;'
            f'margin:3px 3px;letter-spacing:0.04em;">{num}. {name}</span>'
        )

    st.markdown(f'<div style="margin:12px 0 4px;">{badges}</div>', unsafe_allow_html=True)


def _render_latency_pills(latencies: Dict[str, Any]) -> None:
    if not latencies:
        return
    pills = "".join(
        f'<span style="display:inline-block;background:#0d1428;border:1px solid #1e3a5f;'
        f'border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#60a5fa;margin:2px 2px;">'
        f'⏱ {stage}: {ms} ms</span>'
        for stage, ms in latencies.items()
    )
    st.markdown(f'<div style="margin-top:6px;">{pills}</div>', unsafe_allow_html=True)


def _verdict_ui(result: Dict[str, Any]) -> None:
    decision = result.get("decision", "reject")
    trust_score = result.get("trust_score", 0.0)
    cnn_score = result.get("cnn_score", 0.0)
    whisper_score = result.get("whisper_score", 0.0)
    prosody_score = result.get("prosody_score", 0.0)
    transcription = result.get("transcription", "")
    metadata = result.get("forensic_metadata", {})
    latencies = result.get("stage_latencies", {})

    _render_pipeline_steps(result)
    st.divider()
    _render_verdict_banner(decision, trust_score)

    st.markdown("#### Forensic Score Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trust Score", f"{trust_score:.1%}", help="Weighted composite of all forensic signals.")
    with c2:
        st.metric("CNN Detector", f"{cnn_score:.1%}", help="ResNet18 spectrogram-based deepfake classifier.")
    with c3:
        st.metric("Whisper Scan", f"{whisper_score:.1%}", help="Log-probability artifact analysis via Whisper.")
    with c4:
        st.metric("Prosody", f"{prosody_score:.1%}", help="Pitch variability, jitter, shimmer, rhythm naturalness.")

    st.markdown("")
    _render_score_bar("CNN Detector", cnn_score)
    _render_score_bar("Whisper Artifact Scan", whisper_score)
    _render_score_bar("Prosody & Rhythm", prosody_score)
    _render_score_bar("Overall Trust Score", trust_score)
    st.divider()

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("📝 Transcription"):
            if transcription:
                st.markdown(
                    f'<div style="background:#111827;border-radius:8px;padding:14px;'
                    f'color:#e2e8f0;font-style:italic;">"{transcription}"</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("*No speech detected.*")

    with exp_col2:
        if metadata:
            with st.expander("🔬 Forensic Metadata"):
                items = {
                    "Avg Log Probability": round(metadata.get("avg_log_prob", 0.0), 4),
                    "Compression Ratio": round(metadata.get("compression_ratio", 0.0), 4),
                    "No-Speech Probability": round(metadata.get("no_speech_prob", 0.0), 4),
                    "Language": metadata.get("language", "—"),
                    "CNN Inference (ms)": metadata.get("cnn_inference_ms", "—"),
                    "Whisper Inference (ms)": metadata.get("whisper_inference_ms", "—"),
                }
                for k, v in items.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:4px 0;border-bottom:1px solid #1e2d4a;">'
                        f'<span style="color:#8892b0;font-size:0.82rem;">{k}</span>'
                        f'<span style="color:#f0c040;font-size:0.82rem;font-weight:600;">{v}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    if latencies:
        with st.expander("⏱ Processing Latencies"):
            _render_latency_pills(latencies)


def _liveness_ui(challenge_phrase: str, key_suffix: str) -> None:
    """Liveness challenge UI. Takes challenge_phrase directly from session state."""
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1a1000,#2d1f00);
                    border:1px solid #f0c040;border-radius:12px;padding:20px 24px;margin-top:16px;">
            <div style="font-size:1.1rem;font-weight:800;color:#f0c040;letter-spacing:0.08em;
                        text-transform:uppercase;margin-bottom:8px;">
                🔐 Liveness Challenge Required
            </div>
            <div style="color:#8892b0;font-size:0.85rem;">
                Our system requires a liveness verification to confirm you are a real person
                speaking in real time. Please read the phrase below clearly into your microphone.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    if challenge_phrase:
        st.markdown(
            f"""
            <div style="background:#0d1e2d;border:1px solid #3b82f6;border-radius:10px;
                        padding:18px 24px;text-align:center;margin:12px 0;">
                <div style="font-size:0.75rem;color:#60a5fa;text-transform:uppercase;
                            letter-spacing:0.1em;margin-bottom:8px;">Speak this phrase:</div>
                <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;
                            font-style:italic;">"{challenge_phrase}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    response_audio = st.audio_input(
        "🎙 Record your spoken response",
        key=f"liveness_response_{key_suffix}",
    )

    if response_audio is not None:
        if st.button("✅ Submit & Verify Response", key=f"submit_liveness_{key_suffix}"):
            with st.spinner("Analysing liveness response…"):
                details = _validate_liveness(response_audio.read(), challenge_phrase)
            # Store result in session state so it persists after rerun
            st.session_state[f"liveness_result_{key_suffix}"] = details

    # Render result if available (persists across reruns via session state)
    details = st.session_state.get(f"liveness_result_{key_suffix}")
    if details is not None:
        if details.get("error"):
            st.error(f"Validation error: {details['error']}")
            return

        passed = details.get("passed", False)
        similarity = details.get("similarity", 0.0)
        wer = details.get("wer", 1.0)
        transcription = details.get("transcription", "")

        if passed:
            st.markdown(
                """
                <div style="background:linear-gradient(135deg,#052e16,#064e3b);
                            border:2px solid #22c55e;border-radius:14px;
                            padding:28px 32px;text-align:center;margin-top:8px;">
                    <div style="font-size:2rem;margin-bottom:6px;">✅</div>
                    <div style="font-size:1.4rem;font-weight:900;color:#22c55e;
                                letter-spacing:0.12em;text-transform:uppercase;">
                        ACCESS GRANTED
                    </div>
                    <div style="color:#86efac;font-size:0.9rem;margin-top:8px;">
                        Liveness verified — identity confirmed as human. Authentication complete.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:linear-gradient(135deg,#1a0000,#2d0f0f);
                            border:1px solid #ef4444;border-radius:12px;
                            padding:20px 24px;text-align:center;">
                    <div style="font-size:1.2rem;font-weight:800;color:#ef4444;letter-spacing:0.08em;">
                        🚫 &nbsp; LIVENESS CHALLENGE FAILED
                    </div>
                    <div style="color:#fca5a5;font-size:0.85rem;margin-top:6px;">
                        Response did not match the challenge phrase. Please try again.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("Token Similarity", f"{similarity:.1%}",
                      help="Jaccard token overlap between challenge and response.")
        with dc2:
            st.metric("Word Error Rate", f"{wer:.1%}",
                      help="Lower is better. ≤ 20% required.")
        with dc3:
            st.metric("Result", "PASS ✅" if passed else "FAIL ❌")

        if transcription:
            st.markdown(
                f'<div style="background:#111827;border-radius:8px;padding:12px;'
                f'color:#9ca3af;font-size:0.82rem;">🎙 Heard: <em>"{transcription}"</em></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:16px 0;">
                <div style="font-size:2rem;">🏦</div>
                <div style="font-size:1.1rem;font-weight:800;color:#f0c040;letter-spacing:0.08em;">
                    VOICEGUARD
                </div>
                <div style="font-size:0.72rem;color:#8892b0;text-transform:uppercase;
                            letter-spacing:0.1em;margin-top:2px;">
                    Zero-Trust Voice Defense
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Model Status**")
        if MODEL_CHECKPOINT_PATH.exists():
            st.success("Trained checkpoint loaded")
            checkpoint_size = MODEL_CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
            st.caption(f"Checkpoint: {checkpoint_size:.1f} MB")
        else:
            st.warning(
                "No trained checkpoint found.\n\n"
                "Run `python scripts/train.py` first.\n\n"
                "Results will not be meaningful until the model is trained."
            )

        st.divider()

        st.markdown("**Security Level**")
        checkpoint_ok = MODEL_CHECKPOINT_PATH.exists()
        if checkpoint_ok:
            level_label, level_color, level_glow = "HIGH", "#22c55e", "#22c55e44"
            level_desc, level_icon, bars = "Trained model active. All defence layers operational.", "🟢", 3
        else:
            level_label, level_color, level_glow = "LOW", "#ef4444", "#ef444444"
            level_desc, level_icon, bars = "No trained checkpoint. Results are not reliable.", "🔴", 1

        bar_html = "".join(
            f'<div style="flex:1;height:8px;border-radius:4px;margin:0 2px;'
            f'background:{""+level_color if i < bars else "#1e3a5f"};'
            f'box-shadow:{"0 0 6px "+level_glow if i < bars else "none"};"></div>'
            for i in range(3)
        )
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#111827,#1a2640);
                        border:1px solid {level_color}44;border-radius:10px;padding:14px 16px;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <span style="font-size:1.4rem;">{level_icon}</span>
                    <span style="font-size:1rem;font-weight:800;color:{level_color};
                                 letter-spacing:0.1em;">{level_label}</span>
                </div>
                <div style="display:flex;margin-bottom:10px;">{bar_html}</div>
                <div style="font-size:0.74rem;color:#8892b0;line-height:1.5;">{level_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("**Detection Layers**")
        st.markdown(
            """
            <div style="font-size:0.78rem;line-height:2;color:#8892b0;">
                🤖 &nbsp;ResNet18 CNN Detector<br>
                🎙 &nbsp;Whisper Artifact Scan<br>
                📈 &nbsp;Prosody Analyser<br>
                🔐 &nbsp;Dynamic Liveness Challenge<br>
                🧮 &nbsp;Weighted Trust Scorer
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("Built with LangGraph agentic orchestration.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _sidebar()

    st.markdown(
        """
        <div style="text-align:center;padding:24px 0 16px;">
            <div style="font-size:0.78rem;color:#60a5fa;text-transform:uppercase;
                        letter-spacing:0.15em;margin-bottom:6px;">
                🏦 &nbsp; Enterprise Voice Security Platform
            </div>
            <div style="font-size:2.2rem;font-weight:900;color:#f0c040;letter-spacing:0.04em;">
                VoiceGuard Authentication
            </div>
            <div style="font-size:0.95rem;color:#8892b0;margin-top:8px;max-width:560px;
                        margin-left:auto;margin-right:auto;">
                Zero-trust deepfake voice defence — every sample is treated as potentially adversarial
                until verified through multi-layer forensic analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    tab_upload, tab_record, tab_batch = st.tabs(
        ["📂 &nbsp; Upload Audio", "🎙 &nbsp; Live Recording", "📋 &nbsp; Batch Analysis"]
    )

    # ── Tab 1: Upload ────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("#### Submit Audio for Forensic Analysis")
        st.caption(f"Supported formats: {', '.join(f'.{f}' for f in SUPPORTED_FORMATS)}")

        uploaded = st.file_uploader(
            "Drop an audio file here or click to browse",
            type=SUPPORTED_FORMATS,
        )

        if uploaded is not None:
            st.audio(uploaded)
            st.markdown("")

            if st.button("🔍 &nbsp; Run Forensic Analysis", key="btn_upload"):
                # Clear any previous liveness result when re-analysing
                st.session_state.pop("liveness_result_upload", None)
                suffix = f".{uploaded.name.rsplit('.', 1)[-1]}"
                progress_bar = st.progress(0, text="Initialising forensic pipeline…")
                progress_bar.progress(20, text="Preprocessing audio…")
                with st.spinner("Running multi-layer forensic analysis…"):
                    result = _analyse_file(uploaded.read(), suffix=suffix)
                progress_bar.progress(100, text="Analysis complete.")
                progress_bar.empty()
                st.session_state["upload_result"] = result

            # Render persisted result
            result = st.session_state.get("upload_result")
            if result is not None:
                if result.get("error") and not result.get("trust_score"):
                    st.error(f"Pipeline error: {result['error']}")
                else:
                    _verdict_ui(result)
                    if result.get("decision") == "challenge":
                        st.divider()
                        _liveness_ui(
                            result.get("liveness_challenge", ""),
                            key_suffix="upload",
                        )

    # ── Tab 2: Record ────────────────────────────────────────────────────────
    with tab_record:
        st.markdown("#### Live Microphone Recording")
        st.caption("Capture audio directly from your microphone for real-time authentication.")

        recorded_audio = st.audio_input(
            "Click the microphone to start recording",
            key="live_recorder",
        )

        if recorded_audio is not None:
            st.audio(recorded_audio)
            st.markdown("")

            if st.button("🔍 &nbsp; Analyse Recording", key="btn_record"):
                st.session_state.pop("liveness_result_record", None)
                progress_bar = st.progress(0, text="Initialising forensic pipeline…")
                progress_bar.progress(20, text="Preprocessing audio…")
                with st.spinner("Running multi-layer forensic analysis…"):
                    result = _analyse_file(recorded_audio.read(), suffix=".wav")
                progress_bar.progress(100, text="Analysis complete.")
                progress_bar.empty()
                st.session_state["record_result"] = result

        result = st.session_state.get("record_result")
        if result is not None:
            if result.get("error") and not result.get("trust_score"):
                st.error(f"Pipeline error: {result['error']}")
            else:
                _verdict_ui(result)
                if result.get("decision") == "challenge":
                    st.divider()
                    _liveness_ui(
                        result.get("liveness_challenge", ""),
                        key_suffix="record",
                    )

    # ── Tab 3: Batch ─────────────────────────────────────────────────────────
    with tab_batch:
        st.markdown("#### Batch Forensic Analysis")
        st.caption("Upload multiple files to screen them all at once.")

        batch_files = st.file_uploader(
            "Choose audio files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key="batch_uploader",
        )

        if batch_files and st.button("🔍 &nbsp; Analyse All Files", key="btn_batch"):
            import pandas as pd

            rows = []
            progress = st.progress(0.0, text="Analysing…")

            for i, f in enumerate(batch_files):
                suffix = f".{f.name.rsplit('.', 1)[-1]}"
                result = _analyse_file(f.read(), suffix=suffix)
                decision = result.get("decision", "error")
                rows.append({
                    "File": f.name,
                    "Decision": decision.upper(),
                    "Trust Score": f"{result.get('trust_score', 0):.1%}",
                    "CNN Score": f"{result.get('cnn_score', 0):.1%}",
                    "Whisper Score": f"{result.get('whisper_score', 0):.1%}",
                    "Prosody Score": f"{result.get('prosody_score', 0):.1%}",
                    "Total (ms)": result.get("stage_latencies", {}).get("total_ms", "—"),
                    "Error": result.get("error") or "—",
                })
                progress.progress((i + 1) / len(batch_files), text=f"Processed: {f.name}")

            progress.empty()
            st.session_state["batch_rows"] = rows

        rows = st.session_state.get("batch_rows")
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)

            def _colour_decision(val: str) -> str:
                return {"PASS": "color:#22c55e;font-weight:700",
                        "CHALLENGE": "color:#f0c040;font-weight:700",
                        "REJECT": "color:#ef4444;font-weight:700"}.get(val, "")

            st.dataframe(
                df.style.map(_colour_decision, subset=["Decision"]),
                use_container_width=True,
            )

            st.divider()
            st.markdown("#### Batch Summary")
            counts = df["Decision"].value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Files", len(df))
            m2.metric("✅ PASS", counts.get("PASS", 0))
            m3.metric("⚠️ CHALLENGE", counts.get("CHALLENGE", 0))
            m4.metric("🚫 REJECT", counts.get("REJECT", 0))

            try:
                import re
                df["_trust_float"] = df["Trust Score"].apply(
                    lambda x: float(re.sub(r"[%]", "", x)) / 100
                )
                st.markdown("")
                _render_score_bar("Average Trust Score", df["_trust_float"].mean())
            except Exception:
                pass


if __name__ == "__main__":
    main()