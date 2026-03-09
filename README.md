# Zero-Trust Deepfake Voice Defense System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

A multi-layered **Zero-Trust Deepfake Voice Defense System** combining CNN-based audio forensics, LangGraph agentic orchestration, and dynamic liveness verification to detect and defend against AI-generated voice attacks in real time.

---

## Overview / Motivation

Voice-based authentication systems face an escalating threat from AI-generated deepfake audio. Modern voice-cloning tools (ElevenLabs, Bark, XTTS, etc.) can produce highly convincing synthetic speech that defeats simple replay-detection or speaker-verification approaches.

This project adopts a **zero-trust** philosophy: every voice sample is treated as potentially adversarial until proven genuine through multiple independent verification layers. The system combines:

1. **CNN-based audio forensics** — spectral and artifact analysis to detect synthetic speech patterns
2. **LangGraph agentic orchestration** — multi-agent decision pipeline with explicit state management
3. **Dynamic liveness verification** — context-aware, randomised challenges to defeat replay attacks

---

## Key Features

- 🎯 **Multi-dataset support**: ASVspoof 2019, ASVspoof 5, In-The-Wild, and custom samples from modern TTS/voice-cloning APIs
- 🤖 **Agentic pipeline**: LangGraph state machine with forensic, liveness, and decision agents
- 🔐 **Dynamic liveness challenges**: randomised, context-aware prompts — NOT fixed phrases
- ⚡ **Latency-aware design**: per-layer latency budgets, async pipeline, benchmarking utilities
- 🎚️ **Explicit decision thresholds**: pass / liveness-challenge / reject logic with poor-quality vs. synthetic differentiation
- 📊 **Comprehensive evaluation**: EER, accuracy, confusion matrix, latency profiling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Audio Input (Real-Time)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   Audio Preprocessor  │  resampling, normalisation,
          │   + Feature Extractor │  mel-spectrogram, MFCC, LFCC
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    Forensic Agent     │  CNN deepfake detector
          │  (CNN + Whisper)      │  + Whisper artifact analysis
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    Decision Agent     │  threshold engine:
          │  (Trust Scorer)       │  pass / challenge / reject
          └──────┬────────┬───────┘
                 │        │
        PASS     │        │  CHALLENGE / UNCERTAIN
                 │   ┌────▼───────────────┐
                 │   │   Liveness Agent   │  dynamic challenge
                 │   │ (Challenge Gen)    │  + response validation
                 │   └────────────────────┘
                 │
          ┌──────▼──────┐
          │   Decision  │  final trust decision
          └─────────────┘
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rishabhdiwan10/Zero-Trust-Deepfake-Voice-Defense.git
cd Zero-Trust-Deepfake-Voice-Defense

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install all dependencies
make install
# or manually:
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```python
from src.pipeline.realtime_pipeline import RealtimePipeline
from src.utils.config_loader import load_config

# Load configuration
pipeline_cfg = load_config("configs/pipeline_config.yaml")
threshold_cfg = load_config("configs/thresholds_config.yaml")

# Initialise and run
pipeline = RealtimePipeline(pipeline_cfg, threshold_cfg)
result = await pipeline.process("path/to/audio.wav")
print(result)  # {"decision": "pass", "trust_score": 0.92, ...}
```

---

## Project Structure

```
Zero-Trust-Deepfake-Voice-Defense/
├── configs/                    # All YAML configuration files
├── src/
│   ├── data/                   # Dataset loaders, preprocessors, feature extractors
│   ├── models/                 # CNN detector, Whisper analyser, utilities
│   ├── agents/                 # LangGraph orchestrator and individual agents
│   ├── liveness/               # Dynamic challenge generation and validation
│   ├── decision/               # Threshold engine, trust scorer, action router
│   ├── pipeline/               # Real-time async and batch pipelines
│   └── utils/                  # Logging, timing, config loading, audio I/O
├── tests/                      # Unit and integration tests
├── notebooks/                  # Jupyter notebooks for EDA, training, benchmarks
├── scripts/                    # CLI scripts for training, evaluation, benchmarking
├── data/                       # Dataset placeholder (not committed)
├── models/                     # Model checkpoint placeholder (not committed)
└── docs/                       # Architecture, decision logic, API reference docs
```

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/model_config.yaml` | CNN architecture, training hyperparameters |
| `configs/pipeline_config.yaml` | End-to-end pipeline settings, async/streaming options |
| `configs/thresholds_config.yaml` | Pass/challenge/reject thresholds and quality differentiation |
| `configs/liveness_config.yaml` | Challenge prompt types, timeout, difficulty levels |
| `configs/latency_config.yaml` | Per-layer latency budgets, profiling toggles |

---

## Professor Feedback Addressed

| Feedback | How It Is Addressed |
|---|---|
| **Update dataset** | `src/data/dataset_loader.py` supports ASVspoof 2019, ASVspoof 5, In-The-Wild, and custom datasets; `src/data/synthetic_generator.py` generates samples from modern TTS APIs |
| **Clarify decision logic** | `src/decision/threshold_engine.py` and `configs/thresholds_config.yaml` define explicit pass/challenge/reject thresholds and a poor-quality vs. synthetic differentiation path |
| **Dynamic liveness challenges** | `src/liveness/challenge_generator.py` and `src/liveness/prompt_templates.py` implement randomised, context-aware prompts — never fixed phrases |
| **Latency considerations** | `configs/latency_config.yaml` defines per-layer budgets; `src/utils/timer.py` provides profiling decorators; `src/pipeline/realtime_pipeline.py` uses async processing; `scripts/benchmark_latency.py` profiles the full stack |

---

## Development Commands

```bash
make install        # Install dependencies
make test           # Run tests with coverage
make lint           # Lint with flake8 + black + isort
make format         # Auto-format code
make benchmark      # Run latency benchmarks
make train          # Train the CNN detector
make evaluate       # Evaluate model performance
make clean          # Remove build artifacts
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please run `make lint` and `make test` before submitting a PR.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Rishabh Diwan
