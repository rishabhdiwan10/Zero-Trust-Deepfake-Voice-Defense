# System Architecture

## Overview

The Zero-Trust Deepfake Voice Defense System is a multi-layered defence
pipeline that combines CNN-based audio forensics, LangGraph agentic
orchestration, and dynamic liveness verification.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Audio Input (Real-Time)                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │   Audio Preprocessor  │  Resampling (→ 16 kHz mono)
           │   + Feature Extractor │  Mel-spectrogram / MFCC / LFCC
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────────────────────────────────┐
           │                  Forensic Agent                   │
           │  ┌─────────────────┐   ┌─────────────────────┐   │
           │  │  CNN Detector   │   │  Whisper Analyser   │   │
           │  │ (ResNet / ENet) │   │  (Transcription +   │   │
           │  │ cnn_score [0,1] │   │  Artifact Analysis) │   │
           │  └────────┬────────┘   └──────────┬──────────┘   │
           └───────────┼───────────────────────┼───────────────┘
                       │                       │
           ┌───────────▼───────────────────────▼───────────────┐
           │                 Decision Agent                     │
           │  TrustScorer: weighted combination of all scores   │
           │  ThresholdEngine: pass / challenge / reject        │
           └───────┬─────────────────────┬──────────┬──────────┘
                   │                     │          │
                PASS               CHALLENGE     REJECT
                   │                     │          │
                   │          ┌──────────▼──────┐   │
                   │          │  Liveness Agent │   │
                   │          │ ChallengeGen    │   │
                   │          │ ResponseValidator│  │
                   │          └──────────┬──────┘   │
                   │                     │           │
                   │          ┌──────────▼──────┐   │
                   │          │  Decision Agent │   │
                   │          │  (re-evaluate)  │   │
                   │          └──────────┬──────┘   │
                   │                     │           │
           ┌───────▼─────────────────────▼───────────▼──────────┐
           │                   Action Router                     │
           │       "pass" | "challenge" | "reject"               │
           └─────────────────────────────────────────────────────┘
```

## Component Descriptions

### Audio Preprocessor (`src/data/audio_preprocessor.py`)
Handles resampling, mono conversion, normalisation, silence trimming, and
optional data augmentation.

### Feature Extractor (`src/data/feature_extractor.py`)
Converts raw waveforms into CNN-compatible feature maps:
- **Mel-spectrogram** (log-magnitude, 128 bins)
- **MFCC** (40 coefficients)
- **LFCC** (40 coefficients, linear-frequency scale)
- **Combined** (stacked mel + MFCC + LFCC)

### CNN Detector (`src/models/cnn_detector.py`)
ResNet or EfficientNet backbone fine-tuned for binary deepfake detection.
Returns `genuine_prob` and `synthetic_prob`.

### Whisper Analyser (`src/models/whisper_analyzer.py`)
Uses OpenAI Whisper's internal confidence signals (avg_logprob, compression
ratio, no_speech_prob) to detect synthetic speech artefacts beyond raw
transcription.

### LangGraph Orchestrator (`src/agents/orchestrator.py`)
Defines the stateful agent graph. Uses conditional edges so the liveness
branch is only triggered when the trust score is uncertain.

### Decision Engine (`src/decision/`)
Aggregates multi-layer scores into a unified trust score and evaluates it
against configurable thresholds (see `configs/thresholds_config.yaml`).

### Liveness Challenge (`src/liveness/`)
Generates unique, context-aware challenge phrases on every invocation.
Validates user responses using Whisper transcription + WER / token similarity.
