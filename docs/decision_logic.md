# Decision Logic Documentation

## Overview

The Zero-Trust Deepfake Voice Defense System uses a multi-layer scoring
approach with explicit, configurable thresholds to make access control
decisions. This document describes the complete decision logic.

---

## Trust Score Computation

The **TrustScorer** (`src/decision/trust_scorer.py`) aggregates three
per-layer scores into a single trust score in [0, 1]:

```
trust_score = w_cnn × cnn_score
            + w_whisper × whisper_score
            + w_liveness × liveness_score
```

Default weights (configurable in `configs/thresholds_config.yaml`):

| Layer | Weight | Score Source |
|-------|--------|-------------|
| CNN | 0.50 | `CNNDetector.predict()` — genuine probability |
| Whisper | 0.30 | `WhisperAnalyzer.analyze()` — heuristic genuine score |
| Liveness | 0.20 | 1.0 if passed, 0.0 if failed, neutral if not attempted |

---

## Threshold Evaluation

The **ThresholdEngine** (`src/decision/threshold_engine.py`) maps a trust
score to one of three decisions:

```
trust_score ≥ 0.80                          → PASS
0.40 ≤ trust_score < 0.80                  → CHALLENGE
trust_score < 0.40                          → REJECT (default)
                                              (see quality differentiation below)
```

All thresholds are configurable via `configs/thresholds_config.yaml`.

---

## Poor-Quality vs. Synthetic Differentiation

A key challenge in deepfake detection is distinguishing between:
- **Genuine audio degraded by noise / poor conditions** (should CHALLENGE, not REJECT)
- **Synthetic audio** (should REJECT immediately)

The system addresses this with a **quality differentiation path**:

### Logic

```python
if trust_score < challenge_threshold:
    if (snr_db < low_snr_threshold_db OR rms_energy < low_energy_threshold)
       AND cnn_score < poor_quality_cnn_score_max:
        decision = CHALLENGE  # benefit of the doubt
    else:
        decision = REJECT
```

### Parameters (defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `low_snr_threshold_db` | 10.0 dB | SNR below this → "poor quality" |
| `low_energy_threshold` | 0.01 | RMS energy below this → "poor quality" |
| `poor_quality_cnn_score_max` | 0.60 | Max CNN score for quality escalation |

### Rationale

- Synthetic speech from modern TTS tools typically has **high CNN scores**
  (low genuine probability) AND normal energy/SNR.
- Genuine speech in noisy conditions may have low CNN scores but also shows
  low SNR and energy → the system escalates to CHALLENGE instead of REJECT
  to avoid false positives.

---

## Decision Flow Diagram

```
                    ┌─────────────────┐
                    │   Trust Score   │
                    └────────┬────────┘
                             │
               ┌─────────────▼──────────────┐
               │   score ≥ pass_threshold?  │ YES → PASS
               └─────────────┬──────────────┘
                             │ NO
               ┌─────────────▼──────────────┐
               │ score ≥ challenge_threshold?│ YES → CHALLENGE
               └─────────────┬──────────────┘
                             │ NO
               ┌─────────────▼──────────────────────┐
               │  Quality Differentiation Enabled?  │ NO → REJECT
               └─────────────┬──────────────────────┘
                             │ YES
               ┌─────────────▼───────────────────────────────┐
               │  Poor SNR or Low Energy AND CNN score < max?│
               │  → YES: CHALLENGE   NO: REJECT               │
               └─────────────────────────────────────────────┘
```

---

## Liveness Challenge Retry Logic

When a decision of CHALLENGE is reached:

1. The system presents a **dynamic, unique challenge phrase** (via `ChallengeGenerator`)
2. The user speaks the phrase; the response is recorded
3. `ResponseValidator` transcribes with Whisper and computes:
   - Jaccard token similarity ≥ 0.70 (configurable)
   - Word Error Rate ≤ 0.20 (configurable)
4. If the challenge is **passed**: trust score increases by +0.20, pipeline re-evaluates
5. If the challenge is **failed**: trust score decreases by -0.10, retry count increments
6. After `max_liveness_retries` failures → hard REJECT (no further retries)

---

## Configuration Reference

See `configs/thresholds_config.yaml` for all configurable parameters.
