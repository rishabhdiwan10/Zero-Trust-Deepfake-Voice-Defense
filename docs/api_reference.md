# API Reference

## src.data

### `DatasetLoader`
```python
DatasetLoader(dataset_type, root_dir, split="train", max_samples=None)
```
Unified loader for anti-spoofing / deepfake audio datasets.

| Method | Returns | Description |
|--------|---------|-------------|
| `.load()` | `DatasetLoader` | Parse metadata and populate sample list |
| `.samples` | `List[AudioSample]` | All loaded samples |
| `.get_labels()` | `List[int]` | List of integer labels (0/1) |
| `.class_distribution()` | `dict` | Count of genuine/synthetic/total |

### `AudioPreprocessor`
```python
AudioPreprocessor(target_sr=16000, target_duration=None, normalize=True,
                  trim_silence=True, trim_top_db=30.0, augment=False)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.process(waveform, sr)` | `(np.ndarray, int)` | Full preprocessing chain |

### `FeatureExtractor`
```python
FeatureExtractor(feature_type="mel_spectrogram", sample_rate=16000, ...)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.extract(waveform)` | `np.ndarray (C, H, W)` | Extract CNN-ready features |

### `SyntheticGenerator`
```python
SyntheticGenerator(backend="gtts", output_dir="data/synthetic", ...)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.generate(texts)` | `List[GenerationResult]` | Generate audio for text list |
| `.generate_from_file(text_file)` | `List[GenerationResult]` | Generate from file |

---

## src.models

### `CNNDetector`
```python
CNNDetector(backbone="resnet34", num_classes=2, pretrained=True,
            dropout=0.3, device=None)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.build()` | `CNNDetector` | Build model architecture |
| `.predict(feature)` | `dict` | Single-sample inference |
| `.predict_batch(features)` | `List[dict]` | Batch inference |

Prediction dict keys: `genuine_prob`, `synthetic_prob`, `prediction`.

### `WhisperAnalyzer`
```python
WhisperAnalyzer(model_size="base", device=None, fp16=True)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.load()` | `WhisperAnalyzer` | Load Whisper model |
| `.analyze(audio_path)` | `WhisperAnalysisResult` | Analyse audio file |
| `.analyze_waveform(waveform, sr)` | `WhisperAnalysisResult` | Analyse waveform |

---

## src.decision

### `TrustScorer`
```python
TrustScorer(cnn_weight=0.50, whisper_weight=0.30, liveness_weight=0.20,
            liveness_pass_bonus=1.0, liveness_fail_penalty=0.0)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.score(cnn_score, whisper_score, liveness_passed)` | `float` | Unified trust score |
| `.breakdown(...)` | `dict` | Per-layer contribution breakdown |

### `ThresholdEngine`
```python
ThresholdEngine(pass_threshold=0.80, challenge_threshold=0.40,
                enable_quality_differentiation=True, ...)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.evaluate(trust_score, cnn_score, audio_metadata)` | `str` | `"pass"/"challenge"/"reject"` |
| `.get_thresholds()` | `dict` | Configured threshold values |

### `ActionRouter`
```python
ActionRouter(on_pass=None, on_challenge=None, on_reject=None)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.route(decision)` | `str` | Execute action and return description |
| `.register_handler(decision, handler)` | `None` | Register callback |

---

## src.liveness

### `ChallengeGenerator`
```python
ChallengeGenerator(template_bank=None, difficulty="medium",
                   include_session_token=True, seed=None)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.generate(context=None)` | `str` | Single dynamic challenge phrase |
| `.generate_batch(n, context)` | `List[str]` | N distinct challenges |

### `ResponseValidator`
```python
ResponseValidator(min_token_similarity=0.70, max_allowed_wer=0.20,
                  whisper_model_size="base", device=None)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `.validate(challenge, response_audio_path)` | `bool` | Pass/fail decision |
| `.validate_with_details(challenge, response_audio_path)` | `dict` | Detailed result |

---

## src.utils

### `load_config`
```python
load_config(config_path, overrides=None) -> dict
```
Load a YAML config file with optional dot-notation overrides.

### `Timer`
```python
with Timer("label", log=True) as t:
    ...
print(t.elapsed_ms)
```

### `timeit`
```python
@timeit("my_function")
def my_function(): ...
```

### `get_logger`
```python
logger = get_logger(__name__, level=logging.INFO, json_format=False)
```

### `load_audio` / `save_audio`
```python
waveform, sr = load_audio(path, target_sr=16000, mono=True)
save_audio(waveform, path, sample_rate=16000)
```

---

## src.pipeline

### `RealtimePipeline`
```python
RealtimePipeline(orchestrator, pipeline_timeout=5.0)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `await .process(audio_path)` | `dict` | Async end-to-end pipeline |
| `.process_sync(audio_path)` | `dict` | Synchronous wrapper |
| `await .process_stream(audio_chunks, sr)` | `dict` | Streaming input |

### `BatchPipeline`
```python
BatchPipeline(realtime_pipeline, max_concurrent=4)
```
| Method | Returns | Description |
|--------|---------|-------------|
| `await .run_async(audio_paths, labels)` | `List[dict]` | Async batch processing |
| `.run(audio_paths, labels)` | `List[dict]` | Synchronous wrapper |
| `.save_results_csv(results, path)` | `None` | Save to CSV |
| `.compute_metrics(results)` | `dict` | Accuracy, EER etc. |
