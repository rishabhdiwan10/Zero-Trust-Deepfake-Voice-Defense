"""
Tests for src.models.cnn_detector — CNN deepfake audio detector.
"""

import numpy as np
import pytest

from src.models.cnn_detector import CNNDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_feature() -> np.ndarray:
    """Return a random mel-spectrogram feature tensor (1, 128, 128)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, 128, 128)).astype(np.float32)


@pytest.fixture()
def detector_cpu() -> CNNDetector:
    """Build a small ResNet18 detector on CPU (no pretrained weights)."""
    return CNNDetector(
        backbone="resnet18",
        num_classes=2,
        pretrained=False,
        device="cpu",
    ).build()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCNNDetector:
    def test_build_succeeds(self) -> None:
        detector = CNNDetector(
            backbone="resnet18",
            pretrained=False,
            device="cpu",
        )
        detector.build()
        assert detector._model is not None

    def test_predict_returns_expected_keys(
        self, detector_cpu: CNNDetector, mock_feature: np.ndarray
    ) -> None:
        result = detector_cpu.predict(mock_feature)
        assert "genuine_prob" in result
        assert "synthetic_prob" in result
        assert "prediction" in result

    def test_probs_sum_to_one(
        self, detector_cpu: CNNDetector, mock_feature: np.ndarray
    ) -> None:
        result = detector_cpu.predict(mock_feature)
        total = result["genuine_prob"] + result["synthetic_prob"]
        assert abs(total - 1.0) < 1e-5

    def test_probs_in_range(
        self, detector_cpu: CNNDetector, mock_feature: np.ndarray
    ) -> None:
        result = detector_cpu.predict(mock_feature)
        assert 0.0 <= result["genuine_prob"] <= 1.0
        assert 0.0 <= result["synthetic_prob"] <= 1.0

    def test_prediction_is_binary(
        self, detector_cpu: CNNDetector, mock_feature: np.ndarray
    ) -> None:
        result = detector_cpu.predict(mock_feature)
        assert result["prediction"] in (0, 1)

    def test_predict_batch(
        self, detector_cpu: CNNDetector, mock_feature: np.ndarray
    ) -> None:
        batch = np.stack([mock_feature, mock_feature], axis=0)  # (2, 1, 128, 128)
        results = detector_cpu.predict_batch(batch)
        assert len(results) == 2
        for r in results:
            assert "genuine_prob" in r

    def test_predict_without_build_raises(self, mock_feature: np.ndarray) -> None:
        detector = CNNDetector(backbone="resnet18", pretrained=False, device="cpu")
        with pytest.raises(RuntimeError, match="build"):
            detector.predict(mock_feature)

    def test_unsupported_backbone_raises(self) -> None:
        detector = CNNDetector(backbone="vgg16", pretrained=False, device="cpu")
        with pytest.raises(ValueError, match="Unsupported backbone"):
            detector.build()
