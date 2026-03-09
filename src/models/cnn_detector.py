"""
src.models.cnn_detector
========================
CNN-based deepfake audio detector.

Wraps a ResNet or EfficientNet backbone for binary classification of audio
feature maps (mel-spectrogram, MFCC, LFCC) as genuine (0) or synthetic (1).

The detector returns both a binary prediction and a confidence score in [0, 1]
representing the probability that the audio is *genuine*.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class CNNDetector:
    """
    CNN-based deepfake audio detector using a pre-trained backbone.

    Parameters
    ----------
    backbone : str
        Model backbone: ``resnet18`` | ``resnet34`` | ``resnet50`` |
        ``efficientnet_b0`` | ``efficientnet_b2``.
    num_classes : int
        Number of output classes (default: 2).
    pretrained : bool
        Whether to load ImageNet pre-trained backbone weights.
    dropout : float
        Dropout probability for the classifier head.
    device : str | None
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Auto-detected if ``None``.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        device: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout
        self._model = None
        self._device = self._resolve_device(device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build(self) -> "CNNDetector":
        """Build the model architecture and move it to the target device."""
        import torch
        import torch.nn as nn

        self._model = self._build_backbone()
        self._model = self._model.to(self._device)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature: np.ndarray) -> Dict[str, float]:
        """
        Run inference on a single feature map.

        Parameters
        ----------
        feature : np.ndarray
            Feature tensor with shape (C, H, W) or (1, C, H, W).

        Returns
        -------
        dict with keys:
            ``genuine_prob`` (float): probability of genuine audio [0, 1].
            ``synthetic_prob`` (float): probability of synthetic audio [0, 1].
            ``prediction`` (int): 0 = genuine, 1 = synthetic.
        """
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("Model not built. Call .build() first.")

        self._model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(feature).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # add batch dim
            # Ensure 3-channel input expected by ImageNet backbones
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            tensor = tensor.to(self._device)
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return {
            "genuine_prob": float(probs[0]),
            "synthetic_prob": float(probs[1]),
            "prediction": int(np.argmax(probs)),
        }

    def predict_batch(self, features: np.ndarray) -> list[Dict[str, float]]:
        """
        Run inference on a batch of feature maps.

        Parameters
        ----------
        features : np.ndarray
            Batch tensor with shape (N, C, H, W).

        Returns
        -------
        List of dicts (one per sample) with the same keys as ``predict``.
        """
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("Model not built. Call .build() first.")

        self._model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(features).float()
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            tensor = tensor.to(self._device)
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for p in probs:
            results.append(
                {
                    "genuine_prob": float(p[0]),
                    "synthetic_prob": float(p[1]),
                    "prediction": int(np.argmax(p)),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_backbone(self):
        """Construct the backbone + classifier head."""
        import torch.nn as nn

        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for CNNDetector. "
                "Run: pip install torchvision"
            ) from exc

        weights_arg = "DEFAULT" if self.pretrained else None

        if self.backbone.startswith("resnet"):
            size = self.backbone.replace("resnet", "")
            model_fn = getattr(tv_models, f"resnet{size}")
            model = model_fn(weights=weights_arg)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features, self.num_classes),
            )
        elif self.backbone.startswith("efficientnet"):
            variant = self.backbone.replace("efficientnet_", "")
            model_fn = getattr(tv_models, f"efficientnet_{variant}")
            model = model_fn(weights=weights_arg)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features, self.num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        return model

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Auto-detect the best available device if none specified."""
        if device:
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
