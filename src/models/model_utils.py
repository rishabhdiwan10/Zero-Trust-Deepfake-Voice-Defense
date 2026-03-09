"""
src.models.model_utils
=======================
Model loading, checkpointing, and device management utilities shared across
all model components in the Zero-Trust Deepfake Voice Defense System.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> str:
    """
    Return the best available torch device string.

    Parameters
    ----------
    preferred : str | None
        Preferred device (``"cuda"``, ``"cpu"``, ``"mps"``).
        If ``None``, auto-detect.

    Returns
    -------
    str
        Device string suitable for ``torch.device()``.
    """
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str | Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save a model checkpoint to disk.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose state dict will be saved.
    optimizer : torch.optim.Optimizer
        The optimiser whose state dict will be saved.
    epoch : int
        Current training epoch.
    metrics : dict
        Dictionary of metric names → values to record alongside the checkpoint.
    checkpoint_dir : str | Path
        Directory in which to save the checkpoint.
    filename : str | None
        Override checkpoint filename. Defaults to ``checkpoint_epoch_{epoch}.pt``.

    Returns
    -------
    Path
        Absolute path to the saved checkpoint file.
    """
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = filename or f"checkpoint_epoch_{epoch:04d}.pt"
    save_path = checkpoint_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )
    logger.info("Checkpoint saved → %s", save_path)
    return save_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a model checkpoint from disk.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the ``.pt`` checkpoint file.
    model : torch.nn.Module
        Model instance to load weights into.
    optimizer : torch.optim.Optimizer | None
        Optimiser to restore state into (optional).
    device : str | None
        Device to map the checkpoint tensors to.

    Returns
    -------
    dict
        The full checkpoint dictionary (contains ``epoch``, ``metrics``, etc.).
    """
    import torch

    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        "Checkpoint loaded from %s (epoch %s, metrics: %s)",
        checkpoint_path,
        checkpoint.get("epoch", "?"),
        checkpoint.get("metrics", {}),
    )
    return checkpoint


def count_parameters(model) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model) -> None:
    """
    Freeze all backbone parameters, leaving only the classifier head trainable.

    Assumes the model has a ``fc`` or ``classifier`` attribute for the head.
    """
    head_names = {"fc", "classifier", "head"}
    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        param.requires_grad = top_level in head_names

    trainable = count_parameters(model)
    logger.info("Backbone frozen. Trainable parameters: %d", trainable)
