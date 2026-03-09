"""
src.data.dataset_loader
=======================
Unified dataset loader supporting multiple anti-spoofing / deepfake audio
datasets:

  - ASVspoof 2019 (LA and PA partitions)
  - ASVspoof 5
  - In-The-Wild deepfake dataset
  - Custom / self-generated samples produced by modern TTS / voice-cloning
    tools (ElevenLabs, Bark, XTTS, etc.)

The loader returns a consistent interface regardless of the underlying dataset
format, making it straightforward to mix datasets during training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


class DatasetType(str, Enum):
    """Supported dataset identifiers."""

    ASVSPOOF2019 = "asvspoof2019"
    ASVSPOOF5 = "asvspoof5"
    IN_THE_WILD = "in_the_wild"
    CUSTOM = "custom"


class Split(str, Enum):
    """Dataset partition."""

    TRAIN = "train"
    DEV = "dev"
    EVAL = "eval"


@dataclass
class AudioSample:
    """Represents a single labelled audio sample."""

    file_path: Path
    label: int  # 0 = genuine, 1 = synthetic / spoof
    dataset: DatasetType
    speaker_id: Optional[str] = None
    system_id: Optional[str] = None  # TTS/VC system identifier if known
    metadata: dict = field(default_factory=dict)


class DatasetLoader:
    """
    Unified loader for anti-spoofing / deepfake audio datasets.

    Parameters
    ----------
    dataset_type : DatasetType
        Which dataset to load.
    root_dir : str | Path
        Root directory where the dataset is stored.
    split : Split
        Which partition to load (train / dev / eval).
    max_samples : int | None
        If set, limit the number of samples loaded (useful for quick tests).
    """

    def __init__(
        self,
        dataset_type: DatasetType,
        root_dir: str | Path,
        split: Split = Split.TRAIN,
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataset_type = DatasetType(dataset_type)
        self.root_dir = Path(root_dir)
        self.split = Split(split)
        self.max_samples = max_samples
        self._samples: List[AudioSample] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "DatasetLoader":
        """Parse the dataset metadata and populate the sample list."""
        loader_map = {
            DatasetType.ASVSPOOF2019: self._load_asvspoof2019,
            DatasetType.ASVSPOOF5: self._load_asvspoof5,
            DatasetType.IN_THE_WILD: self._load_in_the_wild,
            DatasetType.CUSTOM: self._load_custom,
        }
        loader_fn = loader_map[self.dataset_type]
        loader_fn()
        if self.max_samples is not None:
            self._samples = self._samples[: self.max_samples]
        return self

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self._samples[idx]

    @property
    def samples(self) -> List[AudioSample]:
        """Return the full list of loaded samples."""
        return self._samples

    def get_labels(self) -> List[int]:
        """Return a list of integer labels (0 = genuine, 1 = synthetic)."""
        return [s.label for s in self._samples]

    def class_distribution(self) -> dict:
        """Return a dict with counts for each class."""
        labels = self.get_labels()
        return {
            "genuine": labels.count(0),
            "synthetic": labels.count(1),
            "total": len(labels),
        }

    # ------------------------------------------------------------------
    # Private loaders — one per dataset format
    # ------------------------------------------------------------------

    def _load_asvspoof2019(self) -> None:
        """
        Load ASVspoof 2019 dataset.

        Expected directory layout::

            root_dir/
              LA/
                ASVspoof2019_LA_{split}/
                  flac/
                    *.flac
                ASVspoof2019_LA_cm_protocols/
                  ASVspoof2019.LA.cm.{split}.trl.txt
        """
        protocol_map = {
            Split.TRAIN: "ASVspoof2019.LA.cm.train.trn.txt",
            Split.DEV: "ASVspoof2019.LA.cm.dev.trl.txt",
            Split.EVAL: "ASVspoof2019.LA.cm.eval.trl.txt",
        }
        protocol_file = (
            self.root_dir
            / "LA"
            / "ASVspoof2019_LA_cm_protocols"
            / protocol_map[self.split]
        )
        audio_dir = (
            self.root_dir
            / "LA"
            / f"ASVspoof2019_LA_{self.split.value}"
            / "flac"
        )

        if not protocol_file.exists():
            raise FileNotFoundError(
                f"ASVspoof 2019 protocol file not found: {protocol_file}"
            )

        with open(protocol_file, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id, utt_id, _, system_id, label_str = parts[:5]
                label = 0 if label_str == "bonafide" else 1
                file_path = audio_dir / f"{utt_id}.flac"
                self._samples.append(
                    AudioSample(
                        file_path=file_path,
                        label=label,
                        dataset=DatasetType.ASVSPOOF2019,
                        speaker_id=speaker_id,
                        system_id=system_id,
                    )
                )

    def _load_asvspoof5(self) -> None:
        """
        Load ASVspoof 5 dataset.

        Expected directory layout::

            root_dir/
              flac/
                *.flac
              protocols/
                ASVspoof5.{split}.metadata.txt
        """
        protocol_file = (
            self.root_dir / "protocols" / f"ASVspoof5.{self.split.value}.metadata.txt"
        )
        audio_dir = self.root_dir / "flac"

        if not protocol_file.exists():
            raise FileNotFoundError(
                f"ASVspoof 5 protocol file not found: {protocol_file}"
            )

        with open(protocol_file, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                utt_id, speaker_id, system_id, label_str = parts[:4]
                label = 0 if label_str == "bonafide" else 1
                file_path = audio_dir / f"{utt_id}.flac"
                self._samples.append(
                    AudioSample(
                        file_path=file_path,
                        label=label,
                        dataset=DatasetType.ASVSPOOF5,
                        speaker_id=speaker_id,
                        system_id=system_id,
                    )
                )

    def _load_in_the_wild(self) -> None:
        """
        Load the In-The-Wild deepfake audio dataset.

        Expected directory layout::

            root_dir/
              genuine/
                *.wav
              fake/
                *.wav
        """
        for label, subdir in [(0, "genuine"), (1, "fake")]:
            audio_dir = self.root_dir / subdir
            if not audio_dir.exists():
                continue
            for audio_file in sorted(audio_dir.glob("*.wav")):
                self._samples.append(
                    AudioSample(
                        file_path=audio_file,
                        label=label,
                        dataset=DatasetType.IN_THE_WILD,
                    )
                )

    def _load_custom(self) -> None:
        """
        Load a custom dataset following the simple directory convention::

            root_dir/
              genuine/
                *.wav | *.flac | *.mp3
              synthetic/
                *.wav | *.flac | *.mp3
              metadata.csv  (optional — columns: filename, label, system_id)
        """
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}

        for label, subdir in [(0, "genuine"), (1, "synthetic")]:
            audio_dir = self.root_dir / subdir
            if not audio_dir.exists():
                continue
            for audio_file in sorted(audio_dir.iterdir()):
                if audio_file.suffix.lower() in audio_extensions:
                    self._samples.append(
                        AudioSample(
                            file_path=audio_file,
                            label=label,
                            dataset=DatasetType.CUSTOM,
                        )
                    )
