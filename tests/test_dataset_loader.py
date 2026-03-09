"""
Tests for src.data.dataset_loader — multi-dataset loading support.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.data.dataset_loader import AudioSample, DatasetLoader, DatasetType, Split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def custom_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal custom dataset directory structure."""
    genuine_dir = tmp_path / "genuine"
    synthetic_dir = tmp_path / "synthetic"
    genuine_dir.mkdir()
    synthetic_dir.mkdir()

    # Create dummy audio files
    for i in range(3):
        (genuine_dir / f"genuine_{i}.wav").write_bytes(b"\x00" * 100)
    for i in range(2):
        (synthetic_dir / f"synthetic_{i}.wav").write_bytes(b"\x00" * 100)

    return tmp_path


@pytest.fixture()
def in_the_wild_dir(tmp_path: Path) -> Path:
    """Create a minimal In-The-Wild dataset directory structure."""
    genuine_dir = tmp_path / "genuine"
    fake_dir = tmp_path / "fake"
    genuine_dir.mkdir()
    fake_dir.mkdir()

    for i in range(2):
        (genuine_dir / f"real_{i}.wav").write_bytes(b"\x00" * 100)
    for i in range(3):
        (fake_dir / f"fake_{i}.wav").write_bytes(b"\x00" * 100)

    return tmp_path


# ---------------------------------------------------------------------------
# Custom dataset tests
# ---------------------------------------------------------------------------


class TestCustomDatasetLoader:
    def test_load_returns_correct_count(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
        ).load()
        assert len(loader) == 5

    def test_labels_are_correct(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
        ).load()
        labels = loader.get_labels()
        assert labels.count(0) == 3  # genuine
        assert labels.count(1) == 2  # synthetic

    def test_class_distribution(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
        ).load()
        dist = loader.class_distribution()
        assert dist["total"] == 5
        assert dist["genuine"] == 3
        assert dist["synthetic"] == 2

    def test_max_samples_limit(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
            max_samples=2,
        ).load()
        assert len(loader) == 2

    def test_iteration(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
        ).load()
        for sample in loader:
            assert isinstance(sample, AudioSample)
            assert sample.label in (0, 1)
            assert sample.dataset == DatasetType.CUSTOM

    def test_getitem(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
        ).load()
        sample = loader[0]
        assert isinstance(sample, AudioSample)

    def test_samples_property(self, custom_dataset_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.CUSTOM,
            root_dir=custom_dataset_dir,
        ).load()
        assert isinstance(loader.samples, list)


# ---------------------------------------------------------------------------
# In-The-Wild dataset tests
# ---------------------------------------------------------------------------


class TestInTheWildLoader:
    def test_load_returns_correct_count(self, in_the_wild_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.IN_THE_WILD,
            root_dir=in_the_wild_dir,
        ).load()
        assert len(loader) == 5

    def test_labels_correct(self, in_the_wild_dir: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.IN_THE_WILD,
            root_dir=in_the_wild_dir,
        ).load()
        labels = loader.get_labels()
        assert labels.count(0) == 2
        assert labels.count(1) == 3


# ---------------------------------------------------------------------------
# ASVspoof 2019 protocol parsing
# ---------------------------------------------------------------------------


class TestASVspoof2019Loader:
    def test_missing_protocol_raises(self, tmp_path: Path) -> None:
        loader = DatasetLoader(
            dataset_type=DatasetType.ASVSPOOF2019,
            root_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_from_protocol(self, tmp_path: Path) -> None:
        # Create minimal directory structure
        proto_dir = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols"
        audio_dir = tmp_path / "LA" / "ASVspoof2019_LA_train" / "flac"
        proto_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        # Write a minimal protocol file
        proto_file = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
        proto_file.write_text(
            "LA_0001 LA_T_0000001 - - bonafide\n"
            "LA_0001 LA_T_0000002 - A01 spoof\n"
        )
        # Create dummy audio files
        (audio_dir / "LA_T_0000001.flac").write_bytes(b"\x00" * 100)
        (audio_dir / "LA_T_0000002.flac").write_bytes(b"\x00" * 100)

        loader = DatasetLoader(
            dataset_type=DatasetType.ASVSPOOF2019,
            root_dir=tmp_path,
            split=Split.TRAIN,
        ).load()

        assert len(loader) == 2
        labels = loader.get_labels()
        assert 0 in labels  # bonafide
        assert 1 in labels  # spoof
