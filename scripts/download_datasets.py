"""
scripts/download_datasets.py
===============================
Dataset download and setup helper for the Zero-Trust Deepfake Voice Defense
System.

Supports:
  - ASVspoof 2019 LA (automatic download via Edinburgh DataShare)
  - In-The-Wild dataset (automatic download)
  - Custom dataset directory structure creation

Usage::

    # Download ASVspoof 2019 LA
    python scripts/download_datasets.py --dataset asvspoof2019

    # Download In-The-Wild
    python scripts/download_datasets.py --dataset in_the_wild

    # Set up custom dataset directory structure
    python scripts/download_datasets.py --dataset custom

    # Download all supported datasets
    python scripts/download_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data")

# ASVspoof 2019 LA dataset URLs (Edinburgh DataShare)
ASVSPOOF2019_URLS = {
    "train": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
}

# In-The-Wild dataset
IN_THE_WILD_URL = "https://deepfake-demo.aisec.fraunhofer.de/in_the_wild"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and set up datasets for training."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asvspoof2019", "in_the_wild", "custom", "all"],
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist.",
    )
    return parser.parse_args()


def download_file(url: str, output_path: Path, retries: int = 3) -> bool:
    """Download a file from URL with retry logic."""
    try:
        import requests
    except ImportError:
        # Fallback to wget/curl
        return _download_with_cli(url, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            logger.info("Downloading %s (attempt %d/%d)...", url, attempt + 1, retries)
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        if downloaded % (1024 * 1024 * 10) < 8192:
                            logger.info("  %.1f%% (%d MB)", pct, downloaded // (1024 * 1024))

            logger.info("Downloaded → %s (%d bytes)", output_path, downloaded)
            return True
        except Exception as exc:
            logger.warning("Download attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)

    logger.error("Failed to download %s after %d attempts.", url, retries)
    return False


def _download_with_cli(url: str, output_path: Path) -> bool:
    """Fallback download using wget or curl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try wget
    try:
        subprocess.run(
            ["wget", "-O", str(output_path), url],
            check=True, timeout=600,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try curl
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            check=True, timeout=600,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    logger.error(
        "Neither wget nor curl available. Install requests: pip install requests"
    )
    return False


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract a zip or tar archive."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        logger.info("Extracting ZIP: %s → %s", archive_path, extract_dir)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif archive_path.suffix in (".gz", ".tgz", ".tar"):
        logger.info("Extracting TAR: %s → %s", archive_path, extract_dir)
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        logger.warning("Unknown archive format: %s", archive_path)


def setup_asvspoof2019(data_dir: Path, force: bool = False) -> None:
    """
    Download and set up ASVspoof 2019 LA dataset.

    Expected result::

        data/asvspoof2019/
          LA/
            ASVspoof2019_LA_train/flac/*.flac
            ASVspoof2019_LA_dev/flac/*.flac
            ASVspoof2019_LA_eval/flac/*.flac
            ASVspoof2019_LA_cm_protocols/
              ASVspoof2019.LA.cm.train.trn.txt
              ASVspoof2019.LA.cm.dev.trl.txt
              ASVspoof2019.LA.cm.eval.trl.txt
    """
    target_dir = data_dir / "asvspoof2019"
    zip_path = data_dir / "downloads" / "LA.zip"

    if target_dir.exists() and not force:
        logger.info("ASVspoof 2019 already exists at %s. Use --force to re-download.", target_dir)
        return

    print("\n" + "=" * 70)
    print("ASVspoof 2019 LA Dataset Download")
    print("=" * 70)
    print()
    print("The ASVspoof 2019 LA dataset is hosted on Edinburgh DataShare.")
    print("Due to its size (~6 GB), you may need to download it manually.")
    print()
    print("Option 1: Automatic download (may be slow)")
    print(f"  URL: {ASVSPOOF2019_URLS['train']}")
    print()
    print("Option 2: Manual download")
    print("  1. Go to: https://datashare.ed.ac.uk/handle/10283/3336")
    print("  2. Download LA.zip")
    print(f"  3. Place it at: {zip_path}")
    print(f"  4. Re-run this script")
    print()
    print("Option 3: Use Kaggle (fastest)")
    print("  kaggle datasets download -d awsaf49/asvspoof-2019-dataset")
    print(f"  Then extract to: {target_dir}/")
    print("=" * 70)

    if zip_path.exists():
        logger.info("Found existing download at %s, extracting...", zip_path)
        extract_archive(zip_path, target_dir)
        logger.info("ASVspoof 2019 extracted to %s", target_dir)
    else:
        # Attempt automatic download
        success = download_file(ASVSPOOF2019_URLS["train"], zip_path)
        if success:
            extract_archive(zip_path, target_dir)
            logger.info("ASVspoof 2019 extracted to %s", target_dir)
        else:
            logger.info("Please download manually and re-run.")
            target_dir.mkdir(parents=True, exist_ok=True)


def setup_in_the_wild(data_dir: Path, force: bool = False) -> None:
    """
    Set up In-The-Wild deepfake dataset.

    Expected result::

        data/in_the_wild/
          genuine/*.wav
          fake/*.wav
    """
    target_dir = data_dir / "in_the_wild"

    if target_dir.exists() and not force:
        logger.info("In-The-Wild already exists at %s. Use --force to re-download.", target_dir)
        return

    print("\n" + "=" * 70)
    print("In-The-Wild Deepfake Dataset")
    print("=" * 70)
    print()
    print("The In-The-Wild dataset must be downloaded from:")
    print(f"  {IN_THE_WILD_URL}")
    print()
    print("After downloading, organize files as:")
    print(f"  {target_dir}/genuine/  ← real voice recordings (.wav)")
    print(f"  {target_dir}/fake/     ← deepfake voice samples (.wav)")
    print()
    print("Alternative: Use the release page or contact the authors.")
    print("=" * 70)

    # Create directory structure
    (target_dir / "genuine").mkdir(parents=True, exist_ok=True)
    (target_dir / "fake").mkdir(parents=True, exist_ok=True)
    logger.info("Created directory structure at %s", target_dir)


def setup_custom(data_dir: Path) -> None:
    """
    Create the directory structure for custom datasets.

    Expected result::

        data/custom/
          genuine/   ← your real voice recordings
          synthetic/ ← generated samples (ElevenLabs, Bark, etc.)
    """
    target_dir = data_dir / "custom"

    print("\n" + "=" * 70)
    print("Custom Dataset Setup")
    print("=" * 70)
    print()
    print("Directory structure created for custom samples:")
    print(f"  {target_dir}/genuine/   ← Place real voice recordings here")
    print(f"  {target_dir}/synthetic/ ← Place AI-generated samples here")
    print()
    print("To generate synthetic samples using ElevenLabs:")
    print("  export ELEVENLABS_API_KEY=your_key_here")
    print("  python scripts/generate_synthetic.py --backend elevenlabs \\")
    print(f"      --output-dir {target_dir}/synthetic/elevenlabs")
    print()
    print("To generate using gTTS (free, no API key):")
    print("  python scripts/generate_synthetic.py --backend gtts \\")
    print(f"      --output-dir {target_dir}/synthetic/gtts")
    print("=" * 70)

    (target_dir / "genuine").mkdir(parents=True, exist_ok=True)
    (target_dir / "synthetic").mkdir(parents=True, exist_ok=True)
    logger.info("Created custom dataset structure at %s", target_dir)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("asvspoof2019", "all"):
        setup_asvspoof2019(data_dir, args.force)

    if args.dataset in ("in_the_wild", "all"):
        setup_in_the_wild(data_dir, args.force)

    if args.dataset in ("custom", "all"):
        setup_custom(data_dir)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Download/place dataset files in the directories above")
    print("2. Generate ElevenLabs samples:")
    print("   export ELEVENLABS_API_KEY=your_key")
    print("   python scripts/generate_synthetic.py --backend elevenlabs")
    print("3. Train the model:")
    print("   python scripts/train.py --data-dir data/asvspoof2019 --dataset asvspoof2019")
    print("4. Evaluate:")
    print("   python scripts/evaluate.py --checkpoint models/best_checkpoint.pt \\")
    print("       --data-dir data/asvspoof2019 --dataset asvspoof2019")
    print("5. Calibrate thresholds:")
    print("   python scripts/calibrate_thresholds.py --checkpoint models/best_checkpoint.pt \\")
    print("       --data-dir data/asvspoof2019 --dataset asvspoof2019")
    print("=" * 70)


if __name__ == "__main__":
    main()
