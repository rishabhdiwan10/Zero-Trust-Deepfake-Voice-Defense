"""
src.pipeline.batch_pipeline
=============================
Batch processing pipeline for evaluating the Zero-Trust Deepfake Voice
Defense System across large datasets.

Supports parallel processing with configurable concurrency, progress
tracking, and structured output (CSV / JSON) for offline analysis.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm  # type: ignore

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatchPipeline:
    """
    Batch evaluation pipeline for the deepfake voice defence system.

    Parameters
    ----------
    realtime_pipeline : RealtimePipeline
        Pre-built real-time pipeline to call for each sample.
    max_concurrent : int
        Maximum concurrent pipeline invocations.
    """

    def __init__(
        self,
        realtime_pipeline,
        max_concurrent: int = 4,
    ) -> None:
        self.realtime_pipeline = realtime_pipeline
        self.max_concurrent = max_concurrent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_async(
        self,
        audio_paths: List[str],
        labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a list of audio files asynchronously in batches.

        Parameters
        ----------
        audio_paths : List[str]
            Paths to audio files to evaluate.
        labels : List[int] | None
            Ground-truth labels (0=genuine, 1=synthetic) for computing metrics.

        Returns
        -------
        List[dict]
            One result dict per audio file (same schema as RealtimePipeline).
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []

        async def process_with_semaphore(idx: int, path: str) -> Dict[str, Any]:
            async with semaphore:
                result = await self.realtime_pipeline.process(path)
                result["audio_path"] = path
                if labels is not None and idx < len(labels):
                    result["ground_truth"] = labels[idx]
                return result

        tasks = [
            process_with_semaphore(i, path) for i, path in enumerate(audio_paths)
        ]

        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Batch processing",
        ):
            result = await coro
            results.append(result)

        return results

    def run(
        self,
        audio_paths: List[str],
        labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper around ``run_async``."""
        return asyncio.run(self.run_async(audio_paths, labels))

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_results_csv(
        results: List[Dict[str, Any]], output_path: str | Path
    ) -> None:
        """Save batch results to a CSV file."""
        if not results:
            logger.warning("No results to save.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "audio_path",
            "decision",
            "trust_score",
            "cnn_score",
            "whisper_score",
            "liveness_passed",
            "ground_truth",
            "error",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        logger.info("Results saved to %s", output_path)

    @staticmethod
    def save_results_json(
        results: List[Dict[str, Any]], output_path: str | Path
    ) -> None:
        """Save batch results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

        logger.info("Results saved to %s", output_path)

    @staticmethod
    def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute basic classification metrics from batch results.

        Requires ``ground_truth`` to be set in each result dict.

        Returns
        -------
        dict with keys: ``accuracy``, ``eer`` (placeholder), ``total``.
        """
        y_true, y_pred = [], []
        for r in results:
            gt = r.get("ground_truth")
            decision = r.get("decision", "reject")
            if gt is None:
                continue
            pred = 0 if decision == "pass" else 1
            y_true.append(gt)
            y_pred.append(pred)

        if not y_true:
            return {"accuracy": 0.0, "total": 0}

        correct = sum(t == p for t, p in zip(y_true, y_pred))
        accuracy = correct / len(y_true)

        return {
            "accuracy": round(accuracy, 4),
            "total": len(y_true),
            "correct": correct,
        }
