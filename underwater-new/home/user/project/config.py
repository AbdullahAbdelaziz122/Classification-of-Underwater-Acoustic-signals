from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AudioConfig:
    """Audio preprocessing configuration."""

    sample_rate: int = 16_000
    duration_seconds: int = 5
    n_fft: int = 1_024
    hop_length: int = 512
    n_mels: int = 128
    expected_frames: int = 157

    @property
    def num_samples(self) -> int:
        """Return the target number of audio samples per clip."""
        return self.sample_rate * self.duration_seconds


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters and runtime options."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 40
    early_stopping_patience: int = 7
    validation_split: float = 0.2
    seed: int = 42
    weight_decay: float = 0.0
    min_learning_rate: float = 1e-6


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    ds3500_dir: Path
    noaa_dir: Path
    watkins_dir: Path
    threat_dir: Path
    output_dir: Path
    class_names: Tuple[str, ...] = field(
        default=(
            "Background",
            "Beluga",
            "Dolphin",
            "Narwhal",
            "Seal",
            "Vessel",
            "Walrus",
            "Whale",
            "Submarine",
            "Torpedo",
        )
    )

    @property
    def num_classes(self) -> int:
        """Return the number of target classes."""
        return len(self.class_names)