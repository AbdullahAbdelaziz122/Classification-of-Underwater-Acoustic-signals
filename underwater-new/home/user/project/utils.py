from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample

from config import AudioConfig


class AudioPreprocessor:
    """Centralized audio preprocessing for both training and inference."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.mel_transform = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.db_transform = AmplitudeToDB()
        self._resamplers: Dict[int, Resample] = {}

    def load_waveform(self, audio_path: str | Path) -> Tensor:
        """Load a WAV file using soundfile, convert to mono, resample, and pad/truncate."""
        # Read the audio file directly into a float32 numpy array
        data, sample_rate = sf.read(str(audio_path), dtype="float32")

        # soundfile returns (num_samples, num_channels) or (num_samples,)
        # Convert to PyTorch expected shape: (num_channels, num_samples)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data).t()

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.config.sample_rate:
            resampler = self._resamplers.get(sample_rate)
            if resampler is None:
                resampler = Resample(
                    orig_freq=sample_rate, new_freq=self.config.sample_rate
                )
                self._resamplers[sample_rate] = resampler
            waveform = resampler(waveform)

        num_samples = self.config.num_samples
        if waveform.size(1) > num_samples:
            waveform = waveform[:, :num_samples]
        elif waveform.size(1) < num_samples:
            pad_length = num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform

    def __call__(self, audio_path: str | Path) -> Tensor:
        """Process an audio file into a log-mel spectrogram tensor."""
        waveform = self.load_waveform(audio_path)
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)
        return log_mel_spec


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_batch_size(requested_batch_size: int, device: torch.device) -> int:
    """Adjust batch size based on known hardware VRAM limits."""
    if device.type == "mps" or (
        device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 4e9
    ):
        return min(requested_batch_size, 16)
    return requested_batch_size


def set_seed(seed: int) -> None:
    """Enforce reproducibility across standard random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(payload: Any, output_path: str | Path) -> None:
    """Serialize a dictionary to JSON."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(input_path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    with Path(input_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_class_mapping(class_names: Sequence[str], output_path: str | Path) -> None:
    """Save canonical class-to-index mapping."""
    mapping = {class_name: index for index, class_name in enumerate(class_names)}
    save_json(mapping, output_path)


def setup_logging(
    output_dir: str | Path, log_name: str = "training_log.txt"
) -> logging.Logger:
    """Configure project logging to file and console."""
    output_dir = ensure_dir(output_dir)
    log_path = output_dir / log_name

    logger = logging.getLogger("underwater_acoustic_cnn")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
