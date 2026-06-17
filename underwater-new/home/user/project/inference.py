from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import AudioConfig
from model import UnderwaterAcousticCNN
from utils import AudioPreprocessor, get_device, load_json


def load_model(
    model_path: str | Path,
    class_mapping_path: str | Path,
    device: torch.device | None = None,
) -> Tuple[nn.Module, List[str], torch.device]:
    """Load the trained model, class mapping, and target device."""
    device = device or get_device()
    class_mapping = load_json(class_mapping_path)
    ordered_classes = [class_name for class_name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]

    model = UnderwaterAcousticCNN(num_classes=len(ordered_classes))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, ordered_classes, device


def preprocess_audio(audio_path: str | Path, audio_config: AudioConfig | None = None) -> Tensor:
    """Preprocess a WAV file into model-ready tensor shape (1, 1, 128, 157)."""
    config = audio_config or AudioConfig()
    preprocessor = AudioPreprocessor(config)
    features = preprocessor(audio_path)
    return features.unsqueeze(0)


def _predict_logits(
    model: nn.Module,
    inputs: Tensor,
    device: torch.device,
) -> Tensor:
    """Run a forward pass and return logits."""
    with torch.no_grad():
        logits = model(inputs.to(device))
    return logits


def predict(
    audio_path: str | Path,
    model_path: str | Path,
    class_mapping_path: str | Path,
    device: torch.device | None = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Predict a class for a WAV file and return confidence plus top-k scores."""
    model, class_names, device = load_model(model_path, class_mapping_path, device=device)
    inputs = preprocess_audio(audio_path)
    logits = _predict_logits(model, inputs, device)
    probabilities = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, len(class_names))
    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
    prediction_index = int(top_indices[0].item())

    top_predictions = [
        {"class": class_names[int(index.item())], "prob": float(prob.item())}
        for prob, index in zip(top_probabilities, top_indices, strict=True)
    ]

    return {
        "audio_file": str(audio_path),
        "prediction": class_names[prediction_index],
        "confidence": float(top_probabilities[0].item()),
        "top_k": top_predictions,
    }
