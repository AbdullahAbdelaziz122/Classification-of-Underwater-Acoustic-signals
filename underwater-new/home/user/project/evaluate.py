from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _collect_predictions(
    model: nn.Module,
    data_loader: DataLoader[Tuple[Tensor, int]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a dataloader and collect labels/predictions."""
    model.eval()
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_predictions)


def build_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Construct a confusion matrix using NumPy only."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(targets, predictions, strict=True):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def compute_classification_metrics(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Compute accuracy, precision, recall, F1, and per-class accuracy."""
    true_positives = np.diag(confusion_matrix).astype(np.float64)
    support = confusion_matrix.sum(axis=1).astype(np.float64)
    predicted = confusion_matrix.sum(axis=0).astype(np.float64)

    precision = np.divide(true_positives, predicted, out=np.zeros_like(true_positives), where=predicted > 0)
    recall = np.divide(true_positives, support, out=np.zeros_like(true_positives), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positives),
        where=(precision + recall) > 0,
    )
    per_class_accuracy = recall.copy()
    overall_accuracy = float(true_positives.sum() / max(1.0, confusion_matrix.sum()))

    per_class_metrics = {}
    for index, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "accuracy": float(per_class_accuracy[index]),
            "support": int(support[index]),
        }

    metrics: Dict[str, object] = {
        "accuracy": overall_accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1_score": float(np.mean(f1)),
        "per_class": per_class_metrics,
    }
    return metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a labeled confusion matrix figure."""
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            axis.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader[Tuple[Tensor, int]],
    device: torch.device,
    class_names: Sequence[str],
    output_dir: str | Path,
    logger: logging.Logger,
) -> Dict[str, object]:
    """Evaluate a trained model and save confusion matrix to disk."""
    targets, predictions = _collect_predictions(model, data_loader, device)
    confusion = build_confusion_matrix(targets, predictions, num_classes=len(class_names))
    metrics = compute_classification_metrics(confusion, class_names)
    confusion_path = Path(output_dir) / "confusion_matrix.png"
    plot_confusion_matrix(confusion, class_names, confusion_path)

    logger.info("Evaluation accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro precision: %.4f", metrics["macro_precision"])
    logger.info("Macro recall: %.4f", metrics["macro_recall"])
    logger.info("Macro F1-score: %.4f", metrics["macro_f1_score"])
    for class_name, values in metrics["per_class"].items():
        logger.info(
            "Class=%s | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | support=%d",
            class_name,
            values["accuracy"],
            values["precision"],
            values["recall"],
            values["f1_score"],
            values["support"],
        )

    return metrics
