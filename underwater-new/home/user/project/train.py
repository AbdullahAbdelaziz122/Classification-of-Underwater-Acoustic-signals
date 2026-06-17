from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import AudioConfig, ProjectConfig, TrainingConfig
from dataset import build_dataloaders
from evaluate import evaluate_model
from model import UnderwaterAcousticCNN
from utils import ensure_dir, resolve_batch_size, save_class_mapping


@dataclass
class TrainingArtifacts:
    """Container for trained model and metadata."""

    model: nn.Module
    train_loader: DataLoader[Tuple[Tensor, int]]
    val_loader: DataLoader[Tuple[Tensor, int]]
    class_to_idx: Dict[str, int]
    best_model_path: Path
    effective_batch_size: int


class EarlyStopping:
    """Track validation loss and stop training when it stops improving."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss: float) -> bool:
        """Update early stopping state."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[Tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> Tuple[float, float]:
    """Run one train or validation epoch."""
    training = optimizer is not None
    model.train(mode=training)

    running_loss = 0.0
    correct = 0
    total = 0

    for features, targets in data_loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, targets)

        if training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * features.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    average_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)
    return average_loss, accuracy


def train_model(
    project_config: ProjectConfig,
    audio_config: AudioConfig,
    training_config: TrainingConfig,
    device: torch.device,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Train the unified CNN and save the best checkpoint."""
    output_dir = ensure_dir(project_config.output_dir)
    effective_batch_size = resolve_batch_size(training_config.batch_size, device)
    logger.info("Using device: %s", device)
    logger.info("Effective batch size: %d", effective_batch_size)

    train_loader, val_loader, class_to_idx = build_dataloaders(
        project_config=project_config,
        audio_config=audio_config,
        training_config=training_config,
        batch_size=effective_batch_size,
    )

    save_class_mapping(project_config.class_names, output_dir / "class_mapping.json")

    model = UnderwaterAcousticCNN(num_classes=project_config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=training_config.min_learning_rate,
    )
    early_stopper = EarlyStopping(patience=training_config.early_stopping_patience)
    best_model_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")

    logger.info("Starting training for %d epochs", training_config.epochs)
    for epoch in range(1, training_config.epochs + 1):
        try:
            train_loss, train_accuracy = _run_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and effective_batch_size == 32:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory with batch size 32. Re-run with --batch-size 16 or use a smaller GPU load."
                ) from exc
            raise

        val_loss, val_accuracy = _run_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d/%03d | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.6f",
            epoch,
            training_config.epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best checkpoint to %s", best_model_path)

        if early_stopper.step(val_loss):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    metrics = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        class_names=project_config.class_names,
        output_dir=output_dir,
        logger=logger,
    )
    logger.info("Training complete. Best validation loss: %.4f", best_val_loss)
    logger.info("Final validation accuracy: %.4f", metrics["accuracy"])

    return TrainingArtifacts(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        best_model_path=best_model_path,
        effective_batch_size=effective_batch_size,
    )
