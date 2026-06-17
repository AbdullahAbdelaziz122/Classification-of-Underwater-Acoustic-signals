from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from config import AudioConfig, ProjectConfig, TrainingConfig
from inference import predict
from train import train_model
from utils import get_device, set_seed, setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Unified CNN system for underwater acoustic classification from WAV files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the CNN model.")
    train_parser.add_argument("--ds3500-dir", type=Path, required=True, help="Path to the ds3500 directory.")
    train_parser.add_argument("--noaa-dir", type=Path, required=True, help="Path to the NOAA sanctsound directory.")
    train_parser.add_argument("--watkins-dir", type=Path, required=True, help="Path to the Watkins marine mammals directory.")
    train_parser.add_argument("--threat-dir", type=Path, required=True, help="Path to the threat audio directory.")
    train_parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to save outputs.")
    train_parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=32, choices=[16, 32], help="Training batch size.")
    train_parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")

    predict_parser = subparsers.add_parser("predict", help="Run inference on a WAV file.")
    predict_parser.add_argument("--audio-path", type=Path, required=True, help="Path to the WAV file.")
    predict_parser.add_argument("--model-path", type=Path, required=True, help="Path to best_model.pth.")
    predict_parser.add_argument(
        "--class-mapping-path",
        type=Path,
        required=True,
        help="Path to class_mapping.json.",
    )
    predict_parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to return.")

    return parser


def main() -> None:
    """Entry point for training and inference workflows."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        audio_config = AudioConfig()
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            seed=args.seed,
        )
        project_config = ProjectConfig(
            ds3500_dir=args.ds3500_dir,
            noaa_dir=args.noaa_dir,
            watkins_dir=args.watkins_dir,
            threat_dir=args.threat_dir,
            output_dir=args.output_dir
        )

        set_seed(training_config.seed)
        logger = setup_logging(project_config.output_dir)
        device = get_device()
        train_model(
            project_config=project_config,
            audio_config=audio_config,
            training_config=training_config,
            device=device,
            logger=logger,
        )
        return

    if args.command == "predict":
        result: dict[str, Any] = predict(
            audio_path=args.audio_path,
            model_path=args.model_path,
            class_mapping_path=args.class_mapping_path,
            top_k=args.top_k,
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()