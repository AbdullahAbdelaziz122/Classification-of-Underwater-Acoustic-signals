#!/usr/bin/env python3
"""
Underwater Acoustic Classifier — Raspberry Pi 4B Inference Pipeline
Usage:
    python pi_inference.py --model best_acoustic_cnn.pth --file audio.wav
    python pi_inference.py --model best_resnet18.pth --folder ./test_audio/
    python pi_inference.py --model best_mobilenet_v3_small.pth --benchmark --folder ./test_audio/
    python pi_inference.py --list-models
"""

import os
import sys
import time
import argparse
import warnings
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# ANSI Colors & TUI Helpers
# ─────────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_DARK = "\033[40m"


W = 64  # terminal width


def banner():
    print(f"\n{C.CYAN}{C.BOLD}{'═' * W}")
    print(f"  🔊  Underwater Acoustic Classifier")
    print(f"  Raspberry Pi 4B Inference Pipeline")
    print(f"{'═' * W}{C.RESET}\n")


def section(title: str):
    print(f"\n{C.BLUE}{C.BOLD}┌{'─' * (W - 2)}┐")
    pad = (W - 2 - len(title)) // 2
    print(f"│{' ' * pad}{title}{' ' * (W - 2 - pad - len(title))}│")
    print(f"└{'─' * (W - 2)}┘{C.RESET}")


def row(label: str, value: str, color: str = C.WHITE):
    dots = "." * (28 - len(label))
    print(f"  {C.DIM}{label}{dots}{C.RESET} {color}{value}{C.RESET}")


def divider():
    print(f"  {C.DIM}{'─' * (W - 4)}{C.RESET}")


def prob_bar(label: str, prob: float, width: int = 28, highlight: bool = False):
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{prob * 100:5.1f}%"
    color = C.GREEN if highlight else C.DIM
    star = f" {C.YELLOW}◄{C.RESET}" if highlight else ""
    print(
        f"  {color}{label:<22}{C.RESET} {C.CYAN}{bar}{C.RESET} {color}{pct}{C.RESET}{star}"
    )


def ok(msg):
    print(f"  {C.GREEN}✓{C.RESET}  {msg}")


def warn(msg):
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")


def err(msg):
    print(f"  {C.RED}✗{C.RESET}  {msg}")


# ─────────────────────────────────────────────────────────────
# Audio Config
# ─────────────────────────────────────────────────────────────
@dataclass
class AudioConfig:
    sample_rate: int = 200_000
    duration_ms: float = 5.12
    n_fft: int = 1_024
    hop_length: int = 512
    n_mels: int = 64
    top_db: int = 80

    @property
    def max_samples(self) -> int:
        return int(self.sample_rate * self.duration_ms / 1000)


CFG = AudioConfig()

CLASS_NAMES = [
    "Background",
    "Beluga",
    "Dolphin",
    "Narwhal",
    "Seal",
    "Submarine",
    "Torpedo",
    "Vessel",
    "Walrus",
    "Whale",
]

# Threat classes — highlighted differently in output
THREAT_CLASSES = {"Torpedo", "Submarine"}


# ─────────────────────────────────────────────────────────────
# Model Definitions  (must match training notebook exactly)
# ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(ch // 4, ch),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        h = self.conv(x)
        w = self.se(h).view(h.size(0), h.size(1), 1, 1)
        return F.relu(x + self.drop(h * w), inplace=True)


class AcousticCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_fc: float = 0.4):
        super().__init__()

        def stage(in_ch, out_ch, pool_freq=True, drop=0.1):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ResBlock(out_ch, drop),
            ]
            if pool_freq:
                layers.append(nn.MaxPool2d(kernel_size=(2, 1)))
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            stage(1, 32, pool_freq=True, drop=0.05),
            stage(32, 64, pool_freq=True, drop=0.10),
            stage(64, 128, pool_freq=True, drop=0.15),
            stage(128, 256, pool_freq=False, drop=0.20),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc * 0.75),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.gap(self.encoder(x)))


class PretrainedAcoustic(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str):
        super().__init__()
        import torchvision.models as tv_models

        if backbone_name == "mobilenet_v3_small":
            base = tv_models.mobilenet_v3_small(weights=None)
            c1 = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                1, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=False
            )
            in_feat = base.classifier[3].in_features
            base.classifier[3] = nn.Identity()
            self.backbone = base

        elif backbone_name == "mobilenet_v3_large":
            base = tv_models.mobilenet_v3_large(weights=None)
            c1 = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                1, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=False
            )
            in_feat = base.classifier[3].in_features
            base.classifier[3] = nn.Identity()
            self.backbone = base

        elif backbone_name in ("efficientnet_b0", "efficientnet_b1"):
            base = getattr(tv_models, backbone_name)(weights=None)
            c1 = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                1, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=False
            )
            in_feat = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        elif backbone_name in ("resnet18", "resnet34"):
            base = getattr(tv_models, backbone_name)(weights=None)
            c1 = base.conv1
            base.conv1 = nn.Conv2d(
                1, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=False
            )
            in_feat = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        elif backbone_name == "squeezenet1_1":
            base = tv_models.squeezenet1_1(weights=None)
            c1 = base.features[0]
            base.features[0] = nn.Conv2d(
                1, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=False
            )
            base.classifier[1] = nn.Conv2d(512, 512, kernel_size=1)
            base.num_classes = 512
            in_feat = 512
            self.backbone = base

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.shape[-2] < 32 or x.shape[-1] < 32:
            x = F.interpolate(
                x,
                size=(max(32, x.shape[-2]), max(32, x.shape[-1])),
                mode="bilinear",
                align_corners=False,
            )
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])
        return self.head(feat)


# ─────────────────────────────────────────────────────────────
# Model Registry  — maps filename patterns → architecture
# ─────────────────────────────────────────────────────────────
BACKBONE_KEYS = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "resnet18",
    "resnet34",
    "squeezenet1_1",
]


def detect_backbone(model_path: str) -> str | None:
    name = Path(model_path).stem.lower()
    for key in BACKBONE_KEYS:
        if key in name:
            return key
    return None


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    section("Loading Model")
    row("File", Path(model_path).name)

    backbone = detect_backbone(model_path)

    if backbone is None:
        # Assume AcousticCNN
        row("Architecture", "AcousticCNN (proposed)", C.GREEN)
        model = AcousticCNN(num_classes=num_classes)
    else:
        row("Architecture", f"PretrainedAcoustic [{backbone}]", C.CYAN)
        model = PretrainedAcoustic(num_classes=num_classes, backbone_name=backbone)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    row("Parameters", f"{n_params / 1e6:.2f}M")
    row("Device", str(device))
    ok("Model loaded successfully")
    return model


# ─────────────────────────────────────────────────────────────
# Audio Preprocessing
# ─────────────────────────────────────────────────────────────
_mel_transform = None
_db_transform = None


def get_transforms():
    global _mel_transform, _db_transform
    if _mel_transform is None:
        _mel_transform = T.MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            n_mels=CFG.n_mels,
        )
        _db_transform = T.AmplitudeToDB(top_db=CFG.top_db)
    return _mel_transform, _db_transform


def preprocess(audio_path: str) -> torch.Tensor:
    sig, sr = torchaudio.load(audio_path)
    if sr != CFG.sample_rate:
        sig = T.Resample(sr, CFG.sample_rate)(sig)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    n = CFG.max_samples
    sig = sig[:, :n] if sig.shape[1] >= n else F.pad(sig, (0, n - sig.shape[1]))
    mel, db = get_transforms()
    return db(mel(sig)).unsqueeze(0)  # (1, 1, n_mels, T)


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
def infer(model: nn.Module, spec: torch.Tensor, device: torch.device) -> tuple:
    spec = spec.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(spec)
    latency_ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs, latency_ms


def print_prediction(probs: np.ndarray, latency_ms: float, audio_path: str):
    pred_idx = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    is_threat = pred_class in THREAT_CLASSES

    section("Prediction Result")
    row("File", Path(audio_path).name)
    divider()

    color = C.RED if is_threat else C.GREEN
    label = f"{pred_class}  {'⚠ THREAT DETECTED' if is_threat else ''}"
    row("Predicted Class", label, color)
    row("Confidence", f"{confidence * 100:.1f}%", color)
    row("Latency", f"{latency_ms:.2f} ms")
    divider()

    print(f"\n  {C.BOLD}Class Probabilities:{C.RESET}")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        highlight = i == pred_idx
        c = (
            C.RED
            if CLASS_NAMES[i] in THREAT_CLASSES
            else (C.GREEN if highlight else C.DIM)
        )
        prob_bar(CLASS_NAMES[i], probs[i], highlight=highlight)

    if is_threat:
        print(f"\n  {C.RED}{C.BOLD}{'!' * 56}")
        print(f"  !!  THREAT CLASS DETECTED: {pred_class:<28}!!")
        print(f"  {'!' * 56}{C.RESET}")


# ─────────────────────────────────────────────────────────────
# Benchmark Mode
# ─────────────────────────────────────────────────────────────
def benchmark(
    model: nn.Module, folder: str, device: torch.device, warmup: int = 5, runs: int = 3
):
    section("Benchmark Mode")
    files = [str(p) for p in Path(folder).rglob("*.wav")]
    if not files:
        err(f"No WAV files found in {folder}")
        return

    row("Files found", str(len(files)))
    row("Warmup runs", str(warmup))
    row("Runs per file", str(runs))
    divider()

    # Warmup
    print(f"\n  {C.DIM}Warming up...{C.RESET}")
    for f in files[:warmup]:
        try:
            spec = preprocess(f)
            infer(model, spec, device)
        except Exception:
            pass

    # Timed runs
    latencies = []
    results = []
    class_counts = {c: 0 for c in CLASS_NAMES}

    print(f"  {C.DIM}Running inference...{C.RESET}\n")
    for f in files:
        try:
            spec = preprocess(f)
            file_times = []
            for _ in range(runs):
                probs, lat = infer(model, spec, device)
                file_times.append(lat)
            avg_lat = np.mean(file_times)
            latencies.append(avg_lat)
            pred = CLASS_NAMES[probs.argmax()]
            conf = probs.max()
            class_counts[pred] += 1
            results.append(
                {
                    "file": Path(f).name,
                    "pred": pred,
                    "conf": conf,
                    "latency_ms": avg_lat,
                }
            )
            threat_tag = f"{C.RED} ⚠{C.RESET}" if pred in THREAT_CLASSES else ""
            print(
                f"  {C.DIM}{Path(f).name:<35}{C.RESET} "
                f"{C.CYAN}{pred:<15}{C.RESET} "
                f"{conf * 100:5.1f}%  "
                f"{avg_lat:6.1f}ms{threat_tag}"
            )
        except Exception as e:
            warn(f"{Path(f).name}: {e}")

    # Summary
    section("Benchmark Summary")
    lat = np.array(latencies)
    row("Files processed", str(len(latencies)))
    row("Mean latency", f"{lat.mean():.2f} ms", C.GREEN)
    row("Std latency", f"{lat.std():.2f} ms")
    row("Min latency", f"{lat.min():.2f} ms", C.GREEN)
    row("Max latency", f"{lat.max():.2f} ms", C.YELLOW)
    row("P95 latency", f"{np.percentile(lat, 95):.2f} ms", C.YELLOW)
    row("Throughput", f"{1000 / lat.mean():.1f} samples/sec")
    divider()

    print(f"\n  {C.BOLD}Prediction Distribution:{C.RESET}")
    total = len(latencies)
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        if cnt == 0:
            continue
        color = C.RED if cls in THREAT_CLASSES else C.WHITE
        prob_bar(cls, cnt / total, highlight=(cls in THREAT_CLASSES))

    # Save JSON report
    report = {
        "model": str(Path(args.model).name),
        "files": len(latencies),
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "p95_ms": float(np.percentile(lat, 95)),
        "throughput": float(1000 / lat.mean()),
        "predictions": results,
    }
    out_path = f"benchmark_{Path(args.model).stem}.json"
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    ok(f"Report saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# List available .pth files
# ─────────────────────────────────────────────────────────────
def list_models():
    section("Available Models")
    pth_files = sorted(Path(".").glob("*.pth"))
    if not pth_files:
        warn("No .pth files found in current directory")
        return
    for p in pth_files:
        backbone = detect_backbone(str(p))
        arch = backbone if backbone else "AcousticCNN"
        size_mb = p.stat().st_size / 1e6
        color = C.GREEN if backbone is None else C.CYAN
        print(
            f"  {color}{p.name:<45}{C.RESET} "
            f"{C.DIM}{arch:<25}{C.RESET} "
            f"{size_mb:.1f} MB"
        )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Underwater Acoustic Classifier — Pi Inference"
    )
    p.add_argument("--model", type=str, help="Path to .pth model file")
    p.add_argument("--file", type=str, help="Single WAV file to classify")
    p.add_argument("--folder", type=str, help="Folder of WAV files")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark mode: measure latency over folder",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List all .pth files in current directory",
    )
    p.add_argument(
        "--classes", type=int, default=10, help="Number of output classes (default: 10)"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup runs before benchmarking (default: 5)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Timed runs per file in benchmark (default: 3)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    banner()

    if args.list_models:
        list_models()
        sys.exit(0)

    if not args.model:
        err("--model is required. Use --list-models to see available models.")
        sys.exit(1)

    if not Path(args.model).exists():
        err(f"Model file not found: {args.model}")
        sys.exit(1)

    device = torch.device("cpu")  # Pi runs on CPU
    model = load_model(args.model, num_classes=args.classes, device=device)

    # ── Single file mode ──────────────────────────────────────
    if args.file:
        if not Path(args.file).exists():
            err(f"Audio file not found: {args.file}")
            sys.exit(1)
        section("Preprocessing Audio")
        row("Sample rate", f"{CFG.sample_rate} Hz")
        row("Duration", f"{CFG.duration_ms} ms")
        row("Mel bins", str(CFG.n_mels))
        spec = preprocess(args.file)
        row("Spec shape", str(tuple(spec.shape)))
        probs, lat = infer(model, spec, device)
        print_prediction(probs, lat, args.file)

    # ── Folder mode ───────────────────────────────────────────
    elif args.folder and not args.benchmark:
        files = list(Path(args.folder).rglob("*.wav"))
        if not files:
            err(f"No WAV files in {args.folder}")
            sys.exit(1)
        section(f"Processing {len(files)} files")
        for f in files:
            try:
                spec = preprocess(str(f))
                probs, lat = infer(model, spec, device)
                pred = CLASS_NAMES[probs.argmax()]
                conf = probs.max()
                threat_tag = (
                    f"{C.RED}⚠ THREAT{C.RESET}"
                    if pred in THREAT_CLASSES
                    else f"{C.GREEN}safe{C.RESET}"
                )
                print(
                    f"  {f.name:<40} {C.CYAN}{pred:<15}{C.RESET} "
                    f"{conf * 100:5.1f}%  {lat:5.1f}ms  {threat_tag}"
                )
            except Exception as e:
                warn(f"{f.name}: {e}")

    # ── Benchmark mode ────────────────────────────────────────
    elif args.benchmark:
        if not args.folder:
            err("--benchmark requires --folder")
            sys.exit(1)
        benchmark(model, args.folder, device, warmup=args.warmup, runs=args.runs)

    else:
        err("Provide --file or --folder. Use --help for usage.")

    print(f"\n{C.DIM}{'─' * W}{C.RESET}\n")
