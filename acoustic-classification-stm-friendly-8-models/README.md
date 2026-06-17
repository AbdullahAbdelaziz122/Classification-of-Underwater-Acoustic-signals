# Underwater Acoustic Classifier 🔬🌊

A deep learning pipeline for classifying underwater acoustic signals into marine life families and threat classes (Torpedo, Submarine). Designed for research and edge deployment on Raspberry Pi 4B.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Project Structure](#project-structure)
- [Kaggle Training Pipeline](#kaggle-training-pipeline)
  - [Requirements](#requirements)
  - [Datasets](#datasets)
  - [Audio Configuration](#audio-configuration)
  - [Models](#models)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Raspberry Pi Inference Pipeline](#raspberry-pi-inference-pipeline)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Output Modes](#output-modes)
  - [Benchmark Reports](#benchmark-reports)
- [Results](#results)
- [Paper Notes](#paper-notes)

---

## 🎯 Overview

This project implements a hierarchical underwater acoustic classification system using mel-spectrogram features and convolutional neural networks. The pipeline covers the full lifecycle:

1. **Data ingestion** from three public datasets
2. **Spectrogram precomputation** and caching
3. **Training** a custom proposed CNN (`AcousticCNN`) and 7 pretrained backbone baselines
4. **Evaluation** with per-class metrics, confusion matrices, and latency profiling
5. **Edge deployment** on Raspberry Pi 4B with a TUI inference tool

---

## 🏷️ Classes

| ID | Family | Type |
|----|--------|------|
| 0 | Background | Non-threat |
| 1 | Beluga | Non-threat |
| 2 | Dolphin | Non-threat |
| 3 | Narwhal | Non-threat |
| 4 | Seal | Non-threat |
| 5 | Submarine | ⚠ Threat |
| 6 | Torpedo | ⚠ Threat |
| 7 | Vessel | Non-threat |
| 8 | Walrus | Non-threat |
| 9 | Whale | Non-threat |

---

## 📂 Project Structure

```
.
├── underwater_acoustic_classifier.ipynb   # Kaggle training notebook
├── pi_inference.py                        # Raspberry Pi inference pipeline
├── README.md
├── best_acoustic_cnn.pth                  # Saved AcousticCNN weights
├── best_mobilenet_v3_small.pth
├── best_mobilenet_v3_large.pth
├── best_efficientnet_b0.pth
├── best_efficientnet_b1.pth
├── best_resnet18.pth
├── best_resnet34.pth
├── best_squeezenet1_1.pth
│
└── benchmark_<model_name>.json            # Pi benchmark reports
```

---

## 🚀 Kaggle Training Pipeline

### Requirements

```bash
torch
torchaudio
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
google-cloud-storage          # for NOAA dataset download
```

### Datasets

The pipeline uses three public datasets combined into a single dataframe:

| Dataset | Content | Classes |
|---------|---------|---------|
| **DS3500** | Vessel recordings | Small/Medium/Large Vessels, Passenger Ferries, Background |
| **NOAA SanctSound** | Ocean soundscapes | Background (soundscape, rain, sonar) |
| **Watkins Marine Mammal** | Marine mammal vocalizations | 31 species mapped to Whale, Dolphin, Seal, Beluga, Narwhal, Walrus |
| **Generated Torpedo and Submarine data** | | |


### Audio Configuration

Defined in Section 1 via `AudioConfig`:

```python
@dataclass
class AudioConfig:
    sample_rate: int   = 200_000    # Hz — all audio resampled to this
    duration_ms: float = 5.12     # ms — clips padded/trimmed to this length
    n_fft: int         = 1_024     # FFT window size
    hop_length: int    = 512       # hop between frames
    n_mels: int        = 64        # mel filterbank bins
    top_db: int        = 80        # AmplitudeToDB dynamic range
```

> ⚠️ **Important:** If you change any value in `AudioConfig`, delete `specs_train/` and `specs_val/` and rerun from Section 3. Stale cached spectrograms will silently corrupt training.

### Models

#### AcousticCNN (Proposed)

A custom 4-stage CNN with residual blocks and Squeeze-and-Excitation attention, designed specifically for underwater acoustic spectrograms.

- **Input:** `(B, 1, 64, T)`
- **Pooling:** Frequency-axis only `(2, 1)` — preserves the time axis for short clips
- **Stages:** 1→32→64→128→256 channels with SE-ResBlocks at each stage
- **Head:** 256→256→128→num_classes with BatchNorm and Dropout
- **Parameters:** ~600K

#### Pretrained Baselines

All 7 baselines use the `PretrainedAcoustic` wrapper which:
- Patches the first convolution layer from 3-channel to 1-channel input
- Replaces the original classification head with a lightweight 2-layer head
- Auto-upsamples spectrograms to minimum 32×32 if needed

| Backbone | Params | Notes |
|----------|--------|-------|
| `mobilenet_v3_small` | 2.5M | Primary deployment target for Pi |
| `mobilenet_v3_large` | 5.4M | MobileNet upper bound |
| `efficientnet_b0` | 5.3M | Standard benchmark baseline |
| `efficientnet_b1` | 7.8M | Larger EfficientNet variant |
| `resnet18` | 11M | Classic CNN baseline |
| `resnet34` | 21M | Deeper ResNet variant |
| `squeezenet1_1` | 1.2M | Lightest pretrained option |

### Training

All models are trained with the same engine (Section 6):

- **Loss:** CrossEntropyLoss with `label_smoothing=0.1`
- **Optimizer:** AdamW with `weight_decay=1e-4`
- **Scheduler:** CosineAnnealingLR
- **Early stopping:** patience-based on validation F1
- **Augmentation:** TimeMasking + FrequencyMasking (training only)

Pretrained baselines use **two-phase training**:

```
Phase 1 (warmup_epochs=5):   backbone frozen  → train head only at lr=1e-3
Phase 2 (remaining epochs):  backbone unfrozen → fine-tune all at lr=5e-5
```

Training hyperparameters (Section 9):

```python
WARMUP_EPOCHS = 5
TOTAL_EPOCHS  = 35
PATIENCE      = 8
```

### Evaluation

Section 10 runs evaluation for all models and produces:

- Per-class precision, recall, F1 report
- Normalized confusion matrix (saved as PNG)
- Training history plots (loss, accuracy, F1 over epochs)

Section 11 produces the full comparison table across all models sorted by F1, and a bar chart ready for the paper.

Section 12 (inference/testing) additionally measures:

- Per-sample inference latency (ms) on the training machine (GPU)
- Latency vs F1 scatter plot — efficiency trade-off figure for paper
- JSON benchmark report per model

---

## 🖥️ Raspberry Pi Inference Pipeline

### Setup

Copy the following files to the Raspberry Pi:

```bash
scp pi_inference.py        pi@<pi-ip>:~/acoustic/
scp best_*.pth             pi@<pi-ip>:~/acoustic/
scp your_test_audio/       pi@<pi-ip>:~/acoustic/test_audio/
```

No additional installation needed if the Pi is already configured with PyTorch and torchaudio.

### Usage

```bash
# List all available .pth model files in current directory
python pi_inference.py --list-models

# Classify a single WAV file
python pi_inference.py --model best_acoustic_cnn.pth --file test.wav

# Classify all WAV files in a folder (compact table output)
python pi_inference.py --model best_resnet18.pth --folder ./test_audio/

# Full benchmark — latency stats + JSON report
python pi_inference.py --model best_mobilenet_v3_small.pth \
                       --benchmark \
                       --folder ./test_audio/ \
                       --warmup 10 \
                       --runs 5

# Specify number of classes explicitly (default: 10)
python pi_inference.py --model best_acoustic_cnn.pth --file test.wav --classes 8
```

### Output Modes

#### `--list-models`
Scans the current directory for `.pth` files and displays architecture, and file size:

```
╔══════════════════════════════════════════════════════════════╗
  Available Models
╚══════════════════════════════════════════════════════════════╝
  best_acoustic_cnn.pth               AcousticCNN          4.2 MB
  best_mobilenet_v3_small.pth         mobilenet_v3_small   8.9 MB
  best_resnet18.pth                   resnet18             42.7 MB
```

#### `--file`
Full prediction output with probability bars for every class:

```
  Predicted Class.......... Vessel
  Confidence............... 94.3%
  Latency.................. 38.2 ms

  Class Probabilities:
  Vessel                 ████████████████████████░░░░  94.3%  ◄
  Background             ██░░░░░░░░░░░░░░░░░░░░░░░░░░   3.1%
  Whale                  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.2%
  ...
```

If a threat class (Torpedo or Submarine) is predicted, a high-visibility alert is shown:

```
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!  THREAT CLASS DETECTED: Torpedo                   !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

#### `--folder`
Compact one-line-per-file table:

```
  hydrophone_001.wav               Vessel          91.2%   42.1ms  safe
  hydrophone_002.wav               Torpedo         87.5%   39.8ms  ⚠ THREAT
  hydrophone_003.wav               Dolphin         76.3%   41.0ms  safe
```

#### `--benchmark`
Full latency profiling with summary statistics and prediction distribution:

```
  Files processed.......... 120
  Mean latency............. 41.3 ms
  Std latency.............. 3.2 ms
  Min latency.............. 35.1 ms
  Max latency.............. 58.7 ms
  P95 latency.............. 49.2 ms
  Throughput............... 24.2 samples/sec

  Prediction Distribution:
  Vessel                 ████████████████████░░░░░░░░  68.3%
  Whale                  █████░░░░░░░░░░░░░░░░░░░░░░░  18.4%
  Dolphin                ███░░░░░░░░░░░░░░░░░░░░░░░░░   9.2%
  ...
```

### Benchmark Reports

Each `--benchmark` run saves a JSON report named `benchmark_<model_name>.json`:

```json
{
  "model": "best_mobilenet_v3_small.pth",
  "files": 120,
  "mean_ms": 41.3,
  "std_ms": 3.2,
  "min_ms": 35.1,
  "max_ms": 58.7,
  "p95_ms": 49.2,
  "throughput": 24.2,
  "predictions": [
    {
      "file": "hydrophone_001.wav",
      "pred": "Vessel",
      "conf": 0.912,
      "latency_ms": 42.1
    }
  ]
}
```

Collect one JSON per model then use these to build the Pi latency table for the paper.

---

## 📊 Results

Results from Kaggle training (GPU baseline — update with Pi numbers after benchmarking):

| Model | Params (M) | Accuracy | Precision | Recall | F1 |
|-------|------------|----------|-----------|--------|----|
| AcousticCNN (proposed) | 2.10 | 0.9021 | 0.8751 | 0.8123 | 0.8305 |
| resnet34 | 21.35 | 0.8961 | 0.8326 | 0.8145 | 0.8206 |
| resnet18 | 11.24 | 0.8798 | 0.8239 | 0.7483 | 0.7764 |
| mobilenet_v3_large | 4.37 | 0.7582 | 0.6683 | 0.4593 | 0.4885 |
| efficientnet_b0 | 4.17 | 0.7774 | 0.5611 | 0.4715 | 0.4785 |
| efficientnet_b1 | 6.68 | 0.7493 | 0.4916 | 0.4291 | 0.4254 |
| mobilenet_v3_small | 1.65 | 0.7611 | 0.5746 | 0.4171 | 0.4164 |
| squeezenet1_1 | 1.05 | 0.7878 | 0.4045 | 0.4336 | 0.4160 |

> Pi latency figures to be added after running `--benchmark` on the Raspberry Pi 4B.

---
## Testing

The final test evaluation was conducted on a held-out test set (GPU inference on the training machine). The table below reports per-model classification metrics alongside average per-sample inference latency (ms). These latency figures serve as a baseline for comparison with the Raspberry Pi edge deployment results.

| Model | Accuracy | Precision | Recall | F1 | Latency (ms) |
|-------|----------|-----------|--------|----|--------------|
| AcousticCNN (proposed) | 0.9021 | 0.8751 | 0.8123 | 0.8305 | 0.2759 |
| resnet34 | 0.8961 | 0.8326 | 0.8145 | 0.8206 | 0.3214 |
| resnet18 | 0.8798 | 0.8239 | 0.7483 | 0.7764 | 0.2426 |
| mobilenet_v3_large | 0.7582 | 0.6683 | 0.4593 | 0.4885 | 0.4937 |
| efficientnet_b0 | 0.7774 | 0.5611 | 0.4715 | 0.4785 | 0.6943 |
| efficientnet_b1 | 0.7493 | 0.4916 | 0.4291 | 0.4254 | 1.0174 |
| mobilenet_v3_small | 0.7611 | 0.5746 | 0.4171 | 0.4164 | 0.4759 |
| squeezenet1_1 | 0.7878 | 0.4045 | 0.4336 | 0.4160 | 0.2191 |

**Key observations:**
- AcousticCNN achieves the highest F1 (0.8305) and accuracy (0.9021) while maintaining competitive latency.
- SqueezeNet is the fastest but underperforms significantly in precision.
- The EfficientNet variants show higher latency with lower F1, making them less suitable for real-time edge deployment.
