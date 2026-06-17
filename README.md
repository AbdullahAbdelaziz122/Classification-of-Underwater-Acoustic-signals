# Underwater Acoustic Classifier 🔬🌊

A deep learning pipeline for classifying underwater acoustic signals into marine life families and threat classes (Torpedo, Submarine). Designed for research and edge deployment on resource-constrained hardware such as Raspberry Pi 4B and NVIDIA Jetson Nano.

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
  - [Visualization](#visualization)
- [Raspberry Pi Inference Pipeline](#raspberry-pi-inference-pipeline)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Output Modes](#output-modes)
  - [Benchmark Reports](#benchmark-reports)
- [Results](#results)
- [Future Work](#future-work)

---

## 🎯 Overview

This project implements a hierarchical underwater acoustic classification system using mel-spectrogram features and convolutional neural networks. The pipeline covers the full lifecycle:

1. **Data ingestion** from multiple public datasets
2. **Spectrogram precomputation** and caching
3. **Training** a custom proposed CNN (`AcousticCNN`) and 7 pretrained backbone baselines
4. **Evaluation** with per-class metrics, confusion matrices, and latency profiling
5. **Edge deployment** on Raspberry Pi 4B with a TUI inference tool

The system is engineered to distinguish between marine biological signals (e.g., whales, dolphins) and mechanical threats (e.g., submarines, torpedoes) with high precision, featuring robust memory management for resource-constrained hardware (e.g., 2GB VRAM limits).

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
├── parse_and_plot_logs.py                 # Visualization script for training logs
├── main.py                                # CLI entry point for training and inference
├── README.md
│
├── specs_train/                           # Cached training spectrograms (.npy)
├── specs_val/                             # Cached validation spectrograms (.npy)
│
├── best_acoustic_cnn.pth                  # Saved AcousticCNN weights
├── best_mobilenet_v3_small.pth
├── best_mobilenet_v3_large.pth
├── best_efficientnet_b0.pth
├── best_efficientnet_b1.pth
├── best_resnet18.pth
├── best_resnet34.pth
├── best_squeezenet1_1.pth
├── class_mapping.json                     # Class index mapping
├── training_log.txt                       # Training history
│
├── outputs/
│   └── figures/                           # Evaluation plots (PNG)
│       ├── learning_curves.png
│       └── per_class_metrics.png
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
soundfile                     # for audio I/O
```

### Datasets

The pipeline uses multiple public datasets combined into a single dataframe:

| Dataset | Content | Classes |
|---------|---------|---------|
| **DS3500** | Vessel recordings | Small/Medium/Large Vessels, Passenger Ferries, Background |
| **NOAA SanctSound** | Ocean soundscapes | Background (soundscape, rain, sonar) |
| **Watkins Marine Mammal** | Marine mammal vocalizations | 31 species mapped to Whale, Dolphin, Seal, Beluga, Narwhal, Walrus |
| **Threat Dataset** | Synthetic or collected recordings | Submarine, Torpedo |

### Audio Configuration

Defined via `AudioConfig`:

```python
@dataclass
class AudioConfig:
    sample_rate: int   = 22_050    # Hz — all audio resampled to this
    duration_ms: float = 2_000     # ms — clips padded/trimmed to this length
    n_fft: int         = 1_024     # FFT window size
    hop_length: int    = 512       # hop between frames
    n_mels: int        = 64        # mel filterbank bins
    top_db: int        = 80        # AmplitudeToDB dynamic range
```

This produces spectrograms of shape `(1, 64, ~86)` — 64 mel bins × ~86 time frames per 2-second clip.

> ⚠️ **Important:** If you change any value in `AudioConfig`, delete `specs_train/` and `specs_val/` and rerun the preprocessing. Stale cached spectrograms will silently corrupt training.

### Models

#### AcousticCNN (Proposed)

A custom VGG-style 4-stage CNN with residual blocks, designed specifically for underwater acoustic spectrograms.

- **Input:** `(B, 1, 64, T)`
- **Pooling:** Frequency-axis only `(2, 1)` — preserves the time axis for short clips
- **Stages:** 1→32→64→128→256 channels with BatchNorm and ReLU
- **Head:** Adaptive Average Pooling → MLP (256→128→num_classes) with Dropout
- **Parameters:** ~2.1M

#### Pretrained Baselines

All 7 baselines use the `PretrainedAcoustic` wrapper which:
- Patches the first convolution layer from 3-channel to 1-channel input
- Replaces the original classification head with a lightweight 2-layer head
- Auto-upsamples spectrograms to minimum 32×32 if needed

| Backbone | Params (M) | Notes |
|----------|------------|-------|
| `mobilenet_v3_small` | 1.65 | Lightweight deployment option |
| `mobilenet_v3_large` | 4.37 | MobileNet upper bound |
| `efficientnet_b0` | 4.17 | Standard benchmark baseline |
| `efficientnet_b1` | 6.68 | Larger EfficientNet variant |
| `resnet18` | 11.24 | Classic CNN baseline |
| `resnet34` | 21.35 | Deeper ResNet variant |
| `squeezenet1_1` | 1.05 | Lightest pretrained option |

### Training

All models are trained with the same engine:

- **Loss:** CrossEntropyLoss with `label_smoothing=0.1`
- **Optimizer:** AdamW with `weight_decay=1e-4`
- **Scheduler:** CosineAnnealingLR
- **Early stopping:** patience-based on validation F1
- **Hardware-aware batching:** Automatically resolves effective batch sizes based on available VRAM (CUDA/MPS compatible)
- **Augmentation:** TimeMasking + FrequencyMasking (training only)

Pretrained baselines use **two-phase training**:

```
Phase 1 (warmup_epochs=5):   backbone frozen  → train head only at lr=1e-3
Phase 2 (remaining epochs):  backbone unfrozen → fine-tune all at lr=5e-5
```

Training hyperparameters:

```python
WARMUP_EPOCHS = 5
TOTAL_EPOCHS  = 35
PATIENCE      = 8
```

#### Training via CLI

```bash
# Train the model from scratch
python main.py train \
  --ds3500-dir ./data/DS3500 \
  --noaa-dir ./data/sanctsound/products/sound_clips \
  --watkins-dir ./data/whoi_whale_sounds \
  --threat-dir ./data/submarine_torpedo \
  --output-dir ./outputs \
  --epochs 40 \
  --batch-size 32
```

*Outputs:* Saves `best_model.pth`, `class_mapping.json`, and `training_log.txt` to the specified output directory.

### Evaluation

Evaluation produces:

- Per-class precision, recall, F1 report
- Normalized confusion matrix (saved as PNG)
- Training history plots (loss, accuracy, F1 over epochs)
- Full comparison table across all models sorted by F1
- Per-sample inference latency (ms) on the training machine (GPU)
- Latency vs F1 scatter plot — efficiency trade-off figure

### Visualization

Generate learning curves and final evaluation metric graphs:

```bash
python parse_and_plot_logs.py
```

*Outputs:* Generates high-DPI `.png` files (e.g., `learning_curves.png`, `per_class_metrics.png`) in the `./outputs/figures` directory.

### Inference via CLI

Run a prediction on a single, unseen audio file:

```bash
python main.py predict \
  --audio-path ./data/submarine_torpedo/test_audio.wav \
  --model-path ./outputs/best_model.pth \
  --class-mapping-path ./outputs/class_mapping.json \
  --top-k 3
```

*Outputs:* Returns a JSON payload containing the top predicted class and confidence scores.

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

The final test evaluation was conducted on a held-out test set. The table below reports per-model classification metrics together with average per-sample inference latency on GPU.

| Model | Params (M) | Accuracy | Precision | Recall | F1 | GPU Latency (ms) |
|-------|------------|----------|-----------|--------|----|------------------|
| **AcousticCNN (proposed)** | **2.10** | **0.9866** | **0.9694** | **0.9715** | **0.9689** | **0.1936** |
| resnet18 | 11.24 | 0.9688 | 0.9675 | 0.9699 | 0.9684 | 0.2220 |
| resnet34 | 21.35 | 0.9599 | 0.9448 | 0.9090 | 0.9229 | 0.2996 |
| efficientnet_b0 | 4.17 | 0.9570 | 0.9538 | 0.8771 | 0.8962 | 0.5679 |
| mobilenet_v3_small | 1.65 | 0.9169 | 0.9092 | 0.7592 | 0.7834 | 0.4078 |
| mobilenet_v3_large | 4.37 | 0.9318 | 0.8338 | 0.7494 | 0.7719 | 0.4850 |
| efficientnet_b1 | 6.68 | 0.9021 | 0.9177 | 0.7265 | 0.7661 | 0.7176 |
| squeezenet1_1 | 1.05 | 0.9258 | 0.7359 | 0.7268 | 0.7298 | 0.2236 |

**Key observations:**
- **AcousticCNN** achieves the highest F1 (0.9689) and accuracy (0.9866) while maintaining the lowest latency (0.1936 ms), demonstrating its suitability for real-time edge deployment.
- **ResNet18** performs nearly as well (F1=0.9684) but with slightly higher latency and significantly more parameters.
- **EfficientNet** variants show higher latency and lower F1, making them less favourable for resource-constrained devices.
- **SqueezeNet** is light but underperforms in recall and precision.

> *Note: Rare classes (e.g., Beluga, Walrus) exhibit lower support and recall compared to dominant classes (e.g., Vessel, Dolphin).*

---

## 🔮 Future Work

- **Edge Deployment:** Export the PyTorch model to ONNX/TensorRT for real-time inference on NVIDIA Jetson Nano.
- **Control System Integration:** Link classification outputs to a physical AUV demonstration stand via Sliding Mode Control (SMC) architecture.
- **Data Augmentation:** Expand the dataset with additional synthetic threat samples to improve rare class performance.
- **Real-time Processing:** Implement streaming inference for continuous hydrophone data.
- **Model Quantization:** Apply INT8 quantization to further reduce latency and memory footprint on edge devices.

---

## 📝 Paper Notes

When citing this work in a paper, please refer to:

- **AcousticCNN** as the proposed lightweight architecture tailored for underwater acoustic classification
- **Two-phase training** strategy for fine-tuning pretrained backbones
- **Hardware-aware batching** for efficient training on resource-constrained GPUs
- **Raspberry Pi inference pipeline** with benchmark reports for edge deployment validation

For questions or collaborations, please open an issue in the repository.

---

## 📄 License

This project is for research and educational purposes. Please ensure compliance with dataset licenses when using the provided code.
