# AI-Powered Smart System for Real-Time Underwater Threat Detection and Marine Safety

This repository contains a unified, PyTorch-based Convolutional Neural Network (CNN) designed for single-channel underwater acoustic classification. The system is engineered to distinguish between marine biological signals (e.g., whales, dolphins) and mechanical threats (e.g., submarines, torpedoes) with high precision.

It features robust memory management for resource-constrained hardware (e.g., 2GB VRAM limits), an end-to-end training pipeline, and a lightweight inference engine designed for eventual deployment on edge devices like the NVIDIA Jetson Nano and Raspberry Pi 4.

## Features

* **10-Class Acoustic Classification:** Background, Beluga, Dolphin, Narwhal, Seal, Vessel, Walrus, Whale, Submarine, and Torpedo.
* **Hardware-Aware Batching:** Automatically resolves effective batch sizes based on available VRAM (CUDA/MPS compatible).
* **Robust Preprocessing:** Converts variable-length `.wav` files into fixed-length 2D log-mel spectrograms using `soundfile` and `torchaudio`.
* **Publication-Ready Visualization:** Automated parsing of training logs to generate learning curves and per-class performance metrics.

## Prerequisites

Ensure you have Anaconda or Miniconda installed, then install the required dependencies:

```bash
conda create -n underwater python=3.10
conda activate underwater

# Install PyTorch (Update the command based on your specific OS/CUDA version)
pip install torch torchvision torchaudio

# Install data processing and visualization dependencies
pip install soundfile pandas matplotlib seaborn

```

## Dataset Structure

The training pipeline integrates data from four distinct sources. You must have these datasets downloaded locally:

1. **DS3500:** Contains vessel and background noise.
2. **NOAA SanctSound:** Contains background soundscapes, rain, and sonar.
3. **Watkins Marine Mammal Sound Library:** Contains biological sounds (whales, dolphins, seals, etc.).
4. **Threat Dataset:** Contains `.wav` files for Submarines and Torpedoes.

## Usage

### 1. Training the Model

To train the CNN from scratch, use the `train` command and provide the paths to your local dataset directories.

```bash
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

### 2. Generating Visualizations

Once training is complete, generate the learning curves and final evaluation metric graphs:

```bash
python parse_and_plot_logs.py

```

*Outputs:* Generates high-DPI `.png` files (e.g., `learning_curves.png`, `per_class_metrics.png`) in the `./outputs/figures` directory.

### 3. Running Inference

To run a prediction on a single, unseen audio file, use the `predict` command:

```bash
python main.py predict \
  --audio-path ./data/submarine_torpedo/test_audio.wav \
  --model-path ./outputs/best_model.pth \
  --class-mapping-path ./outputs/class_mapping.json \
  --top-k 3

```

*Outputs:* Returns a JSON payload containing the top predicted class and confidence scores.

## Architecture

The underlying model (`UnderwaterAcousticCNN`) is a custom VGG-style 2D CNN.

* **Feature Extractor:** Four cascading convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool2d), scaling from 32 to 256 filters.
* **Classifier:** An Adaptive Average Pooling layer flattens the spatial dimensions, feeding into a Multi-Layer Perceptron (MLP) with Dropout for regularization, outputting the 10 target logits.

## Performance

The current iteration of the model achieves the following metrics on the stratified validation split:

* **Validation Accuracy:** 96.69%
* **Macro Precision:** 85.53%
* **Macro Recall:** 83.06%
* **Macro F1-Score:** 84.06%

*Note: Rare classes (e.g., Beluga, Walrus) exhibit lower support and recall compared to dominant classes (e.g., Vessel, Dolphin).*

## Future Work

* **Edge Deployment:** Exporting the PyTorch model to ONNX/TensorRT for real-time inference on the NVIDIA Jetson Nano.
* **Control System Integration:** Linking classification outputs to a physical AUV demonstration stand via Sliding Mode Control (SMC) architecture.

---
