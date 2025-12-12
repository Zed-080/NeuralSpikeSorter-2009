
# Neural Spike Sorter (EE32009 Coursework)

This repository implements a full **spike detection and classification pipeline** for extracellular neural recordings, targeting datasets **D2–D6** provided in the coursework.

The system is split into two main stages:

1.  **Spike detector**: A Conv1D "Sequence Labeling" model operating on 120-sample windows, producing per-timestep spike probabilities.
2.  **Spike classifier**: A Conv1D classifier operating on 64-sample waveforms, assigning each detected spike to one of **5 neuron classes**.

The pipeline also includes:
* **Wavelet-based Denoising**: Using `sym4`/`db4` wavelets with dataset-specific thresholding.
* **Matched Filter Enhancement**: Optionally enhances spike visibility using a template derived from D1.
* **Spectral Noise Matching**: Generates synthetic training data by grafting D2-D6 noise profiles onto clean D1 spikes.
* **Automated Tuning**: Grid-searches for the optimal detection threshold and refractory period.

---

## 📋 Prerequisites

* **Python 3.11** (Developed and tested on v3.11.9)
* **Virtual Environment** (Highly recommended to avoid dependency conflicts)

All dependencies are listed in `requirements.txt`.

---

## 🛠️ Quick Setup

**Windows:**
Double-click `setup_env.bat` or run:
```bash
setup_env.bat
````

**Mac / Linux:**
Run the following in your terminal:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

-----

## 🚀 How to Run

### Option A: Run Full Pipeline (Training + Inference)

Use this for the first run. It will generate data, train models, tune thresholds, and produce final predictions.

```bash
# Windows
.venv\Scripts\python main_runner.py

# Mac/Linux
source .venv/bin/activate
python main_runner.py
```

### Option B: Run Inference Only

Use this if you have already trained the models and just want to re-generate predictions for D2-D6.

```bash
# Windows
.venv\Scripts\python main_infer.py

# Mac/Linux
source .venv/bin/activate
python main_infer.py
```

-----

## 📂 Project Structure

```text
NeuralSpikeSorter/
├── data/                       # INPUT DATA
│   ├── D1.mat                  (Required for training)
│   ├── D2.mat ... D6.mat       (Required for inference)
├── outputs/                    # GENERATED ARTIFACTS
│   ├── detector_config.npz     (Saved threshold parameters)
│   ├── spike_detector_model.keras
│   ├── spike_classifier_model.keras
│   ├── predictions/            (Final .mat files for submission)
│   └── submission_datasets/    (Staging area for ZIP)
├── main_runner.py              # Master orchestrator script
├── main_infer.py               # Inference-only script
├── setup_env.bat               # Windows setup
├── setup_env.sh                # Linux/Mac setup
└── spike_pipeline/             # SOURCE CODE
    ├── __init__.py
    ├── data_loader/
    │   ├── build_classifier_data.py  # Cuts 64-sample waveforms
    │   ├── build_detector_data.py    # Cuts 120-sample windows + masks
    │   ├── generate_datasets.py      # Main data generation script
    │   └── load_datasets.py          # .mat loading and normalization
    ├── denoise/
    │   ├── matched_filter.py         # Template matching logic
    │   └── wavelet_denoise.py        # SWT denoising + High-pass
    ├── inference/
    │   ├── detect_spikes.py          # Sliding window logic
    │   ├── extract_waveforms.py      # Window extraction (Pre=20, Post=44)
    │   ├── matching.py               # F1 Score calculation
    │   └── run_pipeline_D2_D6.py     # Main inference loop for evaluation
    ├── models/
    │   ├── classifier_model.py       # 5-Class CNN architecture
    │   └── detector_model.py         # Binary Sequence Labeling CNN
    ├── training/
    │   ├── train_classifier.py       # Training loop for classifier
    │   ├── train_detector.py         # Training loop for detector
    │   └── tune_detector_threshold.py# Grid search for optimal params
    └── utils/
        ├── degradation.py            # Spectral noise synthesis
        └── signal_tools.py           # Filtering, MAD, Normalization
```

-----

## 🧠 Pipeline Details

### 1\. Data Engineering (`spike_pipeline/data_loader`)

  * **Spectral Matching**: To ensure robustness, we do not just train on D1. We analyze the noise spectrum of D2–D6 and inject that specific noise color into clean D1 spikes to create 5 synthetic training sets.
  * **Windowing**:
      * **Detector**: Extracts 120-sample windows. Labels are widened (width=3) to help the model learn the peak.
      * **Classifier**: Extracts 64-sample windows aligned 20 samples before and 44 samples after the peak.

### 2\. Denoising (`spike_pipeline/denoise`)

  * **Matched Filter**: A template is built by averaging all spikes in D1. This template is convolved with the noisy signal to highlight spike-like shapes.
  * **Wavelet Denoising**: Applies Stationary Wavelet Transform (SWT) using `db4` or `sym4` wavelets. Coefficients are soft-thresholded based on a robust noise estimate (MAD).

### 3\. Models (`spike_pipeline/models`)

  * **Detector**: A fully convolutional network that outputs a probability curve (Sequence Labeling). It uses `padding="same"` to preserve temporal resolution.
  * **Classifier**: A standard CNN with Batch Normalization and Global Average Pooling to ensure shift-invariance.

### 4\. Tuning (`spike_pipeline/training`)

  * We run the trained detector on the *clean* D1 dataset and perform a grid search over:
      * **Decision Threshold**: (e.g., 0.70 - 0.95)
      * **Refractory Period**: (e.g., 25 - 60 samples)
  * The combination yielding the highest **F1 Score** is saved to `detector_config.npz` and used for D2–D6.

<!-- end list -->
