# Smart Water Leakage & Theft Detection System

ML-powered pipeline for large-scale water leakage detection using SCADA sensor data. This project tackles severe dataset imbalance and complex time-series anomalies to identify pipeline bursts and leaks in a water distribution network.

---

## Overview

This project utilizes the **BattLeDIM 2018 dataset** (a real-world water distribution benchmark). It processes multi-variate time-series SCADA data (pressures, flows, tank levels, demands) to perform binary classification: **Normal vs. Leak**.

### The Challenge: Extreme Data Imbalance
Throughout the development, a massive class imbalance was discovered: ~98% of the chronological dataset represented various states of water leakage, while only ~2% consisted of truly "Normal" (healthy) network operation. 
Because of this, standard models completely un-fitted or collapsed:
- **Supervised Models (RF, XGB, SVM):** Easily over-fitted to the majority "Leak" class and failed to accurately map normal baseline operations, even with heavy class-weighting.
- **Isolation Forest (IF):** Failed due to the contamination fraction fundamentally breaking the mathematical boundaries of standard anomaly algorithms.

### The Solution: Unsupervised GRU Seq-to-Seq Autoencoder
We pivoted to a purely **Unsupervised Deep Learning** approach. We isolated the healthy normal data and trained a **GRU Sequence-to-Sequence Model** to perfectly memorize and reconstruct normal water physics. 

During inference, if the model fails to reconstruct a sequence (resulting in a high Mean Squared Error), the system flags it as an anomaly/leak. By combining this with **Dynamic Day/Night Thresholding**, the model catches elusive "stealth" leaks without triggering false alarms during peak morning usage.

---

## Repository Structure

```
Leakage-Detection/
├── Dataset/
│   ├── 2018_SCADA.xlsx       # Sensor readings (pressures, flows, levels, demands)
│   └── 2018_Leakages.csv     # Ground truth leakage events per pipe
├── data_preprocess.py        # Alignments, scaling, sequence/sliding windows
├── gru_ae_model.py           # Flagship Model: GRU Seq-to-Seq Autoencoder
├── rf_model.py               # Legacy Baseline: Random Forest
├── xgb_model.py              # Legacy Baseline: XGBoost
├── svm_model.py              # Legacy Baseline: Support Vector Machine (SVC)
├── if_model.py               # Legacy Baseline: Isolation Forest
└── README.md
```

*(Place the dataset files inside the `Dataset/` folder in the project root before running).*

---

## Data Preprocessing (`data_preprocess.py`)

A complete chronological alignment and sequencing pipeline:
1. **Merge & Resample:** Merges all 4 SCADA sheets and resamples down to 5-minute intervals.
2. **Missing Value & Metric Alignment:** Culls invalid European comma-decimals and missing values.
3. **Physical Magnitude Limiting:** Converts total volumes into continuous magnitudes; values below a physical network threshold are ignored as background noise.
4. **Time & Rolling Engineering:** Injects Time-of-Day features to allow the model to learn peak vs off-peak consumption.
5. **Sequence Chunking:** StandardScaler normalizes the data, which is then reshaped into continuous chronologic tensors for PyTorch GRU compatibility.

---

## Models

### 1. Flagship: GRU Seq-to-Seq Autoencoder (`gru_ae_model.py`)
- **Type:** Unsupervised Anomaly Detection.
- **Architecture:** Multi-dimensional PyTorch `nn.GRU` layers. Optimized hidden dimensions (e.g. 80-dim) to strike the "Goldilocks" balance between over-fitting and bottleneck compression.
- **Training Strategy:** Chronological split. Only trained on the subset of data mathematically proven to be 100% healthy `(Y == 0)`.
- **Dynamic Thresholding:** Exports prediction outcomes to `leak_analysis.csv`. Utilizes a 95th-percentile error threshold during the day and a stricter 75th-percentile threshold at night (Minimum Night Flow analysis) to maximize true positive leak catches directly tied to physical fluid dynamics.

### 2. Legacy Baselines (`xgb_model.py`, `rf_model.py`, `svm_model.py`, `if_model.py`)
Standard regressor and classifier iterations that failed to cleanly navigate the massive dataset imbalance. Kept in the repository for benchmark tracking and methodology comparison.

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies
```bash
pip install torch pandas scikit-learn xgboost openpyxl
```

### Run Flagship GRU Model
```bash
python gru_ae_model.py
```
*(This automatically calls `data_preprocess.py`, trains the PyTorch model, outputs continuous metrics, and saves a `leak_analysis.csv` file for physical magnitude crossover inspection).*

---

## Key Technical Milestones

- **Threshold Grid Search & Percentile Bounding:** Transitioned from blind guessing to dynamically extracting the exact anomaly boundary from validation reconstruction errors.
- **Minimum Night Flow (MNF):** Overrode static threshold limits recognizing that low-magnitude stealth leaks look identical to customer shower usage. Enforced "strict" thresholds between 2 AM and 6 AM.
- **Hidden Dimension Scaling:** Experimented deeply across 16-1024 dimensional bounds to solve input projection blending issues, discovering the optimal raw temporal sequence structure without relying on massive compute overhead.

---

## Dataset Reference

> Vrachimis, S.G., et al. (2020). *BattLeDIM: Battle of the Leakage Detection and Isolation Methods*. In *Proc. 2nd International CCWI/WDSA Joint Conference*, Beijing, China.

---

## License

This project is for academic and research purposes.
