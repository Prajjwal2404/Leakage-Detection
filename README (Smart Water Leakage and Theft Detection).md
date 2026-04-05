# Smart Water Leakage & Theft Detection System

ML-powered pipeline anomaly detection using SCADA sensor data. Detects water leaks and abnormal usage patterns in real time to reduce water loss and operational costs.

---

## Overview

This project builds and compares two machine learning models on the **BattLeDIM 2018 dataset** — a real-world water distribution network benchmark. The system processes time-series SCADA data (pressures, flows, tank levels, demands) and predicts whether a leak is occurring at any given 5-minute interval.

Two approaches are implemented and evaluated side by side:

- **Random Forest (Supervised)** — trained with labeled leak/no-leak ground truth
- **Isolation Forest (Unsupervised)** — anomaly detection without relying on labels during training

---

## Repository Structure

```
Leakage-Detection/
├── Dataset/
│   ├── 2018_SCADA.xlsx       # Sensor readings (pressures, flows, levels, demands)
│   └── 2018_Leakages.csv     # Ground truth leakage events per pipe
├── data_preprocess.py        # Data loading, feature engineering, target creation
├── rf_model.py               # Random Forest classifier — train & evaluate
├── if_model.py               # Isolation Forest — train & evaluate
└── .gitignore
```

---

## Dataset

**BattLeDIM (Battle of the Leakage Detection and Isolation Methods) — 2018**

The dataset simulates a real water distribution network with SCADA sensors recording:

| Sheet / File | Description |
|---|---|
| `Pressures (m)` | Pressure readings at network nodes |
| `Flows (m3_h)` | Flow rates across pipes |
| `Levels (m)` | Tank water levels |
| `Demands (L_h)` | Per-node water demand |
| `2018_Leakages.csv` | Ground truth — leak flow rate per pipe per timestamp |

> **Note:** Place the dataset files inside a `Dataset/` folder in the project root before running.

---

## Data Preprocessing (`data_preprocess.py`)

The preprocessing pipeline merges all SCADA sheets, engineers time-aware features, and aligns the target labels:

1. **Load & Merge** — All four SCADA sheets merged on `Timestamp`
2. **Demand Aggregation** — Individual node demands summed into `Total_System_Demand`
3. **Resampling** — Data resampled to `5-minute` intervals (configurable)
4. **Time Features** — `Hour` of day and `Is_Daytime` (1 if 6AM–6PM, else 0)
5. **Rolling Statistics** — For every sensor column, a rolling mean and rolling std are computed over a 36-step window to capture short-term trends
6. **Target Creation** — Leakage CSV is binarized: `Any_Leak = 1` if any pipe has non-zero leak flow at that interval, else `0`
7. **Alignment** — Features (`X`) and labels (`Y`) aligned by timestamp index

**Output shape example:**
```
X shape: (N_samples, N_features)   # sensor readings + rolling stats + time features
Y shape: (N_samples,)              # binary: 0 = normal, 1 = leak detected
```

---

## Models

### Random Forest (`rf_model.py`)

- **Type:** Supervised classification
- **Algorithm:** `RandomForestClassifier` — 100 trees, `class_weight='balanced'` to handle class imbalance
- **Split:** 75% train / 25% test (shuffled, `random_state=42`)
- **Evaluation:** Accuracy, Classification Report (Precision / Recall / F1), Confusion Matrix

```bash
python rf_model.py
```

### Isolation Forest (`if_model.py`)

- **Type:** Unsupervised anomaly detection
- **Algorithm:** `IsolationForest` — `contamination='auto'`
- **Note:** Trained on features only (no labels). Predictions mapped: `-1 → anomaly (leak)`, `1 → normal`
- **Evaluation:** Accuracy, Classification Report, Confusion Matrix (compared against ground truth labels)

```bash
python if_model.py
```

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install pandas scikit-learn openpyxl
```

### Run Preprocessing Only

```bash
python data_preprocess.py
```

### Train & Evaluate Random Forest

```bash
python rf_model.py
```

### Train & Evaluate Isolation Forest

```bash
python if_model.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading, merging, resampling, feature engineering |
| Scikit-learn | Random Forest, Isolation Forest, evaluation metrics |
| OpenPyXL | Reading `.xlsx` SCADA files |

---

## Key Design Decisions

- **5-minute resampling** balances temporal granularity with computational efficiency
- **Rolling window of 36 steps** (= 3 hours at 5-min intervals) captures short-term sensor drift that precedes leaks
- **`class_weight='balanced'`** in Random Forest compensates for the heavily imbalanced dataset (leaks are rare events)
- **Isolation Forest** serves as a label-free baseline — useful for scenarios where labeled data is unavailable

---

## Current Limitations

### Random Forest — Overfitting
The RF model achieves high training accuracy but generalizes poorly on unseen data. Root causes:

- The rolling mean and rolling std features are computed over a 36-step window, which creates high feature correlation across adjacent timestamps. RF memorizes these correlated patterns rather than learning the underlying signal.
- The dataset is heavily imbalanced (leaks are rare). Despite `class_weight='balanced'`, RF still tends to learn the majority class boundary too tightly.
- Shuffled train/test split (`shuffle=True`) breaks temporal ordering, causing data leakage — future sensor states can bleed into the training set, inflating test accuracy artificially.

### Isolation Forest — Underfitting
The IF model fails to reliably distinguish leak intervals from normal operation. Root causes:

- Isolation Forest is a general-purpose anomaly detector and has no notion of temporal context. It treats each 5-minute window as an independent sample, missing the gradual pressure drops and flow deviations that characterize a developing leak.
- `contamination='auto'` assumes a fixed anomaly rate across the dataset, which does not reflect the sporadic and irregular nature of real leakage events.
- The high-dimensional feature space (sensor readings + rolling stats across all nodes) dilutes the anomaly signal, making isolation splits less discriminative.

---

## Roadmap

The current models establish a baseline. The next phase focuses on architectures that are inherently designed for sequential, time-dependent data.

### Phase 2 — Sequence Models

**LSTM (Long Short-Term Memory)**
- Replace the static rolling window approach with a learned temporal context
- Feed sequences of sensor readings as input rather than single flattened feature vectors
- LSTM hidden states will capture gradual pressure decay and flow divergence patterns that precede leaks — something RF cannot model and IF ignores entirely

**Temporal Convolutional Networks (TCN)**
- Use dilated causal convolutions to capture multi-scale temporal patterns efficiently
- Faster to train than LSTM with comparable or better performance on long sequences

**Transformer / Attention-Based Models**
- Apply self-attention across the time dimension to let the model learn which past sensor readings are most predictive of a current leak
- Particularly useful given the multi-sensor nature of the data — attention heads can learn cross-sensor correlations (e.g., pressure drop at node A + flow spike at pipe B = likely leak)
- Explore time-series-specific variants such as **Informer** or **PatchTST**

### Phase 3 — State Space Models (SSMs)

**Mamba / S4**
- State space models offer a principled way to model long-range temporal dependencies with linear complexity
- More efficient than full Transformers on very long sensor sequences (the 2018 dataset spans a full year at 5-minute resolution — ~105,000 timesteps)
- S4 in particular has shown strong results on continuous-time irregularly sampled data, which matches the nature of SCADA sensor streams

### Phase 4 — System Improvements

- **Fix temporal data leakage** — Switch to a chronological train/test split (first 75% of time as train, last 25% as test) to properly simulate real deployment conditions
- **Leak localization** — Extend from binary detection (`Any_Leak`) to pipe-level localization by predicting which specific pipe is leaking
- **Online / streaming inference** — Adapt the pipeline for real-time scoring as new SCADA readings arrive, rather than batch evaluation
- **Explainability** — Integrate SHAP values to surface which sensors are driving predictions, making the system auditable for water utility operators

---

## Dataset Reference

> Vrachimis, S.G., et al. (2020). *BattLeDIM: Battle of the Leakage Detection and Isolation Methods*. In *Proc. 2nd International CCWI/WDSA Joint Conference*, Beijing, China.

---

## License

This project is for academic and research purposes.
