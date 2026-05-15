# Smart Water Leakage & Theft Detection System

ML-powered pipeline anomaly detection using SCADA sensor data. Detects water leaks and abnormal usage patterns in real time to reduce water loss and operational costs.

---

## Overview

This project builds and compares two machine learning models on the **BattLeDIM 2018 dataset** — a real-world water distribution network benchmark. The system processes time-series SCADA data (pressures, flows, tank levels, demands) and predicts the total leakage volume at any given 5-minute interval using regression models.

Three approaches are implemented and evaluated side by side:

- **Random Forest Regressor** — ensemble baseline
- **XGBoost Regressor** — gradient boosting approach
- **Support Vector Machine (SVR)** — kernel-based regression

---

## Repository Structure

```
Leakage-Detection/
├── Dataset/
│   ├── 2018_SCADA.xlsx       # Sensor readings (pressures, flows, levels, demands)
│   └── 2018_Leakages.csv     # Ground truth leakage events per pipe
├── data_preprocess.py        # Data loading, feature engineering, target creation
├── rf_model.py               # Random Forest Regressor — train & evaluate
├── xgb_model.py              # XGBoost Regressor — train & evaluate
├── if_model.py               # SVM Regressor (formerly Isolation Forest) — train & evaluate
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
6. **Target Creation** — Create total leak column by summing all individual pipe leak columns.
7. **Alignment** — Features (`X`) and labels (`Y`) aligned by timestamp index

**Output shape example:**
```
X shape: (N_samples, N_features)   # sensor readings + rolling stats + time features
Y shape: (N_samples,)              # continuous target: total leakage volume
```

---

## Models

### Random Forest (`rf_model.py`)

- **Type:** Supervised regression
- **Algorithm:** `RandomForestRegressor` — 100 trees
- **Split:** 75% train / 25% test (shuffled, `random_state=42`)
- **Evaluation:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2)

```bash
python rf_model.py
```

### XGBoost (`xgb_model.py`)

- **Type:** Supervised regression
- **Algorithm:** `XGBRegressor` — 100 trees
- **Evaluation:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2)

```bash
python xgb_model.py
```

### Support Vector Machine (`if_model.py`)

- **Type:** Supervised regression
- **Algorithm:** `SVR` — RBF kernel
- **Evaluation:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2)

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

### Train & Evaluate XGBoost

```bash
python xgb_model.py
```

### Train & Evaluate SVM

```bash
python if_model.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading, merging, resampling, feature engineering |
| Scikit-learn | Random Forest, SVM (SVR), evaluation metrics |
| XGBoost | High-performance gradient boosting |
| OpenPyXL | Reading `.xlsx` SCADA files |

---

## Key Design Decisions

- **5-minute resampling** balances temporal granularity with computational efficiency
- **Rolling window of 36 steps** (= 3 hours at 5-min intervals) captures short-term sensor drift that precedes leaks
- **Regression approach** — We model the total volume of leaks instead of simply classifying them as present/absent, providing more precise actionability for operators

---

## Current Limitations

### Tree-Based Models
The Random Forest and XGBoost models achieve high test scores but may memorize temporal correlations:

- The rolling mean and rolling std features are computed over a 36-step window, which creates high feature correlation across adjacent timestamps.
- Shuffled train/test split (`shuffle=True`) breaks temporal ordering, causing data leakage — future sensor states can bleed into the training set, inflating test scores artificially.

### SVM — Underfitting
The SVM model with the RBF kernel fails significantly to capture the complexity of the data, predicting close to average values and suffering from very high MSE and negative R-squared. High dimensionality and noise prevent the SVM from finding a strong non-linear decision boundary in raw sensor variance.

---

## Roadmap

- **Fix temporal data leakage** — Switch to a chronological train/test split (first 75% of time as train, last 25% as test) to properly simulate real deployment conditions
- **Leak localization** — Extend from total leakage (`Total_Leak`) to pipe-level localization by predicting which specific pipe is leaking and by how much, enabling targeted repairs
- **Online / streaming inference** — Adapt the pipeline for real-time scoring as new SCADA readings arrive, rather than batch evaluation
- **Explainability** — Integrate SHAP values to surface which sensors are driving predictions, making the system auditable for water utility operators

---

## Dataset Reference

> Vrachimis, S.G., et al. (2020). *BattLeDIM: Battle of the Leakage Detection and Isolation Methods*. In *Proc. 2nd International CCWI/WDSA Joint Conference*, Beijing, China.

---

## License

This project is for academic and research purposes.
