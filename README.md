# COVID-19 Indonesia: Epidemiological Analysis & Prediction System

> An end-to-end machine learning pipeline combining **K-Means Clustering** and **Logistic Regression** to analyze COVID-19 transmission patterns across Indonesian provinces, served through an interactive **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue)

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Unsupervised Learning — K-Means Clustering](#3-unsupervised-learning--k-means-clustering)
  - [4. Supervised Learning — Logistic Regression](#4-supervised-learning--logistic-regression)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## Overview

The COVID-19 pandemic has unfolded unevenly across Indonesia, with some provinces experiencing case counts orders of magnitude higher than others. This project applies a dual-paradigm data mining approach to extract actionable insight from official epidemiological data:

1. **Unsupervised segmentation** — K-Means clustering groups provinces into latent risk tiers (low / medium / high), validated by the Elbow Method and Silhouette Score.
2. **Supervised classification** — A Logistic Regression model predicts whether a given daily observation is a *high-incidence* day (new cases above the national median), with explicit safeguards against target leakage.
3. **Interactive deployment** — A Streamlit dashboard exposes the pipeline to non-technical users, including a symptom-based screening module as a proof-of-concept extension.

The project was developed as the final deliverable for the **Data Mining (BBK2LAB3)** course at Telkom University Jakarta, 2025.

---

## Team

**Kelompok 8 — Program Studi Sistem Informasi, Universitas Telkom Kampus Jakarta**

| Role | Name | NIM |
|------|------|-----|
| **Team Leader / ML Engineer** | Muhammad Abrar Rayhan | 102042330088 |
| Team Member | Khansa Tanaya Putri Daryatmo | 102042300083 |
| Team Member | Princenza Alexandria | 102042300105 |
| Team Member | Tri Novita Sari | 102042300112 |
| Team Member | Angga Prasetya Putra | 102042300205 |
| Team Member | Khansa Khairunnisa | 102042300106 |

**Supervisor:** Arif Rahman Hakim, S.Kom., M.Kom.

---

## Dataset

- **Source:** [COVID-19 Indonesia Time Series — Kaggle (Hendratno)](https://www.kaggle.com/datasets/hendratno/covid19-indonesia)
- **Granularity:** Daily records at the province level, plus a national aggregate (`Indonesia`)
- **Size:** 31,822 rows × 38 columns
- **Coverage:** March 2020 – December 2022
- **Geographic units:** 34 Indonesian provinces + national aggregate

### Selected Features

After schema review, seven attributes were retained as the analytical core:

| Feature | Type | Purpose |
|---------|------|---------|
| `New Cases` | int | Daily newly-confirmed cases — used as the threshold base for the supervised target |
| `New Deaths` | int | Daily newly-reported deaths — proxy for severity |
| `New Recovered` | int | Daily recoveries — indicator of healthcare system throughput |
| `Total Cases` | int | Cumulative confirmed cases |
| `Total Deaths` | int | Cumulative deaths |
| `Total Recovered` | int | Cumulative recoveries |
| `Province_encoded` | int | Label-encoded province identifier |

---

## Methodology

### 1. Data Preprocessing

**Missing value handling.** The selected subset was verified to contain no missing entries across all seven attributes. A median-imputation safeguard was implemented regardless, hardening the pipeline against future data drift.

**Categorical encoding.** The `Province` column was transformed via `sklearn.preprocessing.LabelEncoder`, producing `Province_encoded ∈ {0, …, 34}`. The fitted encoder is persisted to disk for inference-time consistency.

**Feature scaling.** All numerical features were standardized using `StandardScaler` (zero mean, unit variance). This step is essential because:

- K-Means relies on Euclidean distance and is scale-sensitive.
- Logistic Regression coefficients become directly comparable in magnitude, enabling principled feature-importance interpretation.

### 2. Exploratory Data Analysis

Four complementary visualizations were produced to inform downstream modeling decisions:

1. **Distribution of `New Cases`** — reveals a heavily right-skewed distribution (max 64,718 vs. median 27), motivating the use of the **median** (rather than mean) as the decision threshold for supervised labeling.
2. **Correlation heatmap** — quantifies linear relationships. Notable findings:
   - `New Cases` ↔ `New Recovered`: **0.86** (very strong)
   - `Total Cases` ↔ `Total Recovered`: **1.00** (near-perfect, motivating leakage prevention)
   - `Total Cases` ↔ `Total Deaths`: **0.97**
3. **`Total Cases` vs. `Total Deaths` scatter** — confirms a strong monotonic relationship consistent with a stable case fatality rate across provinces.
4. **Temporal trend of daily new cases** — surfaces the characteristic pandemic waves.

### 3. Unsupervised Learning — K-Means Clustering

**Optimal cluster count.** Two complementary criteria were used to select `k`:

- **Elbow Method** on Within-Cluster Sum of Squares (WCSS) across `k ∈ [2, 10]`.
- **Silhouette Score** on the same range, peaking at a value consistent with `k = 3`.

Both criteria converged on **`k = 3`**.

**Training configuration.** `KMeans(n_clusters=3, random_state=42, n_init=10)` was fit on the standardized feature matrix.

**Cluster interpretation (back-transformed means).**

| Cluster | Profile | Mean New Cases | Mean New Deaths | Mean Total Cases | Interpretation |
|--------:|---------|---------------:|----------------:|-----------------:|----------------|
| **0** | Low-incidence provinces | 353 | 9 | 137,079 | Majority of provinces outside Java |
| **1** | National-aggregate regime | 9,775 | 228 | 4,902,906 | Dominated by the `Indonesia` national row |
| **2** | Medium-incidence provinces | 189 | 5 | 49,489 | Transitional provinces with moderate load |

**Dimensionality reduction for visualization.** Principal Component Analysis (PCA → 2D) was applied post-clustering to project the 7-dimensional feature space into an interpretable scatter plot, with cluster membership color-encoded.

### 4. Supervised Learning — Logistic Regression

**Target definition.** A binary label `High_Cases` was engineered as:

```
High_Cases = 1 if New Cases > median(New Cases) else 0
```

The median (27) was chosen instead of the mean (402) because the distribution is heavily right-skewed — the median yields a class-balanced target (~50/50), avoiding the need for resampling.

**Leakage prevention.** `New Cases` (source of the target) and `Total Cases` (deterministically correlated with `New Cases` via cumulative summation) were **explicitly dropped** from the feature set. This is a critical methodological choice — omitting it would inflate accuracy to near-perfect values via circular reasoning.

**Final feature set (post-leakage-check):** `New Deaths`, `New Recovered`, `Total Deaths`, `Total Recovered`, `Province_encoded`.

**Training.** A 70/30 stratified train-test split (`random_state=42`) with `LogisticRegression(max_iter=1000)` was used.

**Feature importance** (absolute coefficient magnitudes on standardized features):

| Rank | Feature | \|Coefficient\| |
|-----:|---------|-----------:|
| 1 | New Recovered | 27.20 |
| 2 | New Deaths | 22.34 |
| 3 | Total Recovered | 1.18 |
| 4 | Total Deaths | 0.42 |
| 5 | Province_encoded | 0.06 |

The dominance of `New Recovered` and `New Deaths` reflects the tight temporal coupling between these daily-flow variables and same-day case counts — cumulative totals contribute far less discriminative signal once the daily-flow features are available.

---

## Results

### Classification Performance

| Metric | Class 0 (Low) | Class 1 (High) | Macro Avg |
|--------|--------------:|---------------:|----------:|
| Precision | 0.76 | 0.94 | 0.85 |
| Recall | 0.95 | 0.70 | 0.82 |
| F1-Score | 0.84 | 0.80 | 0.82 |

**Overall accuracy: 82.4%** on a held-out test set of 9,547 samples.

### Confusion Matrix

|                     | Predicted Low | Predicted High |
|---------------------|--------------:|---------------:|
| **Actual Low**      | 4,519 (TN)    | 232 (FP)       |
| **Actual High**     | 1,452 (FN)    | 3,344 (TP)     |

The model exhibits **high precision on the high-incidence class (0.94)** — when it flags a day as high-incidence, it is almost always correct. Recall on the same class is lower (0.70), meaning some high-incidence days are missed. This trade-off is acceptable for a screening tool where false alarms are more costly than missed detections; threshold tuning is recommended if the opposite priority applies.

### Clustering Quality

Silhouette Score at `k = 3`: approximately **0.62**, indicating well-separated and cohesive clusters.

### Hypothesis Validation

| Hypothesis | Result |
|------------|--------|
| Provinces cluster into 3–4 groups | ✅ Confirmed (3 clusters optimal) |
| Logistic Regression achieves > 80% accuracy | ✅ Confirmed (82.4%) |
| `Total Cases` and `New Deaths` are strong predictors | ⚠️ Partial — `New Deaths` confirmed, but `New Recovered` emerged as the strongest predictor after leakage-aware feature selection |

---

## Tech Stack

**Core ML & Data**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — `KMeans`, `LogisticRegression`, `StandardScaler`, `LabelEncoder`, `PCA`, `train_test_split`, metrics
- `joblib` — model persistence

**Visualization**
- `matplotlib`, `seaborn` — static plots
- `plotly` — interactive charts in the dashboard

**Deployment**
- `streamlit` — interactive web dashboard
- `pyngrok` — public tunneling from Google Colab for demo purposes

---

## Project Structure

```
covid19-indonesia-datmin/
├── notebook/
│   └── covid19_analysis.ipynb       # Full pipeline: EDA → preprocessing → K-Means → LogReg
├── app/
│   └── app.py                       # Streamlit dashboard (symptom-based screening demo)
├── models/
│   ├── covid-19_model.pkl           # Trained Logistic Regression model
│   ├── scaler.pkl                   # Fitted StandardScaler
│   └── encoder.pkl                  # Fitted LabelEncoder
├── data/
│   └── covid_19_indonesia_time_series_all.csv
├── docs/
│   └── LAPORAN_TUBES_DATMIN_KELOMPOK_8.pdf
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/covid19-indonesia-datmin.git
cd covid19-indonesia-datmin

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis Notebook

```bash
jupyter notebook notebook/covid19_analysis.ipynb
```

### Launch the Streamlit Dashboard

```bash
streamlit run app/app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## Deployment

The dashboard was developed and demoed from Google Colab using **ngrok** for public URL tunneling. To reproduce:

```python
from pyngrok import ngrok
import os

# Kill any existing ngrok tunnels
ngrok.kill()

# Add your ngrok authtoken (get one free at https://dashboard.ngrok.com/)
!ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>

# Launch Streamlit in the background
get_ipython().system_raw('streamlit run app.py --server.port 8501 &')

# Create the public tunnel
public_url = ngrok.connect(addr="8501", proto="http")
print(f"Dashboard accessible at: {public_url}")
```

For production deployment, consider **Streamlit Community Cloud**, **Hugging Face Spaces**, or a containerized deployment on **Railway / Render**.

---

## Screenshots

> Placeholder section — replace with actual screenshots once captured.

**1. Symptom Input Sidebar**
![Sidebar Input](docs/screenshots/sidebar_input.png)

**2. Prediction Result — Positive Case**
![Positive Result](docs/screenshots/result_positive.png)

**3. Probability Chart**
![Probability Bar Chart](docs/screenshots/probability_chart.png)

**4. PCA Cluster Visualization**
![Cluster PCA](docs/screenshots/cluster_pca.png)

**5. Confusion Matrix**
![Confusion Matrix](docs/screenshots/confusion_matrix.png)

---

## Future Improvements

**Data expansion**
- Extend granularity to the regency/city level for finer spatial resolution.
- Integrate external covariates: vaccination coverage, mobility indices (Google / Apple), NPI policy indicators, and healthcare capacity.

**Methodological extensions**
- Benchmark Logistic Regression against ensemble methods (Random Forest, XGBoost, LightGBM) and compare calibration quality, not just accuracy.
- Explore time-series-aware approaches: LSTM, temporal convolutional networks, or Prophet for forecasting.
- Apply hierarchical or density-based clustering (HDBSCAN) to relax K-Means' spherical-cluster assumption.

**Evaluation rigor**
- Perform temporal cross-validation (rolling-origin) instead of random splits, since i.i.d. assumptions are violated in time-series data.
- Add ROC-AUC, PR-AUC, and calibration plots to the evaluation suite.
- External validation on a held-out period or a different country's dataset.

**Deployment & MLOps**
- Replace the dummy symptom dataset in the Streamlit app with a properly-sourced clinical dataset (e.g., Israeli Ministry of Health COVID-19 symptom dataset) and re-train.
- Containerize with Docker for reproducibility.
- Add automated retraining pipelines as new data becomes available.
- Implement real-time early-warning alerts for provinces approaching high-incidence thresholds.

**Dashboard enhancements**
- Interactive choropleth map of Indonesia with cluster coloring.
- Temporal slider to visualize cluster membership drift across pandemic waves.
- Downloadable PDF reports per province.

---

## Acknowledgments

- **Hendratno** for maintaining the open COVID-19 Indonesia time-series dataset on Kaggle.
- **Arif Rahman Hakim, S.Kom., M.Kom.** for supervision and methodological guidance throughout the course.
- **Telkom University Jakarta — Program Studi Sistem Informasi** for providing the academic context.
- The open-source maintainers of `scikit-learn`, `pandas`, `streamlit`, and `plotly`.

---

## Disclaimer

This project is developed strictly for **academic and educational purposes**. The symptom-based screening demo in the Streamlit dashboard uses synthetic placeholder data and must **not** be used for medical diagnosis. Individuals experiencing symptoms should consult a qualified healthcare professional.

---

<div align="center">

**Telkom University Jakarta · Program Studi Sistem Informasi · 2025**

</div>
