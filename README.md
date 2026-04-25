# 🍽️ Restaurant Rating Prediction — End-to-End ML Pipeline

> A production-grade, end-to-end machine learning system for predicting restaurant aggregate ratings. Built on a global Zomato dataset of **9,551 restaurants across 15 countries**, the project spans structured EDA, feature engineering, a modular training pipeline with drift detection, MLflow experiment tracking, and an interactive Streamlit prediction app.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [What This Enables](#-what-this-enables)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Pipeline Deep-Dive](#-pipeline-deep-dive)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Model Evaluation](#-model-evaluation)
- [Key Findings & Implications](#-key-findings--implications)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)

---

## 🎯 Project Overview

This project builds a complete ML pipeline — from raw data ingestion out of MongoDB, through multi-stage validation, feature engineering, and model selection, to a deployable Streamlit web application. The pipeline is fully modular: each stage produces typed artifact dataclasses consumed by the next, making the system easy to extend, debug, and re-run.

**What makes this project production-oriented:**
- MongoDB-backed data ingestion with schema-driven validation
- Multi-stage validation: structural checks → data drift (KS test) → post-transform integrity
- Target encoding for high-cardinality features (city, country, cuisine) instead of naive one-hot encoding
- MLflow experiment tracking for every trained model
- A serialized preprocessing pipeline (`preprocessor.pkl`) for consistent train/inference transforms
- Streamlit app with real-time single prediction and CSV-based batch prediction modes

> **Scope note:** The model predicts expected ratings for *active, reviewable restaurants* — those with existing operational signals like price tier, cuisine type, and city context. It is not designed for cold-start scenarios where a restaurant has zero prior reviews or presence.

---

## 💡 What This Enables

Training on the Zomato dataset produces a generalizable rating predictor that goes beyond academic benchmarking. Here are concrete use cases the trained model supports:

**New restaurant owners** can use single-prediction mode to benchmark their expected rating *before launch* — given their planned price tier, cuisine, and city. This gives actionable pre-opening intelligence: for example, the model quantifies how much adding table booking shifts the predicted rating, or whether a cuisine choice is disadvantageous in a specific market.

**Food delivery platforms and aggregators** can apply batch prediction to populate "predicted rating" placeholders for newly listed restaurants with no review history. Rather than showing a blank or suppressing discovery, they can surface a model-estimated quality tier to help new restaurants compete from day one.

**Franchise operators and investors** can run portfolio-level analysis: given a target city and cuisine category, which combination yields the highest predicted rating ceiling? The city-level target encodings and cuisine average ratings make this a direct query on the trained mappings.

**Restaurant consultants** can simulate how operational changes affect predicted ratings — modelling scenarios like upgrading from mid to premium pricing, adding table booking, or expanding cuisine offerings — without waiting for actual review data to accumulate.

**Data and product teams at food-tech companies** can extend the pipeline to their own proprietary datasets. The modular architecture (schema-driven validation, drift detection, serialized preprocessor) means retraining on a new regional dataset requires only swapping the MongoDB collection and re-running `run_training.py`.

---

## 📺 Live Demo

| App Preview — Single Prediction | App Preview — Batch Prediction |
|---|---|
| ![App Preview 1](app/app_preview_1.png) | ![App Preview 2](app/app_preview_2.png) |

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Source** | Zomato Restaurant Data |
| **File** | `Restaurant_Data/Dataset.csv` |
| **Records** | 9,551 restaurants |
| **Features** | 21 columns (8 numerical, 13 categorical) |
| **Countries** | 15 |
| **Cities** | 141 |
| **Target Variable** | `Aggregate rating` (0.0 – 5.0) |
| **Encoding** | UTF-8 with BOM (`utf-8-sig`) |

**Key input features:** Country Code, City, Cuisines, Average Cost for Two, Currency, Has Table Booking, Has Online Delivery, Is Delivering Now, Price Range, Votes.

**Rating Scale:**

| Color | Label | Range |
|---|---|---|
| 🟢 Dark Green | Excellent | 4.5 – 5.0 |
| 🟩 Green | Very Good | 4.0 – 4.4 |
| 🟡 Yellow | Good | 3.5 – 3.9 |
| 🟠 Orange | Average | 3.0 – 3.4 |
| 🔴 Red | Poor | 2.5 – 2.9 |
| ⚪ White | Not Rated | 0.0 |

> **Note:** 2,148 restaurants (22.5%) carry a rating of 0.0 ("Not rated"). These are genuine missing values, not true zeros — and were excluded from all modelling and rating-based analyses.

---

## 🗂️ Project Structure

```
Restaurant-Rating/
│
├── Restaurant_Data/
│   ├── Dataset.csv                         # Raw Zomato dataset (9,551 records)
│   └── README.md                           # Dataset column reference & notes
│
├── Notebooks/
│   ├── EDA/
│   │   ├── 01_EDA.ipynb                    # Data exploration & preprocessing
│   │   ├── 02_EDA.ipynb                    # Table booking, delivery & price analysis
│   │   ├── 03_EDA.ipynb                    # Cuisine & customer preference analysis
│   │   ├── 04_regression_analysis.ipynb    # Feature engineering & model benchmarking
│   │   ├── rating_dashboard.py             # Streamlit EDA dashboard
│   │   └── utils/
│   │       └── rating_histogram.py         # Histogram utility
│   ├── processed_data/
│   │   └── Dataset_filtered.csv            # Filtered dataset (output of Notebook 01)
│   └── reports/                            # All EDA visualizations (.png exports)
│
├── src/                                    # Core ML package
│   ├── components/
│   │   ├── data_ingestion.py               # MongoDB fetch → train/test split
│   │   ├── data_validation.py              # Primary, drift, and final validation
│   │   ├── data_transformation.py          # Feature engineering & preprocessing
│   │   └── model_trainer.py               # Multi-model benchmarking + MLflow tracking
│   ├── pipeline/
│   │   ├── training_pipeline.py            # Orchestrates all pipeline stages
│   │   └── batch_prediction.py             # Batch inference from MongoDB
│   ├── entity/
│   │   ├── config_entity.py                # Typed config dataclasses per stage
│   │   └── artifact_entity.py              # Typed artifact dataclasses per stage
│   ├── constants/
│   │   ├── training_pipeline/__init__.py   # All pipeline constants & path definitions
│   │   └── models/__init__.py              # Model registry & hyperparameter search space
│   ├── utils/
│   │   ├── main_utils/utils.py             # File I/O, DB fetch, YAML helpers
│   │   └── ml_utils/
│   │       ├── metric/regression_metric.py # MAE, RMSE, R² computation + model evaluation
│   │       └── model/estimator.py          # RatingPredictor class for batch inference
│   ├── exception/exception.py              # Custom exception with traceback formatting
│   └── logging/logger.py                   # Timestamped rotating file logger
│
├── app/
│   ├── rating_app.py                       # Streamlit UI (single & batch prediction)
│   ├── backend.py                          # Inference backend: transform + predict
│   ├── data.yaml                           # UI dropdown options (cities, currencies, etc.)
│   ├── styles/styles.py                    # Custom CSS for the Streamlit app
│   └── templates/templates.py             # HTML templates for app components
│
├── scripts/
│   ├── push_data.py                        # Upload raw CSVs to MongoDB collections
│   ├── run_training.py                     # CLI entry point for training + batch prediction
│   ├── run_inference.py                    # Standalone inference runner
│   └── test_mongodb_connection.py          # MongoDB connectivity check
│
├── data_schema/
│   └── schema.yaml                         # Column types, target, drop list, encoding config
│
├── historical_data/
│   └── base_df.csv                         # Baseline data for KS-based drift detection
│
├── batch_prediction_data/
│   └── batch_input.csv                     # Sample batch input for inference
│
├── final_model/                            # ← Generated at runtime
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── binary_mapping.yaml
│   ├── City_mapping.yaml
│   ├── Country Code_mapping.yaml
│   ├── Currency_mapping.yaml
│   └── cuisine_mapping.yaml
│
├── Artifacts/                              # ← Generated at runtime, timestamped per run
│   └── MM_DD_YYYY_HH_MM_SS/
│       ├── data_ingestion/
│       ├── data_validation/
│       ├── data_transformation/
│       └── model_trainer/
│
├── logs/                                   # ← Auto-generated timestamped log files
│   └── MM_DD_YYYY_HH_MM_SS.log
│
├── .env.example                            # Environment variable template
├── requirements.txt
└── setup.py
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RESTAURANT RATING PREDICTION SYSTEM                      │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     push_data.py      ┌──────────────────────────────────────┐
  │  Raw CSVs    │ ──────────────────►   │           MongoDB Atlas              │
  │              │                       │  ┌─────────────────────────────────┐ │
  │ Dataset.csv  │                       │  │  DB: Restaurant_DB              │ │
  │ batch_input  │                       │  │  ├─ restaurant_details (train)  │ │
  │ base_df.csv  │                       │  │  ├─ batch_collection            │ │
  └──────────────┘                       │  │  └─ historical_collection       │ │
                                         │  └─────────────────────────────────┘ │
                                         └──────────────┬───────────────────────┘
                                                        │
                                         ┌──────────────▼──────────────────────┐
                                         │        TRAINING PIPELINE            │
                                         │   (scripts/run_training.py)         │
                                         └──────────────┬──────────────────────┘
                                                        │
              ┌─────────────────────────────────────────▼───────────────────────┐
              │                                                                 │
              │  ①  DATA INGESTION          ②  DATA VALIDATION                  │
              │  ─────────────────          ─────────────────                   │
              │  • Fetch 9,551 rows         • Schema checks (cols, types)       │
              │    from MongoDB             • Missing value threshold ≤10%      │
              │  • Export feature store     • KS drift test per feature         │
              │  • Stratified train/test    • Post-transform array integrity    │
              │    split (80/20)            • ✅ ALL PASSED (run: 04_18_2026)   │
              │    → 7,640 train                                                │
              │    → 1,911 test                                                 │
              │                                                                 │
              │  ③  DATA TRANSFORMATION     ④  MODEL TRAINER                    │
              │  ──────────────────────     ──────────────                      │
              │  • Drop 9 irrelevant cols   • Benchmark 3 ensemble models       │
              │  • Drop "Not Rated" rows    • RandomizedSearchCV per model      │
              │  • Binary encoding          • Select best by test R²            │
              │  • Feature engineering      • Track all runs in MLflow          │
              │    Cuisine_Count            • 🏆 Best: GradientBoost            │
              │    Cuisine_avg_rating         R²=0.9604, MAE=0.199              │
              │  • Target encoding          • Save best_model.pkl               │
              │  • log1p + RobustScaler     • Save preprocessor.pkl             │
              │                                                                 │
              └─────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────────┐
                    │                    │                        │
         ┌──────────▼──────┐   ┌─────────▼─────────┐   ┌──────────▼─────────┐
         │  STREAMLIT APP  │   │ BATCH PREDICTION  │   │  MLFLOW TRACKING   │
         │  (rating_app.py)│   │(batch_prediction  │   │  (localhost:5000)  │
         │                 │   │     .py)          │   │                    │
         │ Single predict  │   │ Fetch from MongoDB│   │  R², MAE, RMSE     │
         │ Batch CSV upload│   │ Transform + infer │   │  per run           │
         │                 │   │ Save output.csv   │   │                    │
         └─────────────────┘   └───────────────────┘   └────────────────────┘
```

---

## 🔬 Pipeline Deep-Dive

### 1. Data Flow & Database Connection

Before training, all datasets must be pushed to MongoDB Atlas using `scripts/push_data.py`. The script reads three CSVs and inserts each into its own collection inside the `Restaurant_DB` database.

```
MongoDB Atlas
└── Restaurant_DB
    ├── restaurant_details        ← main training data  (Dataset.csv)
    ├── batch_collection          ← batch inference inputs  (batch_input.csv)
    └── historical_collection     ← drift detection baseline  (base_df.csv)
```

**Connection mechanism** (`src/components/data_ingestion.py`):
```python
# Environment-driven — no credentials in code
load_dotenv()
mongodb_url = os.getenv("MONGO_DB_URL")
client = MongoClient(mongodb_url, server_api=ServerApi('1'), tlsCAFile=certifi.where())
```

The `.env` file (populated from `.env.example`) supplies all secrets at runtime:

```env
MONGO_DB_URL="mongodb+srv://<user>:<pass>@cluster.mongodb.net/"
DATABSE="Restaurant_DB"
DATA_FILE_PATH="Restaurant_Data/Dataset.csv"
DATA_COLLECTION_NAME="restaurant_details"
BATCH_FILE_PATH="batch_prediction_data/batch_input.csv"
BATCH_COLLECTION_NAME="batch_collection"
HISTORICAL_DATA_FILE_PATH="historical_data/base_df.csv"
HISTORICAL_COLLECTION_NAME="historical_collection"
AUTHOR_NAME="your_name"
AUTHOR_MAIL="your_email"
```

---

### 2. Data Ingestion

**Class:** `src/components/data_ingestion.py → DataIngestion`

Fetches 9,551 rows from MongoDB, exports a full feature store snapshot, then performs a **stratified 80/20 train/test split** on binned rating buckets (`low`, `average`, `good`, `excellent`) to preserve class balance.

| Split | Rows | File |
|---|---|---|
| Train | 7,640 | `ingested/train.csv` |
| Test | 1,911 | `ingested/test.csv` |

---

### 3. Data Validation

**Classes:** `PrimaryDataValidation`, `DriftValidation`, `FinalDataValidation` in `src/components/data_validation.py`

Validation runs in three sequential gates. Training halts if any gate fails.

**Primary validation** checks ingested CSVs against `data_schema/schema.yaml`:

| Check | Required | Result |
|---|---|---|
| Column count | 21 columns | ✅ PASSED |
| Numerical columns | 8 columns | ✅ PASSED |
| Categorical columns | 13 columns | ✅ PASSED |
| Missing value threshold | ≤ 10% per column | ✅ PASSED |

**Drift validation** uses the **Kolmogorov-Smirnov two-sample test** (p > 0.05 threshold) to compare freshly transformed training data against the historical baseline. All 12 features passed with no drift detected. **Training gate: OPEN.**

**Final validation** checks post-transform `.npy` arrays for shape integrity and absence of NaN/Inf values before they reach the model trainer:

| Array | Shape | Status |
|---|---|---|
| Train | (7,632 × 12) | ✅ |
| Test | (1,910 × 12) | ✅ |

---

### 4. Data Transformation

**Class:** `src/components/data_transformation.py → DataTransformation`

Every transformation is fit on train data only and applied to both splits to prevent leakage. The feature space is reduced from 21 raw columns to **11 input features + 1 target**.

| Step | Transformation | Applied To |
|---|---|---|
| 1 | Drop irrelevant columns | `Rating color`, `Rating text`, `Locality`, `Address`, `Longitude`, `Latitude`, `Restaurant Name`, `Switch to order menu`, `Restaurant ID` |
| 2 | Drop "Not Rated" rows | Rows with `Aggregate rating == 0.0` |
| 3 | Binary encoding | `Has Table booking`, `Has Online delivery`, `Is delivering now` → `Yes=1`, `No=0` |
| 4 | Feature engineering | `Cuisine_Count` = number of distinct cuisines per restaurant |
| 5 | Cuisine avg rating | Explode multi-cuisine entries → compute per-cuisine mean rating → merge back as `Cuisine_avg_rating` |
| 6 | Target (mean) encoding | `City`, `Currency`, `Country Code` → replaced with their mean target rating |
| 7 | `log1p` transform | `Average Cost for two`, `Votes` — corrects severe right-skew |
| 8 | RobustScaler | `Average Cost for two`, `Votes` — median+IQR, immune to extreme outliers |

**Final feature set:**
```
Average Cost for two | Votes | Country Code | City | Currency | Has Table booking
Has Online delivery  | Is delivering now | Price range | Cuisine_Count | Cuisine_avg_rating
```

**Serialized artifacts saved to `final_model/`:**

| File | Contents |
|---|---|
| `preprocessor.pkl` | Fitted `sklearn` ColumnTransformer (RobustScaler) |
| `binary_mapping.yaml` | `Yes/No` → `1/0` mapping |
| `City_mapping.yaml` | City name → mean rating (141 entries) |
| `Country Code_mapping.yaml` | Country code → mean rating (15 entries) |
| `Currency_mapping.yaml` | Currency → mean rating |
| `cuisine_mapping.yaml` | Cuisine name → mean rating |

---

### 5. Model Training & Selection

**Class:** `src/components/model_trainer.py → ModelTrainer`

All models are evaluated via `RandomizedSearchCV`, then the best performer is re-trained with its optimal parameters on the full training set. Selection criterion: highest R² on the held-out test set, subject to a train/test R² delta of ≤ 0.05. Every run is tracked via **MLflow**.

**Output artifacts:**
- `Artifacts/<timestamp>/model_trainer/trained_model/best_model.pkl`
- `Artifacts/<timestamp>/model_trainer/model_evaluation/` — full evaluation plots
- `final_model/best_model.pkl` — stable production copy, read by `backend.py` at inference

---

## 📈 Exploratory Data Analysis

Four sequential notebooks in `Notebooks/EDA/` cover the full analytical journey. Notebook 01 produces `processed_data/Dataset_filtered.csv` which notebooks 02–04 consume.

### Notebook 01 — Data Exploration & Preprocessing

**Missing Value Analysis:**

![Missing Value Analysis](Notebooks/reports/missing_value_analysis.png)

The dataset is remarkably clean. Only `Cuisines` contains missing values (9 rows, 0.09%) — dropped since cuisine imputation would introduce arbitrary noise.

**Target Variable Distribution:**

![Rating Histogram](Notebooks/reports/rating_histogram.png)

22.5% of restaurants carry a rating of 0.0 ("Not rated") and were excluded from modelling. Among rated restaurants, the distribution centres around 3.0–3.6 ("Average" band), with very few scoring above 4.5.

**Univariate Analysis — Categorical Features:**

![Univariate Analysis](Notebooks/reports/Univariate_analysis_categorical_features.png)

India (Country Code 1) accounts for ~89% of all restaurants. `Is delivering now` and `Switch to order menu` are near-zero and carry no useful signal.

---

### Notebook 02 — Table Booking, Online Delivery & Price Analysis

| Chart | Key Insight |
|---|---|
| ![Table Booking](Notebooks/reports/Avg_Rating_With_vs_Without_Table_Booking.png) | Restaurants with table booking average **3.59** vs **3.41** — a consistent 0.18-point premium. |
| ![Online Delivery by Price](Notebooks/reports/Online_Delivery_Availability_by_Price_Range.png) | Online delivery peaks at the **Mid tier (41.3%)** and is nearly absent in Luxury (9.0%). |
| ![Price Range Rating](Notebooks/reports/Avg_Rating_for_different_price_range.png) | Clear monotonic relationship: Budget 3.24 → Mid 3.38 → Premium 3.78 → Luxury 3.89. |

---

### Notebook 03 — Customer Preference & Cuisine Analysis

| Chart | Key Insight |
|---|---|
| ![Top 10 Rated](Notebooks/reports/Top_10_Cuisines_by_Average_Rating.png) | **Brazilian cuisine leads at 4.34** average rating. All top-10 cuisines exceed 4.0 — none are high-volume. |
| ![Top 10 by Votes](Notebooks/reports/Top_10_Most_Popular_Cuisines_by_Votes.png) | **North Indian accumulates 595,981 votes**. Volume and quality are largely decoupled. |
| ![Quality vs Scale](Notebooks/reports/cuisine_analysis_charts.png) | Only 11 of 145 cuisine types maintain a 4.0+ average with sufficient representation. |

---

### Notebook 04 — Feature Engineering & Regression Modelling

**Distribution of Numerical Features:**

![Distribution Analysis](Notebooks/reports/Distribution_Analysis_of_numerical_features.png)

`Average Cost for two` and `Votes` are severely right-skewed — confirming the need for `log1p` transformation.

**Inter-Feature Correlation Heatmap:**

![Inter-correlation](Notebooks/reports/Inter-correlation_of_input_features.png)

`Country Code` ↔ `Currency` shows r=0.99 (near-perfectly redundant). `Price range` ↔ `Has Table booking` at r=0.51 reinforces table booking as a quality proxy.

**Feature Correlation with Target:**

![Correlation with Target](Notebooks/reports/correlation_with_target.png)

`Price range` (r=0.44) and the engineered `Cuisine_avg_rating` (r=0.42) are the two strongest predictors. City-level target encoding ranks third at r=0.41.

---

## ⚙️ Feature Engineering

| Transformation | Applied To | Reason |
|---|---|---|
| Row drop | `Cuisines` nulls (9 rows) | Too few to impute reliably |
| Stratified train/test split (80/20) | All rows | Preserves rating bucket distribution |
| Binary encoding (`Yes→1`, `No→0`) | `Has Table booking`, `Has Online delivery`, `Is delivering now` | Nominal binary columns |
| Target (mean) encoding | `City`, `Currency`, `Country Code` | High-cardinality; preserves rating signal without exploding dimensionality |
| Cuisine avg rating feature | `Cuisines` column (exploded) | Custom feature — per-cuisine mean rating; r=0.42 with target |
| `Cuisine_Count` feature | `Cuisines` column | Number of cuisines served — captures menu breadth |
| `log1p` transform | `Average Cost for two`, `Votes` | Corrects severe right-skew and dampens outlier influence |
| RobustScaler | `Average Cost for two`, `Votes` | Median+IQR scaling; immune to extreme outliers |

---

## 📉 Model Evaluation

### Model Comparison

![Model Comparison](static_for_readme/model_comparison.png)

| Model | Train R² | Test R² | Δ R² | Train MAE | Test MAE | Train RMSE | Test RMSE |
|---|---|---|---|---|---|---|---|
| 🏆 **Gradient Boosting** | **0.9623** | **0.9604** | **0.0019** | **0.1933** | **0.1992** | **0.2945** | **0.3010** |
| Random Forest | 0.9936 | 0.9591 | 0.0345 | 0.0765 | 0.1985 | 0.1213 | 0.3058 |
| XGBoost | 0.9833 | 0.9565 | 0.0268 | 0.1259 | 0.2082 | 0.1958 | 0.3153 |

> **Why Gradient Boosting was selected:** It achieved the best test R² (0.9604) with a near-zero train/test gap (Δ=0.0019) — the smallest of all three models and well inside the 0.05 overfitting threshold. Random Forest had a higher train R² (0.9936) but a wider gap (Δ=0.0345), indicating mild overfitting.

---

### Actual vs Predicted

| Gradient Boosting | Random Forest | XGBoost |
|---|---|---|
| ![GB Actual vs Pred](static_for_readme/GradientBoost_Regressor_actual_vs_pred.png) | ![RF Actual vs Pred](static_for_readme/Random_Forest_Regressor_actual_vs_pred.png) | ![XGB Actual vs Pred](static_for_readme/XGBoost_Regressor_actual_vs_pred.png) |

All three models show strong alignment across the rating range. Gradient Boosting shows the tightest clustering with minimal spread — consistent with its lowest test MAE (0.199).

---

### Residual Plots

| Gradient Boosting | Random Forest | XGBoost |
|---|---|---|
| ![GB Residuals](static_for_readme/GradientBoost_Regressor_residuals.png) | ![RF Residuals](static_for_readme/Random_Forest_Regressor_residuals.png) | ![XGB Residuals](static_for_readme/XGBoost_Regressor_residuals.png) |

Residuals are tightly centered around zero with symmetric spread and no visible pattern, confirming no systematic bias across the prediction range.

---

### Feature Importance

| Gradient Boosting | Random Forest | XGBoost |
|---|---|---|
| ![GB Importance](static_for_readme/GradientBoost_Regressor_feature_importance.png) | ![RF Importance](static_for_readme/Random_Forest_Regressor_feature_importance.png) | ![XGB Importance](static_for_readme/XGBoost_Regressor_feature_importance.png) |

Across all three models, **`Cuisine_avg_rating`** and **`Price range`** consistently emerge as the top two contributors — directly validating the EDA finding that cuisine quality tier and price tier are the dominant drivers of restaurant ratings.

---

## 💡 Key Findings & Implications

**`Cuisine_avg_rating` is the top feature across all three models** — built entirely from target encoding, not present in the raw data. This confirms that thoughtful feature engineering delivered more predictive signal than any raw column. For practitioners: when extending this pipeline to a new regional dataset, investing in cuisine-level and city-level encoding is more valuable than collecting additional raw features.

**Price range is the strongest raw predictor** (r = 0.44), with the sharpest quality jump from Mid → Premium (+0.40 points). For new restaurant owners, this means the pricing tier decision at launch has a larger measurable impact on expected rating than cuisine choice alone — a counterintuitive but actionable result.

**Gradient Boosting generalizes best with the tightest train/test gap** (Δ R²=0.0019 vs 0.0345 for Random Forest). For deployment contexts where the model will be retrained periodically on evolving data, this stability under the generalization threshold is more operationally valuable than a marginally higher training-set score.

**All 12 features cleared the KS drift test**, confirming the current training data is statistically consistent with the historical baseline. This also validates that the drift gate is functional — a pipeline health check as much as a data quality check.

**Cuisine quality and popularity are largely decoupled.** Brazilian cuisine leads on average rating (4.34) across ~20 restaurants; North Indian leads on vote volume (596K) across 3,960 restaurants. For aggregator product teams, this means surfacing "top rated" and "most popular" as distinct signals — and not assuming that high vote counts proxy for quality.

**Table booking is a consistent quality signal** (+0.18 points on average). It acts as a proxy for overall establishment quality — not a direct driver of better ratings. Restaurants considering whether to add a reservation system can treat this as a lower-bound estimate of the associated rating uplift.

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.13 |
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Machine learning | `scikit-learn`, `xgboost` |
| Experiment tracking | `mlflow` |
| Database | `pymongo`, `MongoDB Atlas`, `certifi` |
| App framework | `streamlit` |
| Config management | `python-dotenv`, `pyyaml` |
| Statistical testing | `scipy` (KS test) |
| Packaging | `setuptools` |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.13+
- A MongoDB Atlas account (free tier is sufficient)
- Conda or `venv`

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/restaurant-rating.git
cd restaurant-rating
```

### 2. Create & Activate Environment
```bash
# Option A — Conda (recommended)
conda create -p venv python==3.13 -y
conda activate ./venv

# Option B — venv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .   # installs the src package in editable mode
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your MongoDB connection string, collection names, and file paths
```

---

## 🚀 Usage Guide

### Step 1 — Verify Database Connection
```bash
python scripts/test_mongodb_connection.py
# Expected: "Pinged your deployment. You successfully connected to MongoDB!"
```

### Step 2 — Push Data to MongoDB
```bash
python scripts/push_data.py
# Uploads: Dataset.csv, batch_input.csv, base_df.csv to their respective collections
```

### Step 3 — Run the Training Pipeline
```bash
python scripts/run_training.py
# Runs all stages: Ingestion → Validation → Transform → Drift → Final Validation → Train
# Saves final_model/ and a timestamped Artifacts/ directory with all evaluation plots
```

### Step 4 — Launch the Prediction App
```bash
streamlit run app/rating_app.py
```

### Step 5 — Run Batch Predictions (optional)
```bash
python scripts/run_inference.py
# Fetches batch data from MongoDB, runs inference, saves output.csv
```

### Step 6 — EDA Dashboard (optional)
```bash
streamlit run Notebooks/EDA/rating_dashboard.py
```

### Step 7 — MLflow Experiment Tracking (optional)
```bash
mlflow ui
# Open http://localhost:5000 to browse all training runs and compare metrics
```

### Running Notebooks in Order
```bash
jupyter notebook Notebooks/EDA/
```
> Run in order: `01_EDA.ipynb` → `02_EDA.ipynb` → `03_EDA.ipynb` → `04_regression_analysis.ipynb`
>
> Notebooks 02–04 depend on `Notebooks/processed_data/Dataset_filtered.csv` produced by Notebook 01.

---

<div align="center">

  Built with 🐍Python 3.13 · ⚙️📊scikit-learn · 🌳⚡XGBoost · 🍃MongoDB Atlas · 📈🔁MLflow · 🔺Streamlit · 📓🟠Jupyter Notebook

  *If this project helped you, consider giving it a ⭐ on GitHub!*
</div>
