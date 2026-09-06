# Customer Churn Prediction (Classification)

Predicting which telecom customers are likely to churn, using account, service, and billing data — so retention efforts can be targeted at the customers most at risk.

## Problem

Customer churn is expensive: acquiring a new customer typically costs far more than retaining an existing one. This project builds a classifier that flags at-risk customers *before* they leave, using only information available at the time of prediction (no future/leaked data).

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 customers, 21 columns (demographics, account info, subscribed services), binary target `Churn` (Yes/No).

The raw CSV is **not committed to this repo**. The notebook downloads it automatically at runtime (with a fallback mirror if the primary source is unreachable) and caches it locally under `data/`, which is git-ignored. This keeps the repo lightweight and ensures anyone who clones it gets a working pipeline with zero manual data-download steps.

## Approach

1. **Data cleaning** — coerce `TotalCharges` to numeric, resolve blanks (new customers with 0 tenure), drop the non-predictive `customerID`.
2. **EDA** — churn rate, distribution of tenure/charges by churn status, churn rate by contract/internet/payment type.
3. **Preprocessing** — `ColumnTransformer` (standard scaling for numeric features, one-hot encoding for categorical features) wrapped in an sklearn `Pipeline`, so there's no train/test leakage.
4. **Modeling** — two models trained and compared:
   - **Logistic Regression** (`class_weight="balanced"`) — interpretable baseline
   - **XGBoost** — stronger non-linear model
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves. Accuracy alone is not used to judge the models since the classes are imbalanced (~27% churn).
6. **Feature importance** — top drivers of churn according to the XGBoost model.

## Results

See the notebook's Section 7 (`results_df`) for the full metrics table generated on each run — this file intentionally doesn't hardcode numbers here so the README never drifts out of sync with the code.

## Reproducibility

- Fixed random seed (`RANDOM_SEED = 42`) used for the train/test split and both models.
- Dependency versions pinned in `requirements.txt`.
- No absolute file paths — all paths are relative to the repo root.
- Data loading is a documented function (`load_churn_data`) with an explicit fallback source, rather than a manual download step.

Re-running the notebook top to bottom reproduces the same split, models, and metrics.

## How to run

**Option A — Google Colab (recommended, no local setup):**
1. Upload `Customer_Churn_Prediction.ipynb` to [Google Colab](https://colab.research.google.com/).
2. In the first code cell, install dependencies: `!pip install -r requirements.txt` (or just run the notebook — Colab has most of these pre-installed; only `xgboost` may need `!pip install xgboost`).
3. Run all cells (`Runtime` → `Run all`). Data downloads automatically.

**Option B — Local:**
\```bash
git clone https://github.com/<your-username>/customer-churn-classification.git
cd customer-churn-classification
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
\```

## Repository structure

\```
customer-churn-classification/
├── Customer_Churn_Prediction.ipynb   # Full analysis, pipeline, models, evaluation
├── requirements.txt                  # Pinned dependencies
└── README.md
\```

## Possible extensions

- Hyperparameter tuning (`GridSearchCV`, `Optuna`)
- SHAP values for per-customer explanations
- Probability calibration for cost-sensitive decision thresholds
