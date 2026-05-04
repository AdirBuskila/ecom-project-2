# Supervised Learning Home Project

A two-part machine learning project covering both **regression** and **classification** problems using the full supervised learning workflow: data preparation, exploratory analysis, model training with hyperparameter tuning, evaluation, and deployment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Part 1 — Regression: Predicting GDP from COVID-19 Data](#part-1--regression-predicting-gdp-from-covid-19-data)
- [Part 2 — Classification: Predicting Customer Churn](#part-2--classification-predicting-customer-churn)
- [Shared Methodology](#shared-methodology)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Results Summary](#results-summary)

---

## Project Overview

This project tackles two real-world supervised learning problems end-to-end:

1. **Regression** — Predict a country's GDP based on COVID-19 statistics and economic indicators.
2. **Classification** — Predict whether a customer will churn (cancel their subscription) based on demographic and behavioral features.

Both notebooks follow an identical, rigorous workflow so the engineering patterns — splitting, scaling, tuning, evaluating, exporting — transfer cleanly between problem types.

---

## Project Structure

```
ecom-project-2/
├── data/                                       # Datasets
│   ├── Covid19_With_GDP_Values.csv             # Regression dataset
│   └── customer_churn_dataset.csv              # Classification dataset
├── models/                                     # Exported model artifacts
│   ├── final_gdp_model.joblib                  # Regression model (Ridge)
│   ├── scaler_gdp.joblib                       # GDP feature scaler
│   ├── polynomial_converter_gdp.joblib         # Polynomial transformer (only created when Polynomial wins selection)
│   ├── final_churn_model.joblib                # Classification model (Random Forest)
│   └── final_churn_scaler.joblib               # Churn feature scaler
├── docs/
│   └── Supervised Learning - Home Project.pdf  # Original assignment brief
├── covid_gdp.ipynb                             # Part 1 — Regression notebook
├── classification.ipynb                        # Part 2 — Classification notebook
└── README.md
```

Each notebook defines `DATA_DIR = Path("data")` and `MODELS_DIR = Path("models")` immediately after its imports, so every read/write below those constants flows through the folders above. Move the data or model folders only by updating those two constants.

---

## Part 1 — Regression: Predicting GDP from COVID-19 Data

**Notebook:** [covid_gdp.ipynb](covid_gdp.ipynb)

### Goal
Predict a country's GDP based on COVID-19 case counts, deaths, recoveries, unemployment rate, and CPI.

### Dataset
`Covid19_With_GDP_Values.csv` — country-level observations from 2021 and 2022, including COVID statistics and economic indicators.

### Pipeline
1. **Data Preparation**
   - Dropped `Unnamed: 0` (CSV export artifact)
   - Removed rows with missing values (`dropna`) and duplicates
   - Aggregated the two yearly records per country using `groupby().mean()` so each country contributes a single row
   - **Log-transformed** `GDP`, `Confirmed`, `Deaths`, and `Recovered` with `np.log1p` to compress their multi-order-of-magnitude range so a linear model can fit the relationship
2. **Data Exploration**
   - Pairplot + individual feature scatter plots
   - Correlation heatmap
   - Key finding: strong multicollinearity between Confirmed, Deaths, and Recovered
3. **Models Trained**
   - Linear Regression (baseline, with 5-fold CV)
   - Ridge Regression (RidgeCV, alpha ∈ [1, 999])
   - Lasso Regression (LassoCV)
   - Polynomial Regression (degrees 1–6, chosen via train/test RMSE curves)
4. **Evaluation metrics:** MAE, MSE, RMSE (in log-GDP units, since the target is log-transformed)
5. **Deployment:** Final model + scaler exported to joblib

### Final Model
**Ridge Regression** (alpha = 11) — the lowest MAE/MSE/RMSE across all four candidates. The small L2 penalty stabilizes the coefficients of the multicollinear COVID features (`Confirmed`, `Deaths`, `Recovered`), which plain Linear Regression cannot do. Polynomial regression overfit at higher degrees due to the small sample size (~118 training rows vs. up to 126 polynomial features at degree 4).

---

## Part 2 — Classification: Predicting Customer Churn

**Notebook:** [classification.ipynb](classification.ipynb)

### Goal
Predict whether a customer will churn (label = 1) based on demographics, tenure, usage frequency, support calls, payment delays, subscription type, and contract length.

### Dataset
`customer_churn_dataset.csv` — 64,374 customer records with 11 features and a binary `Churn` label.

### Pipeline
1. **Data Preparation**
   - Dropped `CustomerID` (unique identifier with no predictive value)
   - Verified no missing values and no duplicates
   - One-hot encoded categorical features (`Gender`, `Subscription Type`, `Contract Length`) with `drop_first=True`
2. **Data Exploration**
   - Correlation heatmap identified Support Calls, Payment Delay (positive) and Usage Frequency (negative) as the strongest churn indicators
   - Focused pairplot on the three top correlates, colored by Churn, confirmed clear class separation
3. **Models Trained**
   - Logistic Regression (`LogisticRegressionCV`, L1 penalty, saga solver, 10 Cs)
   - KNN (elbow method for K ∈ [1, 11])
   - SVM (`GridSearchCV` over C and kernel — tuned on a 5k stratified sample due to O(n²) scaling)
   - Random Forest (`GridSearchCV` over `n_estimators`, `max_features`, `max_depth` — tuned on a 10k sample)
4. **Evaluation metrics:** Accuracy, Recall, F1-Score + confusion matrix heatmap
5. **Deployment:** Final model + StandardScaler exported to joblib

### Final Model
**Random Forest** — best across all three metrics (Accuracy ≈ 0.999, Recall ≈ 0.998, F1 ≈ 0.999). Outperformed the other three models because churn in this dataset is driven by non-linear interactions between features that tree-based models capture naturally.

### Reusable Evaluation Helper
The classification notebook defines an `evaluate_model()` helper that, for any fitted classifier:
- Prints predictions and per-class probabilities (when available)
- Computes Accuracy, Recall, and F1-Score
- Plots the confusion matrix as a heatmap
- Stores results in a shared dict for the final comparison table

This keeps every model's evaluation section to a single line.

---

## Shared Methodology

Both notebooks follow the same disciplined workflow:

| Step | Regression (covid_gdp) | Classification (classification) |
|------|------------------------|----------------------------------|
| Data Cleaning | Drop junk cols, `dropna`, deduplicate | Drop ID col, null/dup checks |
| Feature Engineering | Aggregate per country, log-transform skewed columns | One-hot encode categoricals |
| Train/Test Split | `test_size=0.3`, `random_state=42` | `test_size=0.3`, `random_state=42` |
| Scaling | `StandardScaler` fit on train only | `StandardScaler` fit on train only |
| Tuning | CV / RidgeCV / LassoCV | CV / GridSearchCV / elbow method |
| Evaluation | MAE, MSE, RMSE | Accuracy, Recall, F1 + confusion matrix |
| Deployment | joblib export + verified reload | joblib export + verified reload |

**Key engineering principles applied throughout:**
- **No data leakage** — scalers and polynomial converters are always fit on training data only, never on the full dataset before splitting.
- **Sampling for expensive tuning** — SVM and Random Forest hyperparameter searches use stratified subsamples to stay tractable on a 64k-row dataset, then refit the best parameters on the full training set.
- **Reproducibility** — `random_state=42` everywhere, optimal parameters printed explicitly, final models exported with all preprocessing artifacts.
- **Round-trip verification** — every exported artifact is reloaded and tested with a real prediction to confirm the deployment bundle is complete.

---

## How to Run

1. Clone the repository and navigate to the project folder.
2. Ensure the `data/` folder is in the project root with both CSVs present.
3. Open either notebook in Jupyter or VS Code (run from the repo root so the `DATA_DIR` / `MODELS_DIR` constants resolve correctly).
4. Run cells top-to-bottom. Both notebooks are self-contained and will regenerate the joblib files in `models/` when the deployment cells run (the `models/` folder is created automatically on first run if missing).

**Estimated run times:**
- `covid_gdp.ipynb` — under 1 minute
- `classification.ipynb` — 5–10 minutes (SVM and Random Forest tuning on stratified samples + final fit on full 64k rows)

---

## Dependencies

```
pandas
numpy
seaborn
matplotlib
scikit-learn
joblib
```

Any recent Anaconda / scikit-learn ≥ 1.3 environment will work. Development was done on scikit-learn 1.8.

---

## Results Summary

### Regression (GDP Prediction)

Metrics are reported on the log-GDP scale (target was log-transformed with `np.log1p` during preparation).

| Model | MAE | MSE | RMSE |
|---|---|---|---|
| Linear Regression | 1.083 | 1.953 | 1.398 |
| **Ridge** (alpha = 11, selected) | **1.045** | **1.832** | **1.354** |
| Lasso | 1.062 | 1.869 | 1.367 |

### Classification (Customer Churn)

| Model | Accuracy | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.829 | 0.826 | 0.821 |
| KNN (K=8) | 0.915 | 0.915 | 0.911 |
| SVM (C=10, rbf) | 0.952 | 0.961 | 0.950 |
| **Random Forest** (selected) | **0.999** | **0.998** | **0.999** |

---

## Author

Adir Buskila — Supervised Learning Home Project.
