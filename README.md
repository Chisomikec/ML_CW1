# Machine Learning Coursework 1 (ML_CW1)

This repository contains the implementation of the **Machine Learning CW1** regression task. The objective is to develop the best possible model for predicting `outcome` using the provided dataset

## Exploratory Data Analysis
- **Initial dataset exploration:** Checked for missing values, data types, and duplicate rows.
- **Feature transformations:**
  - Encoded categorical variables (`cut`, `color`, `clarity`) using One-Hot and Ordinal Encoding.
  - Created new features (`xyz = x*y*z`, `price_per_carat`) based on correlation analysis.
- **Outlier analysis:** Removal reduced R², so high-end diamonds were retained to preserve information.
- **Dimensionality reduction:** PCA and feature selection were tested but negatively impacted R2, so they were discarded.

 _More details are available in [eda.ipynb](eda.ipynb)_

 ---

## Model Selection

- Compared **linear and tree-based models** using two datasets:
  - **Cleaned dataset (cleaned_train.csv)**
  - **Lasso-selected dataset (final_lasso_feature.csv)**
- **Results:**
  - Linear models underperformed due to high feature dimensionality.
  - Random Forest worked best with Lasso-selected features(final_lasso_feature.csv).
  - XGBoost performed best on the cleaned_train.csv and was selected.

_More details on model selection are in [compare_models.py](compare_models.py)_

---

## Model Training & Hyperparameter Tuning

XGBoost was fine-tuned to prevent overfitting:
- **Learning rate:** `0.023`
- **n_estimators:** `550`
- **Max depth:** `3`
- **Min child weight:** `8`
- **Subsample:** `0.7`
- **colsample_bytree:** `0.7`
- **Reg_Alpha (L1):** `0.5`
- **Reg_Lambda (L2):** `0.5`

_More details on training are in [models.py](models.py)_

---

## Final Model Evaluation
**Final Model: XGBoost trained on Cleaned Dataset (cleaned_train.csv)** with `R2 = 0.4640`

---

## Prediction for CW1_test.csv
- The predictions are saved in CW1_submission_k23086553.csv .
- These predictions were generated using the final_xgboost_model2.pkl model.
