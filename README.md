# Interpretable ML for Robotic Choreography Evaluation

This repository contains all code, data and results for our three‐part study on predicting and explaining audience evaluations of humanoid‐robot choreographies:

1. **Classification** – Bin-arize audience scores (1–3 → 0, 4–5 → 1) and train four model families (Logistic Regression, Random Forest, XGBoost, CatBoost) to predict “high vs. low” ratings across seven targets.  
2. **Regression** – Predict the original 1–5 Likert scores with linear (Ridge) and tree-based regressors.  
3. **SHAP** – Interpret the final models using a four-phase pipeline of SHAP value computation, dependence and interaction plots, stability analysis, subgroup analyses and decision plots.

---

## 📂 Classification

**Folder:** `classification/`

### Sub-folders

- **`catboost_info/`**  
  Contains CatBoost training logs, feature importances and class-weight settings.
- **`classification_models_data/`**  
  Joblib-serialized pipelines, hyperparameter grids and saved model objects for each target.
- **`saved_overall_best_models/`**  
  The final chosen models (XGBoost & CatBoost) serialized via joblib for deployment.
- **`saved_tuned_models/`**  
  All grid-search results and best estimators (joblib format) before final selection.
---

## 📂 Regression

> *(Folder structure present, content to be filled.)*

---

## 📂 SHAP

> *(Folder structure present, content to be filled.)*

---

## 📂 Data

> **Note:** You will find both pre-processed (cleaned) and post-processed (split, encoded) versions organized under `data/` so you can reproduce each ML pipeline exactly.

