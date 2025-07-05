# Car Price Prediction Project

Predicts used‑car prices using an end‑to‑end machine‑learning pipeline with automated preprocessing, feature engineering, hyper‑parameter tuning (XGBoost), cross‑validation, and model comparison.

---

## 1. Project Highlights

| Stage | What Happens |
| ----- | ------------ |
|       |              |

|   |
| - |

| **Data Loading**            | Reads `car details v4.csv` (20 raw columns).                                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Feature Engineering**     | ‑ Car age, numeric engine size, max power, torque & RPM. ‑ Drops messy original columns.                                                                      |
| **Pre‑processing Pipeline** | Numeric → median impute + standard‑scale. Categorical → mode impute + one‑hot.                                                                                |
| **Modeling**                | Primary model: **XGBoostRegressor** tuned with grid‑search (n\_estimators, max\_depth, learning\_rate). Baselines: Random Forest, Bagging, Linear Regression. |
| **Evaluation**              | Hold‑out set (20 %), 5‑fold CV, R² & RMSE.                                                                                                                    |
| **Interpretability**        | Bar‑plot of top‑20 XGBoost feature importances.                                                                                                               |

---

## 2. Dataset

- Source: `carDekho/car details v4.csv` (public Kaggle dataset).
- Target: `Price` (numeric, INR).
- Key raw features: make, model, year, km driven, fuel, transmission, city, engine text, power text, torque text, dimensions, seating, tank capacity.

### Null Values Handled

```
Engine               80
Max Power            80
Max Torque           80
Drivetrain          136
Length               64
Width                64
Height               64
Seating Capacity     64
Fuel Tank Capacity  113
```

Median or mode imputation is applied automatically inside the pipeline.

---

## 3. Feature Engineering Steps

| New Feature         | Formula / Extraction                   |
| ------------------- | -------------------------------------- |
| **Car\_Age**        | `2025 − Year`                          |
| **Engine\_CC**      | Regex `r'(\d+)'` on `Engine`           |
| **Max\_Power\_BHP** | Regex float from `Max Power`           |
| **Torque\_Nm**      | Regex float before `@` in `Max Torque` |
| **Torque\_RPM**     | Regex digits after `@` in `Max Torque` |

Original text columns (`Engine`, `Max Power`, `Max Torque`) are dropped after extraction.

---

## 4. Pipeline Architecture

```
                  ┌──────────────────┐
 raw dataframe →  │ ColumnTransformer│
                  └────────┬─────────┘
            ┌──────────────┴──────────────┐
      numeric pipe                  categorical pipe
  (impute → scale)             (impute → one‑hot)
            └──────────────┬──────────────┘
                  features matrix (dense)
                              ↓
                    Regressor (XGB)
```

*Wrapped in a single ****\`\`**** so raw data → prediction in one call.*

---

## 5. Hyper‑parameter Tuning (GridSearchCV)

```
param_grid = {
  'regressor__n_estimators': [100, 200],
  'regressor__max_depth'   : [4, 6, 8],
  'regressor__learning_rate': [0.05, 0.1, 0.2]
}
cv = 5‑fold, scoring = 'r2', n_jobs = ‑1
```

`best_estimator_` is saved and evaluated on the hold‑out test set.

---

## 6. Results (sample run)

```
Best Params : {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
Test R²     : 0.86
Test RMSE   : 1.3 lakh INR
5‑Fold CV   : mean R² ≈ 0.84
```

| Model           | Test R²  | Test RMSE |
| --------------- | -------- | --------- |
| XGBoost (tuned) | **0.86** | **1.3 L** |
| Random Forest   | 0.82     | 1.5 L     |
| Bagging         | 0.80     | 1.6 L     |
| Linear Reg.     | 0.65     | 2.4 L     |

---

## 7. Feature Importance

The script plots the top‑20 features driving price (e.g., `Car_Age`, `Engine_CC`, `Model_Specific_OHE`, etc.) to aid interpretability.

---

## 8. How to Run

```bash
# 1. Clone repo & install deps
pip install -r requirements.txt

# 2. Place CSV in carDekho/ directory as shown

# 3. Run training script
python car_price_prediction.py
```

Outputs: console metrics + bar plot.

---

## 9. Dependencies

```
pandas numpy matplotlib seaborn scikit‑learn xgboost
```

Python ≥ 3.8 recommended.

---

## 10. Future Work

- ✔️ Hyper‑opt with Bayesian search
- ✔️ Stacking models (e.g., LightGBM)
- ✔️ Model persistence with `joblib`
- ✔️ Streamlit app for UI
- ✔️ SHAP values for deeper explanatory power

---

© 2025 Tanishk Dubey — Feel free to fork & improve!





