# Car Price Prediction Project with Hyperparameter Tuning and Cross-Validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load data
df = pd.read_csv("Dataset/car details v4.csv")

# --------------------------
# Feature Engineering
# --------------------------

# Drop rows with missing price (if any)
df.dropna(subset=['Price'], inplace=True)

# Car Age
df['Car_Age'] = 2025 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

# Extract numeric Engine (cc)
df['Engine_CC'] = df['Engine'].str.extract(r'(\d+)')[0].astype(float)

# Extract Max Power (bhp)
df['Max_Power_BHP'] = df['Max Power'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)

# Extract Torque (Nm)
df['Torque_Nm'] = df['Max Torque'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)

# Extract Torque RPM
rpm_extract = df['Max Torque'].str.extract(r'@(\s*\d+)')
df['Torque_RPM'] = rpm_extract[0].str.replace(" ", "").astype(float)

# Drop original columns with messy data
df.drop(['Engine', 'Max Power', 'Max Torque'], axis=1, inplace=True)

# --------------------------
# Handle Nulls
# --------------------------
numeric_cols = ['Engine_CC', 'Max_Power_BHP', 'Torque_Nm', 'Torque_RPM', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']
categorical_cols = ['Make', 'Model', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Drivetrain']

y = df['Price']
df.drop(['Price'], axis=1, inplace=True)

# --------------------------
# Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# --------------------------
# Preprocessing Pipeline
# --------------------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# --------------------------
# Hyperparameter Tuning for XGBoost
# --------------------------
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42 ))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [4, 6, 8],
    'regressor__learning_rate': [0.05, 0.1, 0.2]
}

grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nTuned XGBoost Performance")
print("----------------------------")
print(f"Best Params: {grid_search.best_params_}")
print(f"R2 Score: {r2_score(y_test, y_pred_best):.2f}")
print(f"RMES: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}")

# Cross-Validation Score for Tuned Model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print("\nCross-validation R2 Scores:", cv_scores)
print("Average R2 Score:", np.mean(cv_scores))

# --------------------------
# Feature Importance for XGBoost
# --------------------------
regressor = best_model.named_steps['regressor']
importances = regressor.feature_importances_
ohe_features = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols)
all_features = numeric_cols + list(ohe_features)

feat_imp = pd.Series(importances, index=all_features)
feat_imp.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10, 8))
plt.title('Top 20 Feature Importances (XGBoost)')
plt.show()

# --------------------------
# Other Models
# --------------------------
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Bagging": BaggingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

for name, reg in models.items():
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Performance")
    print("------------------")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")