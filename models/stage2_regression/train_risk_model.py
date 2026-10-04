import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, f1_score

os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("models/evaluation/results", exist_ok=True)

print("=" * 70)
print("STAGE 2: Train Risk Score Regressor & Ablation Test")
print("=" * 70)

# Load data
train_df = pd.read_csv("dataset/processed/merged_train.csv")
test_df = pd.read_csv("dataset/processed/merged_test.csv")

y_class = train_df["casualty_severity"]
y_reg = train_df["number_of_casualties"]
groups = train_df["collision_index"]

# Drop target and collision_index
drop_cols = ["casualty_severity", "number_of_casualties", "collision_index"]
X_train = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=drop_cols)

print(f"[1] Tuning Regressors on {X_train.shape[1]} features...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Ridge
ridge_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_search = GridSearchCV(Ridge(random_state=42), ridge_grid, scoring='r2', cv=sgkf)
ridge_search.fit(X_train, y_reg, groups=groups)
print(f"  Ridge Best Alpha: {ridge_search.best_params_['alpha']} -> R2: {ridge_search.best_score_:.4f}")

# Decision Tree Regressor
dt_grid = {'max_depth': [3, 5, 7, 10]}
dt_search = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_grid, scoring='r2', cv=sgkf)
dt_search.fit(X_train, y_reg, groups=groups)
print(f"  Decision Tree Best Depth: {dt_search.best_params_['max_depth']} -> R2: {dt_search.best_score_:.4f}")

if ridge_search.best_score_ > dt_search.best_score_:
    print("\n[2] Winner: Ridge Regression. Generating OOF predictions...")
    best_regressor = ridge_search.best_estimator_
    best_r2 = ridge_search.best_score_
else:
    print("\n[2] Winner: Decision Tree Regressor. Generating OOF predictions...")
    best_regressor = dt_search.best_estimator_
    best_r2 = dt_search.best_score_

# Generate OOF predictions for risk_score
oof_risk_score = cross_val_predict(best_regressor, X_train, y_reg, groups=groups, cv=sgkf)

# Now generate test predictions
best_regressor.fit(X_train, y_reg)
test_risk_score = best_regressor.predict(X_test)

print("\n[3] Risk Score Ablation Test (Classification)")
# Does adding risk_score improve a classification DecisionTree over just number_of_casualties?

clf = DecisionTreeClassifier(random_state=42)

# Base: features + number_of_casualties
X_base = X_train.copy()
X_base["number_of_casualties"] = y_reg
f1_base = np.mean(cross_val_score(clf, X_base, y_class, groups=groups, cv=sgkf, scoring='f1_macro'))
print(f"  Macro-F1 (with number_of_casualties): {f1_base:.4f}")

# Alt: features + risk_score + number_of_casualties
X_alt = X_base.copy()
X_alt["risk_score"] = oof_risk_score
f1_alt = np.mean(cross_val_score(clf, X_alt, y_class, groups=groups, cv=sgkf, scoring='f1_macro'))
print(f"  Macro-F1 (with risk_score + number_of_casualties): {f1_alt:.4f}")

# Alt2: features + risk_score (dropping number_of_casualties)
X_alt2 = X_train.copy()
X_alt2["risk_score"] = oof_risk_score
f1_alt2 = np.mean(cross_val_score(clf, X_alt2, y_class, groups=groups, cv=sgkf, scoring='f1_macro'))
print(f"  Macro-F1 (with risk_score ONLY): {f1_alt2:.4f}")

if f1_alt > f1_base and f1_alt > f1_alt2:
    print("\n  => CONCLUSION: risk_score + number_of_casualties yields best performance. Keeping BOTH.")
    train_df["risk_score"] = oof_risk_score
    test_df["risk_score"] = test_risk_score
elif f1_alt2 > f1_base:
    print("\n  => CONCLUSION: risk_score ONLY yields best performance. Replacing number_of_casualties.")
    train_df["risk_score"] = oof_risk_score
    test_df["risk_score"] = test_risk_score
    train_df.drop(columns=["number_of_casualties"], inplace=True)
    test_df.drop(columns=["number_of_casualties"], inplace=True)
else:
    print("\n  => CONCLUSION: risk_score does NOT improve classification. DROPPING risk_score entirely.")

# Save modified datasets
train_df.to_csv("dataset/processed/merged_train.csv", index=False)
test_df.to_csv("dataset/processed/merged_test.csv", index=False)
print("[OK] Saved merged datasets with finalized features.")

import joblib
joblib.dump(best_regressor, "models/saved_models/stage1_best_regressor.pkl")
print("[OK] Saved best regressor model.")
