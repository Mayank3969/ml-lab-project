import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("=" * 70)
print("STAGE 3A: Tuned Base Classifiers & Meta-Feature Generation")
print("=" * 70)

# Load data
train_df = pd.read_csv("dataset/processed/merged_train.csv")
test_df = pd.read_csv("dataset/processed/merged_test.csv")

y_train = train_df["casualty_severity"]
groups = train_df["collision_index"]

# Drop non-feature columns
drop_cols = ["casualty_severity", "collision_index"]
# If number_of_casualties wasn't dropped in Stage 2 ablation, it will be here. Same for risk_score.
X_train = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=drop_cols)

from sklearn.preprocessing import StandardScaler

# We use 5-fold CV to generate meta-features (OOF predictions) for the stacking Stage 4.
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n[1] Profiling {X_train.shape[1]} features for model constraints...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/saved_models/scaler.pkl")

print("\n[2] GridSearch Tuning Base Models (Macro-F1)")
tuning_log = []

# 2.1 Logistic Regression (on scaled features)
print("  => Tuning Logistic Regression...")
lr_grid = {'C': [0.1, 1.0, 10.0], 'class_weight': [None, 'balanced']}
lr_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_grid, scoring='f1_macro', cv=sgkf, n_jobs=-1)
lr_search.fit(X_train_scaled, y_train, groups=groups)
print(f"     Best LR params: {lr_search.best_params_} -> F1: {lr_search.best_score_:.4f}")
tuning_log.append(f"**Logistic Regression**: {lr_search.best_params_} (F1: {lr_search.best_score_:.4f})")
best_lr = lr_search.best_estimator_

# 2.2 Decision Tree (on raw features)
print("  => Tuning Decision Tree...")
dt_grid = {'max_depth': [10, 15, 20, None], 'min_samples_leaf': [1, 5, 10], 'class_weight': [None, 'balanced']}
dt_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_grid, scoring='f1_macro', cv=sgkf, n_jobs=-1)
dt_search.fit(X_train, y_train, groups=groups)
print(f"     Best DT params: {dt_search.best_params_} -> F1: {dt_search.best_score_:.4f}")
tuning_log.append(f"**Decision Tree**: {dt_search.best_params_} (F1: {dt_search.best_score_:.4f})")
best_dt = dt_search.best_estimator_

# 2.3 K-Nearest Neighbors (on scaled features via kd_tree)
print("  => Tuning K-Nearest Neighbors (KD-Tree on frequency-encoded features)...")
knn_grid = {'n_neighbors': [5, 15, 25], 'weights': ['uniform', 'distance']}
knn_search = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree'), knn_grid, scoring='f1_macro', cv=sgkf, n_jobs=-1)
knn_search.fit(X_train_scaled, y_train, groups=groups)
print(f"     Best KNN params: {knn_search.best_params_} -> F1: {knn_search.best_score_:.4f}")
tuning_log.append(f"**K-Nearest Neighbors**: {knn_search.best_params_} (F1: {knn_search.best_score_:.4f})")
best_knn = knn_search.best_estimator_

# Log tuning results
with open("docs/model_tuning_log.md", "w") as f:
    f.write("# Stage 3A Base Model Tuning\n\n")
    f.write("\n".join(tuning_log))

print("\n[3] Generating Out-Of-Fold Meta Features for Stage 4...")
# Out-of-fold probability predictions
lr_oof = cross_val_predict(best_lr, X_train_scaled, y_train, groups=groups, cv=sgkf, method='predict_proba', n_jobs=-1)
dt_oof = cross_val_predict(best_dt, X_train, y_train, groups=groups, cv=sgkf, method='predict_proba', n_jobs=-1)
knn_oof = cross_val_predict(best_knn, X_train_scaled, y_train, groups=groups, cv=sgkf, method='predict_proba', n_jobs=-1)

# Build meta_train.csv (9 probabilities + target)
meta_train = pd.DataFrame()
for i, c in enumerate([1, 2, 3]):
    meta_train[f"lr_prob_{c}"] = lr_oof[:, i]
    meta_train[f"dt_prob_{c}"] = dt_oof[:, i]
    meta_train[f"knn_prob_{c}"] = knn_oof[:, i]

if "risk_score" in train_df.columns:
    meta_train["risk_score"] = train_df["risk_score"].values
if "number_of_casualties" in train_df.columns:
    meta_train["number_of_casualties"] = train_df["number_of_casualties"].values

meta_train["casualty_severity"] = y_train.values

print("\n[4] Training Final Base Models on Full Train Set & Generating Test Meta Features...")
best_lr.fit(X_train_scaled, y_train)
best_dt.fit(X_train, y_train)
best_knn.fit(X_train_scaled, y_train)

lr_test_probs = best_lr.predict_proba(X_test_scaled)
dt_test_probs = best_dt.predict_proba(X_test)
knn_test_probs = best_knn.predict_proba(X_test_scaled)

meta_test = pd.DataFrame()
for i, c in enumerate([1, 2, 3]):
    meta_test[f"lr_prob_{c}"] = lr_test_probs[:, i]
    meta_test[f"dt_prob_{c}"] = dt_test_probs[:, i]
    meta_test[f"knn_prob_{c}"] = knn_test_probs[:, i]

if "risk_score" in test_df.columns:
    meta_test["risk_score"] = test_df["risk_score"].values
if "number_of_casualties" in test_df.columns:
    meta_test["number_of_casualties"] = test_df["number_of_casualties"].values

meta_test["casualty_severity"] = test_df["casualty_severity"].values

# Save everything
meta_train.to_csv("dataset/processed/meta_train.csv", index=False)
meta_test.to_csv("dataset/processed/meta_test.csv", index=False)
joblib.dump(best_lr, "models/saved_models/stage2_logreg_final.pkl")
joblib.dump(best_dt, "models/saved_models/stage2_dt_final.pkl")
joblib.dump(best_knn, "models/saved_models/stage2_knn_final.pkl")

print("[OK] Meta datasets and finalized base models (LR, DT, KNN) saved.")

