"""
Stage 3A: Phase B — Final Base Models Trainer & Meta-Test Generator
- Trains 3 final base models on the entire training set (merged_train.csv).
- Generates direct probability predictions on the held-out test set (merged_test.csv).
- Serializes final models to models/saved_models/ for downstream stages and Streamlit app.
- Writes dataset/processed/meta_test.csv containing 9 probability columns + risk_score + label.
"""
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Path setup
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "models"))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report

from utils.config import (
    PROCESSED,
    MODELS_DIR,
    SEED,
    TARGET,
    CLASS_LABELS,
    CLASS_NAMES,
    STAGE3_PROBA_COLS,
    STAGE4_FEATURES
)

print("=" * 70)
print("STAGE 3A — Phase B: Train Final Base Models & Generate meta_test.csv")
print("=" * 70)

# [1] Load merged train and test datasets
print("\n[1] Loading train and test datasets...")
train_df = pd.read_csv(PROCESSED / "merged_train.csv", converters={"collision_index": str})
test_df  = pd.read_csv(PROCESSED / "merged_test.csv",  converters={"collision_index": str})

print(f"  Train samples: {len(train_df):,} rows")
print(f"  Test samples : {len(test_df):,} rows")

# Features: everything except collision_index and target
feature_cols = [c for c in train_df.columns if c not in ["collision_index", TARGET]]
print(f"  Using {len(feature_cols)} input features: {feature_cols}")

X_train, y_train = train_df[feature_cols], train_df[TARGET].astype(int)
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET].astype(int)

# [2] Train Final Base Models on full train set
print("\n[2] Training 3 Final Base Models on full training set...")

# 1. Final Logistic Regression
print("  Fitting final Logistic Regression...", end="", flush=True)
t0 = time.time()
final_logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
])
final_logreg.fit(X_train, y_train)
print(f" done in {time.time() - t0:.1f}s.")

# 2. Final K-Nearest Neighbors
print("  Fitting final KNN (k=15, kd_tree)...", end="", flush=True)
t0 = time.time()
final_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=-1))
])
final_knn.fit(X_train, y_train)
print(f" done in {time.time() - t0:.1f}s.")

# 3. Final Decision Tree
print("  Fitting final Decision Tree...", end="", flush=True)
t0 = time.time()
final_dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=SEED
)
final_dt.fit(X_train, y_train)
print(f" done in {time.time() - t0:.1f}s.")

# [3] Save final trained models to saved_models/
print("\n[3] Serializing models to models/saved_models/...")
joblib.dump(final_logreg, MODELS_DIR / "stage2_logreg_final.pkl")
joblib.dump(final_knn,    MODELS_DIR / "stage2_knn_final.pkl")
joblib.dump(final_dt,     MODELS_DIR / "stage2_dt_final.pkl")
print("  [OK] Saved stage2_logreg_final.pkl")
print("  [OK] Saved stage2_knn_final.pkl")
print("  [OK] Saved stage2_dt_final.pkl")

# [4] Generate Probability Predictions on Test Set
print("\n[4] Generating probability predictions on held-out test set...")
p_test_logreg = final_logreg.predict_proba(X_test)
p_test_knn    = final_knn.predict_proba(X_test)
p_test_dt     = final_dt.predict_proba(X_test)

# [5] Standalone Test Set Metrics
print("\n[5] Standalone Base Model Performance on Test Set:")
for name, probs in [("Logistic Regression", p_test_logreg), ("KNN", p_test_knn), ("Decision Tree", p_test_dt)]:
    preds = np.argmax(probs, axis=1) + 1
    macro_f1 = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    fatal_f1 = f1_score(y_test == 1, preds == 1, zero_division=0)
    print(f"  {name:<22s} | Macro-F1: {macro_f1:.4f} | Accuracy: {acc*100:5.2f}% | Fatal-F1: {fatal_f1:.4f}")

# [6] Assemble meta_test dataframe
print("\n[6] Assembling meta_test.csv...")
meta_test = pd.DataFrame({
    "collision_index": test_df["collision_index"],
    "P_logreg_fatal":   p_test_logreg[:, 0],
    "P_logreg_serious": p_test_logreg[:, 1],
    "P_logreg_slight":  p_test_logreg[:, 2],
    "P_knn_fatal":      p_test_knn[:, 0],
    "P_knn_serious":    p_test_knn[:, 1],
    "P_knn_slight":     p_test_knn[:, 2],
    "P_dt_fatal":       p_test_dt[:, 0],
    "P_dt_serious":     p_test_dt[:, 1],
    "P_dt_slight":      p_test_dt[:, 2],
    "risk_score":       test_df["risk_score"],
    TARGET:             test_df[TARGET]
})

# [7] Validation Checks
print("\n[7] Running Validation Checks against skills.md spec:")
assert meta_test.shape[0] == len(test_df), f"Row mismatch: {meta_test.shape[0]} vs {len(test_df)}"
assert meta_test[STAGE4_FEATURES].shape[1] == 10, f"Expected 10 features, got {meta_test[STAGE4_FEATURES].shape[1]}"
assert meta_test.isna().sum().sum() == 0, "Found NaN values in meta_test!"
print(f"  [OK] Row count verified: {meta_test.shape[0]:,} rows")
print(f"  [OK] Meta-feature count verified: exactly 10 features ({STAGE4_FEATURES})")
print(f"  [OK] Zero NaN values verified")

# [8] Save meta_test.csv
out_path = PROCESSED / "meta_test.csv"
meta_test.to_csv(out_path, index=False)
print(f"\n[8] Successfully saved: {out_path}")
print("=" * 70)
print("Stage 3A complete. Both meta_train.csv and meta_test.csv are ready for Stage 4!")
print("=" * 70)
