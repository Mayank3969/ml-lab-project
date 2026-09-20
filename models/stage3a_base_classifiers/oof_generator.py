"""
Stage 3A: Phase A — Out-of-Fold (OOF) Generator
- Performs 5-fold cross-validation using StratifiedGroupKFold on collision_index.
- Trains 3 base classifiers: Logistic Regression, KNN, Decision Tree (No Perceptron).
- Generates out-of-fold probability predictions on held-out folds (zero leakage).
- Cleans up temporary models after each fold.
- Writes dataset/processed/meta_train.csv containing 9 probability columns + risk_score + label.
"""
import sys
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Path setup
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "models"))

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report

from utils.config import (
    PROCESSED,
    SEED,
    TARGET,
    CLASS_LABELS,
    CLASS_NAMES,
    STAGE3_PROBA_COLS,
    STAGE4_FEATURES
)

print("=" * 70)
print("STAGE 3A — Phase A: 5-Fold Out-Of-Fold (OOF) Meta-Feature Generator")
print("=" * 70)

# [1] Load merged training data
print("\n[1] Loading training dataset...")
train_df = pd.read_csv(PROCESSED / "merged_train.csv", converters={"collision_index": str})
print(f"  Loaded {len(train_df):,} rows from merged_train.csv")

# Feature selection: all columns except collision_index and target
feature_cols = [c for c in train_df.columns if c not in ["collision_index", TARGET]]
print(f"  Using {len(feature_cols)} input features (including risk_score):")
print(f"  {feature_cols}")

X = train_df[feature_cols]
y = train_df[TARGET].astype(int)
groups = train_df["collision_index"]

# [2] Pre-allocate OOF prediction arrays (N rows x 3 classes)
n_rows = len(train_df)
oof_logreg = np.zeros((n_rows, 3), dtype=np.float64)
oof_knn    = np.zeros((n_rows, 3), dtype=np.float64)
oof_dt     = np.zeros((n_rows, 3), dtype=np.float64)

# [3] 5-Fold Stratified Group K-Fold Loop
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
print("\n[2] Starting 5-Fold Stratified Group CV (Grouping by collision_index)...")

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
    fold_start = time.time()
    print(f"\n--- Fold {fold} / 5 ---")
    print(f"  Train samples: {len(train_idx):,} | Val samples: {len(val_idx):,}")
    
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. Logistic Regression
    print("  Training Logistic Regression...", end="", flush=True)
    m_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    ])
    m_lr.fit(X_tr, y_tr)
    oof_logreg[val_idx] = m_lr.predict_proba(X_val)
    print(" done.")
    
    # 2. K-Nearest Neighbors (KD-Tree for fast inference)
    print("  Training KNN (k=15)...", end="", flush=True)
    m_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=-1))
    ])
    m_knn.fit(X_tr, y_tr)
    oof_knn[val_idx] = m_knn.predict_proba(X_val)
    print(" done.")
    
    # 3. Decision Tree
    print("  Training Decision Tree...", end="", flush=True)
    m_dt = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED
    )
    m_dt.fit(X_tr, y_tr)
    oof_dt[val_idx] = m_dt.predict_proba(X_val)
    print(" done.")
    
    # Fold-level cleanup: delete models to prevent memory leak and strict no-reuse
    del m_lr, m_knn, m_dt
    gc.collect()
    print(f"  Fold {fold} finished in {time.time() - fold_start:.1f}s. Temporary models destroyed.")

total_time = time.time() - start_time
print(f"\n[3] 5-Fold cross validation finished in {total_time:.1f}s.")

# [4] OOF Evaluation Metrics
print("\n[4] Overall Out-Of-Fold Baseline Performance (Macro-F1):")
for name, probs in [("Logistic Regression", oof_logreg), ("KNN", oof_knn), ("Decision Tree", oof_dt)]:
    preds = np.argmax(probs, axis=1) + 1  # classes are 1, 2, 3
    macro_f1 = f1_score(y, preds, average="macro")
    acc = accuracy_score(y, preds)
    fatal_f1 = f1_score(y == 1, preds == 1, zero_division=0)
    print(f"  {name:<22s} | Macro-F1: {macro_f1:.4f} | Accuracy: {acc*100:5.2f}% | Fatal-F1: {fatal_f1:.4f}")

# [5] Assemble meta_train dataframe
print("\n[5] Assembling meta_train.csv...")
meta_train = pd.DataFrame({
    "collision_index": train_df["collision_index"],
    "P_logreg_fatal":   oof_logreg[:, 0],
    "P_logreg_serious": oof_logreg[:, 1],
    "P_logreg_slight":  oof_logreg[:, 2],
    "P_knn_fatal":      oof_knn[:, 0],
    "P_knn_serious":    oof_knn[:, 1],
    "P_knn_slight":     oof_knn[:, 2],
    "P_dt_fatal":       oof_dt[:, 0],
    "P_dt_serious":     oof_dt[:, 1],
    "P_dt_slight":      oof_dt[:, 2],
    "risk_score":       train_df["risk_score"],
    TARGET:             train_df[TARGET]
})

# [6] Validation Checks
print("\n[6] Running Validation Checks against skills.md spec:")
assert meta_train.shape[0] == len(train_df), f"Row count mismatch: {meta_train.shape[0]} vs {len(train_df)}"
assert meta_train[STAGE4_FEATURES].shape[1] == 10, f"Expected 10 meta-features, got {meta_train[STAGE4_FEATURES].shape[1]}"
assert meta_train.isna().sum().sum() == 0, "Found NaN values in meta_train!"
print(f"  [OK] Row count verified: {meta_train.shape[0]:,} rows")
print(f"  [OK] Meta-feature count verified: exactly 10 features ({STAGE4_FEATURES})")
print(f"  [OK] Zero NaN values verified")

# [7] Save meta_train.csv
out_path = PROCESSED / "meta_train.csv"
meta_train.to_csv(out_path, index=False)
print(f"\n[7] Successfully saved: {out_path}")
print("=" * 70)
print("Phase A complete. Proceed to final_models_trainer.py (Phase B).")
print("=" * 70)
