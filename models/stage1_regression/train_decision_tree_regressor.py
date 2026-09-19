"""
Stage 1B: Train Decision Tree Regressor to predict number_of_casualties.
Uses 5-fold out-of-fold predictions to avoid leakage.
Saves: model, OOF predictions, and metrics.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Path setup
MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS_DIR))

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from utils.config import SEED, STAGE1_FEATURES, STAGE1_TARGET, STAGE1_DIR, MODELS_DIR as SAVE_DIR
from utils.preprocessing import load_merged, collapse_to_collision, build_stage1_preprocessor

print("=" * 60)
print("STAGE 1B: Decision Tree Regressor")
print("=" * 60)

# Load and prepare data
train, _ = load_merged()
train_col = collapse_to_collision(train)
print(f"  Collision-level train rows: {len(train_col):,}")

X = train_col[STAGE1_FEATURES]
y = train_col[STAGE1_TARGET].astype(float)

# Build model
model = Pipeline([
    ("pre", build_stage1_preprocessor()),
    ("reg", DecisionTreeRegressor(max_depth=8, random_state=SEED)),
])

# 5-fold out-of-fold predictions
print("  Running 5-fold OOF...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = cross_val_predict(model, X, y, cv=kf)

# Metrics
rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
r2 = float(r2_score(y, oof_preds))
print(f"  RMSE: {rmse:.4f}")
print(f"  R2:   {r2:.4f}")

# Save metrics
metrics = {"model": "DecisionTreeRegressor", "RMSE": rmse, "R2": r2}
with open(STAGE1_DIR / "dt_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save OOF predictions
oof_df = pd.DataFrame({
    "collision_index": train_col["collision_index"],
    "oof_pred": oof_preds
})
oof_df.to_csv(STAGE1_DIR / "dt_oof_predictions.csv", index=False)

# Fit on full train and save model
model.fit(X, y)
import joblib
joblib.dump(model, SAVE_DIR / "stage1_dt_model.pkl")

print("  Saved: dt_metrics.json, dt_oof_predictions.csv, stage1_dt_model.pkl")
print("  Done.")