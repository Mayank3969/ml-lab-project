"""
Stage 1C: Compare both regressors, pick the best, generate risk_score.
- Reads metrics from both train scripts
- Selects winner by R2
- Maps risk_score back to every casualty row
- Saves updated merged files + best model
"""
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Path setup
MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS_DIR))

from utils.config import STAGE2_DIR, MODELS_DIR as SAVE_DIR, RESULTS_DIR, PROCESSED

print("=" * 60)
print("STAGE 1C: Compare Regressors + Generate risk_score")
print("=" * 60)

# [1] Load metrics from both models
print("\n[1] Loading metrics...")
with open(STAGE2_DIR / "lr_metrics.json") as f:
    lr_metrics = json.load(f)
with open(STAGE2_DIR / "dt_metrics.json") as f:
    dt_metrics = json.load(f)

# [2] Comparison table
print("\n[2] Comparison:")
print(f"  {'Model':<25s} {'RMSE':>8s} {'R2':>8s}")
print(f"  {'-'*41}")
print(f"  {lr_metrics['model']:<25s} {lr_metrics['RMSE']:8.4f} {lr_metrics['R2']:8.4f}")
print(f"  {dt_metrics['model']:<25s} {dt_metrics['RMSE']:8.4f} {dt_metrics['R2']:8.4f}")

# [3] Pick best by R2
best_name = "lr" if lr_metrics["R2"] >= dt_metrics["R2"] else "dt"
best_metrics = lr_metrics if best_name == "lr" else dt_metrics
print(f"\n[3] Winner: {best_metrics['model']} (R2 = {best_metrics['R2']:.4f})")

# Save comparison to results
results_df = pd.DataFrame([lr_metrics, dt_metrics])
results_df.to_csv(RESULTS_DIR / "stage1_metrics.csv", index=False)

# [4] Generate risk_score
print("\n[4] Generating risk_score...")

# Load OOF predictions for the best model (train risk_score = OOF, no leakage)
oof_df = pd.read_csv(STAGE2_DIR / f"{best_name}_oof_predictions.csv",
                     converters={"collision_index": str})
train_risk_map = dict(zip(oof_df["collision_index"], oof_df["oof_pred"]))

# Load the best full-train model for test predictions
best_model = joblib.load(SAVE_DIR / f"stage1_{best_name}_model.pkl")

# Load merged data
train = pd.read_csv(PROCESSED / "merged_train.csv", converters={"collision_index": str})
test = pd.read_csv(PROCESSED / "merged_test.csv", converters={"collision_index": str})

# Map train risk_score from OOF
train["risk_score"] = train["collision_index"].map(train_risk_map).astype(float)

# For test: predict using the full-train model on collision-level features
from utils.config import STAGE1_FEATURES
from utils.preprocessing import collapse_to_collision
test_col = collapse_to_collision(test)
test_preds = best_model.predict(test_col[STAGE1_FEATURES])
test_risk_map = dict(zip(test_col["collision_index"], test_preds))
test["risk_score"] = test["collision_index"].map(test_risk_map).astype(float)

# [5] Verify
print(f"  Train risk_score: min={train['risk_score'].min():.3f}, max={train['risk_score'].max():.3f}")
print(f"  Test  risk_score: min={test['risk_score'].min():.3f}, max={test['risk_score'].max():.3f}")
print(f"  Missing (train): {train['risk_score'].isna().sum()}")
print(f"  Missing (test):  {test['risk_score'].isna().sum()}")

# [6] Save
train.to_csv(PROCESSED / "merged_train.csv", index=False)
test.to_csv(PROCESSED / "merged_test.csv", index=False)

# Copy best model to the expected name
joblib.dump(best_model, SAVE_DIR / "stage1_best_regressor.pkl")

print(f"\n  Saved updated merged files with risk_score (now 19 columns)")
print(f"  Saved stage1_best_regressor.pkl")
print("\n  Stage 1 complete. Ready for Stage 2.")