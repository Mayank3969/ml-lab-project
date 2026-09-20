"""
Stage 5: Baseline vs Multi-Stage Pipeline Evaluation
Compares:
1. Stage 1 regression models (Linear Regression vs Decision Tree)
2. Stage 3A standalone base classifiers (Logistic Regression, KNN, Decision Tree)
3. Best standalone classifier vs Stage 4 Meta-Classifier (Headline Delta)
4. Class-level recall, especially for the critical Fatal minority class.
Saves metrics tables to models/evaluation/results/.
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "models"))

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from utils.config import PROCESSED, RESULTS_DIR, MODELS_DIR, STAGE2_DIR, TARGET, CLASS_NAMES, STAGE4_FEATURES

print("=" * 70)
print("STAGE 5: Baseline vs Multi-Stage Pipeline Evaluation")
print("=" * 70)

# [1] Load Stage 1 Regression metrics
print("\n[1] Regression Models Comparison (Stage 1):")
with open(STAGE2_DIR / "lr_metrics.json") as f:
    lr_m = json.load(f)
with open(STAGE2_DIR / "dt_metrics.json") as f:
    dt_m = json.load(f)

print(f"  {'Model':<25s} {'RMSE':>8s} {'R2':>8s}")
print(f"  {'-'*43}")
print(f"  {lr_m['model']:<25s} {lr_m['RMSE']:8.4f} {lr_m['R2']:8.4f}")
print(f"  {dt_m['model']:<25s} {dt_m['RMSE']:8.4f} {dt_m['R2']:8.4f}")

# [2] Load Test Set & Meta Test Set
meta_test = pd.read_csv(PROCESSED / "meta_test.csv")
y_true = meta_test[TARGET].astype(int)

# Load final models
logreg = joblib.load(MODELS_DIR / "stage2_logreg_final.pkl")
knn    = joblib.load(MODELS_DIR / "stage2_knn_final.pkl")
dt     = joblib.load(MODELS_DIR / "stage2_dt_final.pkl")
meta   = joblib.load(MODELS_DIR / "stage3_meta_classifier.pkl")

# [3] Evaluate Standalone Base Classifiers
print("\n[2] Base Classifiers Standalone vs Full Pipeline on Held-out Test Set:")

models_eval = []

# Base models
preds_logreg = np.argmax(meta_test[["P_logreg_fatal", "P_logreg_serious", "P_logreg_slight"]].values, axis=1) + 1
preds_knn    = np.argmax(meta_test[["P_knn_fatal", "P_knn_serious", "P_knn_slight"]].values, axis=1) + 1
preds_dt     = np.argmax(meta_test[["P_dt_fatal", "P_dt_serious", "P_dt_slight"]].values, axis=1) + 1

# Pipeline meta-classifier
preds_meta   = meta.predict(meta_test[STAGE4_FEATURES])

eval_dict = {
    "Logistic Regression (Standalone)": preds_logreg,
    "KNN (Standalone)": preds_knn,
    "Decision Tree (Standalone)": preds_dt,
    "Full Multi-Stage Stacking Pipeline": preds_meta
}

for name, preds in eval_dict.items():
    macro_f1 = f1_score(y_true, preds, average="macro")
    acc = accuracy_score(y_true, preds)
    fatal_rec = f1_score(y_true == 1, preds == 1, zero_division=0)
    serious_rec = f1_score(y_true == 2, preds == 2, zero_division=0)
    slight_rec = f1_score(y_true == 3, preds == 3, zero_division=0)
    
    models_eval.append({
        "Model": name,
        "Macro_F1": macro_f1,
        "Accuracy": acc,
        "Fatal_F1": fatal_rec,
        "Serious_F1": serious_rec,
        "Slight_F1": slight_rec
    })

eval_df = pd.DataFrame(models_eval)

print(f"\n  {'Model':<38s} {'Macro-F1':>10s} {'Accuracy':>10s} {'Fatal-F1':>10s}")
print(f"  {'-'*72}")
for _, row in eval_df.iterrows():
    print(f"  {row['Model']:<38s} {row['Macro_F1']:10.4f} {row['Accuracy']*100:9.2f}% {row['Fatal_F1']:10.4f}")

# [4] Compute Headline Result Delta
best_standalone = eval_df.iloc[:3].sort_values("Macro_F1", ascending=False).iloc[0]
pipeline_row = eval_df.iloc[3]
delta_f1 = pipeline_row["Macro_F1"] - best_standalone["Macro_F1"]

print("\n" + "=" * 60)
print("HEADLINE RESULT: Best Standalone vs Full Pipeline")
print("=" * 60)
print(f"  Best Standalone Model : {best_standalone['Model']}")
print(f"  Standalone Macro-F1   : {best_standalone['Macro_F1']:.4f}")
print(f"  Full Pipeline Macro-F1: {pipeline_row['Macro_F1']:.4f}")
print(f"  Headline Delta        : {delta_f1:+.4f} Macro-F1")

# Fatal Recall
cm = confusion_matrix(y_true, preds_meta)
fatal_total = (y_true == 1).sum()
fatal_caught = cm[0, 0]
print(f"\n  Fatal Class Safety Recall:")
print(f"  Caught {fatal_caught} out of {fatal_total} fatal accidents ({fatal_caught / fatal_total * 100:.1f}%)")

# [5] Save results table
eval_df.to_csv(RESULTS_DIR / "model_comparison_table.csv", index=False)
print(f"\nSaved metrics summary to: {RESULTS_DIR / 'model_comparison_table.csv'}")
print("=" * 70)
