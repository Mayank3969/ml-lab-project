"""
Stage 4: Meta-Classifier Trainer
- Trains stacking meta-classifier (Logistic Regression) on the 10 meta-features (9 probas + risk_score).
- Trains on meta_train.csv and evaluates on meta_test.csv.
- Computes Macro-F1, accuracy, and per-class performance metrics.
- Serializes final model to models/saved_models/stage3_meta_classifier.pkl.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "models"))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from utils.config import (
    PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    TARGET,
    CLASS_NAMES,
    STAGE4_FEATURES
)

print("=" * 70)
print("STAGE 4: Train Stacking Meta-Classifier")
print("=" * 70)

# [1] Load meta train and test datasets
print("\n[1] Loading meta datasets...")
meta_train = pd.read_csv(PROCESSED / "meta_train.csv")
meta_test  = pd.read_csv(PROCESSED / "meta_test.csv")

print(f"  meta_train: {meta_train.shape[0]:,} rows")
print(f"  meta_test : {meta_test.shape[0]:,} rows")

# [2] Prepare input features (strictly 10 columns) and target
print(f"\n[2] Feature verification:")
print(f"  Meta-features ({len(STAGE4_FEATURES)} cols): {STAGE4_FEATURES}")
assert len(STAGE4_FEATURES) == 10, f"Expected 10 features, got {len(STAGE4_FEATURES)}"

X_meta_train = meta_train[STAGE4_FEATURES]
y_meta_train = meta_train[TARGET].astype(int)

X_meta_test = meta_test[STAGE4_FEATURES]
y_meta_test = meta_test[TARGET].astype(int)

# [3] Train Meta-Classifier (Logistic Regression with class_weight='balanced')
print("\n[3] Fitting Meta-Classifier (Logistic Regression)...")
meta_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
])
meta_model.fit(X_meta_train, y_meta_train)
print("  Model fit complete.")

# [4] Predict on held-out test set
print("\n[4] Evaluating on meta_test.csv...")
y_pred = meta_model.predict(X_meta_test)
y_prob = meta_model.predict_proba(X_meta_test)

macro_f1 = f1_score(y_meta_test, y_pred, average="macro")
acc = accuracy_score(y_meta_test, y_pred)
fatal_f1 = f1_score(y_meta_test == 1, y_pred == 1, zero_division=0)
serious_f1 = f1_score(y_meta_test == 2, y_pred == 2, zero_division=0)
slight_f1 = f1_score(y_meta_test == 3, y_pred == 3, zero_division=0)

print("\n" + "=" * 50)
print("STAGE 4 META-CLASSIFIER RESULTS:")
print("=" * 50)
print(f"  Macro-F1 Score : {macro_f1:.4f}")
print(f"  Accuracy       : {acc*100:.2f}%")
print(f"  Fatal F1       : {fatal_f1:.4f}")
print(f"  Serious F1     : {serious_f1:.4f}")
print(f"  Slight F1      : {slight_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_meta_test, y_pred, target_names=CLASS_NAMES))

print("Confusion Matrix:")
cm = confusion_matrix(y_meta_test, y_pred)
print(cm)

# Save evaluation results
cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
cm_df.to_csv(RESULTS_DIR / "meta_classifier_confusion_matrix.csv")

# [5] Save final meta-model artifact
model_out = MODELS_DIR / "stage3_meta_classifier.pkl"
joblib.dump(meta_model, model_out)
print(f"\n[5] Saved model artifact to: {model_out}")
print("=" * 70)
