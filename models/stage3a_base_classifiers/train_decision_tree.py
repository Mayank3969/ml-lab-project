"""
train_decision_tree.py
Standalone training and evaluation of Decision Tree base classifier.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "models"))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score

from utils.config import PROCESSED, SEED, TARGET, CLASS_NAMES

print("=" * 60)
print("Standalone Base Model: Decision Tree")
print("=" * 60)

train_df = pd.read_csv(PROCESSED / "merged_train.csv", converters={"collision_index": str})
test_df  = pd.read_csv(PROCESSED / "merged_test.csv",  converters={"collision_index": str})

feature_cols = [c for c in train_df.columns if c not in ["collision_index", TARGET]]

X_train, y_train = train_df[feature_cols], train_df[TARGET].astype(int)
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET].astype(int)

model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=SEED
)

print("Training Decision Tree...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nTest Evaluation:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print(f"Macro-F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
