"""
train_knn.py
Standalone training and evaluation of K-Nearest Neighbors (KNN) base classifier.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "models"))

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

from utils.config import PROCESSED, TARGET, CLASS_NAMES

print("=" * 60)
print("Standalone Base Model: K-Nearest Neighbors (KNN)")
print("=" * 60)

train_df = pd.read_csv(PROCESSED / "merged_train.csv", converters={"collision_index": str})
test_df  = pd.read_csv(PROCESSED / "merged_test.csv",  converters={"collision_index": str})

feature_cols = [c for c in train_df.columns if c not in ["collision_index", TARGET]]

X_train, y_train = train_df[feature_cols], train_df[TARGET].astype(int)
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET].astype(int)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=-1))
])

print("Training KNN (k=15)...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nTest Evaluation:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print(f"Macro-F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
