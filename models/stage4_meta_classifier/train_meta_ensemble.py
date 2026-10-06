"""Train the final ensemble using manual parameters and one selected risk score."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from utils.config import META_FEATURES, MODELS_DIR, PROCESSED, RESULTS_DIR, TARGET
from utils.manual_parameters import META_ENSEMBLE_PARAMS, META_RISK_FEATURE


def metrics(y_true: pd.Series, prediction: pd.Series) -> dict:
    """Return all metrics used in the final comparison."""
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision_macro": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, prediction, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, prediction, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "fatal_recall": float(recall_score(y_true, prediction, labels=[1], average=None, zero_division=0)[0]),
        "serious_recall": float(recall_score(y_true, prediction, labels=[2], average=None, zero_division=0)[0]),
        "slight_recall": float(recall_score(y_true, prediction, labels=[3], average=None, zero_division=0)[0]),
    }


def main() -> None:
    """Fit and evaluate the manually selected final ensemble."""
    train = pd.read_csv(PROCESSED / "meta_train.csv", dtype={"collision_index": str})
    test = pd.read_csv(PROCESSED / "meta_test.csv", dtype={"collision_index": str})
    if META_RISK_FEATURE not in train.columns:
        raise ValueError(f"Selected risk feature is unavailable: {META_RISK_FEATURE}")
    features = META_FEATURES + [META_RISK_FEATURE]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(random_state=19, **META_ENSEMBLE_PARAMS)),
    ])
    model.fit(train[features], train[TARGET].astype(int))
    prediction = model.predict(test[features])
    result = {
        "model": "meta_ensemble_3",
        "risk_feature": META_RISK_FEATURE,
        "features": features,
        "params": META_ENSEMBLE_PARAMS,
        **metrics(test[TARGET].astype(int), prediction),
        "classification_report": classification_report(
            test[TARGET], prediction, labels=[1, 2, 3], output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(test[TARGET], prediction, labels=[1, 2, 3]).tolist(),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "meta_ensemble_3.pkl")
    (MODELS_DIR / "meta_contract.json").write_text(
        json.dumps({"features": features, "risk_feature": META_RISK_FEATURE, "class_labels": [1, 2, 3]}, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "meta_ensemble_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(test[TARGET], prediction, labels=[1, 2, 3]),
        index=["True_Fatal", "True_Serious", "True_Slight"],
        columns=["Pred_Fatal", "Pred_Serious", "Pred_Slight"],
    ).to_csv(RESULTS_DIR / "meta_ensemble_confusion_matrix.csv")
    print(classification_report(test[TARGET], prediction, labels=[1, 2, 3], target_names=["Fatal", "Serious", "Slight"]))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
