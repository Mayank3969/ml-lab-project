"""Evaluate all three base models and the final ensemble together."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from utils.config import META_FEATURES, MODEL_FEATURES, MODELS_DIR, PROCESSED, RESULTS_DIR, TARGET


def metrics(y_true: pd.Series, prediction: pd.Series) -> dict[str, float]:
    """Calculate the comparison metrics shown in Streamlit."""
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
    """Evaluate saved stage-2 models and the stage-3 ensemble."""
    train = pd.read_csv(PROCESSED / "merged_train.csv", dtype={"collision_index": str})
    test = pd.read_csv(PROCESSED / "merged_test.csv", dtype={"collision_index": str})
    y = test[TARGET].astype(int)
    rows = []
    reports = {}
    for model_name, features in MODEL_FEATURES.items():
        model = joblib.load(MODELS_DIR / f"{model_name}_2.pkl")
        prediction = model.predict(test[features])
        rows.append({"model": model_name, **metrics(y, prediction)})
        reports[model_name] = classification_report(y, prediction, labels=[1, 2, 3], output_dict=True, zero_division=0)

    ensemble = joblib.load(MODELS_DIR / "meta_ensemble_3.pkl")
    contract = json.loads((MODELS_DIR / "meta_contract.json").read_text(encoding="utf-8"))
    meta_test = pd.read_csv(PROCESSED / "meta_test.csv", dtype={"collision_index": str})
    prediction = ensemble.predict(meta_test[contract["features"]])
    rows.append({"model": "meta_ensemble_3", **metrics(y, prediction)})
    reports["meta_ensemble_3"] = classification_report(y, prediction, labels=[1, 2, 3], output_dict=True, zero_division=0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    (RESULTS_DIR / "classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for name, report in reports.items():
        model_prediction = (
            ensemble.predict(meta_test[contract["features"]])
            if name == "meta_ensemble_3"
            else joblib.load(MODELS_DIR / f"{name}_2.pkl").predict(test[MODEL_FEATURES[name]])
        )
        pd.DataFrame(
            confusion_matrix(y, model_prediction, labels=[1, 2, 3]),
            index=["True_Fatal", "True_Serious", "True_Slight"],
            columns=["Pred_Fatal", "Pred_Serious", "Pred_Slight"],
        ).to_csv(RESULTS_DIR / f"{name}_confusion_matrix.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
