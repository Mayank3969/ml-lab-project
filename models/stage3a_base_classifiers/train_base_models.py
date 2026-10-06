"""Train the three base classifiers with manual parameters and create OOF data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from utils.config import META_CANDIDATE_FEATURES, MODEL_FEATURES, MODELS_DIR, PROCESSED, RESULTS_DIR, TARGET
from utils.manual_parameters import (
    DECISION_TREE_PARAMS,
    KNN_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
)
from utils.preprocessing import build_preprocessor, grouped_splits, load_processed


ARTIFACT_NAMES = {
    "logistic_regression": "logistic_regression_2.pkl",
    "knn": "knn_2.pkl",
    "decision_tree": "decision_tree_2.pkl",
}


def build_model(model_name: str) -> Pipeline:
    """Create one classifier with its model-specific preprocessing."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(random_state=19, **LOGISTIC_REGRESSION_PARAMS)
    elif model_name == "knn":
        estimator = KNeighborsClassifier(**KNN_PARAMS)
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=19, **DECISION_TREE_PARAMS)
    else:
        raise KeyError(model_name)
    steps = [("preprocessor", build_preprocessor(model_name))]
    if model_name == "knn":
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def ordered_probabilities(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Align probabilities to Fatal, Serious, Slight regardless of class order."""
    raw = model.predict_proba(frame[features])
    classes = list(model.classes_)
    output = np.zeros((len(frame), 3), dtype=float)
    for position, label in enumerate([1, 2, 3]):
        if label in classes:
            output[:, position] = raw[:, classes.index(label)]
    return output


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    """Calculate the full comparison metric set."""
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


def load_risk_scores(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the two collision-level OOF risk sources to casualty rows."""
    train_out = pd.DataFrame({"collision_index": train["collision_index"].astype(str)})
    test_out = pd.DataFrame({"collision_index": test["collision_index"].astype(str)})
    for model_name in ["linear_regression", "decision_tree_regressor"]:
        name = f"risk_score_{model_name}"
        train_risk = pd.read_csv(PROCESSED / f"{name}_train.csv", dtype={"collision_index": str})
        test_risk = pd.read_csv(PROCESSED / f"{name}_test.csv", dtype={"collision_index": str})
        train_out[name] = train_out["collision_index"].map(train_risk.set_index("collision_index")["risk_score"])
        test_out[name] = test_out["collision_index"].map(test_risk.set_index("collision_index")["risk_score"])
        if train_out[name].isna().any() or test_out[name].isna().any():
            raise ValueError(f"Missing {name} values")
    return train_out, test_out


def main() -> None:
    """Train all base models, evaluate them, and write stacking inputs."""
    train, test = load_processed()
    risk_train, risk_test = load_risk_scores(train, test)
    splits = grouped_splits(train)
    meta_train = risk_train.copy()
    meta_test = risk_test.copy()
    y = train[TARGET].astype(int)
    result_rows = []

    for model_name, features in MODEL_FEATURES.items():
        oof = np.zeros((len(train), 3), dtype=float)
        for fold, (fit_index, validation_index) in enumerate(splits, start=1):
            fold_model = clone(build_model(model_name))
            fold_model.fit(train.iloc[fit_index][features], y.iloc[fit_index])
            oof[validation_index] = ordered_probabilities(fold_model, train.iloc[validation_index], features)
            print(f"{model_name}: classifier fold {fold}/5 complete")

        final_model = build_model(model_name).fit(train[features], y)
        test_probabilities = ordered_probabilities(final_model, test, features)
        predictions = np.argmax(test_probabilities, axis=1) + 1
        oof_predictions = np.argmax(oof, axis=1) + 1
        row = {"model": model_name}
        row.update({f"oof_{key}": value for key, value in metrics(y, oof_predictions).items()})
        row.update({f"test_{key}": value for key, value in metrics(test[TARGET].astype(int), predictions).items()})
        row["classification_report"] = classification_report(
            test[TARGET], predictions, labels=[1, 2, 3], output_dict=True, zero_division=0
        )
        result_rows.append(row)
        for column, values in zip(
            [f"{model_name}_fatal", f"{model_name}_serious", f"{model_name}_slight"], oof.T
        ):
            meta_train[column] = values
        for column, values in zip(
            [f"{model_name}_fatal", f"{model_name}_serious", f"{model_name}_slight"], test_probabilities.T
        ):
            meta_test[column] = values
        joblib.dump(final_model, MODELS_DIR / ARTIFACT_NAMES[model_name])
        print(json.dumps(row, indent=2, default=float))

    meta_train[TARGET] = y.to_numpy()
    meta_test[TARGET] = test[TARGET].astype(int).to_numpy()
    meta_train.to_csv(PROCESSED / "meta_train.csv", index=False)
    meta_test.to_csv(PROCESSED / "meta_test.csv", index=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in row.items() if k != "classification_report"} for row in result_rows]).to_csv(
        RESULTS_DIR / "base_model_metrics.csv", index=False
    )
    (RESULTS_DIR / "base_model_reports.json").write_text(json.dumps(result_rows, indent=2, default=float), encoding="utf-8")
    missing = sorted(set(META_CANDIDATE_FEATURES) - set(meta_train.columns))
    if missing:
        raise ValueError(f"Missing meta features: {missing}")
    print("Saved meta_train.csv and meta_test.csv")


if __name__ == "__main__":
    main()
