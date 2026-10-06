"""Shared helpers used by the active training and inference code."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import (
    CLASS_LABELS,
    FATAL_TOLERANCE,
    FEATURE_MANIFEST_PATH,
    META_FEATURES,
    MODEL_FEATURES,
    SEED,
    SPLIT_FOLDS,
    TARGET,
)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the prepared train/test files while preserving collision IDs."""
    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / "dataset/processed/merged_train.csv", dtype={"collision_index": str})
    test = pd.read_csv(root / "dataset/processed/merged_test.csv", dtype={"collision_index": str})
    return train, test


def load_feature_manifest() -> dict:
    """Return the feature contract written by the merge stage."""
    with FEATURE_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_feature_names(model_name: str) -> list[str]:
    """Return the ordered raw columns used by one base model."""
    if model_name not in MODEL_FEATURES:
        raise KeyError(f"Unknown model: {model_name}")
    return list(MODEL_FEATURES[model_name])


def categorical_and_numeric(columns: list[str]) -> tuple[list[str], list[str]]:
    """Classify STATS19 columns for model-specific preprocessing."""
    numeric_names = {
        "age_of_casualty", "age_of_driver", "engine_capacity_cc",
        "number_of_vehicles", "speed_limit",
    }
    numeric = [column for column in columns if column in numeric_names]
    categorical = [column for column in columns if column not in numeric_names]
    return categorical, numeric


def build_preprocessor(model_name: str) -> ColumnTransformer:
    """Build a safe, fold-local encoder for a named model."""
    columns = model_feature_names(model_name)
    categorical, numeric = categorical_and_numeric(columns)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    if model_name == "knn":
        # KNN needs comparable numeric and binary-category distances.
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = StandardScaler() if model_name == "logistic_regression" else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, categorical),
            ("numeric", numeric_transformer, numeric),
        ],
        remainder="drop",
    )


def build_regression_preprocessor() -> ColumnTransformer:
    """Build the collision-level encoder used by both risk regressors."""
    from utils.config import REGRESSION_FEATURES

    numeric_names = {"number_of_vehicles", "speed_limit"}
    numeric = [column for column in REGRESSION_FEATURES if column in numeric_names]
    categorical = [column for column in REGRESSION_FEATURES if column not in numeric_names]
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler(), numeric),
    ])


def grouped_splits(
    frame: pd.DataFrame,
    n_splits: int = SPLIT_FOLDS,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified, collision-group-aware folds and validate their shape."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = frame[TARGET].astype(int)
    groups = frame["collision_index"].astype(str)
    splits = list(splitter.split(frame, y, groups))
    global_rates = y.value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
    for fold, (_, validation_index) in enumerate(splits):
        validation = frame.iloc[validation_index]
        if validation[TARGET].nunique() != len(CLASS_LABELS):
            raise ValueError(f"Fold {fold} does not contain all severity classes")
        rates = validation[TARGET].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
        rate_errors = (rates - global_rates).abs()
        if rate_errors.max() > FATAL_TOLERANCE:
            raise ValueError(
                f"Fold {fold} class-rate deviation is {rate_errors.max():.4%}; "
                f"expected <= {FATAL_TOLERANCE:.2%}. Choose a better grouped seed."
            )
        train_groups = set(groups.iloc[splits[fold][0]])
        validation_groups = set(groups.iloc[validation_index])
        if train_groups & validation_groups:
            raise ValueError(f"Collision leakage detected in fold {fold}")
    return splits


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Return the metrics used consistently by notebooks and reports."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "fatal_recall": float(recall_score(y_true, predictions, labels=[1], average=None, zero_division=0)[0]),
        "serious_recall": float(recall_score(y_true, predictions, labels=[2], average=None, zero_division=0)[0]),
        "slight_recall": float(recall_score(y_true, predictions, labels=[3], average=None, zero_division=0)[0]),
    }


__all__ = [
    "META_FEATURES", "build_preprocessor", "build_regression_preprocessor", "categorical_and_numeric",
    "evaluate_predictions", "grouped_splits", "load_feature_manifest",
    "load_processed", "model_feature_names",
]
