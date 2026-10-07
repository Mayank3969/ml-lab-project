"""Train both collision-level risk regressors using manual parameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from utils.config import (
    HYPERPARAMETER_RESULTS,
    MODELS_DIR,
    PROCESSED,
    REGRESSION_MODEL_FEATURES,
    REGRESSION_TARGET,
    REGRESSION_TEST_PATH,
    REGRESSION_TRAIN_PATH,
    SEED,
)
from utils.manual_parameters import DECISION_TREE_REGRESSOR_PARAMS, LINEAR_REGRESSION_PARAMS
from utils.preprocessing import build_regression_preprocessor

RESULTS = HYPERPARAMETER_RESULTS / "manual_pipeline_regression"
RESULTS.mkdir(parents=True, exist_ok=True)

def build_model(model_name: str) -> Pipeline:
    """Create one regression pipeline from manual_parameters.py."""
    if model_name == "linear_regression":
        params = {k.removeprefix("regressor__"): v for k, v in LINEAR_REGRESSION_PARAMS.items()}
        estimator = LinearRegression(**params)
    elif model_name == "decision_tree_regressor":
        params = {k.removeprefix("regressor__"): v for k, v in DECISION_TREE_REGRESSOR_PARAMS.items()}
        estimator = DecisionTreeRegressor(random_state=SEED, **params)
    else:
        raise KeyError(model_name)
    return Pipeline([
        ("preprocessor", build_regression_preprocessor(REGRESSION_MODEL_FEATURES[model_name])),
        ("regressor", estimator),
    ])


def train_one(model_name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit one regressor, generate collision-level OOF scores, and save outputs."""
    features = REGRESSION_MODEL_FEATURES[model_name]
    X = train[features]
    y = train[REGRESSION_TARGET].astype(float)
    X_test = test[features]
    y_test = test[REGRESSION_TARGET].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    base_model = build_model(model_name)
    oof = np.zeros(len(train), dtype=float)
    for fold, (fit_index, validation_index) in enumerate(cv.split(X), start=1):
        fold_model = clone(base_model)
        fold_model.fit(X.iloc[fit_index], y.iloc[fit_index])
        oof[validation_index] = fold_model.predict(X.iloc[validation_index])
        print(f"{model_name}: regression fold {fold}/5 complete")

    final_model = build_model(model_name).fit(X, y)
    test_prediction = final_model.predict(X_test)
    result = {
        "model": model_name,
        "features": features,
        "params": (LINEAR_REGRESSION_PARAMS if model_name == "linear_regression" else DECISION_TREE_REGRESSOR_PARAMS),
        "oof_r2": float(r2_score(y, oof)),
        "oof_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "oof_mae": float(mean_absolute_error(y, oof)),
        "test_r2": float(r2_score(y_test, test_prediction)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_prediction))),
        "test_mae": float(mean_absolute_error(y_test, test_prediction)),
    }
    risk_column = f"risk_score_{model_name}"
    pd.DataFrame({"collision_index": train["collision_index"], "risk_score": oof}).to_csv(
        PROCESSED / f"{risk_column}_train.csv", index=False
    )
    pd.DataFrame({"collision_index": test["collision_index"], "risk_score": test_prediction}).to_csv(
        PROCESSED / f"{risk_column}_test.csv", index=False
    )
    joblib.dump(final_model, MODELS_DIR / f"{model_name}_1.pkl")
    (RESULTS / f"{model_name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    """Train both risk regressors and write a comparison table."""
    train = pd.read_csv(REGRESSION_TRAIN_PATH, dtype={"collision_index": str})
    test = pd.read_csv(REGRESSION_TEST_PATH, dtype={"collision_index": str})
    results = [
        train_one("linear_regression", train, test),
        train_one("decision_tree_regressor", train, test),
    ]
    pd.DataFrame(results).to_csv(RESULTS / "regressor_comparison.csv", index=False)


if __name__ == "__main__":
    main()
