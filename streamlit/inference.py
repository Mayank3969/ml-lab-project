"""Small inference wrapper used by the Streamlit application."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "saved_models"
import sys

sys.path.insert(0, str(ROOT / "models"))
from utils.config import CLASS_LABELS, META_FEATURES, MODEL_FEATURES, REGRESSION_FEATURES, RISK_FEATURES


class ModelArtifacts:
    """Load the three model-specific pipelines and the meta pipeline once."""

    def __init__(self) -> None:
        self.models: dict[str, object] = {}
        self.regressors: dict[str, object] = {}
        self.meta = None
        self.meta_features: list[str] = []
        self.loaded = False
        self.error = None

    def load(self) -> "ModelArtifacts":
        try:
            self.models = {
                name: joblib.load(MODELS / f"{name}_2.pkl")
                for name in MODEL_FEATURES
            }
            self.regressors = {
                "risk_score_linear_regression": joblib.load(MODELS / "linear_regression_1.pkl"),
                "risk_score_decision_tree_regressor": joblib.load(MODELS / "decision_tree_regressor_1.pkl"),
            }
            self.meta = joblib.load(MODELS / "meta_ensemble_3.pkl")
            contract_path = MODELS / "meta_contract.json"
            if not contract_path.exists():
                raise FileNotFoundError("meta_contract.json is missing; run meta_ensemble_tuning.ipynb")
            import json
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.meta_features = contract["features"]
            if set(self.meta_features) != set(META_FEATURES + [contract["risk_feature"]]):
                raise ValueError("Saved meta contract does not match the active meta feature contract")
            for name, model in self.models.items():
                trained_columns = list(getattr(model, "feature_names_in_", MODEL_FEATURES[name]))
                if trained_columns != MODEL_FEATURES[name]:
                    raise ValueError(f"Saved {name} columns differ from the active feature contract")
            self.loaded = True
            self.error = None
        except Exception as error:  # pragma: no cover - displayed by the app
            self.loaded = False
            self.error = str(error)
        return self


_artifacts: ModelArtifacts | None = None


def get_models() -> ModelArtifacts:
    """Return cached artifacts, loading them on first use."""
    global _artifacts
    if _artifacts is None or not _artifacts.loaded:
        _artifacts = ModelArtifacts().load()
    return _artifacts


def required_columns() -> list[str]:
    """Return the union of all raw columns required by the base models."""
    all_columns = list(column for columns in MODEL_FEATURES.values() for column in columns)
    all_columns.extend(REGRESSION_FEATURES)
    return list(dict.fromkeys(all_columns))


def _ordered_probabilities(model: object, values: np.ndarray) -> np.ndarray:
    """Put estimator probabilities into the fixed Fatal/Serious/Slight order."""
    output = np.zeros((len(values), 3), dtype=float)
    classes = list(model.classes_)
    for position, label in enumerate(CLASS_LABELS):
        if label in classes:
            output[:, position] = values[:, classes.index(label)]
    return output


def predict_pipeline(input_df: pd.DataFrame) -> dict:
    """Run all base pipelines and the stacking model on raw input rows."""
    artifacts = get_models()
    if not artifacts.loaded:
        raise RuntimeError(
            f"Models are unavailable: {artifacts.error}. Run the three tuning notebooks first."
        )
    missing = sorted(set(required_columns()) - set(input_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if input_df.empty:
        raise ValueError("Input contains no records")
    input_df = input_df.copy()
    for column in required_columns():
        input_df[column] = pd.to_numeric(input_df[column], errors="raise").fillna(-1)
    base_probabilities = {}
    meta_frame = pd.DataFrame(index=input_df.index)
    for name, model in artifacts.models.items():
        raw_probabilities = model.predict_proba(input_df[MODEL_FEATURES[name]])
        probabilities = _ordered_probabilities(model, raw_probabilities)
        base_probabilities[name] = probabilities
        for column, values in zip(
            [f"{name}_fatal", f"{name}_serious", f"{name}_slight"], probabilities.T
        ):
            meta_frame[column] = values
    for risk_name, model in artifacts.regressors.items():
        meta_frame[risk_name] = model.predict(input_df[REGRESSION_FEATURES])
    final_predictions = artifacts.meta.predict(meta_frame[artifacts.meta_features])
    final_probabilities = artifacts.meta.predict_proba(meta_frame[artifacts.meta_features])
    return {
        "final_predictions": np.asarray(final_predictions, dtype=int),
        "final_probabilities": final_probabilities,
        "base_probabilities": base_probabilities,
        "meta_features_df": meta_frame[artifacts.meta_features],
    }


def predict_single(feature_dict: dict) -> dict:
    """Run one dictionary through the same path used by batch predictions."""
    result = predict_pipeline(pd.DataFrame([feature_dict]))
    return {
        "prediction": int(result["final_predictions"][0]),
        "probabilities": result["final_probabilities"][0],
        "base_probabilities": {name: values[0] for name, values in result["base_probabilities"].items()},
        "meta_features": result["meta_features_df"].iloc[0].to_dict(),
    }
