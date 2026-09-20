"""
inference.py — End-to-End Pipeline Inference Engine
Loads all saved model artifacts and executes the 4-stage prediction flow
for both single-row inputs and bulk DataFrames (CSV / Excel).
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths setup
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "saved_models"

# Exact features used at each stage
STAGE1_REG_FEATURES = [
    "speed_limit", "road_type", "junction_detail", "light_conditions", "weather_conditions"
]

BASE_CLASSIFIER_FEATURES = [
    "casualty_class", "casualty_type", "number_of_vehicles", "number_of_casualties",
    "day_of_week", "first_road_class", "road_type", "speed_limit", "junction_detail",
    "junction_control", "light_conditions", "weather_conditions", "road_surface_conditions",
    "special_conditions_at_site", "carriageway_hazards", "urban_or_rural_area", "risk_score"
]

META_FEATURES = [
    "P_logreg_fatal", "P_logreg_serious", "P_logreg_slight",
    "P_knn_fatal",    "P_knn_serious",    "P_knn_slight",
    "P_dt_fatal",     "P_dt_serious",     "P_dt_slight",
    "risk_score"
]


class ModelArtifacts:
    """Holds all serialized models in memory."""
    def __init__(self):
        self.regressor = None
        self.logreg = None
        self.knn = None
        self.dt = None
        self.meta_classifier = None
        self.loaded = False
        self.error = None

    def load(self):
        try:
            self.regressor = joblib.load(MODELS_DIR / "stage1_best_regressor.pkl")
            self.logreg = joblib.load(MODELS_DIR / "stage2_logreg_final.pkl")
            self.knn = joblib.load(MODELS_DIR / "stage2_knn_final.pkl")
            self.dt = joblib.load(MODELS_DIR / "stage2_dt_final.pkl")
            self.meta_classifier = joblib.load(MODELS_DIR / "stage3_meta_classifier.pkl")
            self.loaded = True
            self.error = None
        except Exception as e:
            self.loaded = False
            self.error = str(e)
        return self


# Global singleton
_artifacts = None

def get_models():
    """Returns the cached ModelArtifacts object."""
    global _artifacts
    if _artifacts is None or not _artifacts.loaded:
        _artifacts = ModelArtifacts().load()
    return _artifacts


def predict_pipeline(input_df: pd.DataFrame):
    """
    Executes the entire 4-stage pipeline on an input DataFrame.
    
    Expected columns in input_df (the 16 raw features):
        ['casualty_class', 'casualty_type', 'number_of_vehicles', 'number_of_casualties',
         'day_of_week', 'first_road_class', 'road_type', 'speed_limit', 'junction_detail',
         'junction_control', 'light_conditions', 'weather_conditions', 'road_surface_conditions',
         'special_conditions_at_site', 'carriageway_hazards', 'urban_or_rural_area']
    
    Returns:
        dict containing:
            - 'final_predictions': array of int (1=Fatal, 2=Serious, 3=Slight)
            - 'final_probabilities': array of shape (N, 3) for [Fatal, Serious, Slight]
            - 'risk_scores': array of float
            - 'base_probabilities': dict of arrays for logreg, knn, dt
            - 'meta_features_df': DataFrame of shape (N, 10)
    """
    models = get_models()
    if not models.loaded:
        raise RuntimeError(f"Models could not be loaded: {models.error}. Run 'python pipeline.py' first.")

    # 1. Validate required columns
    cols_16 = [c for c in BASE_CLASSIFIER_FEATURES if c != "risk_score"]
    missing = [c for c in cols_16 if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    df = input_df[cols_16].copy()

    # 2. Stage 2: Predict risk_score using best regressor
    X_reg = df[STAGE1_REG_FEATURES]
    risk_scores = models.regressor.predict(X_reg)
    df["risk_score"] = risk_scores

    # 3. Stage 3A: Predict class probabilities using 3 final base models
    X_base = df[BASE_CLASSIFIER_FEATURES]
    p_logreg = models.logreg.predict_proba(X_base)
    p_knn = models.knn.predict_proba(X_base)
    p_dt = models.dt.predict_proba(X_base)

    # 4. Assemble 10 meta-features
    meta_df = pd.DataFrame({
        "P_logreg_fatal":   p_logreg[:, 0],
        "P_logreg_serious": p_logreg[:, 1],
        "P_logreg_slight":  p_logreg[:, 2],
        "P_knn_fatal":      p_knn[:, 0],
        "P_knn_serious":    p_knn[:, 1],
        "P_knn_slight":     p_knn[:, 2],
        "P_dt_fatal":       p_dt[:, 0],
        "P_dt_serious":     p_dt[:, 1],
        "P_dt_slight":      p_dt[:, 2],
        "risk_score":       risk_scores
    })

    # 5. Stage 4: Predict final severity using stacking meta-classifier
    final_preds = models.meta_classifier.predict(meta_df[META_FEATURES])
    final_probs = models.meta_classifier.predict_proba(meta_df[META_FEATURES])

    return {
        "final_predictions": final_preds,
        "final_probabilities": final_probs,
        "risk_scores": risk_scores,
        "base_probabilities": {
            "Logistic Regression": p_logreg,
            "K-Nearest Neighbors": p_knn,
            "Decision Tree": p_dt
        },
        "meta_features_df": meta_df
    }


def predict_single(feature_dict: dict):
    """Helper for a single record prediction."""
    df = pd.DataFrame([feature_dict])
    res = predict_pipeline(df)
    return {
        "prediction": int(res["final_predictions"][0]),
        "probabilities": res["final_probabilities"][0],
        "risk_score": float(res["risk_scores"][0]),
        "base_probabilities": {
            k: v[0] for k, v in res["base_probabilities"].items()
        },
        "meta_features": res["meta_features_df"].iloc[0].to_dict()
    }
