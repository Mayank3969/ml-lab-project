"""Small, shared configuration for the active pipeline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
PROCESSED = ROOT / "dataset" / "processed"
MODELS_DIR = ROOT / "models" / "saved_models"
RESULTS_DIR = ROOT / "models" / "evaluation" / "results"
HYPERPARAMETER_DIR = ROOT / "hyper parameter"
HYPERPARAMETER_RESULTS = HYPERPARAMETER_DIR / "results"
HYPERPARAMETER_BEST = HYPERPARAMETER_DIR / "best"

for directory in [PROCESSED, MODELS_DIR, RESULTS_DIR, HYPERPARAMETER_RESULTS, HYPERPARAMETER_BEST]:
    directory.mkdir(parents=True, exist_ok=True)


def _load_best_features(model_name: str) -> list[str]:
    """Load and validate the feature contract selected for one model."""
    path = HYPERPARAMETER_BEST / f"{model_name}_best_params.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing best-parameter file for {model_name}: {path}"
        ) from error

    if payload.get("model") != model_name:
        raise ValueError(f"{path} declares model {payload.get('model')!r}, expected {model_name!r}")
    features = payload.get("features")
    if not isinstance(features, list) or not features or not all(isinstance(feature, str) for feature in features):
        raise ValueError(f"{path} must contain a non-empty string list named 'features'")
    if len(features) != len(set(features)):
        raise ValueError(f"{path} contains duplicate feature names")
    return features

SEED = 19
TARGET = "casualty_severity"
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ["Fatal", "Serious", "Slight"]
SPLIT_FOLDS = 5
FATAL_TOLERANCE = 0.001
REGRESSION_TARGET = "number_of_casualties"

# The JSON files produced by the tuning notebooks are the source of truth for
# raw model inputs.  Keeping these lists here makes every training/inference
# module import the same ordered contract instead of silently drifting from the
# selected feature set.
BEST_FEATURES = {
    model_name: _load_best_features(model_name)
    for model_name in [
        "linear_regression",
        "decision_tree_regressor",
        "logistic_regression",
        "knn",
        "decision_tree",
    ]
}

REGRESSION_MODEL_FEATURES = {
    "linear_regression": BEST_FEATURES["linear_regression"],
    "decision_tree_regressor": BEST_FEATURES["decision_tree_regressor"],
}
# The merge stage keeps the union because the two regressors may legitimately
# have different selected inputs; each regressor later receives only its own
# list from REGRESSION_MODEL_FEATURES.
REGRESSION_FEATURES = list(dict.fromkeys(
    feature
    for features in REGRESSION_MODEL_FEATURES.values()
    for feature in features
))

# These fields are known immediately after a collision.  Confirmed severity,
# casualty totals, and police attendance are intentionally excluded from the
# early-after-collision contract because they can be outcome proxies.
LOGISTIC_FEATURES = BEST_FEATURES["logistic_regression"]
KNN_FEATURES = BEST_FEATURES["knn"]
DECISION_TREE_FEATURES = BEST_FEATURES["decision_tree"]

MODEL_FEATURES = {
    "logistic_regression": LOGISTIC_FEATURES,
    "knn": KNN_FEATURES,
    "decision_tree": DECISION_TREE_FEATURES,
}

META_FEATURES = [
    "logistic_regression_fatal", "logistic_regression_serious",
    "logistic_regression_slight", "knn_fatal", "knn_serious", "knn_slight",
    "decision_tree_fatal", "decision_tree_serious", "decision_tree_slight",
]
RISK_FEATURES = ["risk_score_linear_regression", "risk_score_decision_tree_regressor"]
META_CANDIDATE_FEATURES = META_FEATURES + RISK_FEATURES

FEATURE_MANIFEST_PATH = PROCESSED / "feature_manifest.json"
SPLIT_AUDIT_PATH = PROCESSED / "split_audit.csv"
TRAIN_PATH = PROCESSED / "merged_train.csv"
TEST_PATH = PROCESSED / "merged_test.csv"
REGRESSION_TRAIN_PATH = PROCESSED / "regression_train.csv"
REGRESSION_TEST_PATH = PROCESSED / "regression_test.csv"
