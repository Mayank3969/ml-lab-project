from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]

# Paths
PROCESSED   = ROOT / "dataset" / "processed"
MODELS_DIR  = ROOT / "models" / "saved_models"
RESULTS_DIR = ROOT / "models" / "evaluation" / "results"
STAGE2_DIR  = ROOT / "models" / "stage2_regression"
STAGE3_DIR  = ROOT / "models" / "stage3a_base_classifiers"
STAGE4_DIR  = ROOT / "models" / "stage4_meta_classifier"

# Create dirs if they don't exist
for d in [PROCESSED, MODELS_DIR, RESULTS_DIR, STAGE2_DIR, STAGE3_DIR, STAGE4_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants
SEED = 42
TARGET = "casualty_severity"
STAGE1_TARGET = "number_of_casualties"

# Stage 1 dynamically selected features path
SELECTED_FEATURES_PATH = PROCESSED / "selected_features.json"

# Classification
CLASS_NAMES = ["Fatal", "Serious", "Slight"]
CLASS_LABELS = [1, 2, 3]

# Stage 3A Probability Columns (exactly 9 columns)
STAGE3_PROBA_COLS = [
    "P_logreg_fatal", "P_logreg_serious", "P_logreg_slight",
    "P_knn_fatal",    "P_knn_serious",    "P_knn_slight",
    "P_dt_fatal",     "P_dt_serious",     "P_dt_slight"
]

# Stage 4 Meta-classifier input features (9 base model probabilities + risk_score = 10 columns)
STAGE4_FEATURES = STAGE3_PROBA_COLS + ["risk_score"]