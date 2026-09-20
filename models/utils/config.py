from pathlib import Path

# Project root (ml-lab-project/)
ROOT = Path(__file__).resolve().parents[2]

# Paths
PROCESSED   = ROOT / "dataset" / "processed"
MODELS_DIR  = ROOT / "models" / "saved_models"
RESULTS_DIR = ROOT / "models" / "evaluation" / "results"
STAGE2_DIR  = ROOT / "models" / "stage2_regression"

# Create dirs if they don't exist
for d in [PROCESSED, MODELS_DIR, RESULTS_DIR, STAGE2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants
SEED = 42
TARGET = "casualty_severity"
STAGE1_TARGET = "number_of_casualties"

# Stage 1 features (collision-level, environmental/road)
STAGE1_NUM = ["speed_limit"]
STAGE1_CAT = ["road_type", "junction_detail", "light_conditions", "weather_conditions"]
STAGE1_FEATURES = STAGE1_NUM + STAGE1_CAT

# Classification
CLASS_NAMES = ["Fatal", "Serious", "Slight"]
CLASS_LABELS = [1, 2, 3]