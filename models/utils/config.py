"""Small, shared configuration for the active pipeline."""

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

SEED = 19
TARGET = "casualty_severity"
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ["Fatal", "Serious", "Slight"]
SPLIT_FOLDS = 5
FATAL_TOLERANCE = 0.001
REGRESSION_TARGET = "number_of_casualties"
REGRESSION_FEATURES = [
    "number_of_vehicles", "day_of_week", "first_road_class", "road_type",
    "speed_limit", "junction_detail", "junction_control", "light_conditions",
    "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
    "special_conditions_at_site", "carriageway_hazards",
]

# These fields are known immediately after a collision.  Confirmed severity,
# casualty totals, and police attendance are intentionally excluded from the
# early-after-collision contract because they can be outcome proxies.
LOGISTIC_FEATURES = [
    "casualty_class", "sex_of_casualty", "age_of_casualty", "casualty_type",
    "casualty_imd_decile", "casualty_distance_banding", "number_of_vehicles",
    "day_of_week", "first_road_class", "road_type", "speed_limit",
    "junction_detail", "junction_control", "pedestrian_crossing",
    "light_conditions", "weather_conditions", "road_surface_conditions",
    "urban_or_rural_area", "vehicle_type", "vehicle_manoeuvre",
    "vehicle_direction_from", "vehicle_direction_to", "junction_location",
    "skidding_and_overturning", "hit_object_in_carriageway",
    "vehicle_leaving_carriageway", "hit_object_off_carriageway",
    "first_point_of_impact", "sex_of_driver", "age_of_driver",
    "engine_capacity_cc", "propulsion_code", "journey_purpose_of_driver",
]

KNN_FEATURES = [
    "casualty_class", "sex_of_casualty", "age_of_casualty", "casualty_type",
    "number_of_vehicles", "first_road_class", "road_type", "speed_limit",
    "light_conditions", "weather_conditions", "urban_or_rural_area",
    "vehicle_type", "vehicle_manoeuvre", "vehicle_leaving_carriageway",
    "first_point_of_impact", "skidding_and_overturning",
    "hit_object_in_carriageway", "hit_object_off_carriageway",
    "age_of_driver", "engine_capacity_cc",
]

DECISION_TREE_FEATURES = [
    "casualty_class", "sex_of_casualty", "age_of_casualty", "casualty_type",
    "number_of_vehicles", "day_of_week", "first_road_class", "road_type",
    "speed_limit", "junction_detail", "junction_control", "light_conditions",
    "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
    "vehicle_type", "vehicle_manoeuvre", "vehicle_direction_from",
    "vehicle_direction_to", "vehicle_leaving_carriageway",
    "first_point_of_impact", "hit_object_off_carriageway", "age_of_driver",
    "engine_capacity_cc", "journey_purpose_of_driver",
]

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
