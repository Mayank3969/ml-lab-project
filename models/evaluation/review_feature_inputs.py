"""Reproducible input ablations and grouped hyperparameter search for the review.

Reads raw data and existing split IDs. Writes only to a separate review directory.
Run from the repository root; see ``action plan.md`` for the interpretation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
TARGET = "casualty_severity"
KEY = "collision_index"
ATTENDANCE = "did_police_officer_attend_scene_of_accident"
COUNT = "number_of_casualties"
NUMERIC = {
    "age_of_casualty", "age_of_driver", "age_of_vehicle", "engine_capacity_cc",
    "speed_limit", "number_of_vehicles", COUNT,
    "age_band_of_casualty", "age_band_of_driver",
}
EXCLUDED = {
    TARGET, KEY, "collision_severity", "enhanced_severity_collision",
    "collision_injury_based", "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight", "enhanced_casualty_severity",
    "casualty_injury_based", "casualty_adjusted_severity_serious",
    "casualty_adjusted_severity_slight", "location_easting_osgr",
    "location_northing_osgr", "longitude", "latitude", "lsoa_of_accident_location",
    "lsoa_of_casualty", "lsoa_of_driver", "local_authority_district",
    "local_authority_ons_district", "local_authority_highway",
    "local_authority_highway_current", "police_force", "collision_ref_no",
    "vehicle_reference", "casualty_reference", "first_road_number",
    "second_road_number", "generic_make_model", "date", "time", "collision_year",
    "collision_year_col", "collision_ref_no_col", "collision_year_veh",
    "collision_ref_no_veh", "age_of_vehicle",
}
CURRENT = [
    "age_of_casualty", "engine_capacity_cc", "age_of_driver", "day_of_week",
    "casualty_type", "casualty_imd_decile", "vehicle_direction_to",
    "vehicle_direction_from", "driver_imd_decile", ATTENDANCE,
    "journey_purpose_of_driver", COUNT,
]
PERMUTATION = [
    "casualty_type", ATTENDANCE, "age_of_casualty", "engine_capacity_cc",
    "vehicle_leaving_carriageway", "age_of_driver", "speed_limit",
    "first_point_of_impact", "vehicle_manoeuvre", "urban_or_rural_area",
    "vehicle_direction_from", "vehicle_direction_to", "journey_purpose_of_driver",
    "first_road_class", "hit_object_off_carriageway", "number_of_vehicles", COUNT,
]
DOMAIN = [
    "casualty_class", "sex_of_casualty", "age_of_casualty", "casualty_type",
    "number_of_vehicles", COUNT, "day_of_week", "first_road_class", "road_type",
    "speed_limit", "junction_detail", "junction_control", "light_conditions",
    "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
    ATTENDANCE, "vehicle_type", "vehicle_manoeuvre", "vehicle_direction_from",
    "vehicle_direction_to", "vehicle_leaving_carriageway", "first_point_of_impact",
    "hit_object_off_carriageway", "age_of_driver", "engine_capacity_cc",
    "journey_purpose_of_driver",
]
COMPACT = [
    "casualty_type", "age_of_casualty", "age_of_driver", "engine_capacity_cc",
    "speed_limit", "urban_or_rural_area", "vehicle_manoeuvre",
    "vehicle_leaving_carriageway", "first_point_of_impact", "number_of_vehicles",
]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Unsupervised category frequencies fitted separately inside each fold."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X[c].value_counts(normalize=True).to_dict() for c in X]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([
            X[c].map(mapping).fillna(0).to_numpy(dtype=float)
            for c, mapping in zip(X.columns, self.maps_)
        ])


def load_data():
    raw = ROOT / "dataset/raw"
    cas = pd.read_csv(raw / "casualty.csv", dtype={KEY: str, "vehicle_reference": str}, low_memory=False)
    col = pd.read_csv(raw / "collision.csv", dtype={KEY: str}, low_memory=False)
    veh = pd.read_csv(raw / "vehicle.csv", dtype={KEY: str, "vehicle_reference": str}, low_memory=False)
    if cas[[KEY, "vehicle_reference", "casualty_reference"]].duplicated().any():
        raise ValueError("Duplicate casualty keys: investigate before joining.")
    merged = cas.merge(col, on=KEY, how="left", suffixes=("", "_col"), validate="many_to_one", indicator="_collision_join")
    if not merged["_collision_join"].eq("both").all():
        raise ValueError("Unmatched casualty/collision keys.")
    merged = merged.drop(columns="_collision_join").merge(
        veh, on=[KEY, "vehicle_reference"], how="left", suffixes=("", "_veh"),
        validate="many_to_one", indicator="_vehicle_join",
    )
    unmatched_vehicles = int(merged["_vehicle_join"].ne("both").sum())
    merged = merged.drop(columns="_vehicle_join")
    if len(merged) != len(cas):
        raise ValueError("Joining changed the casualty row count.")
    merged = merged.loc[merged[TARGET].isin([1, 2, 3])].copy()
    all_safe = [c for c in merged if c not in EXCLUDED and not c.endswith("_historic")]
    # Match the existing deterministic zero-variance exclusion, not target selection.
    all_safe = [c for c in all_safe if merged[c].nunique(dropna=False) > 1]
    for c in all_safe:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(-1)
    split_dir = ROOT / "dataset/processed"
    train_ids = set(pd.read_csv(split_dir / "merged_train.csv", usecols=[KEY], dtype={KEY: str})[KEY])
    test_ids = set(pd.read_csv(split_dir / "merged_test.csv", usecols=[KEY], dtype={KEY: str})[KEY])
    if train_ids & test_ids:
        raise ValueError("Train/test collision groups overlap.")
    if train_ids | test_ids != set(merged[KEY]):
        raise ValueError("Existing split IDs do not cover the raw casualty data.")
    train = merged.loc[merged[KEY].isin(train_ids)].reset_index(drop=True)
    test = merged.loc[merged[KEY].isin(test_ids)].reset_index(drop=True)
    sets = {
        "current12": CURRENT,
        "permutation17": PERMUTATION,
        "domain27": DOMAIN,
        "all_safe": all_safe,
        "compact10": COMPACT,
        "domain_no_attendance": [c for c in DOMAIN if c != ATTENDANCE],
        "domain_no_attendance_count": [c for c in DOMAIN if c not in [ATTENDANCE, COUNT]],
    }
    audit = {
        "raw_rows": {"casualty": len(cas), "collision": len(col), "vehicle": len(veh)},
        "merged_rows": len(merged), "train_rows": len(train), "test_rows": len(test),
        "train_groups": len(train_ids), "test_groups": len(test_ids),
        "group_overlap": 0, "unmatched_vehicle_rows": unmatched_vehicles,
        "train_class_counts": train[TARGET].value_counts().sort_index().to_dict(),
        "test_class_counts": test[TARGET].value_counts().sort_index().to_dict(),
        "input_sha256": {name: hashlib.sha256((raw / f"{name}.csv").read_bytes()).hexdigest()
                         for name in ["casualty", "collision", "vehicle"]},
        "feature_sets": sets,
    }
    return train, test, sets, audit


def make_pipeline(model, features, encoding="frequency"):
    nums = [c for c in features if c in NUMERIC]
    cats = [c for c in features if c not in NUMERIC]
    categorical = (FrequencyEncoder() if encoding == "frequency" else
                   OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True))
    # Frequency branch deliberately reproduces existing -1 and global standardization
    # semantics. One-hot branch treats numeric -1 as missing and imputes within folds.
    numeric = ("passthrough" if encoding == "frequency" else Pipeline([
        ("imputer", SimpleImputer(missing_values=-1, strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]))
    pre = ColumnTransformer([
        ("num", numeric, nums), ("cat", categorical, cats),
    ], sparse_threshold=1.0)
    if model == "logreg":
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)
    elif model == "dt":
        clf = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, class_weight=None, random_state=42)
    else:
        clf = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto", n_jobs=1)
    steps = [("pre", pre)]
    if model != "dt" and encoding == "frequency":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def metrics(y, prediction):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=[1, 2, 3], average="macro", zero_division=0)),
        "fatal_f1": float(f1_score(y == 1, prediction == 1, zero_division=0)),
        "fatal_recall": float(recall_score(y, prediction, labels=[1], average=None, zero_division=0)[0]),
        "serious_recall": float(recall_score(y, prediction, labels=[2], average=None, zero_division=0)[0]),
    }


def screen(train, sets, args):
    """Select inputs only on development-set CV; the test labels are unused."""
    cv = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42).split(
        train, train[TARGET], train[KEY]))
    rows = []
    for name in args.feature_sets:
        features = sets[name]
        for model in args.models:
            for encoding in args.encodings:
                # All-safe one-hot KNN is intentionally outside the compact screen.
                if model == "knn" and encoding == "onehot" and name != "compact10":
                    continue
                scores = []
                start = time.time()
                for i, j in cv:
                    assert not set(train.iloc[i][KEY]) & set(train.iloc[j][KEY])
                    estimator = make_pipeline(model, features, encoding)
                    estimator.fit(train.iloc[i][features], train.iloc[i][TARGET])
                    scores.append(metrics(train.iloc[j][TARGET], estimator.predict(train.iloc[j][features])))
                row = {"model": model, "feature_set": name, "features": len(features), "encoding": encoding}
                for metric in scores[0]:
                    values = [s[metric] for s in scores]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_std"] = float(np.std(values))
                row["seconds"] = round(time.time() - start, 2)
                rows.append(row)
                print(json.dumps(row), flush=True)
    return pd.DataFrame(rows)


def tune(train, test, sets, args, output):
    features = sets[args.tune_features]
    if not args.allow_attendance:
        features = [c for c in features if c != ATTENDANCE]
    if not args.allow_casualty_count:
        features = [c for c in features if c != COUNT]
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "macro_f1": make_scorer(
        f1_score, labels=[1, 2, 3], average="macro", zero_division=0),
        "balanced_accuracy": "balanced_accuracy"}
    summaries = []
    for model in args.models:
        estimator = make_pipeline(model, features, args.tune_encoding)
        if model == "logreg":
            space = {"clf__C": loguniform(1e-3, 1e2), "clf__class_weight": [None, "balanced", {1: 5, 2: 2, 3: 1}]}
        elif model == "dt":
            space = {
                "clf__max_depth": [3, 5, 8, 12, 16, 20, None],
                "clf__min_samples_leaf": [1, 5, 10, 20, 50, 100],
                "clf__min_samples_split": [2, 10, 30, 100],
                "clf__max_features": [None, "sqrt", 0.7],
                "clf__criterion": ["gini", "entropy"],
                "clf__ccp_alpha": [0, 1e-5, 1e-4, 1e-3],
                "clf__class_weight": [None, "balanced", {1: 5, 2: 2, 3: 1}],
            }
        else:
            space = {"clf__n_neighbors": [5, 15, 31, 51, 101, 201],
                     "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}
        search = RandomizedSearchCV(
            estimator, space, n_iter=args.iterations, scoring=scoring, refit=args.refit,
            cv=cv, random_state=42, n_jobs=args.jobs, pre_dispatch=args.jobs,
            return_train_score=True, error_score="raise",
        )
        search.fit(train[features], train[TARGET], groups=train[KEY])
        stem = f"{model}_{args.refit}"
        pd.DataFrame(search.cv_results_).to_csv(output / f"{stem}_search.csv", index=False)
        joblib.dump(search.best_estimator_, output / f"{stem}_pipeline.pkl")
        summary = {
            "model": model, "refit_metric": args.refit, "features": features,
            "encoding": args.tune_encoding, "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "test_evaluated": args.evaluate_test,
        }
        # Opt in once after feature/parameter choices are frozen. No test-based search.
        if args.evaluate_test:
            prediction = search.predict(test[features])
            summary["test_metrics"] = metrics(test[TARGET], prediction)
            summary["test_report"] = classification_report(
                test[TARGET], prediction, labels=[1, 2, 3],
                target_names=["Fatal", "Serious", "Slight"], output_dict=True, zero_division=0)
            summary["test_confusion_matrix"] = confusion_matrix(test[TARGET], prediction, labels=[1, 2, 3]).tolist()
            summary["accuracy_90_reached"] = summary["test_metrics"]["accuracy"] >= 0.9
        print(json.dumps(summary), flush=True)
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["features", "tune"], default="features")
    parser.add_argument("--models", nargs="+", choices=["logreg", "dt", "knn"], default=["logreg", "dt", "knn"])
    parser.add_argument("--feature-sets", nargs="+", default=["current12", "permutation17", "domain27", "all_safe", "compact10", "domain_no_attendance", "domain_no_attendance_count"])
    parser.add_argument("--encodings", nargs="+", choices=["frequency", "onehot"], default=["frequency"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--mode-name", default="feature_screen")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/evaluation/results/input_review")
    parser.add_argument("--tune-features", default="domain_no_attendance_count")
    parser.add_argument("--tune-encoding", choices=["frequency", "onehot"], default="onehot")
    parser.add_argument("--refit", choices=["accuracy", "macro_f1", "balanced_accuracy"], default="macro_f1")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--allow-attendance", action="store_true")
    parser.add_argument("--allow-casualty-count", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.folds < 2 or args.iterations < 1 or args.jobs < 1:
        parser.error("folds >= 2, iterations >= 1, and jobs >= 1 are required")
    train, test, sets, audit = load_data()
    for name in args.feature_sets + [args.tune_features]:
        if name not in sets:
            parser.error(f"Unknown feature set {name!r}. Choices: {list(sets)}")
    output = args.output_dir.resolve()
    if output == (ROOT / "models/saved_models").resolve() or output == (ROOT / "dataset/processed").resolve():
        parser.error("Use a separate experiment output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.mode == "features":
        screen(train, sets, args).to_csv(output / f"{args.mode_name}.csv", index=False)
    else:
        (output / f"{args.mode_name}.json").write_text(
            json.dumps(tune(train, test, sets, args, output), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
