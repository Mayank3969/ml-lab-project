"""Build the leakage-safe, immediately-after-collision modelling tables.

This replaces the old feature-ablation scripts.  It keeps collision groups
together, preserves the measured class proportions, and writes raw numeric
STATS19 codes.  Model-specific encoders are fitted later inside each notebook.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from utils.config import (
    CLASS_LABELS,
    DECISION_TREE_FEATURES,
    FATAL_TOLERANCE,
    FEATURE_MANIFEST_PATH,
    KNN_FEATURES,
    LOGISTIC_FEATURES,
    SEED,
    REGRESSION_FEATURES,
    REGRESSION_MODEL_FEATURES,
    REGRESSION_TARGET,
    REGRESSION_TEST_PATH,
    REGRESSION_TRAIN_PATH,
    SPLIT_AUDIT_PATH,
    SPLIT_FOLDS,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)

RAW = ROOT / "dataset" / "raw"

REQUIRED_OUTPUTS = [
    TRAIN_PATH,
    TEST_PATH,
    REGRESSION_TRAIN_PATH,
    REGRESSION_TEST_PATH,
    FEATURE_MANIFEST_PATH,
    SPLIT_AUDIT_PATH,
]

# These fields either identify a row, encode the label, or are unavailable in
# the immediately-after-collision prediction contract.
DROP_COLUMNS = {
    "collision_year", "collision_ref_no", "vehicle_reference", "casualty_reference",
    "date", "time", "location_easting_osgr", "location_northing_osgr",
    "longitude", "latitude", "police_force", "local_authority_district",
    "local_authority_ons_district", "local_authority_highway",
    "local_authority_highway_current", "first_road_number", "second_road_number",
    "generic_make_model", "lsoa_of_accident_location", "lsoa_of_casualty",
    "lsoa_of_driver", "age_of_vehicle", "did_police_officer_attend_scene_of_accident",
    "number_of_casualties", "collision_severity", "enhanced_severity_collision",
    "collision_injury_based", "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight", "enhanced_casualty_severity",
    "casualty_injury_based", "casualty_adjusted_severity_serious",
    "casualty_adjusted_severity_slight",
}


def read_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three STATS19 tables with stable join-key types."""
    casualty = pd.read_csv(
        RAW / "casualty.csv",
        dtype={"collision_index": str, "vehicle_reference": str},
        low_memory=False,
    )
    collision = pd.read_csv(RAW / "collision.csv", dtype={"collision_index": str}, low_memory=False)
    vehicle = pd.read_csv(
        RAW / "vehicle.csv",
        dtype={"collision_index": str, "vehicle_reference": str},
        low_memory=False,
    )
    return casualty, collision, vehicle


def join_tables(casualty: pd.DataFrame, collision: pd.DataFrame, vehicle: pd.DataFrame) -> pd.DataFrame:
    """Join casualty, collision, and vehicle rows without multiplying records."""
    if casualty[["collision_index", "vehicle_reference", "casualty_reference"]].duplicated().any():
        raise ValueError("Duplicate casualty keys found before joining")
    if vehicle[["collision_index", "vehicle_reference"]].duplicated().any():
        raise ValueError("Duplicate vehicle keys would multiply casualty rows")

    merged = casualty.merge(
        collision,
        on="collision_index",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_collision"),
    )
    merged = merged.merge(
        vehicle,
        on=["collision_index", "vehicle_reference"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_vehicle"),
    )
    if len(merged) != len(casualty):
        raise ValueError(f"Join changed row count: {len(casualty):,} -> {len(merged):,}")
    return merged


def clean_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage/ID fields and coerce remaining model inputs to numbers."""
    merged = merged.loc[merged[TARGET].isin(CLASS_LABELS)].copy()
    historic = {column for column in merged.columns if column.endswith("_historic")}
    drop = (DROP_COLUMNS | historic) - {"collision_index", TARGET}
    for column in merged.columns:
        if column.endswith(("_collision", "_vehicle")):
            base_name = column.rsplit("_", 1)[0]
            if base_name in DROP_COLUMNS:
                drop.add(column)
    keep = [
        column for column in merged.columns
        if column not in drop and column not in {"collision_index", TARGET}
    ]
    cleaned = merged[["collision_index", TARGET] + keep].copy()
    for column in cleaned.columns:
        if column not in {"collision_index", TARGET}:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(-1)
    zero_variance = [
        column for column in cleaned.columns
        if column not in {"collision_index", TARGET} and cleaned[column].nunique(dropna=False) <= 1
    ]
    return cleaned.drop(columns=zero_variance)


def choose_group_split(frame: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Choose the stable grouped seed found during the class-balance audit."""
    y = frame[TARGET].astype(int)
    groups = frame["collision_index"].astype(str)
    global_rates = y.value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
    best: tuple[tuple[float, float], int, list[tuple[np.ndarray, np.ndarray]]] | None = None

    # A small deterministic seed search is cheap compared with model training
    # and prevents a new raw-data release from inheriting a poor first fold.
    for seed in range(SEED + 1):
        splitter = StratifiedGroupKFold(n_splits=SPLIT_FOLDS, shuffle=True, random_state=seed)
        splits = list(splitter.split(frame, y, groups))
        deviations = []
        valid = True
        for _, validation_index in splits:
            rates = y.iloc[validation_index].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
            deviations.extend((rates - global_rates).abs().tolist())
            if y.iloc[validation_index].nunique() != len(CLASS_LABELS):
                valid = False
        if not valid:
            continue
        score = (max(deviations), sum(deviations))
        if best is None or score < best[0]:
            best = (score, seed, splits)
    if best is None:
        raise ValueError("Could not create grouped folds containing all classes")
    return best[2], best[1]


def audit_splits(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]], seed: int) -> pd.DataFrame:
    """Write counts, rates, and group-overlap checks for every partition."""
    rows = []
    global_rates = frame[TARGET].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
    groups = frame["collision_index"].astype(str)
    for fold, (train_index, validation_index) in enumerate(splits):
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        if train_groups & validation_groups:
            raise ValueError(f"Collision group overlap in fold {fold}")
        validation = frame.iloc[validation_index]
        counts = validation[TARGET].value_counts().reindex(CLASS_LABELS, fill_value=0)
        rates = validation[TARGET].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
        row = {
            "split_seed": seed,
            "fold": fold,
            "rows": len(validation),
            "collision_groups": validation["collision_index"].nunique(),
            "fatal_count": int(counts.loc[1]),
            "serious_count": int(counts.loc[2]),
            "slight_count": int(counts.loc[3]),
            "fatal_rate": float(rates.loc[1]),
            "serious_rate": float(rates.loc[2]),
            "slight_rate": float(rates.loc[3]),
            "fatal_rate_delta": float(rates.loc[1] - global_rates.loc[1]),
            "group_overlap": 0,
        }
        rows.append(row)
    audit = pd.DataFrame(rows)
    for label, column in zip(CLASS_LABELS, ["fatal_rate", "serious_rate", "slight_rate"]):
        global_rate = float(global_rates.loc[label])
        if (audit[column] - global_rate).abs().max() > FATAL_TOLERANCE:
            raise ValueError(f"Class {label} rate exceeds the configured tolerance")
    if audit["fatal_rate_delta"].abs().max() > FATAL_TOLERANCE:
        raise ValueError("Fatal class balance exceeds the configured tolerance")
    return audit


def build_regression_tables(
    collision: pd.DataFrame,
    train_groups: set[str],
    test_groups: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one-row-per-collision regression tables for risk-score tuning."""
    missing = sorted(set(REGRESSION_FEATURES + [REGRESSION_TARGET]) - set(collision.columns))
    if missing:
        raise ValueError(f"Missing regression columns in collision table: {missing}")
    regression = collision[["collision_index"] + REGRESSION_FEATURES + [REGRESSION_TARGET]].copy()
    regression = regression.drop_duplicates(subset=["collision_index"])
    for column in REGRESSION_FEATURES + [REGRESSION_TARGET]:
        regression[column] = pd.to_numeric(regression[column], errors="coerce").fillna(-1)
    train = regression[regression["collision_index"].isin(train_groups)].reset_index(drop=True)
    test = regression[regression["collision_index"].isin(test_groups)].reset_index(drop=True)
    if len(train) + len(test) != len(regression):
        raise ValueError("Regression groups do not cover every collision")
    return train, test


def save_outputs(
    frame: pd.DataFrame,
    collision: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> None:
    """Save train/test data, fold labels, feature manifests, and the audit."""
    audit = audit_splits(frame, splits, seed)
    test_index = splits[0][1]
    train_index = splits[0][0]
    train = frame.iloc[train_index].copy()
    test = frame.iloc[test_index].copy()
    regression_train, regression_test = build_regression_tables(
        collision,
        set(train["collision_index"]),
        set(test["collision_index"]),
    )
    partition_audit = pd.DataFrame([
        {
            "partition": "train",
            "split_seed": seed,
            "fold": "partition",
            "rows": len(train),
            "collision_groups": train["collision_index"].nunique(),
            "fatal_count": int(train[TARGET].eq(1).sum()),
            "serious_count": int(train[TARGET].eq(2).sum()),
            "slight_count": int(train[TARGET].eq(3).sum()),
            "fatal_rate": float(train[TARGET].eq(1).mean()),
            "serious_rate": float(train[TARGET].eq(2).mean()),
            "slight_rate": float(train[TARGET].eq(3).mean()),
            "fatal_rate_delta": float(train[TARGET].eq(1).mean() - frame[TARGET].eq(1).mean()),
            "group_overlap": int(bool(set(train["collision_index"]) & set(test["collision_index"]))),
        },
        {
            "partition": "test",
            "split_seed": seed,
            "fold": "partition",
            "rows": len(test),
            "collision_groups": test["collision_index"].nunique(),
            "fatal_count": int(test[TARGET].eq(1).sum()),
            "serious_count": int(test[TARGET].eq(2).sum()),
            "slight_count": int(test[TARGET].eq(3).sum()),
            "fatal_rate": float(test[TARGET].eq(1).mean()),
            "serious_rate": float(test[TARGET].eq(2).mean()),
            "slight_rate": float(test[TARGET].eq(3).mean()),
            "fatal_rate_delta": float(test[TARGET].eq(1).mean() - frame[TARGET].eq(1).mean()),
            "group_overlap": int(bool(set(train["collision_index"]) & set(test["collision_index"]))),
        },
    ])
    if partition_audit["group_overlap"].any():
        raise ValueError("Collision groups overlap between train and test")
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    regression_train.to_csv(REGRESSION_TRAIN_PATH, index=False)
    regression_test.to_csv(REGRESSION_TEST_PATH, index=False)
    pd.concat([partition_audit, audit.assign(partition="fold")], ignore_index=True).to_csv(
        SPLIT_AUDIT_PATH, index=False
    )

    manifest = {
        "prediction_timing": "immediately_after_collision",
        "target": TARGET,
        "group_column": "collision_index",
        "split_seed": seed,
        "class_labels": CLASS_LABELS,
        "global_class_counts": frame[TARGET].value_counts().sort_index().to_dict(),
        "excluded_columns": sorted(DROP_COLUMNS),
        "model_features": {
            "logistic_regression": LOGISTIC_FEATURES,
            "knn": KNN_FEATURES,
            "decision_tree": DECISION_TREE_FEATURES,
        },
        "regression_features": REGRESSION_FEATURES,
        "regression_model_features": REGRESSION_MODEL_FEATURES,
        "regression_target": REGRESSION_TARGET,
        "all_safe_columns": [
            column for column in frame.columns if column not in {"collision_index", TARGET}
        ],
        "split_audit": pd.concat(
            [partition_audit, audit.assign(partition="fold")],
            ignore_index=True,
        ).to_dict(orient="records"),
    }
    FEATURE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved {len(train):,} training rows to {TRAIN_PATH}")
    print(f"Saved {len(test):,} locked test rows to {TEST_PATH}")
    print(f"Fatal rate: {frame[TARGET].eq(1).mean():.4%}")
    print(f"Regression train: {REGRESSION_TRAIN_PATH}")
    print(f"Regression test: {REGRESSION_TEST_PATH}")
    print(f"Fold audit: {SPLIT_AUDIT_PATH}")


def main() -> None:
    """Run the complete data preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild processed files even when they exist")
    args = parser.parse_args()
    if not args.force and all(path.exists() for path in REQUIRED_OUTPUTS):
        print("Prepared database already exists; reusing processed files.")
        print(f"Train data: {TRAIN_PATH}")
        print(f"Test data: {TEST_PATH}")
        print("Use `python pipeline.py --clean` to rebuild it from raw CSV files.")
        return
    casualty, collision, vehicle = read_raw_tables()
    merged = clean_features(join_tables(casualty, collision, vehicle))
    splits, seed = choose_group_split(merged)
    save_outputs(merged, collision, splits, seed)


if __name__ == "__main__":
    main()
