"""Create the static data, OOF, and model-diagnostic graphs.

The script is intentionally separate from Streamlit.  Run it manually after
the relevant pipeline artifacts exist:

    python models/stage1_merge/create_graphs.py

Images are written to ``docs/graphs`` and the dashboard only reads those
already-rendered files.  ``--scope data`` can be used immediately after the
merge stage; ``--scope all`` is the normal command after a complete pipeline
run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import clean_features, join_tables, read_raw_tables
from utils.config import (
    CLASS_LABELS,
    CLASS_NAMES,
    MODEL_FEATURES,
    MODELS_DIR,
    PROCESSED,
    REGRESSION_MODEL_FEATURES,
    RESULTS_DIR,
    TARGET,
)
from utils.preprocessing import grouped_splits


GRAPH_DIR = ROOT / "docs" / "graphs"
TRAIN_PATH = PROCESSED / "merged_train.csv"
TEST_PATH = PROCESSED / "merged_test.csv"
META_TRAIN_PATH = PROCESSED / "meta_train.csv"
META_TEST_PATH = PROCESSED / "meta_test.csv"
REGRESSION_TRAIN_PATH = PROCESSED / "regression_train.csv"
REGRESSION_TEST_PATH = PROCESSED / "regression_test.csv"

MODEL_ORDER = ["logistic_regression", "knn", "decision_tree"]
ALL_MODEL_ORDER = MODEL_ORDER + ["meta_ensemble_3"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "meta_ensemble_3": "Meta Ensemble",
}
RISK_MODEL_ORDER = ["linear_regression", "decision_tree_regressor"]
RISK_LABELS = {
    "linear_regression": "Linear Regression",
    "decision_tree_regressor": "Decision Tree Regressor",
}
COLORS = {1: "#b2182b", 2: "#ef8a62", 3: "#2166ac"}
SEVERITY_LABELS = dict(zip(CLASS_LABELS, CLASS_NAMES))
PROBABILITY_COLUMNS = {
    model_name: [f"{model_name}_fatal", f"{model_name}_serious", f"{model_name}_slight"]
    for model_name in MODEL_ORDER
}

GRAPH_CATALOG = [
    ("data", "class_distribution.png", "Class distribution"),
    ("data", "missing_values_by_table.png", "Missing values by source table"),
    ("data", "severity_by_casualty_features.png", "Severity by casualty features"),
    ("data", "severity_by_collision_conditions.png", "Severity by collision conditions"),
    ("data", "severity_by_vehicle_impact.png", "Severity by vehicle and impact features"),
    ("data", "numeric_features_by_severity.png", "Numeric features by severity"),
    ("data", "collision_group_size_by_severity.png", "Collision group size by severity"),
    ("derived", "train_test_class_balance.png", "Train/test class balance"),
    ("derived", "oof_fold_class_balance.png", "OOF fold class balance"),
    ("derived", "model_feature_contracts.png", "Model feature contracts"),
    ("derived", "risk_scores_by_severity.png", "Risk scores by severity"),
    ("derived", "oof_predicted_vs_actual_distribution.png", "OOF predicted versus actual classes"),
    ("derived", "oof_fold_metrics.png", "OOF fold metrics"),
    ("derived", "oof_confusion_matrices.png", "OOF confusion matrices"),
    ("derived", "probability_calibration.png", "OOF probability calibration"),
    ("model", "model_metric_comparison.png", "Model metric comparison"),
    ("model", "per_class_metrics.png", "Per-class test metrics"),
    ("model", "test_confusion_matrices.png", "Test confusion matrices"),
    ("model", "precision_recall_curves.png", "Test precision-recall curves"),
    ("model", "regressor_diagnostics.png", "Risk regressor diagnostics"),
    ("model", "meta_feature_coefficients.png", "Meta-model feature coefficients"),
]

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, filename: str) -> None:
    """Save one finished figure with a consistent static format."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {GRAPH_DIR / filename}")


def _target_names(frame: pd.DataFrame) -> pd.Series:
    """Return display labels for the numeric target column."""
    return frame[TARGET].astype(int).map(SEVERITY_LABELS).fillna("Unknown")


def _top_categories(frame: pd.DataFrame, column: str, limit: int = 10) -> pd.Series:
    """Keep the common values and group the long tail as Other."""
    values = frame[column].astype(str).fillna("Missing")
    common = set(values.value_counts().head(limit).index)
    return values.where(values.isin(common), "Other")


def _severity_rates(frame: pd.DataFrame, column: str, limit: int = 10) -> pd.DataFrame:
    """Calculate row-normalized severity rates for one categorical column."""
    values = _top_categories(frame, column, limit)
    rates = pd.crosstab(values, frame[TARGET].astype(int), normalize="index")
    return rates.reindex(columns=CLASS_LABELS, fill_value=0.0)


def _plot_rate_panels(
    frame: pd.DataFrame,
    columns: list[str],
    filename: str,
    title: str,
) -> None:
    """Plot one normalized severity-rate panel per categorical feature."""
    available = [column for column in columns if column in frame.columns]
    columns_per_row = 3
    rows = max(1, int(np.ceil(len(available) / columns_per_row)))
    fig, axes = plt.subplots(rows, columns_per_row, figsize=(18, 5 * rows))
    axes = np.asarray(axes).ravel()
    for axis, column in zip(axes, available):
        rates = _severity_rates(frame, column)
        rates.columns = [SEVERITY_LABELS[label] for label in CLASS_LABELS]
        rates.plot(
            kind="bar",
            stacked=True,
            ax=axis,
            color=[COLORS[label] for label in CLASS_LABELS],
            width=0.85,
        )
        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("Share within feature value")
        axis.tick_params(axis="x", labelrotation=45)
        axis.legend().set_visible(False)
    for axis in axes[len(available):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, y=1.01, fontsize=16)
    _save(fig, filename)


def _read_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read raw tables, the validated join, and the cleaned processed frame."""
    casualty, collision, vehicle = read_raw_tables()
    merged = join_tables(casualty, collision, vehicle)
    processed = clean_features(merged)
    return casualty, collision, vehicle, merged, processed


def create_data_graphs() -> None:
    """Create the seven graphs based on raw and merged data."""
    casualty, collision, vehicle, raw_merged, merged = _read_source_data()

    counts = merged[TARGET].astype(int).value_counts().reindex(CLASS_LABELS, fill_value=0)
    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        [SEVERITY_LABELS[label] for label in CLASS_LABELS],
        counts.to_numpy(),
        color=[COLORS[label] for label in CLASS_LABELS],
    )
    axis.set_title("Casualty severity distribution")
    axis.set_ylabel("Rows")
    axis.set_xlabel("Severity")
    axis.bar_label(bars, labels=[f"{value:,}" for value in counts], padding=3)
    for bar, value in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value / counts.sum():.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _save(fig, "class_distribution.png")

    missing_rows = []
    for table_name, table in {
        "Casualty": casualty,
        "Collision": collision,
        "Vehicle": vehicle,
        "Merged": raw_merged,
    }.items():
        missing = (table.isna().mean() * 100).sort_values(ascending=False).head(12)
        for column, percentage in missing.items():
            if percentage > 0:
                missing_rows.append({"table": table_name, "column": column, "missing_percent": percentage})
    missing_frame = pd.DataFrame(missing_rows)
    fig, axis = plt.subplots(figsize=(14, 8))
    if missing_frame.empty:
        axis.text(0.5, 0.5, "No missing values found", ha="center", va="center")
        axis.axis("off")
    else:
        sns.barplot(
            data=missing_frame,
            x="missing_percent",
            y="column",
            hue="table",
            ax=axis,
        )
        axis.set_title("Highest missing-value percentages by source table")
        axis.set_xlabel("Missing rows (%)")
        axis.set_ylabel("")
    _save(fig, "missing_values_by_table.png")

    _plot_rate_panels(
        merged,
        ["casualty_class", "sex_of_casualty", "casualty_type", "age_band_of_casualty", "casualty_distance_banding"],
        "severity_by_casualty_features.png",
        "Severity rates by casualty features",
    )
    _plot_rate_panels(
        merged,
        [
            "first_road_class", "road_type", "speed_limit", "junction_detail",
            "junction_control", "light_conditions", "weather_conditions",
            "road_surface_conditions", "urban_or_rural_area",
        ],
        "severity_by_collision_conditions.png",
        "Severity rates by collision conditions",
    )
    _plot_rate_panels(
        merged,
        [
            "vehicle_type", "vehicle_manoeuvre", "first_point_of_impact",
            "skidding_and_overturning", "hit_object_in_carriageway",
            "vehicle_leaving_carriageway", "hit_object_off_carriageway",
        ],
        "severity_by_vehicle_impact.png",
        "Severity rates by vehicle and impact features",
    )

    numeric_columns = [
        "age_of_casualty", "age_of_driver", "engine_capacity_cc", "speed_limit", "number_of_vehicles",
    ]
    available_numeric = [column for column in numeric_columns if column in merged.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = np.asarray(axes).ravel()
    for axis, column in zip(axes, available_numeric):
        values = pd.to_numeric(merged[column], errors="coerce")
        plot_frame = pd.DataFrame({"severity": _target_names(merged), "value": values})
        plot_frame = plot_frame.loc[plot_frame["value"].notna() & (plot_frame["value"] >= 0)]
        sns.boxplot(
            data=plot_frame,
            x="severity",
            y="value",
            order=CLASS_NAMES,
            ax=axis,
            palette=[COLORS[label] for label in CLASS_LABELS],
            hue="severity",
            legend=False,
        )
        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("Value")
    for axis in axes[len(available_numeric):]:
        axis.axis("off")
    fig.suptitle("Numeric input distributions by severity", y=1.02, fontsize=16)
    _save(fig, "numeric_features_by_severity.png")

    group_sizes = merged.groupby("collision_index").size().rename("casualties_in_collision")
    group_frame = merged[["collision_index", TARGET]].copy()
    group_frame["casualties_in_collision"] = group_frame["collision_index"].map(group_sizes)
    group_frame["severity"] = _target_names(group_frame)
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=group_frame,
        x="severity",
        y="casualties_in_collision",
        order=CLASS_NAMES,
        palette=[COLORS[label] for label in CLASS_LABELS],
        hue="severity",
        legend=False,
        ax=axis,
    )
    axis.set_title("Number of casualties in a collision by casualty severity")
    axis.set_xlabel("Severity")
    axis.set_ylabel("Casualty rows in collision group")
    _save(fig, "collision_group_size_by_severity.png")


def _read_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read processed classifier and stacking tables."""
    train = pd.read_csv(TRAIN_PATH, dtype={"collision_index": str})
    test = pd.read_csv(TEST_PATH, dtype={"collision_index": str})
    meta_train = pd.read_csv(META_TRAIN_PATH, dtype={"collision_index": str})
    meta_test = pd.read_csv(META_TEST_PATH, dtype={"collision_index": str})
    return train, test, meta_train, meta_test


def _oof_predictions(
    train: pd.DataFrame,
    meta_train: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Return OOF argmax predictions and the exact active fold indexes."""
    splits = grouped_splits(train)
    predictions = {
        model_name: np.argmax(meta_train[columns].to_numpy(), axis=1) + 1
        for model_name, columns in PROBABILITY_COLUMNS.items()
    }
    return predictions, splits


def create_derived_graphs() -> None:
    """Create processed-data, risk-score, and OOF diagnostic graphs."""
    train, test, meta_train, _ = _read_processed()
    y_train = train[TARGET].astype(int).to_numpy()

    split_frame = pd.concat([
        train[TARGET].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0).rename("Train"),
        test[TARGET].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0).rename("Test"),
    ], axis=1)
    split_frame.index = [SEVERITY_LABELS[label] for label in CLASS_LABELS]
    fig, axis = plt.subplots(figsize=(9, 5))
    split_frame.mul(100).plot(kind="bar", ax=axis, color=["#4daf4a", "#984ea3"])
    axis.set_title("Train and locked-test class proportions")
    axis.set_xlabel("Severity")
    axis.set_ylabel("Rows (%)")
    axis.tick_params(axis="x", labelrotation=0)
    axis.legend(title="Partition")
    _save(fig, "train_test_class_balance.png")

    _, splits = _oof_predictions(train, meta_train)
    fold_rows = []
    for fold, (_, validation_index) in enumerate(splits, start=1):
        rates = train.iloc[validation_index][TARGET].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0)
        for label in CLASS_LABELS:
            fold_rows.append({"fold": f"Fold {fold}", "severity": SEVERITY_LABELS[label], "share": rates[label] * 100})
    fold_frame = pd.DataFrame(fold_rows)
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=fold_frame, x="fold", y="share", hue="severity", palette=[COLORS[label] for label in CLASS_LABELS], ax=axis)
    axis.set_title("Class proportions in OOF validation folds")
    axis.set_xlabel("Validation fold")
    axis.set_ylabel("Rows (%)")
    axis.legend(title="Severity")
    _save(fig, "oof_fold_class_balance.png")

    contract_rows = []
    contracts = {**REGRESSION_MODEL_FEATURES, **MODEL_FEATURES}
    for model_name, features in contracts.items():
        contract_rows.append({"model": MODEL_LABELS.get(model_name, RISK_LABELS.get(model_name, model_name)), "features": len(features)})
    contract_frame = pd.DataFrame(contract_rows).sort_values("features")
    base_sets = {name: set(features) for name, features in MODEL_FEATURES.items()}
    overlap = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for left in MODEL_ORDER:
        for right in MODEL_ORDER:
            union = base_sets[left] | base_sets[right]
            overlap.loc[left, right] = len(base_sets[left] & base_sets[right]) / len(union)
    overlap.index = [MODEL_LABELS[name] for name in overlap.index]
    overlap.columns = [MODEL_LABELS[name] for name in overlap.columns]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    sns.barplot(data=contract_frame, x="features", y="model", color="#377eb8", ax=axes[0])
    axes[0].set_title("Selected features per model")
    axes[0].set_xlabel("Feature count")
    axes[0].set_ylabel("")
    sns.heatmap(overlap, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Base-model feature Jaccard overlap")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    _save(fig, "model_feature_contracts.png")

    risk_columns = [f"risk_score_{name}" for name in RISK_MODEL_ORDER]
    risk_frame = meta_train[[TARGET] + risk_columns].copy()
    risk_frame["severity"] = _target_names(risk_frame)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, column, name in zip(axes, risk_columns, RISK_MODEL_ORDER):
        sns.boxplot(
            data=risk_frame,
            x="severity",
            y=column,
            order=CLASS_NAMES,
            palette=[COLORS[label] for label in CLASS_LABELS],
            hue="severity",
            legend=False,
            ax=axis,
        )
        axis.set_title(RISK_LABELS[name])
        axis.set_xlabel("Severity")
        axis.set_ylabel("OOF risk score")
    fig.suptitle("Collision-level OOF risk scores by casualty severity", y=1.02, fontsize=16)
    _save(fig, "risk_scores_by_severity.png")

    predictions, splits = _oof_predictions(train, meta_train)
    actual_counts = pd.Series(y_train).value_counts().reindex(CLASS_LABELS, fill_value=0)
    distribution_rows = [
        {"model": "Actual", "severity": SEVERITY_LABELS[label], "share": actual_counts[label] / len(y_train) * 100}
        for label in CLASS_LABELS
    ]
    for model_name, prediction in predictions.items():
        counts = pd.Series(prediction).value_counts().reindex(CLASS_LABELS, fill_value=0)
        distribution_rows.extend(
            {"model": MODEL_LABELS[model_name], "severity": SEVERITY_LABELS[label], "share": counts[label] / len(prediction) * 100}
            for label in CLASS_LABELS
        )
    distribution_frame = pd.DataFrame(distribution_rows)
    fig, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(data=distribution_frame, x="model", y="share", hue="severity", palette=[COLORS[label] for label in CLASS_LABELS], ax=axis)
    axis.set_title("Actual versus OOF-predicted class distributions")
    axis.set_xlabel("")
    axis.set_ylabel("Rows (%)")
    axis.tick_params(axis="x", labelrotation=20)
    axis.legend(title="Severity")
    _save(fig, "oof_predicted_vs_actual_distribution.png")

    metric_rows = []
    for model_name, prediction in predictions.items():
        for fold, (_, validation_index) in enumerate(splits, start=1):
            report = classification_report(
                y_train[validation_index],
                prediction[validation_index],
                labels=CLASS_LABELS,
                output_dict=True,
                zero_division=0,
            )
            metric_rows.extend([
                {"model": MODEL_LABELS[model_name], "fold": fold, "metric": "Accuracy", "value": report["accuracy"]},
                {"model": MODEL_LABELS[model_name], "fold": fold, "metric": "Fatal recall", "value": report["1"]["recall"]},
                {"model": MODEL_LABELS[model_name], "fold": fold, "metric": "Serious recall", "value": report["2"]["recall"]},
                {"model": MODEL_LABELS[model_name], "fold": fold, "metric": "Slight recall", "value": report["3"]["recall"]},
            ])
    metric_frame = pd.DataFrame(metric_rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for axis, metric in zip(axes.ravel(), ["Accuracy", "Fatal recall", "Serious recall", "Slight recall"]):
        subset = metric_frame.loc[metric_frame["metric"] == metric]
        sns.lineplot(data=subset, x="fold", y="value", hue="model", marker="o", ax=axis)
        axis.set_title(metric)
        axis.set_xlabel("OOF fold")
        axis.set_ylabel("Score")
        axis.set_xticks(range(1, len(splits) + 1))
        axis.set_ylim(0, 1)
        axis.legend(title="Model")
    fig.suptitle("Base-model OOF fold metrics", y=1.02, fontsize=16)
    _save(fig, "oof_fold_metrics.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, model_name in zip(axes, MODEL_ORDER):
        matrix = confusion_matrix(y_train, predictions[model_name], labels=CLASS_LABELS).astype(float)
        matrix /= matrix.sum(axis=1, keepdims=True)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axis,
        )
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.suptitle("Row-normalized OOF confusion matrices", y=1.02, fontsize=16)
    _save(fig, "oof_confusion_matrices.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, label in zip(axes, CLASS_LABELS):
        for model_name in MODEL_ORDER:
            probabilities = meta_train[PROBABILITY_COLUMNS[model_name][label - 1]].to_numpy()
            fraction_true, fraction_pred = calibration_curve(
                y_train == label,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
            axis.plot(fraction_pred, fraction_true, marker="o", label=MODEL_LABELS[model_name])
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
        axis.set_title(f"{SEVERITY_LABELS[label]} probability")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed frequency")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend()
    fig.suptitle("OOF probability calibration", y=1.02, fontsize=16)
    _save(fig, "probability_calibration.png")


def _load_reports() -> dict:
    """Load the common locked-test classification reports."""
    path = RESULTS_DIR / "classification_reports.json"
    if not path.exists():
        path = RESULTS_DIR / "base_model_reports.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _align_probabilities(model: object, raw: np.ndarray) -> np.ndarray:
    """Align a saved estimator's probability columns to labels 1, 2, 3."""
    output = np.zeros((len(raw), len(CLASS_LABELS)), dtype=float)
    for position, label in enumerate(CLASS_LABELS):
        if label in model.classes_:
            output[:, position] = raw[:, list(model.classes_).index(label)]
    return output


def _load_test_probabilities(meta_test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load base and meta probabilities for the locked test set."""
    probabilities = {
        model_name: meta_test[columns].to_numpy()
        for model_name, columns in PROBABILITY_COLUMNS.items()
    }
    model_path = MODELS_DIR / "meta_ensemble_3.pkl"
    contract_path = MODELS_DIR / "meta_contract.json"
    if model_path.exists() and contract_path.exists():
        model = joblib.load(model_path)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        raw = model.predict_proba(meta_test[contract["features"]])
        probabilities["meta_ensemble_3"] = _align_probabilities(model, raw)
    return probabilities


def create_model_graphs() -> None:
    """Create locked-test comparison, PR, regressor, and meta-model graphs."""
    _, test, _, meta_test = _read_processed()
    y_test = test[TARGET].astype(int).to_numpy()
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        metric_columns = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
        metric_columns = [column for column in metric_columns if column in comparison.columns]
        metric_long = comparison.melt(id_vars="model", value_vars=metric_columns, var_name="metric", value_name="score")
        metric_long["model"] = metric_long["model"].map(MODEL_LABELS).fillna(metric_long["model"])
        fig, axis = plt.subplots(figsize=(14, 7))
        sns.barplot(data=metric_long, x="model", y="score", hue="metric", ax=axis)
        axis.set_title("Locked-test model comparison")
        axis.set_xlabel("")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", labelrotation=15)
        axis.legend(title="Metric")
        _save(fig, "model_metric_comparison.png")

    reports = _load_reports()
    report_rows = []
    for model_name in ALL_MODEL_ORDER:
        report = reports.get(model_name)
        if not report:
            continue
        for label in CLASS_LABELS:
            values = report.get(str(label), {})
            report_rows.extend([
                {"model": MODEL_LABELS[model_name], "severity": SEVERITY_LABELS[label], "metric": "Precision", "score": values.get("precision", 0)},
                {"model": MODEL_LABELS[model_name], "severity": SEVERITY_LABELS[label], "metric": "Recall", "score": values.get("recall", 0)},
                {"model": MODEL_LABELS[model_name], "severity": SEVERITY_LABELS[label], "metric": "F1", "score": values.get("f1-score", 0)},
            ])
    if report_rows:
        report_frame = pd.DataFrame(report_rows)
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        for axis, severity in zip(axes, CLASS_NAMES):
            subset = report_frame.loc[report_frame["severity"] == severity]
            sns.barplot(data=subset, x="model", y="score", hue="metric", ax=axis)
            axis.set_title(f"{severity}")
            axis.set_xlabel("")
            axis.set_ylabel("Score")
            axis.set_ylim(0, 1)
            axis.tick_params(axis="x", labelrotation=30)
            axis.legend(title="Metric")
        fig.suptitle("Locked-test per-class metrics", y=1.02, fontsize=16)
        _save(fig, "per_class_metrics.png")

    matrix_names = [
        ("logistic_regression", "logistic_regression_confusion_matrix.csv"),
        ("knn", "knn_confusion_matrix.csv"),
        ("decision_tree", "decision_tree_confusion_matrix.csv"),
        ("meta_ensemble_3", "meta_ensemble_3_confusion_matrix.csv"),
    ]
    available_matrices = []
    for model_name, filename in matrix_names:
        path = RESULTS_DIR / filename
        if path.exists():
            matrix = pd.read_csv(path, index_col=0).to_numpy()
            available_matrices.append((model_name, matrix))
    if available_matrices:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        for axis, (model_name, matrix) in zip(axes.ravel(), available_matrices):
            sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
            axis.set_title(MODEL_LABELS[model_name])
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
        for axis in axes.ravel()[len(available_matrices):]:
            axis.axis("off")
        fig.suptitle("Locked-test confusion matrices", y=1.02, fontsize=16)
        _save(fig, "test_confusion_matrices.png")

    probabilities = _load_test_probabilities(meta_test)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, label in zip(axes, [1, 2]):
        for model_name, scores in probabilities.items():
            precision, recall, _ = precision_recall_curve(y_test == label, scores[:, label - 1])
            average_precision = average_precision_score(y_test == label, scores[:, label - 1])
            axis.plot(recall, precision, label=f"{MODEL_LABELS[model_name]} (AP {average_precision:.3f})")
        prevalence = np.mean(y_test == label)
        axis.axhline(prevalence, linestyle="--", color="black", label=f"Prevalence {prevalence:.1%}")
        axis.set_title(f"{SEVERITY_LABELS[label]} one-vs-rest")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend()
    fig.suptitle("Locked-test precision-recall curves", y=1.02, fontsize=16)
    _save(fig, "precision_recall_curves.png")

    regression_test = pd.read_csv(REGRESSION_TEST_PATH, dtype={"collision_index": str})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for column_index, model_name in enumerate(RISK_MODEL_ORDER):
        risk_path = PROCESSED / f"risk_score_{model_name}_test.csv"
        if not risk_path.exists():
            continue
        risk = pd.read_csv(risk_path, dtype={"collision_index": str})
        joined = regression_test[["collision_index", "number_of_casualties"]].merge(risk, on="collision_index", how="inner")
        actual = joined["number_of_casualties"].to_numpy(dtype=float)
        predicted = joined["risk_score"].to_numpy(dtype=float)
        scatter_axis = axes[0, column_index]
        scatter_axis.scatter(actual, predicted, alpha=0.18, s=10)
        limit = max(float(np.max(actual)), float(np.max(predicted)))
        scatter_axis.plot([0, limit], [0, limit], linestyle="--", color="black")
        scatter_axis.set_title(f"{RISK_LABELS[model_name]}: predicted vs actual")
        scatter_axis.set_xlabel("Actual casualties")
        scatter_axis.set_ylabel("Predicted risk score")
        residual_axis = axes[1, column_index]
        residual_axis.hist(predicted - actual, bins=40, color="#4daf4a", alpha=0.8)
        residual_axis.axvline(0, linestyle="--", color="black")
        residual_axis.set_title(f"{RISK_LABELS[model_name]} residuals")
        residual_axis.set_xlabel("Predicted - actual")
        residual_axis.set_ylabel("Rows")
    fig.suptitle("Locked-test risk-regressor diagnostics", y=1.02, fontsize=16)
    _save(fig, "regressor_diagnostics.png")

    contract_path = MODELS_DIR / "meta_contract.json"
    meta_path = MODELS_DIR / "meta_ensemble_3.pkl"
    if contract_path.exists() and meta_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        model = joblib.load(meta_path)
        classifier = model.named_steps.get("classifier", model)
        coefficients = np.asarray(classifier.coef_)
        classes = list(getattr(classifier, "classes_", CLASS_LABELS))
        coefficient_frame = pd.DataFrame(
            coefficients.T,
            index=contract["features"],
            columns=[SEVERITY_LABELS.get(int(label), str(label)) for label in classes],
        )
        fig, axis = plt.subplots(figsize=(12, 8))
        sns.heatmap(coefficient_frame, center=0, cmap="coolwarm", annot=True, fmt=".2f", ax=axis)
        axis.set_title("Meta-model coefficients after feature scaling")
        axis.set_xlabel("Meta-model class")
        axis.set_ylabel("Meta feature")
        _save(fig, "meta_feature_coefficients.png")


def write_manifest() -> None:
    """Write the graph catalog so the dashboard can discover static files."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_DIR / "graph_manifest.json").write_text(
        json.dumps(
            [
                {"section": section, "filename": filename, "title": title}
                for section, filename, title in GRAPH_CATALOG
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Create the requested graph scope without modifying the model pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["data", "derived", "model", "all"],
        default="all",
        help="Which graph group to create; default is all available groups.",
    )
    args = parser.parse_args()
    if args.scope in {"data", "all"}:
        create_data_graphs()
    if args.scope in {"derived", "all"}:
        create_derived_graphs()
    if args.scope in {"model", "all"}:
        create_model_graphs()
    write_manifest()
    print(f"Graph generation complete: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
