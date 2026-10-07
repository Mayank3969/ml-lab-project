"""Simple dashboard for workflow explanation and model testing."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Make imports work the same from PowerShell, VS Code, and `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "models"))
from inference import MODEL_FEATURES, predict_pipeline, required_columns
from utils.config import REGRESSION_MODEL_FEATURES

TRAIN_PATH = ROOT / "dataset" / "processed" / "merged_train.csv"
GRAPH_DIR = ROOT / "docs" / "graphs"
LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}
NUMERIC = {"age_of_casualty", "age_of_driver", "engine_capacity_cc", "number_of_vehicles", "speed_limit"}

GRAPH_SECTIONS = {
    "data": "Base and merged data",
    "derived": "Processed and OOF diagnostics",
    "model": "Final model diagnostics",
}
GRAPH_FALLBACK = [
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


@st.cache_data
def load_reference() -> pd.DataFrame:
    """Load training rows only to create safe form defaults and category options."""
    return pd.read_csv(TRAIN_PATH)


def render_prediction_result(result: dict) -> None:
    """Show one consistent result panel for form and CSV predictions."""
    predictions = result["final_predictions"]
    probabilities = result["final_probabilities"]
    if len(predictions) == 1:
        label = LABELS[int(predictions[0])]
        st.subheader(f"Prediction: {label}")
        st.metric("Model confidence", f"{probabilities[0].max():.1%}")
    else:
        counts = pd.Series(predictions).map(LABELS).value_counts().reindex(LABELS.values(), fill_value=0)
        st.subheader("Batch prediction summary")
        st.dataframe(counts.rename("rows"), use_container_width=True)
    probability_frame = pd.DataFrame(
        probabilities,
        columns=["Fatal probability", "Serious probability", "Slight probability"],
    )
    st.dataframe(probability_frame.head(20), use_container_width=True)


def render_single_form(reference: pd.DataFrame) -> None:
    """Build a compact form from the exact saved-model feature contract."""
    st.subheader("Test one record")
    st.caption("The form uses the same raw columns and saved preprocessing pipelines as batch testing.")
    columns = required_columns()
    values: dict[str, object] = {}
    with st.form("single_prediction"):
        form_columns = st.columns(3)
        for index, column in enumerate(columns):
            target = form_columns[index % len(form_columns)]
            series = pd.to_numeric(reference[column], errors="coerce").dropna()
            with target:
                if column in NUMERIC:
                    default = float(series.median()) if not series.empty else 0.0
                    values[column] = st.number_input(column, value=default)
                else:
                    options = sorted(reference[column].dropna().unique().tolist())
                    if not options:
                        options = [-1]
                    values[column] = st.selectbox(column, options=options, format_func=str)
        submitted = st.form_submit_button("Run prediction")
    if submitted:
        try:
            render_prediction_result(predict_pipeline(pd.DataFrame([values])))
        except Exception as error:
            st.error(str(error))


def render_csv_testing() -> None:
    """Accept a CSV, run predictions, and provide a downloadable result."""
    st.subheader("Test a CSV")
    st.caption("Upload raw feature columns. The target and collision_index columns are not required.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return
    frame = pd.read_csv(uploaded)
    missing = sorted(set(required_columns()) - set(frame.columns))
    if missing:
        st.error(f"Missing required columns: {missing}")
        return
    st.dataframe(frame.head(10), use_container_width=True)
    if st.button("Run CSV predictions"):
        try:
            result = predict_pipeline(frame)
            render_prediction_result(result)
            output = frame.copy()
            output["Predicted_Severity_Code"] = result["final_predictions"]
            output["Predicted_Severity"] = [LABELS[int(value)] for value in result["final_predictions"]]
            output["P_Fatal"] = result["final_probabilities"][:, 0]
            output["P_Serious"] = result["final_probabilities"][:, 1]
            output["P_Slight"] = result["final_probabilities"][:, 2]
            st.download_button(
                "Download predictions",
                output.to_csv(index=False),
                file_name="collision_severity_predictions.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(str(error))


@st.cache_data
def load_graph_manifest() -> list[dict[str, str]]:
    """Load the offline graph catalog without creating any plots."""
    manifest_path = GRAPH_DIR / "graph_manifest.json"
    if manifest_path.exists():
        import json

        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        {"section": section, "filename": filename, "title": title}
        for section, filename, title in GRAPH_FALLBACK
    ]


def render_saved_graphs() -> None:
    """Display pre-rendered graph files from docs/graphs."""
    st.header("Data and model graphs")
    st.caption(
        "These images are generated manually by "
        "models/stage1_merge/create_graphs.py. Streamlit only reads the saved files."
    )
    if not GRAPH_DIR.exists():
        st.info("No saved graphs found. Run `python models/stage1_merge/create_graphs.py` first.")
        return

    manifest = load_graph_manifest()
    available = [entry for entry in manifest if (GRAPH_DIR / entry["filename"]).exists()]
    if not available:
        st.info("No saved graph images found. Run the manual graph-generation command first.")
        return

    section_options = ["All sections"] + list(GRAPH_SECTIONS.values())
    selected_section = st.selectbox("Graph section", section_options)
    selected_key = next(
        (key for key, label in GRAPH_SECTIONS.items() if label == selected_section),
        None,
    )
    if selected_key is not None:
        available = [entry for entry in available if entry["section"] == selected_key]

    for section_key, section_title in GRAPH_SECTIONS.items():
        entries = [entry for entry in available if entry["section"] == section_key]
        if not entries:
            continue
        st.subheader(section_title)
        for start in range(0, len(entries), 2):
            columns = st.columns(2)
            for column, entry in zip(columns, entries[start:start + 2]):
                with column:
                    st.image(str(GRAPH_DIR / entry["filename"]), caption=entry["title"], use_container_width=True)


st.set_page_config(page_title="Collision Severity", page_icon="🚗", layout="wide")
st.title("Collision Severity Prediction")
st.caption("A grouped, leakage-aware model for Fatal, Serious, and Slight casualty severity.")

models = st.sidebar.radio("Page", ["Home", "Workflow", "Model metrics", "Graphs", "Single prediction", "CSV testing"])
artifacts = None
try:
    from inference import get_models

    artifacts = get_models()
except Exception:
    pass

if artifacts is not None and artifacts.loaded:
    st.sidebar.success("All model artifacts loaded")
else:
    st.sidebar.warning("Run the three tuning notebooks and the stacking stage first")

if models == "Home":
    st.header("Project overview")
    st.write(
        "This project joins UK STATS19 casualty, vehicle, and collision tables. "
        "It keeps collision groups together during validation and uses three "
        "model-specific input contracts before combining their out-of-fold probabilities."
    )
    st.subheader("Topics")
    st.markdown("- Leakage-safe data preparation\n- Fatal/Serious/Slight class imbalance\n- Logistic Regression, KNN, and Decision Tree\n- Grouped cross-validation and stacking\n- Single-record and CSV testing")
    if TRAIN_PATH.exists():
        reference = load_reference()
        st.metric("Prepared training rows", f"{len(reference):,}")
        st.dataframe(reference[required_columns()].head(5), use_container_width=True)

elif models == "Workflow":
    st.header("System workflow")
    st.markdown(
        """```mermaid
flowchart LR
    A[collision.csv] --> D[Validated casualty-level join]
    B[vehicle.csv] --> D
    C[casualty.csv] --> D
    D --> E[Leakage removal + grouped split]
    E --> F[Collision-level regression tables]
    F --> G1[Linear Regression - Stage 1]
    F --> G2[Decision Tree Regressor - Stage 1]
    G1 --> H1[OOF risk score]
    G2 --> H2[OOF risk score]
    E --> I[Classifier train/test rows]
    I --> J1[Logistic Regression - Stage 2]
    I --> J2[KNN - Stage 2]
    I --> J3[Decision Tree - Stage 2]
    J1 --> K[Nine OOF probabilities]
    J2 --> K
    J3 --> K
    H1 --> L[Final ensemble - Stage 3]
    H2 --> L
    K --> L
    L --> M[Severity prediction]
```""",
        unsafe_allow_html=False,
    )
    st.code(
        "Raw tables -> validated merge -> grouped split -> two risk regressors\n"
        "           -> three base classifiers -> nine OOF probabilities\n"
        "           -> final ensemble (nine probabilities + selected risk score)\n"
        "           -> Fatal / Serious / Slight prediction",
        language="text",
    )
    stages = [
        ("1. Prepare", "Join the three raw tables, remove identifiers and target-derived fields, and create grouped train/test data."),
        ("2. Tune", "Run one Windows notebook per model. Each notebook saves every CV result and its best complete preprocessing pipeline."),
        ("3. OOF probabilities", "Fit clones of the tuned pipelines on grouped folds so each training row receives a prediction from an unseen model."),
        ("4. Stack", "Train the meta-classifier only on those nine out-of-fold probability columns."),
        ("5. Test", "Use this dashboard for one raw record or a CSV with the required model input columns."),
    ]
    for title, description in stages:
        st.subheader(title)
        st.write(description)
    st.subheader("Model-specific inputs")
    for name, columns in REGRESSION_MODEL_FEATURES.items():
        st.write(f"**{name}:** {', '.join(columns)}")
    for name, columns in MODEL_FEATURES.items():
        st.write(f"**{name}:** {', '.join(columns)}")

elif models == "Model metrics":
    st.header("Model comparison")
    metrics_path = ROOT / "models" / "evaluation" / "results" / "model_comparison.csv"
    if not metrics_path.exists():
        st.info("Run `python pipeline.py --clean` after setting manual parameters to generate metrics.")
    else:
        comparison = pd.read_csv(metrics_path)
        st.dataframe(comparison, use_container_width=True)
        base_metrics_path = ROOT / "models" / "evaluation" / "results" / "base_model_metrics.csv"
        if base_metrics_path.exists():
            st.subheader("Base-model OOF versus locked-test metrics")
            st.dataframe(pd.read_csv(base_metrics_path), use_container_width=True)
        st.subheader("Metric comparison")
        chart_columns = [
            column for column in [
                "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
            ] if column in comparison.columns
        ]
        if chart_columns:
            st.bar_chart(comparison.set_index("model")[chart_columns])
        report_path = ROOT / "models" / "evaluation" / "results" / "classification_reports.json"
        if report_path.exists():
            import json
            reports = json.loads(report_path.read_text(encoding="utf-8"))
            selected = st.selectbox("Confusion matrix model", list(reports))
            matrix_path = ROOT / "models" / "evaluation" / "results" / f"{selected}_confusion_matrix.csv"
            if matrix_path.exists():
                st.dataframe(pd.read_csv(matrix_path, index_col=0), use_container_width=True)
            st.json(reports[selected])

elif models == "Graphs":
    render_saved_graphs()

elif models == "Single prediction":
    if not TRAIN_PATH.exists():
        st.error("Prepared data is missing. Run models/stage1_merge/build_dataset.py first.")
    else:
        render_single_form(load_reference())

else:
    render_csv_testing()
