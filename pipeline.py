"""
pipeline.py — Master Pipeline Runner
Executes all stages in the correct sequence and logs every step's metadata,
inputs, outputs, metrics, timing, and console output into a single
pipeline_log.json file that grows as each step completes.

Usage:
    python pipeline.py              # full run from raw data
    python pipeline.py --from 6     # resume from step 6 onwards
"""
import subprocess
import sys
import json
import time
import os
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # same interpreter that launched this script
LOG_FILE = ROOT / "pipeline_log.json"

# ---------------------------------------------------------------------------
# Step definitions — the single source of truth for execution order
# ---------------------------------------------------------------------------
STEPS = [
    {
        "step": 1,
        "name": "Stage 1 — Merge Raw Tables",
        "script": "models/stage1_merge/step3_build_dataset.py",
        "description": "Joins collision.csv + vehicle.csv + casualty.csv at casualty level. "
                       "Drops 9 leakage columns and ~35 irrelevant columns. "
                       "Splits 80/20 grouped by collision_index (GroupShuffleSplit). "
                       "Converts all features to numeric.",
        "why": "Creates the clean, analysis-ready dataset from 3 raw sources. "
               "Group-aware split prevents same-crash casualties leaking across train/test.",
        "inputs": [
            "dataset/raw/collision.csv",
            "dataset/raw/vehicle.csv",
            "dataset/raw/casualty.csv",
        ],
        "outputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
            "docs/column_removal_log.md",
        ],
    },
    {
        "step": 2,
        "name": "Stage 1 — Fix Junk Columns",
        "script": "models/stage1_merge/step3_fix.py",
        "description": "Removes 4 residual columns (collision_year_col, collision_ref_no_col, "
                       "collision_year_veh, collision_ref_no_veh) that survived the merge "
                       "due to pandas suffix collisions.",
        "why": "These columns are merge artifacts with no predictive value. "
               "After this step merged CSVs have exactly 18 columns.",
        "inputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
        ],
        "outputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
            "docs/column_removal_log.md",
        ],
    },
    {
        "step": 3,
        "name": "Stage 2 — Train Linear Regression",
        "script": "models/stage2_regression/train_linear_regression.py",
        "description": "Trains Linear Regression to predict number_of_casualties from "
                       "5 collision-level features (speed_limit, road_type, junction_detail, "
                       "light_conditions, weather_conditions). Uses 5-fold OOF predictions "
                       "to avoid leakage. Saves model, OOF predictions, and metrics.",
        "why": "One of two candidate models for generating the risk_score feature. "
               "OOF ensures the risk_score on training rows was never seen during training.",
        "inputs": [
            "dataset/processed/merged_train.csv",
        ],
        "outputs": [
            "models/stage2_regression/lr_metrics.json",
            "models/stage2_regression/lr_oof_predictions.csv",
            "models/saved_models/stage1_lr_model.pkl",
        ],
        "metrics_file": "models/stage2_regression/lr_metrics.json",
    },
    {
        "step": 4,
        "name": "Stage 2 — Train Decision Tree Regressor",
        "script": "models/stage2_regression/train_decision_tree_regressor.py",
        "description": "Trains Decision Tree Regressor on the same 5 collision-level features. "
                       "Same 5-fold OOF approach as Linear Regression.",
        "why": "Second candidate model. Compared against LR by R-squared to pick the winner.",
        "inputs": [
            "dataset/processed/merged_train.csv",
        ],
        "outputs": [
            "models/stage2_regression/dt_metrics.json",
            "models/stage2_regression/dt_oof_predictions.csv",
            "models/saved_models/stage1_dt_model.pkl",
        ],
        "metrics_file": "models/stage2_regression/dt_metrics.json",
    },
    {
        "step": 5,
        "name": "Stage 2 — Compare Regressors & Generate risk_score",
        "script": "models/stage2_regression/compare_regressor.py",
        "description": "Compares LR vs DT by R-squared, picks the winner (LR, R2=0.023). "
                       "Maps OOF risk_score onto train rows (no leakage). "
                       "Retrains winner on full train set and predicts risk_score for test rows. "
                       "Overwrites merged CSVs with the new risk_score column (18 -> 19 cols).",
        "why": "The risk_score feature is the bridge between Stage 2 regression and "
               "Stage 3A classification. It gives the classifiers a continuous signal "
               "about expected casualty count.",
        "inputs": [
            "models/stage2_regression/lr_metrics.json",
            "models/stage2_regression/dt_metrics.json",
            "models/stage2_regression/lr_oof_predictions.csv",
            "models/saved_models/stage1_lr_model.pkl",
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
        ],
        "outputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
            "models/saved_models/stage1_best_regressor.pkl",
            "models/evaluation/results/stage1_metrics.csv",
        ],
    },
    {
        "step": 6,
        "name": "Stage 3A Phase A — Out-of-Fold Meta-Feature Generation",
        "script": "models/stage3a_base_classifiers/oof_generator.py",
        "description": "5-fold StratifiedGroupKFold CV with 3 base classifiers "
                       "(Logistic Regression, KNN k=15, Decision Tree). "
                       "Each training row gets probability predictions from a model "
                       "that never saw it. All 15 temporary models (3 x 5 folds) are destroyed. "
                       "Output: 9 probability columns + risk_score + target.",
        "why": "Creates leak-free training data for the meta-classifier. "
               "If we predicted on training data directly, the meta-classifier would "
               "learn to trust overconfident base model predictions and fail on test data.",
        "inputs": [
            "dataset/processed/merged_train.csv",
        ],
        "outputs": [
            "dataset/processed/meta_train.csv",
        ],
    },
    {
        "step": 7,
        "name": "Stage 3A Phase B — Train Final Base Models & Generate meta_test",
        "script": "models/stage3a_base_classifiers/final_models_trainer.py",
        "description": "Trains 3 final base models on the full training set (no folding). "
                       "Predicts class probabilities on the held-out test set. "
                       "Saves the 3 model artifacts for Streamlit inference.",
        "why": "These final models produce the test-side meta-features and are the "
               "actual models loaded at inference time in the Streamlit app.",
        "inputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
        ],
        "outputs": [
            "dataset/processed/meta_test.csv",
            "models/saved_models/stage2_logreg_final.pkl",
            "models/saved_models/stage2_knn_final.pkl",
            "models/saved_models/stage2_dt_final.pkl",
        ],
    },
    {
        "step": 8,
        "name": "Stage 4 — Train Meta-Classifier",
        "script": "models/stage4_meta_classifier/train_meta_classifier.py",
        "description": "Trains stacking meta-classifier (Logistic Regression, class_weight=balanced) "
                       "on exactly 10 features: 9 base model probabilities + risk_score. "
                       "Evaluates on held-out test set. Saves confusion matrix.",
        "why": "This is the final prediction model. It learns the optimal weighting "
               "of 3 classifiers' opinions plus the regression risk score.",
        "inputs": [
            "dataset/processed/meta_train.csv",
            "dataset/processed/meta_test.csv",
        ],
        "outputs": [
            "models/saved_models/stage3_meta_classifier.pkl",
            "models/evaluation/results/meta_classifier_confusion_matrix.csv",
        ],
    },
    {
        "step": 9,
        "name": "Stage 5 — Baseline vs Pipeline Evaluation",
        "script": "models/evaluation/baseline_vs_pipeline.py",
        "description": "Compares standalone base classifiers against the full stacking pipeline. "
                       "Reports Macro-F1, Accuracy, and per-class F1 (Fatal, Serious, Slight). "
                       "Computes headline delta: best standalone vs pipeline.",
        "why": "This is the payoff — the answer to 'does the multi-stage pipeline beat "
               "a single model?' Also checks Fatal-class recall, the most safety-critical metric.",
        "inputs": [
            "models/stage2_regression/lr_metrics.json",
            "models/stage2_regression/dt_metrics.json",
            "dataset/processed/meta_test.csv",
            "models/saved_models/stage2_logreg_final.pkl",
            "models/saved_models/stage2_knn_final.pkl",
            "models/saved_models/stage2_dt_final.pkl",
            "models/saved_models/stage3_meta_classifier.pkl",
        ],
        "outputs": [
            "models/evaluation/results/model_comparison_table.csv",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def file_info(path_str):
    """Return dict with file path, existence, size, and row count (for CSVs)."""
    p = ROOT / path_str
    info = {"path": path_str, "exists": p.exists()}
    if p.exists():
        info["size_bytes"] = p.stat().st_size
        info["size_human"] = _human_size(p.stat().st_size)
        if path_str.endswith(".csv"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                info["rows"] = lines - 1  # subtract header
                # grab column names from first line
                with open(p, "r", encoding="utf-8") as f:
                    header = f.readline().strip()
                info["columns"] = header.split(",")
                info["num_columns"] = len(info["columns"])
            except Exception:
                pass
        elif path_str.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    info["content"] = json.load(f)
            except Exception:
                pass
    return info


def _human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_log():
    """Load existing pipeline log or start fresh."""
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"pipeline": "Collision Severity Stacked Ensemble", "steps": []}


def save_log(log):
    """Write pipeline log to disk."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def run_step(step_def, log):
    """Execute one pipeline step, capture everything, append to log."""
    step_num = step_def["step"]
    script = step_def["script"]
    script_path = ROOT / script

    print()
    print("=" * 70)
    print(f"  STEP {step_num}/9 : {step_def['name']}")
    print(f"  Script : {script}")
    print("=" * 70)

    # Snapshot inputs BEFORE running
    inputs_before = [file_info(p) for p in step_def["inputs"]]

    # Run the script (force utf-8 so child scripts don't crash on Windows cp1252)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    duration = round(time.time() - t0, 2)
    success = result.returncode == 0

    # Print console output live (ascii-safe for Windows terminals)
    if result.stdout:
        print(result.stdout.encode("ascii", errors="replace").decode("ascii"))
    if result.stderr and not success:
        print("[STDERR]", result.stderr.encode("ascii", errors="replace").decode("ascii"))

    # Snapshot outputs AFTER running
    outputs_after = [file_info(p) for p in step_def["outputs"]]

    # Load metrics file if one is specified
    metrics = None
    if "metrics_file" in step_def:
        mf = ROOT / step_def["metrics_file"]
        if mf.exists():
            with open(mf, "r", encoding="utf-8") as f:
                metrics = json.load(f)

    # Also try to grab metrics from evaluation results if this is step 8 or 9
    if step_num == 8:
        cm_path = ROOT / "models/evaluation/results/meta_classifier_confusion_matrix.csv"
        if cm_path.exists():
            metrics = {"confusion_matrix_file": str(cm_path)}
    if step_num == 9:
        comp_path = ROOT / "models/evaluation/results/model_comparison_table.csv"
        if comp_path.exists():
            try:
                import csv
                with open(comp_path, "r") as f:
                    reader = csv.DictReader(f)
                    metrics = {"model_comparison": [row for row in reader]}
            except Exception:
                metrics = {"comparison_file": str(comp_path)}

    # Build log entry
    entry = {
        "step": step_num,
        "name": step_def["name"],
        "script": script,
        "description": step_def["description"],
        "why": step_def["why"],
        "status": "SUCCESS" if success else "FAILED",
        "return_code": result.returncode,
        "duration_seconds": duration,
        "started_at": datetime.now().isoformat(),
        "inputs": inputs_before,
        "outputs": outputs_after,
        "console_output": result.stdout.strip() if result.stdout else "",
        "stderr": result.stderr.strip() if result.stderr else "",
    }
    if metrics:
        entry["metrics"] = metrics

    # Append to log and save immediately (so partial runs are preserved)
    log["steps"].append(entry)
    log["last_updated"] = datetime.now().isoformat()
    save_log(log)

    if success:
        print(f"  [OK] Step {step_num} completed in {duration:.1f}s")
    else:
        print(f"  [FAILED] Step {step_num} failed with exit code {result.returncode}")
        print(f"  Check pipeline_log.json for details.")

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run the full collision severity pipeline and log everything."
    )
    parser.add_argument(
        "--from", dest="start_step", type=int, default=1,
        help="Resume from this step number (default: 1 = full run)"
    )
    parser.add_argument(
        "--to", dest="end_step", type=int, default=9,
        help="Stop after this step number (default: 9 = run all)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  COLLISION SEVERITY PREDICTION — MASTER PIPELINE")
    print(f"  Running steps {args.start_step} to {args.end_step}")
    print(f"  Log file: {LOG_FILE}")
    print("=" * 70)

    # Start fresh log if running from step 1, otherwise load existing
    if args.start_step == 1:
        log = {
            "pipeline": "Collision Severity Stacked Ensemble",
            "python": PYTHON,
            "started_at": datetime.now().isoformat(),
            "steps": [],
        }
    else:
        log = load_log()
        # Remove any previous entries for steps we're about to re-run
        log["steps"] = [s for s in log["steps"] if s["step"] < args.start_step]

    total_start = time.time()
    failed_at = None

    for step_def in STEPS:
        if step_def["step"] < args.start_step:
            continue
        if step_def["step"] > args.end_step:
            break

        success = run_step(step_def, log)
        if not success:
            failed_at = step_def["step"]
            break

    total_time = round(time.time() - total_start, 2)

    # Final summary
    print()
    print("=" * 70)
    if failed_at:
        print(f"  PIPELINE STOPPED at step {failed_at} (FAILED)")
        print(f"  To resume: python pipeline.py --from {failed_at}")
    else:
        print("  PIPELINE COMPLETE — ALL STEPS PASSED")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Full log saved to: {LOG_FILE}")
    print("=" * 70)

    # Update final metadata
    log["total_duration_seconds"] = total_time
    log["completed_at"] = datetime.now().isoformat()
    log["final_status"] = "FAILED" if failed_at else "COMPLETE"
    if failed_at:
        log["failed_at_step"] = failed_at
    save_log(log)


if __name__ == "__main__":
    main()
