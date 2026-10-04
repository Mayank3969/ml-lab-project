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
        "name": "Stage 1 — Iterative Feature Selection & Stratified Split",
        "script": "models/stage1_merge/step3_build_dataset.py",
        "description": "Joins 3 raw tables. Runs 5-fold CV ablation study to dynamically select features. "
                       "Splits 80/20 using StratifiedGroupKFold to maintain class proportions.",
        "why": "Creates an analysis-ready dataset with optimal features backed by performance metrics, "
               "while guaranteeing no leakage across train/test and balanced class ratios.",
        "inputs": [
            "dataset/raw/collision.csv",
            "dataset/raw/vehicle.csv",
            "dataset/raw/casualty.csv",
        ],
        "outputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
            "docs/ablation_log.md",
            "dataset/processed/selected_features.json"
        ],
    },
    {
        "step": 2,
        "name": "Stage 2 — Train Risk Score Regressor & Ablation Test",
        "script": "models/stage2_regression/train_risk_model.py",
        "description": "Tunes Ridge and DecisionTree Regressor to predict number_of_casualties via CV GridSearch. "
                       "Generates risk_score using OOF predictions. Tests if risk_score improves "
                       "macro-F1 classification vs just number_of_casualties.",
        "why": "Determines the absolute best model for generating the risk_score continuous signal, "
               "and rigorously proves whether that signal is mathematically justified before keeping it.",
        "inputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
        ],
        "outputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
            "models/saved_models/stage1_best_regressor.pkl",
        ],
    },
    {
        "step": 3,
        "name": "Stage 3A — Profile Features for Model Suitability",
        "script": "models/stage3a_base_classifiers/step1_profile_data.py",
        "description": "Profiles final dimensionality and cardinality for theoretical suitability.",
        "why": "Avoids blindly throwing algorithms at sparse high-dimensional data.",
        "inputs": [
            "models/saved_models/feature_selection.json",
            "dataset/processed/merged_train.csv"
        ],
        "outputs": [
            "docs/data_profile.md",
        ],
    },
    {
        "step": 4,
        "name": "Stage 3A — Train & Tune Base Models",
        "script": "models/stage3a_base_classifiers/train_base_models.py",
        "description": "GridSearchCV tuning of LR, DecisionTree, and KNN. Generates OOF probabilities.",
        "why": "Trains optimal base classifiers and creates leak-free meta-features for Stage 4.",
        "inputs": [
            "dataset/processed/merged_train.csv",
            "dataset/processed/merged_test.csv",
        ],
        "outputs": [
            "dataset/processed/meta_train.csv",
            "dataset/processed/meta_test.csv",
            "models/saved_models/stage2_logreg_final.pkl",
            "models/saved_models/stage2_dt_final.pkl",
            "models/saved_models/stage2_knn_final.pkl",
        ],
    },
    {
        "step": 5,
        "name": "Stage 4 — Train Meta-Classifier",
        "script": "models/stage4_meta_classifier/train_meta_classifier.py",
        "description": "Trains stacking meta-classifier on 9 probabilities + risk_score + number_of_casualties.",
        "why": "Final model that learns to blend the base classifier opinions.",
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
        "step": 6,
        "name": "Stage 5 — Baseline vs Pipeline Evaluation",
        "script": "models/evaluation/baseline_vs_pipeline.py",
        "description": "Compares standalone base classifiers against the full stacking pipeline.",
        "why": "The final test of whether the pipeline beats a standalone model.",
        "inputs": [
            "dataset/processed/meta_test.csv",
            "models/saved_models/stage2_logreg_final.pkl",
            "models/saved_models/stage2_dt_final.pkl",
            "models/saved_models/stage2_knn_final.pkl",
            "models/saved_models/stage3_meta_classifier.pkl",
        ],
        "outputs": [
            "models/evaluation/results/model_comparison_table.csv",
            "models/evaluation/results/multiseed_evaluation_table.csv",
            "docs/stage5_evaluation_report.md",
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


def run_step(step_def, log, refresh_features=False):
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

    # Run the script and stream output live
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    
    cmd = [PYTHON, str(script_path)]
    if refresh_features and step_num == 1:
        cmd.append("--refresh-features")
        
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    
    output_lines = []
    # Print console output live (ascii-safe for Windows terminals)
    for line in process.stdout:
        output_lines.append(line)
        print(line.encode("ascii", errors="replace").decode("ascii"), end="")
        
    process.wait()
    full_output = "".join(output_lines)
    
    duration = round(time.time() - t0, 2)
    success = process.returncode == 0

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
        "return_code": process.returncode,
        "duration_seconds": duration,
        "started_at": datetime.now().isoformat(),
        "inputs": inputs_before,
        "outputs": outputs_after,
        "console_output": full_output.strip() if full_output else "",
        "stderr": "",
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
        print(f"  [FAILED] Step {step_num} failed with exit code {process.returncode}")
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
    parser.add_argument(
        "--refresh-features", action="store_true",
        help="Force re-run of Stage 1 feature ablation (ignore cache)"
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

        success = run_step(step_def, log, refresh_features=args.refresh_features)
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
