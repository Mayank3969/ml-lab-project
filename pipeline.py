"""Run the complete faculty demonstration pipeline from raw data.

The tuning notebooks are separate. After copying selected values into
``models/utils/manual_parameters.py``, run:

    python pipeline.py --clean

This removes generated artifacts, rebuilds the grouped data, trains both risk
regressors, trains the three base classifiers, fits the final ensemble, and
writes one comparison table containing every model's metrics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = {
    1: ("Prepare validated grouped data", ROOT / "models/stage1_merge/build_dataset.py"),
    2: ("Train both risk regressors", ROOT / "models/stage1_regression/train_risk_models.py"),
    3: ("Train base classifiers and OOF features", ROOT / "models/stage3a_base_classifiers/train_base_models.py"),
    4: ("Train final ensemble", ROOT / "models/stage4_meta_classifier/train_meta_ensemble.py"),
    5: ("Evaluate all models", ROOT / "models/evaluation/evaluate_pipeline.py"),
}


def run_stage(number: int, force_rebuild: bool = False) -> None:
    """Run one stage and stop with a non-zero code if it fails."""
    name, script = STAGES[number]
    print(f"\n=== Stage {number}/{len(STAGES)}: {name} ===", flush=True)
    command = [sys.executable, str(script)]
    if number == 1 and force_rebuild:
        command.append("--force")
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"Stage {number} failed with exit code {completed.returncode}")


def main() -> None:
    """Run the requested stage range, optionally resetting generated files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=int, default=1, choices=STAGES)
    parser.add_argument("--to", dest="end", type=int, default=max(STAGES), choices=STAGES)
    parser.add_argument("--clean", action="store_true", help="Delete generated artifacts before Stage 1")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--from must be less than or equal to --to")
    if args.clean:
        subprocess.run([sys.executable, "reset_generated.py", "--yes"], cwd=ROOT, check=True)
        args.start = 1
    for number in range(args.start, args.end + 1):
        run_stage(number, force_rebuild=args.clean)
    print("\nPipeline completed successfully.")
    print("Final comparison: models/evaluation/results/model_comparison.csv")


if __name__ == "__main__":
    main()
