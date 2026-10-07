"""Safely remove generated datasets, models, tuning outputs, and reports.

Raw CSV files, source code, notebooks, and documentation are preserved.
Use ``--dry-run`` to inspect the deletion list. Use ``--yes`` for a scripted
faculty demonstration reset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def generated_files() -> list[Path]:
    """Return generated files that can be recreated by the pipeline."""
    files: list[Path] = []
    for directory, patterns in {
        ROOT / "dataset" / "processed": ["*.csv", "*.json"],
        ROOT / "models" / "saved_models": ["*.pkl", "*.json"],
        ROOT / "models" / "evaluation" / "results": ["*"],
        ROOT / "hyper parameter" / "results": ["*"],
        ROOT / "hyper parameter" / "best": ["*"],
        ROOT / "docs" / "graphs": ["*.png", "graph_manifest.json"],
    }.items():
        for pattern in patterns:
            matches = directory.rglob("*") if pattern == "*" else directory.glob(pattern)
            files.extend(
                path for path in matches
                if path.is_file() and path.name not in {".gitkeep", "README.md"}
            )
    files.extend([
        ROOT / "models" / "stage2_regression" / name
        for name in ["lr_metrics.json", "dt_metrics.json", "lr_oof_predictions.csv", "dt_oof_predictions.csv"]
    ])
    files.extend([ROOT / "pipeline_log.json"])
    return sorted({path for path in files if path.exists()})


def main() -> None:
    """Preview or delete generated artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List files without deleting")
    parser.add_argument("--yes", action="store_true", help="Delete without asking")
    args = parser.parse_args()
    files = generated_files()
    if not files:
        print("No generated artifacts found.")
        return
    print("Generated files selected for deletion:")
    for path in files:
        print(f"  {path.relative_to(ROOT)}")
    if args.dry_run:
        print("Dry run complete; nothing was deleted.")
        return
    if not args.yes and input("Delete these files? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("Deletion cancelled.")
        return
    for path in files:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"Deleted {len(files)} generated files. Raw data and source code were preserved.")


if __name__ == "__main__":
    main()
