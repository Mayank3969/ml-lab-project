"""
clean_generated.py — Safe Artifact Cleaner
Deletes all generated files, cached models, and intermediate CSVs
produced by the pipeline, leaving raw data and code untouched.

Usage:
    python clean_generated.py          # Interactive prompt before deletion
    python clean_generated.py --dry-run # Preview what would be deleted
    python clean_generated.py --yes    # Force delete without prompt
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Exact list of patterns and directories for generated artifacts
ARTIFACTS_PATTERNS = [
    # 1. Processed datasets
    "dataset/processed/*.csv",
    # 2. Saved model pickles
    "models/saved_models/*.pkl",
    # 3. Stage 2 regression intermediate outputs
    "models/stage2_regression/*.json",
    "models/stage2_regression/*.csv",
    # 4. Evaluation results
    "models/evaluation/results/*.csv",
    # 5. Generated documentation
    "docs/column_removal_log.md",
    # 6. Pipeline execution log
    "pipeline_log.json",
    # 7. Temporary scratch scripts
    "scratch/*",
]

# Directories that should be cleared of Python bytecode
CACHE_DIRS = [
    "**/__pycache__",
    "**/.pytest_cache",
]

PROTECTED_PATHS = [
    ROOT / "dataset" / "raw",
    ROOT / "models" / "stage1_merge",
    ROOT / "models" / "stage2_regression",
    ROOT / "models" / "stage3a_base_classifiers",
    ROOT / "models" / "stage4_meta_classifier",
    ROOT / "models" / "evaluation",
    ROOT / "models" / "utils",
    ROOT / "docs",
]


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def find_files_to_clean():
    files_to_delete = []
    
    for pattern in ARTIFACTS_PATTERNS:
        for p in ROOT.glob(pattern):
            # Guard against deleting raw data or source code
            if "dataset/raw" in str(p).replace("\\", "/"):
                continue
            if p.suffix == ".py":
                continue
            if p.is_file():
                files_to_delete.append(p)

    return sorted(list(set(files_to_delete)))


def find_caches_to_clean():
    dirs_to_delete = []
    for pattern in CACHE_DIRS:
        for p in ROOT.glob(pattern):
            if p.is_dir() and "venv" not in str(p) and ".git" not in str(p):
                dirs_to_delete.append(p)
    return sorted(list(set(dirs_to_delete)))


def main():
    parser = argparse.ArgumentParser(
        description="Clean all generated artifacts from the Collision Severity pipeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate and print what files would be deleted without deleting anything."
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Execute deletion immediately without interactive confirmation."
    )
    args = parser.parse_args()

    files = find_files_to_clean()
    caches = find_caches_to_clean()

    total_bytes = sum(f.stat().st_size for f in files if f.exists())

    print("=" * 70)
    print("  COLLISION SEVERITY PIPELINE — ARTIFACT CLEANER")
    print("=" * 70)

    if not files and not caches:
        print("  No generated artifacts found. Workspace is already clean.")
        return

    print(f"\nFound {len(files)} generated file(s) ({human_size(total_bytes)}):")
    for f in files:
        rel = f.relative_to(ROOT)
        size = human_size(f.stat().st_size)
        print(f"  [-] {str(rel):<55} ({size})")

    if caches:
        print(f"\nFound {len(caches)} cache directory/directories:")
        for c in caches:
            rel = c.relative_to(ROOT)
            print(f"  [dir] {str(rel)}")

    print("\nProtected files (NEVER touched):")
    print("  [+] dataset/raw/collision.csv")
    print("  [+] dataset/raw/vehicle.csv")
    print("  [+] dataset/raw/casualty.csv")
    print("  [+] All Python source code (.py) & configurations")

    if args.dry_run:
        print("\n[DRY RUN] No files were deleted.")
        return

    if not args.yes:
        print()
        response = input(f"Are you sure you want to permanently delete these {len(files)} files? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Operation aborted. No files were removed.")
            return

    # Perform deletion
    deleted_count = 0
    deleted_bytes = 0
    for f in files:
        try:
            sz = f.stat().st_size
            f.unlink()
            deleted_count += 1
            deleted_bytes += sz
        except Exception as e:
            print(f"  [ERROR] Could not delete {f.name}: {e}")

    for c in caches:
        try:
            shutil.rmtree(c, ignore_errors=True)
        except Exception:
            pass

    print()
    print("=" * 70)
    print(f"  CLEANUP COMPLETE: Removed {deleted_count} files ({human_size(deleted_bytes)}).")
    print("  To retrain everything from raw data:")
    print("      python pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
