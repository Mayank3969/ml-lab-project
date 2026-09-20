# Pipeline Reference Guide

> **What each file does, what data it reads, what it outputs, and the exact execution order.**

---

## Answer: Yes, All 5 Stages Are Done

| Stage | Status | Script(s) | Output |
|-------|--------|-----------|--------|
| Stage 1 — Merge & Clean | Done | `step3_build_dataset.py` | `merged_train.csv`, `merged_test.csv` |
| Stage 2 — Regression | Done | `train_linear_regression.py`, `train_decision_tree_regressor.py`, `compare_regressor.py` | `risk_score` column added, `stage1_best_regressor.pkl` |
| Stage 3A Phase A — OOF | Done | `oof_generator.py` | `meta_train.csv` |
| Stage 3A Phase B — Final | Done | `final_models_trainer.py` | `meta_test.csv`, 3 `.pkl` models |
| Stage 4 — Meta-Classifier | Done | `train_meta_classifier.py` | `stage3_meta_classifier.pkl` |
| Stage 5 — Evaluation | Done | `baseline_vs_pipeline.py` | Metrics CSVs + confusion matrix |

---

## If You Delete Everything and Start From Raw Data Only

If you wipe `dataset/processed/` and `models/saved_models/`, here is the **exact order** to rebuild everything from scratch using only the 3 raw CSV files:

```
Step 1:  python models/stage1_merge/step3_build_dataset.py
Step 2:  python models/stage1_merge/step3_fix.py
Step 3:  python models/stage2_regression/train_linear_regression.py
Step 4:  python models/stage2_regression/train_decision_tree_regressor.py
Step 5:  python models/stage2_regression/compare_regressor.py
Step 6:  python models/stage3a_base_classifiers/oof_generator.py
Step 7:  python models/stage3a_base_classifiers/final_models_trainer.py
Step 8:  python models/stage4_meta_classifier/train_meta_classifier.py
Step 9:  python models/evaluation/baseline_vs_pipeline.py
```

> [!CAUTION]
> **Order matters!** Each step depends on the output of the previous one. You cannot skip steps or run them out of order.

---

## Every File Explained

---

### `models/utils/config.py`
**Purpose:** Central configuration file. Every other script imports paths, constants, and column lists from here.

| What | Value |
|------|-------|
| Defines | `ROOT`, `PROCESSED`, `MODELS_DIR`, `RESULTS_DIR`, `STAGE2_DIR` paths |
| Defines | `SEED=42`, `TARGET="casualty_severity"`, `CLASS_LABELS=[1,2,3]` |
| Defines | `STAGE1_FEATURES` (5 collision-level columns for regression) |
| Defines | `STAGE3_PROBA_COLS` (9 probability column names) |
| Defines | `STAGE4_FEATURES` (9 probas + risk_score = 10 meta-features) |
| Creates | All output directories if they don't exist |

**You never "run" this file directly. It's imported by everything else.**

---

### `models/utils/preprocessing.py`
**Purpose:** Shared helper functions for data loading, collision-level deduplication, and building the Stage 1 preprocessor.

| Function | What It Does |
|----------|-------------|
| `load_merged()` | Loads `merged_train.csv` and `merged_test.csv` with `collision_index` forced to string |
| `collapse_to_collision(df)` | Deduplicates rows to one-per-collision (Stage 2 regression operates at collision level, not casualty level) |
| `build_stage1_preprocessor()` | Returns a `ColumnTransformer` that passes through `speed_limit` and one-hot-encodes 4 categorical features |

**You never "run" this file directly. It's imported by Stage 2 scripts.**

---

### `models/stage1_merge/step1_peek.py` (Optional/EDA)
**Purpose:** Quick exploratory peek at the 3 raw CSV files — row counts, column counts, header detection.

| Reads From | Writes To |
|------------|-----------|
| `dataset/raw/collision.csv` | Nothing (prints to console only) |
| `dataset/raw/vehicle.csv` | |
| `dataset/raw/casualty.csv` | |

**Why:** Sanity check before merging. Not required for the pipeline — purely exploratory.

---

### `models/stage1_merge/step2_target_and_joins.py` (Optional/EDA)
**Purpose:** Validates that join keys connect across the 3 tables, checks the target distribution (Fatal/Serious/Slight), and identifies leakage columns.

| Reads From | Writes To |
|------------|-----------|
| `dataset/raw/collision.csv` | Nothing (prints to console only) |
| `dataset/raw/vehicle.csv` | |
| `dataset/raw/casualty.csv` | |

**Why:** Pre-merge diagnostics. Not required for the pipeline — purely exploratory.

---

### `models/stage1_merge/step3_build_dataset.py` ★ RUN FIRST
**Purpose:** The big merge script. Joins 3 raw tables, drops 9 leakage columns, drops ~35 irrelevant columns, converts dtypes, splits 80/20 grouped by `collision_index`.

| Reads From | Writes To |
|------------|-----------|
| `dataset/raw/collision.csv` | `dataset/processed/merged_train.csv` (~94k rows) |
| `dataset/raw/vehicle.csv` | `dataset/processed/merged_test.csv` (~24k rows) |
| `dataset/raw/casualty.csv` | `docs/column_removal_log.md` |

**Why:** This is where raw data becomes a usable ML dataset. The `GroupShuffleSplit` ensures no collision appears in both train and test sets.

**Output columns (~18):** `collision_index`, `casualty_class`, `casualty_severity` (target), `casualty_type`, `number_of_vehicles`, `number_of_casualties`, `day_of_week`, `first_road_class`, `road_type`, `speed_limit`, `junction_detail`, `junction_control`, `light_conditions`, `weather_conditions`, `road_surface_conditions`, `special_conditions_at_site`, `carriageway_hazards`, `urban_or_rural_area`.

---

### `models/stage1_merge/step3_fix.py` ★ RUN SECOND
**Purpose:** Post-merge cleanup. Removes 4 junk columns (`collision_year_col`, `collision_ref_no_col`, etc.) that survived the merge due to suffix collisions.

| Reads From | Writes To |
|------------|-----------|
| `dataset/processed/merged_train.csv` | `dataset/processed/merged_train.csv` (overwritten, fewer columns) |
| `dataset/processed/merged_test.csv` | `dataset/processed/merged_test.csv` (overwritten, fewer columns) |
| | `docs/column_removal_log.md` (regenerated) |

**Why:** Fixes artifacts from the merge step. After this, merged files have exactly 18 columns.

---

### `models/stage2_regression/train_linear_regression.py` ★ RUN 3rd
**Purpose:** Trains a Linear Regression model to predict `number_of_casualties` from 5 collision-level features. Uses 5-fold OOF to generate leak-free predictions.

| Reads From | Writes To |
|------------|-----------|
| `dataset/processed/merged_train.csv` | `models/stage2_regression/lr_metrics.json` (RMSE, R²) |
| | `models/stage2_regression/lr_oof_predictions.csv` (collision_index + OOF predictions) |
| | `models/saved_models/stage1_lr_model.pkl` (full-train model) |

**Why:** Generates a continuous `risk_score` feature. The OOF predictions ensure no data leakage when this score is used downstream.

---

### `models/stage2_regression/train_decision_tree_regressor.py` ★ RUN 4th
**Purpose:** Same as above but uses a Decision Tree Regressor instead.

| Reads From | Writes To |
|------------|-----------|
| `dataset/processed/merged_train.csv` | `models/stage2_regression/dt_metrics.json` |
| | `models/stage2_regression/dt_oof_predictions.csv` |
| | `models/saved_models/stage1_dt_model.pkl` |

**Why:** Provides a second regression model to compare against Linear Regression.

---

### `models/stage2_regression/compare_regressor.py` ★ RUN 5th
**Purpose:** Compares both regressors by R², selects the winner, maps the `risk_score` column onto every casualty row in both train and test.

| Reads From | Writes To |
|------------|-----------|
| `models/stage2_regression/lr_metrics.json` | `dataset/processed/merged_train.csv` (overwritten — now has `risk_score` column) |
| `models/stage2_regression/dt_metrics.json` | `dataset/processed/merged_test.csv` (overwritten — now has `risk_score` column) |
| `models/stage2_regression/lr_oof_predictions.csv` (or dt) | `models/saved_models/stage1_best_regressor.pkl` |
| `models/saved_models/stage1_lr_model.pkl` (or dt) | `models/evaluation/results/stage1_metrics.csv` |
| `dataset/processed/merged_train.csv` | |
| `dataset/processed/merged_test.csv` | |

**Why:** After this step, `merged_train.csv` and `merged_test.csv` gain a 19th column: `risk_score`. This is the engineered feature that feeds into Stage 3A.

---

### `models/stage3a_base_classifiers/oof_generator.py` ★ RUN 6th
**Purpose:** Phase A of stacking. Runs 5-fold `StratifiedGroupKFold` CV with 3 base classifiers (Logistic Regression, KNN, Decision Tree). Generates out-of-fold probability predictions for every training row.

| Reads From | Writes To |
|------------|-----------|
| `dataset/processed/merged_train.csv` (with `risk_score`) | `dataset/processed/meta_train.csv` (93,975 rows × 12 cols) |

**Output columns:** `collision_index`, 9 probability columns (`P_logreg_fatal`, `P_logreg_serious`, `P_logreg_slight`, `P_knn_fatal`, `P_knn_serious`, `P_knn_slight`, `P_dt_fatal`, `P_dt_serious`, `P_dt_slight`), `risk_score`, `casualty_severity`.

**Why:** The meta-classifier needs training data where every probability prediction came from a model that *never saw that row*. This is the core anti-leakage mechanism of stacking. All 15 temporary models (3 × 5 folds) are destroyed after use.

---

### `models/stage3a_base_classifiers/final_models_trainer.py` ★ RUN 7th
**Purpose:** Phase B of stacking. Trains 3 final base models on the *entire* training set (no folding). Predicts probabilities on the held-out test set.

| Reads From | Writes To |
|------------|-----------|
| `dataset/processed/merged_train.csv` | `dataset/processed/meta_test.csv` (23,528 rows × 12 cols) |
| `dataset/processed/merged_test.csv` | `models/saved_models/stage2_logreg_final.pkl` |
| | `models/saved_models/stage2_knn_final.pkl` |
| | `models/saved_models/stage2_dt_final.pkl` |

**Why:** These 3 final `.pkl` models are the ones the Streamlit app loads at inference time. `meta_test.csv` has the same 12-column shape as `meta_train.csv`.

---

### `models/stage4_meta_classifier/train_meta_classifier.py` ★ RUN 8th
**Purpose:** Trains the stacking meta-classifier (Logistic Regression) on exactly 10 input features (9 base model probabilities + `risk_score`).

| Reads From | Writes To |
|------------|-----------|
| `dataset/processed/meta_train.csv` | `models/saved_models/stage3_meta_classifier.pkl` |
| `dataset/processed/meta_test.csv` | `models/evaluation/results/meta_classifier_confusion_matrix.csv` |

**Why:** This is the final model. It learns how to optimally combine the 3 base classifiers' opinions + the regression risk score into one prediction.

---

### `models/evaluation/baseline_vs_pipeline.py` ★ RUN 9th (LAST)
**Purpose:** The headline comparison. Benchmarks each standalone base classifier against the full multi-stage stacking pipeline.

| Reads From | Writes To |
|------------|-----------|
| `models/stage2_regression/lr_metrics.json` | `models/evaluation/results/model_comparison_table.csv` |
| `models/stage2_regression/dt_metrics.json` | |
| `dataset/processed/meta_test.csv` | |
| `models/saved_models/stage2_logreg_final.pkl` | |
| `models/saved_models/stage2_knn_final.pkl` | |
| `models/saved_models/stage2_dt_final.pkl` | |
| `models/saved_models/stage3_meta_classifier.pkl` | |

**Why:** This produces the "headline delta" — the entire reason the pipeline exists. It answers: *does stacking actually beat using a single model?*

---

### Standalone Base Model Scripts (Optional)

These 3 scripts train and evaluate individual base classifiers independently. They are **not part of the pipeline** — they exist so you can run a single model and see its report in isolation.

| Script | Model |
|--------|-------|
| `models/stage3a_base_classifiers/train_logistic_regression.py` | Logistic Regression |
| `models/stage3a_base_classifiers/train_knn.py` | KNN (k=15) |
| `models/stage3a_base_classifiers/train_decision_tree.py` | Decision Tree |

Each reads `merged_train.csv` + `merged_test.csv`, trains, predicts, and prints a `classification_report`. They don't save models or CSVs.

---

### Concept Learning Scripts (Optional, Standalone Demos)

| Script | Purpose |
|--------|---------|
| `models/concept_learning/find_s.py` | Find-S algorithm demo (not part of the pipeline) |
| `models/concept_learning/candidate_elimination.py` | Candidate Elimination demo (not part of the pipeline) |

---

## Data Flow Diagram

```
dataset/raw/                          dataset/processed/              models/saved_models/
 collision.csv ──┐
 vehicle.csv  ───┤ step3_build_dataset.py
 casualty.csv ───┘       │
                         ▼
                  merged_train.csv ◄──── step3_fix.py (cleanup)
                  merged_test.csv  ◄──── step3_fix.py (cleanup)
                         │
                         │  train_linear_regression.py ──► stage1_lr_model.pkl
                         │  train_decision_tree_regressor.py ──► stage1_dt_model.pkl
                         │
                         ▼
                  compare_regressor.py
                    │ Picks best regressor (LR won)
                    │ Adds risk_score column to both CSVs
                    └──► stage1_best_regressor.pkl
                         │
                         ▼
                  merged_train.csv (now 19 cols, has risk_score)
                  merged_test.csv  (now 19 cols, has risk_score)
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   oof_generator.py              final_models_trainer.py
   (Phase A: 5-fold OOF)        (Phase B: full-train models)
          │                             │
          ▼                             ▼
   meta_train.csv                meta_test.csv
   (93k × 12 cols)              (23k × 12 cols)
          │                             │  ──► stage2_logreg_final.pkl
          │                             │  ──► stage2_knn_final.pkl
          │                             │  ──► stage2_dt_final.pkl
          └──────────┬──────────────────┘
                     ▼
          train_meta_classifier.py
              │
              ▼
          stage3_meta_classifier.pkl
              │
              ▼
          baseline_vs_pipeline.py ──► model_comparison_table.csv
```

---

## All Saved Model Files

| File | Created By | Used By | What's Inside |
|------|-----------|---------|--------------|
| `stage1_lr_model.pkl` | `train_linear_regression.py` | `compare_regressor.py` | LinearRegression pipeline (preprocessor + model) |
| `stage1_best_regressor.pkl` | `compare_regressor.py` | Streamlit app (Stage 6) | Copy of the winning regressor (LR) |
| `stage2_logreg_final.pkl` | `final_models_trainer.py` | `baseline_vs_pipeline.py`, Streamlit | LogisticRegression pipeline (scaler + clf) |
| `stage2_knn_final.pkl` | `final_models_trainer.py` | `baseline_vs_pipeline.py`, Streamlit | KNN pipeline (scaler + clf), ~27 MB |
| `stage2_dt_final.pkl` | `final_models_trainer.py` | `baseline_vs_pipeline.py`, Streamlit | DecisionTreeClassifier |
| `stage3_meta_classifier.pkl` | `train_meta_classifier.py` | `baseline_vs_pipeline.py`, Streamlit | LogisticRegression meta-model (scaler + clf) |

---

## All Generated Data Files

| File | Created By | Used By | Shape |
|------|-----------|---------|-------|
| `merged_train.csv` | `step3_build_dataset.py` → `compare_regressor.py` | Stage 2, Stage 3A | 93,975 × 19 |
| `merged_test.csv` | `step3_build_dataset.py` → `compare_regressor.py` | Stage 2, Stage 3A | 23,528 × 19 |
| `meta_train.csv` | `oof_generator.py` | `train_meta_classifier.py` | 93,975 × 12 |
| `meta_test.csv` | `final_models_trainer.py` | `train_meta_classifier.py`, `baseline_vs_pipeline.py` | 23,528 × 12 |
| `lr_oof_predictions.csv` | `train_linear_regression.py` | `compare_regressor.py` | 81,220 × 2 |
| `dt_oof_predictions.csv` | `train_decision_tree_regressor.py` | `compare_regressor.py` | 81,220 × 2 |
