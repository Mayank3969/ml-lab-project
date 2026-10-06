# Collision Severity Accuracy Review and Action Plan

## 0. Scope and conclusion

This plan is based on the source code, the raw tables, the processed CSVs, the
saved model metadata, and the existing evaluation artifacts. It does not use a
feature or model choice merely because it improves one number on one split.
Every candidate must be checked with collision-group-aware validation and then
reported on the untouched held-out test set.

The immediate conclusion is that the low overall result is caused by several
pipeline failures acting together:

1. **The feature-selection result is too narrow.** The selected 11 columns
   were chosen mainly git **merical representation even though**
2. **their geometry is different.** Frequency encoding is compact, but it turns
   a category into its prevalence rather than preserving the category identity.
   That is a poor representation for a linear boundary and for distances.
3. **The metric objective is inconsistent with the reported expectation.** The
   training and stacking models use `class_weight="balanced"` and the evaluation
   report uses macro-F1, while the reported expectation is approximately 90%
   accuracy. With 77.09% Slight cases, a model can increase accuracy by mostly
   predicting Slight while becoming unsafe for Fatal and Serious cases.
4. **The saved models and the inference application do not share a schema.**
   The training runner currently saves models trained on 12 frequency-encoded
   columns. Streamlit sends 16 raw columns plus a raw `risk_score`. The saved
   meta-model expects `lr_prob_*`, `dt_prob_*`, `knn_prob_*`, and
   `number_of_casualties`, while Streamlit sends `P_*` columns and
   `risk_score`.
5. **The repository contains two incompatible pipeline implementations.**
   `pipeline.py` calls `train_risk_model.py` and `train_base_models.py`, while
   `oof_generator.py`, `final_models_trainer.py`, `train_meta_classifier.py`,
   `compare_regressor.py`, `config.py`, the documentation, and Streamlit expect
   different stages and column names. Only one implementation should remain
   authoritative.

The target of 90%+ accuracy is therefore a tuning target, not a result that can
be promised in advance. Reaching it by retaining target-derived columns such as
`collision_severity` would be leakage and would make the model invalid. The
plan below tries to improve accuracy without allowing that shortcut.

---

## 1. Findings from the data and current results

### 1.1 Data audit

The raw data contains:

| Table                         |    Rows | Purpose                                           |
| ----------------------------- | ------: | ------------------------------------------------- |
| `dataset/raw/collision.csv` | 101,525 | Collision-level road and environment data         |
| `dataset/raw/vehicle.csv`   | 183,948 | Vehicle and impact data                           |
| `dataset/raw/casualty.csv`  | 127,883 | Casualty features and`casualty_severity` target |

The raw join used by the project produced 127,883 casualty rows with no
duplicate `(collision_index, vehicle_reference, casualty_reference)` keys. The
current processed split contains 102,305 train rows and 25,578 test rows. The
collision groups do not overlap between train and test.

The training class distribution is:

| Label | Meaning |   Rows |  Share |
| ----: | ------- | -----: | -----: |
|     1 | Fatal   |  1,230 |  1.20% |
|     2 | Serious | 22,206 | 21.71% |
|     3 | Slight  | 78,869 | 77.09% |

This imbalance is the reason accuracy, macro-F1, and Fatal recall tell very
different stories. All three must be reported together.

### 1.2 Existing model results

The current saved evaluation artifact reports:

| Model                  | Accuracy | Macro-F1 | Fatal-F1 |
| ---------------------- | -------: | -------: | -------: |
| Logistic Regression    |   51.70% |   0.3467 |   0.0547 |
| Decision Tree          |   70.40% |   0.3791 |   0.0456 |
| KNN                    |   73.59% |   0.3719 |   0.0251 |
| Full stacking pipeline |   53.00% |   0.3523 |   0.0584 |

The 3-seed report also shows that full stacking has lower macro-F1 than the
standalone Decision Tree and KNN. Therefore, adding more models to the current
meta-classifier is not currently improving the overall classifier.

### 1.3 Verified runtime and schema failures

The repository was checked before changing the plan:

- The system `python3` does not contain pandas or scikit-learn. The bundled
  `venv` is a Windows environment and cannot run as a Linux environment. Use a
  Linux virtual environment or the documented `uv run` equivalent before
  retraining.
- `pipeline.py --from 1 --to 1` starts the child process but fails because of
  the missing dependency. The master process still exits with status 0. The
  runner must return a non-zero status when a child fails.
- `models/utils/preprocessing.py` imports `STAGE1_FEATURES`, `STAGE1_NUM`, and
  `STAGE1_CAT`, but those constants are absent from the current
  `models/utils/config.py`. The older regression scripts are consequently not
  runnable as written.
- The active `train_base_models.py` scales the training arrays before saving
  Logistic Regression and KNN, but `streamlit/inference.py` sends unscaled raw
  data to the saved estimators. A direct check of the saved Logistic model on
  raw rather than scaled inputs fell to approximately 2.30% accuracy.
- The active artifacts use 12 numeric inputs and a meta-feature named
  `number_of_casualties`; Streamlit requires 16 different raw inputs, computes a
  different `risk_score`, and asks the meta-model for `P_logreg_*` and
  `P_dt_*` columns. The artifact, training code, and inference code must be
  regenerated as one version.
- `train_risk_model.py` found that its generated `risk_score` did not improve
  macro-F1 over `number_of_casualties`, and it dropped the risk score. The
  current meta-model uses `number_of_casualties`, not `risk_score`. Do not add
  the risk score back unless a grouped validation experiment proves a gain.

---

# Part 1 — Change and test input columns for each model

## 1.1 Input rules

Use these rules while rebuilding the merged data:

1. Join at casualty level using `collision_index` and
   `(collision_index, vehicle_reference)`.
2. Validate that the collision join is many-to-one and the vehicle join is
   many-to-one. Fail if a join multiplies rows or leaves unexpected unmatched
   keys.
3. Remove identifiers, coordinates, dates, historic fields, and every
   target-derived field before feature selection. In particular, never use
   `collision_severity`, `enhanced_casualty_severity`,
   `casualty_injury_based`, adjusted-severity fields, or their collision-level
   equivalents.
4. Keep `number_of_casualties` only when it is genuinely available at the time
   the prediction is requested. It is a valid incident-context feature for a
   post-collision casualty report, but it is not a valid early-warning input if
   the total is not known yet.
5. Treat `did_police_officer_attend_scene_of_accident` as a separate availability
   decision. It is predictive, but it may be known only after the event and can
   act as a severity proxy. Run both the with-attendance and
   without-attendance variants.
6. Fit every frequency map, imputer, one-hot encoder, and scaler on the fold's
   training rows only. The transformer must be inside the scikit-learn
   `Pipeline` used by cross-validation.

## 1.2 Candidate feature sets

The following sets were tested. `current12` reproduces the active pipeline's
11 selected features plus `number_of_casualties`.

### `compact10` — recommended compact candidate

```text
casualty_type
age_of_casualty
age_of_driver
engine_capacity_cc
speed_limit
urban_or_rural_area
vehicle_manoeuvre
vehicle_leaving_carriageway
first_point_of_impact
number_of_vehicles
```

This set keeps the strongest casualty, driver, road-speed, and impact signals
identified by the existing permutation analysis while avoiding the 52-column
noise benchmark. Run a companion `compact11` variant with
`number_of_casualties` when that value is available.

### `domain27` — recommended broad, interpretable candidate

```text
casualty_class, sex_of_casualty, age_of_casualty, casualty_type,
number_of_vehicles, number_of_casualties, day_of_week, first_road_class,
road_type, speed_limit, junction_detail, junction_control, light_conditions,
weather_conditions, road_surface_conditions, urban_or_rural_area,
did_police_officer_attend_scene_of_accident, vehicle_type, vehicle_manoeuvre,
vehicle_direction_from, vehicle_direction_to, vehicle_leaving_carriageway,
first_point_of_impact, hit_object_off_carriageway, age_of_driver,
engine_capacity_cc, journey_purpose_of_driver
```

Create `domain_no_attendance_count` by removing both
`did_police_officer_attend_scene_of_accident` and `number_of_casualties` for an
early-availability model.

### `permutation17` — targeted recovery candidate

This is the smallest recovery set used in the experiment. It adds the
permutation-supported impact and road fields to the strongest existing
features:

```text
casualty_type, did_police_officer_attend_scene_of_accident,
age_of_casualty, engine_capacity_cc, vehicle_leaving_carriageway,
age_of_driver, speed_limit, first_point_of_impact, vehicle_manoeuvre,
urban_or_rural_area, vehicle_direction_from, vehicle_direction_to,
journey_purpose_of_driver, first_road_class, hit_object_off_carriageway,
number_of_vehicles, number_of_casualties
```

### `all_safe` — benchmark only

Use every remaining non-leakage, non-historic, non-ID, non-zero-variance
column. This is a benchmark to check whether compact selection removed signal;
it is not automatically the production input. It contained 52 fields in this
review.

## 1.3 Input experiment that was run

`models/evaluation/review_feature_inputs.py` was added as the reproducible
review experiment. It reconstructs the raw join, checks the split group IDs,
fits transformations inside each fold, and records accuracy, balanced
accuracy, macro-F1, Fatal-F1, Fatal recall, and Serious recall.

The table below is the verified two-fold `StratifiedGroupKFold` screening run
using the existing tuned base-model parameters. It is a development screen;
the untouched test labels were not used to choose the final production model.

| Input / encoding              | Model               |         Accuracy | Balanced accuracy |         Macro-F1 | Fatal recall |
| ----------------------------- | ------------------- | ---------------: | ----------------: | ---------------: | -----------: |
| `current12` / frequency     | Logistic Regression |           51.99% |            0.5031 |           0.3476 |       0.6463 |
| `current12` / frequency     | Decision Tree       |           69.16% |            0.3798 |           0.3814 |       0.0480 |
| `current12` / frequency     | KNN                 |           73.84% |            0.3716 |           0.3743 |       0.0146 |
| `permutation17` / frequency | Logistic Regression |           55.08% |            0.5527 |           0.3726 |       0.7000 |
| `permutation17` / frequency | Decision Tree       |           69.23% |            0.3841 |           0.3857 |       0.0561 |
| `permutation17` / frequency | KNN                 |           73.92% |            0.3808 |           0.3863 |       0.0236 |
| `domain27` / frequency      | Logistic Regression |           55.50% |            0.5597 |           0.3759 |       0.7065 |
| `domain27` / frequency      | Decision Tree       |           69.27% |            0.3889 |           0.3899 |       0.0618 |
| `domain27` / frequency      | KNN                 |           74.07% |            0.3797 |           0.3860 |       0.0285 |
| `current12` / one-hot       | Logistic Regression |           54.94% |            0.5142 |           0.3672 |       0.5992 |
| `current12` / one-hot       | Decision Tree       |           71.40% |            0.3663 |           0.3682 |       0.0228 |
| `domain27` / one-hot        | Logistic Regression | **57.78%** |            0.5614 | **0.3901** |       0.6545 |
| `domain27` / one-hot        | Decision Tree       |           71.23% |            0.3832 |           0.3873 |       0.0528 |
| `compact10` / one-hot       | Logistic Regression |           54.70% |            0.5473 |           0.3735 |       0.6642 |
| `compact10` / one-hot       | Decision Tree       |           72.31% |            0.3813 |           0.3847 |       0.0610 |
| `compact10` / one-hot       | KNN                 | **74.22%** |            0.3842 | **0.3913** |       0.0358 |

A separate three-fold frequency-encoding check gave `all_safe` Logistic
Regression 56.10% accuracy / 0.3809 macro-F1 and `all_safe` Decision Tree
70.16% accuracy / 0.3929 macro-F1. This confirms that the 11-column Gini
selection is not a safe assumption, but it does not justify sending all 52
fields to every model.

## 1.4 Part 1 decision and next test

Use model-specific candidate inputs rather than forcing every model to use the
same columns:

- **Logistic Regression:** start with `domain27` one-hot encoded and scaled.
  It was the strongest tested linear representation. Also train
  `domain_no_attendance_count` for an early-availability comparison.
- **Decision Tree:** start with `compact10` one-hot encoded for an accuracy
  track, and retain `domain27`/`all_safe` as macro-F1 challengers. Trees do not
  need standard scaling. Select between these with five-fold grouped CV, not
  the final test set.
- **KNN:** start with `compact10` one-hot encoded and scaled. It gave the best
  tested KNN macro-F1 and avoids allowing weak dimensions to dominate distance.
  Keep frequency encoding as a memory/performance baseline.
- **Meta-classifier:** do not pass raw base inputs to the meta layer. Generate
  leak-free OOF probabilities from each model's own transformer and pass only
  those probabilities plus `number_of_casualties` when it is an approved
  prediction-time input. Exclude `risk_score` until it demonstrates a positive
  grouped-CV gain.

The next run must use five folds, with the final test set locked until the
feature choice is frozen. Store the winning feature list and the complete
transformer in one artifact per model. Streamlit must use those artifacts
directly rather than rebuilding a different input DataFrame.

---

# Part 2 — Hyperparameter tuning

## 2.1 Tuning protocol

1. Freeze the feature-availability policy first: post-event model may include
   casualty count and police attendance; early-warning model may not.
2. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with
   `collision_index` as the group. No collision may occur in two folds.
3. Put encoding, missing-value handling, and scaling inside the estimator
   pipeline. Do not call `fit_transform` on the entire training table before
   cross-validation.
4. Search on the development train set only. Select by the agreed refit metric:
   `accuracy` for an accuracy product or `macro_f1` for a balanced safety
   product. Always log the other metrics.
5. Fit the selected pipeline on all development rows, evaluate exactly once on
   the locked test set, and save the confusion matrix and per-class report.
6. Tune the meta-classifier only after the base-model OOF predictions are
   regenerated with the new model-specific inputs.

## 2.2 Search spaces

### Logistic Regression

Search `C` on a logarithmic scale from `1e-3` to `1e2`, `class_weight` in
`None`, `balanced`, and modest explicit weights such as `{1: 5, 2: 2, 3: 1}`.
Compare L1 and L2 penalties only when the selected solver supports them. The
balanced option is expected to lower accuracy while improving Fatal recall, so
it must not be selected silently when accuracy is the requirement.

### Decision Tree

Search `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`,
`criterion`, `ccp_alpha`, and `class_weight`. A fully grown tree can fit the
large training set very well while generalizing poorly; report the train/CV
gap and do not select on train accuracy.

### KNN

Search `n_neighbors` across small and large neighborhoods, `weights` in
`uniform`/`distance`, and Minkowski `p` in `1`/`2`. Use one-hot plus scaling for
the compact candidate. Use `algorithm="auto"`; force `brute` only if sparse
one-hot matrices make the tree backend slower or invalid.

### Meta-classifier

Search Logistic Regression `C`, class weight, and the final decision policy.
A separate threshold/utility experiment should compare default argmax with a
Fatal-recall constrained rule. The chosen rule must still report accuracy,
macro-F1, Fatal-F1, Fatal recall, Serious recall, and Slight recall.

## 2.3 Executable tuning implementation

The reusable implementation is in
`models/evaluation/review_feature_inputs.py`. It includes:

- raw three-table merge validation;
- model-specific feature sets;
- fold-local frequency encoding or one-hot encoding;
- grouped feature screening;
- randomized searches for Logistic Regression, Decision Tree, and KNN;
- multi-metric scoring;
- saved search results, selected pipelines, classification reports, and
  confusion matrices;
- an explicit `accuracy_90_reached` field rather than an unsupported claim.

Run the input screen first:

```bash
uv run --with pandas --with scipy --with scikit-learn --with joblib \
  python models/evaluation/review_feature_inputs.py \
  --mode features \
  --models logreg dt knn \
  --feature-sets current12 permutation17 domain27 compact10 all_safe \
  --encodings frequency onehot \
  --folds 5 \
  --output-dir models/evaluation/results/input_review
```

After the feature policy is frozen, run the tuning search. This example tunes
the early-availability set for balanced quality; change `--refit accuracy` for
an accuracy-first model, and use `--allow-casualty-count` only when the count is
available at inference time:

```bash
uv run --with pandas --with scipy --with scikit-learn --with joblib \
  python models/evaluation/review_feature_inputs.py \
  --mode tune \
  --models logreg dt knn \
  --tune-features domain_no_attendance_count \
  --tune-encoding onehot \
  --refit macro_f1 \
  --folds 5 \
  --iterations 60 \
  --jobs 2 \
  --output-dir models/evaluation/results/input_review/tuning
```

The essential tuning pattern is:

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from scipy.stats import loguniform

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": make_scorer(
        f1_score, labels=[1, 2, 3], average="macro", zero_division=0
    ),
}

search = RandomizedSearchCV(
    estimator=model_pipeline,                 # encoder + scaler + classifier
    param_distributions={
        "clf__C": loguniform(1e-3, 1e2),
        "clf__class_weight": [None, "balanced", {1: 5, 2: 2, 3: 1}],
    },
    n_iter=60,
    scoring=scoring,
    refit="accuracy",                         # or "macro_f1"
    cv=cv,
    random_state=42,
    n_jobs=2,
    return_train_score=True,
)

search.fit(X_development, y_development, groups=collision_index)
test_prediction = search.predict(X_locked_test)
print(search.best_params_)
print(search.best_score_)
print(accuracy_score(y_locked_test, test_prediction))
print(f1_score(y_locked_test, test_prediction, average="macro"))
```

For the Decision Tree and KNN, replace the classifier search space with the
spaces described above. The actual file contains all three spaces and saves
the complete fitted pipelines, so the saved object includes the exact input
transformer required by inference.

## 2.4 What counts as success

Do not call a model successful based on accuracy alone:

- **Accuracy track:** beat the 77.09% Slight-majority baseline without reducing
  Fatal recall to zero, then check whether 90% is reached on the locked test.
- **Balanced track:** improve macro-F1 and Fatal/Serious recall over the current
  stack while retaining acceptable Slight recall.
- **Operational track:** report a confusion matrix, class-specific recall,
  calibration, and performance across at least three group-aware seeds.

If no leakage-safe experiment reaches 90%, record the highest honest result and
the metric trade-off. A 90% number produced by target-derived fields is not an
acceptable improvement.

---

# Part 3 — Miscellaneous corrections and delivery plan

## 3.1 Consolidate the pipeline

Choose one active execution path. The complete demonstration order is:

1. `build_dataset.py`: raw joins, leakage removal, regression tables, feature
   manifest, and locked grouped split.
2. `train_risk_models.py`: manually selected Linear Regression and Decision
   Tree Regressor risk stages.
3. `train_base_models.py`: manually selected Logistic Regression, KNN, and
   Decision Tree with grouped OOF probabilities.
4. `train_meta_ensemble.py`: final model using nine probabilities plus the
   manually selected risk score.
5. `evaluate_pipeline.py`: locked test metrics and confusion matrices.
6. Streamlit inference: load the six stage-numbered artifacts and feature
   contract.

Deprecate or rewrite the duplicate regression and classifier scripts after the
new path is verified. Do not leave scripts with different feature names in the
documented run order.

## 3.2 Repair the artifact contract

For each saved model, serialize one pipeline containing:

- ordered raw feature names;
- imputation policy;
- encoder or frequency map;
- scaler, if required;
- classifier and selected hyperparameters;
- training data hash and random seed.

The meta artifact must declare its exact input names. Use one naming convention,
for example `logreg_p_fatal`, `logreg_p_serious`, `logreg_p_slight`, and the
equivalent names for KNN and Decision Tree. Update Streamlit to read the
manifest instead of maintaining a second hard-coded feature list.

## 3.3 Repair evaluation and logging

- Make `pipeline.py` return `sys.exit(1)` when any child step fails.
- Replace the `1/9` banner with the real number of active stages.
- Make the output snapshot describe the exact generated schema.
- Add assertions for row count, join multiplicity, group overlap, missing
  values, feature order, and class labels.
- Remove the current scaler leakage: a scaler must be fitted inside each CV
  fold, not on the entire training table before `cross_val_predict`.
- Keep the locked test split out of feature and hyperparameter selection.
- Re-run reports after every artifact regeneration so old metrics cannot be
  displayed beside new models.

## 3.4 Review the risk-score stage

The current casualty-count regressor has weak predictive value and its
ablation did not improve macro-F1. Keep `number_of_casualties` only if it is a
valid input at prediction time. Otherwise remove both the count and the
derived `risk_score` from the early-warning feature contract. Do not create a
second model stage merely because it exists in the original architecture.

## 3.5 Streamlit changes

Before deploying the dashboard:

1. Replace the 16-column form/batch contract with the selected manifest for
   the chosen model version.
2. Add the casualty and vehicle fields needed by `compact10` or
   `domain_no_attendance_count`, or retrain on a genuinely smaller UI schema.
3. Call each saved pipeline with raw columns in manifest order; never apply a
   second manual scaler or frequency map.
4. Change the sample-test loader to use the same schema used by training.
5. Display the locked-test metrics from the current model version, not stale
   values such as the older 62.2% Fatal recall text.

## 3.6 Final execution order

```text
1. Install a Linux-compatible environment and run compile/import checks.
2. Rebuild the raw join with validation and freeze the group split.
3. Run Part 1 feature/encoding screens with five grouped folds.
4. Freeze separate input manifests for LR, DT, KNN, and the meta layer.
5. Run Part 2 hyperparameter searches on development data only.
6. Regenerate leak-free OOF base probabilities.
7. Tune and fit the meta-classifier.
8. Evaluate once on the locked test and repeat across three seeds.
9. Regenerate all artifacts and reports from that one pipeline.
10. Run Streamlit smoke tests using a held-out sample with the saved manifests.
```

The first implementation milestone is complete when the selected model inputs,
transformers, and metrics are reproducible from a clean checkout. The 90%+
accuracy claim should only be made if it is achieved on the locked test with
the same leakage-safe input contract used by the application.

---

# Implementation file and function map

The active implementation uses an **immediately-after-collision** contract. It
does not use confirmed severity, `number_of_casualties`, or police attendance.
The raw data currently contains 1,538 Fatal rows out of 127,883 rows, or
**1.20%**. The split builder searches deterministic grouped seeds and selects a
split whose Fatal rate stays within 0.10 percentage points of the observed
rate. It also writes the exact fold counts to `dataset/processed/split_audit.csv`.

## Active Python files

| File                                                     | Main responsibility                                                                                                                                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models/stage1_merge/build_dataset.py`                 | Validates the three joins, removes identifiers and target-derived columns, selects the immediately-after-collision contract, creates the grouped split, and writes the manifest/audit. |
| `models/stage1_regression/train_risk_models.py`        | Trains the manually selected Linear Regression and Decision Tree Regressor, creates collision-level OOF risk scores, and writes stage-1 metrics.                                       |
| `models/utils/config.py`                               | Stores paths, labels, split settings, and the three model-specific feature lists.                                                                                                      |
| `models/utils/manual_parameters.py`                    | Single editable parameter file used by the full demonstration run after notebook tuning.                                                                                               |
| `models/utils/preprocessing.py`                        | Loads prepared data, builds fold-local encoders, creates grouped folds, and calculates shared metrics.                                                                                 |
| `models/stage3a_base_classifiers/train_base_models.py` | Trains the three manually selected classifiers, creates nine leak-free OOF probabilities, and writes base metrics.                                                                     |
| `models/stage4_meta_classifier/train_meta_ensemble.py` | Fits the final ensemble on nine probabilities plus the selected risk score.                                                                                                            |
| `models/evaluation/evaluate_pipeline.py`               | Produces the common Accuracy/Precision/Recall/F1 comparison and confusion matrices.                                                                                                    |
| `streamlit/inference.py`                               | Loads the three classifiers, two regressors, and final ensemble, validates raw columns, and creates the meta vector.                                                                   |
| `streamlit/app.py`                                     | Displays project information/workflow and provides simple single-record and CSV testing.                                                                                               |
| `reset_generated.py`                                   | Deletes generated data, models, tuning results, and reports while preserving source and raw data.                                                                                      |
| `pipeline.py`                                          | Runs all five authoritative stages, optionally resets generated files, and exits non-zero on failure.                                                                                  |

## Active function map

### `build_dataset.py`

- `read_raw_tables()` reads raw STATS19 files with string join keys.
- `join_tables()` performs validated many-to-one joins and rejects row
  multiplication.
- `clean_features()` removes IDs, historic fields, target-derived fields,
  post-event proxy fields, and zero-variance columns.
- `choose_group_split()` tests deterministic grouped stratification seeds and
  chooses the closest class-proportion result.
- `audit_splits()` records class counts, rates, Fatal deviations, and group
  overlap for every fold.
- `save_outputs()` writes train/test CSVs, the feature manifest, and the split
  audit.
- `main()` runs those steps in order.

### `preprocessing.py`

- `load_processed()` reads the prepared CSVs.
- `load_feature_manifest()` reads the active contract.
- `model_feature_names()` returns the ordered raw columns for one model.
- `categorical_and_numeric()` separates columns by model preprocessing needs.
- `build_preprocessor()` builds the encoder/scaler inside the model pipeline.
- `grouped_splits()` creates and validates collision-group-aware folds.
- `evaluate_predictions()` returns the common accuracy, macro-F1, and recall
  metrics.

### `train_risk_models.py`

- `build_model()` creates the selected regression pipeline.
- `train_one()` generates OOF/test risk scores and regression metrics.
- `main()` trains both risk regressors and writes the comparison CSV.

### `train_base_models.py`

- `build_model()` creates each model's own preprocessing/classifier pipeline.
- `ordered_probabilities()` aligns estimator class order to Fatal/Serious/Slight.
- `metrics()` calculates the common base-model metric set.
- `load_risk_scores()` maps collision-level regression OOF values to casualty rows.
- `main()` creates OOF predictions, final base models, meta tables, and metrics.

### `train_meta_ensemble.py`

- `metrics()` calculates the final ensemble comparison metrics.
- `main()` fits the final ensemble using `manual_parameters.py`, saves the
  contract, metrics, and confusion matrix.

### `evaluate_pipeline.py`

- `metrics()` calculates Accuracy, balanced accuracy, macro/weighted Precision,
  Recall, F1, and per-class recall.
- `main()` evaluates all base models and the final ensemble together.

### `inference.py`

- `ModelArtifacts.load()` loads model-specific pipelines and the meta model.
- `required_columns()` returns the union of raw input fields for validation.
- `_ordered_probabilities()` aligns probability columns.
- `predict_pipeline()` executes the full model workflow for a DataFrame.
- `predict_single()` adapts one dictionary to the same workflow.

## Hyperparameter notebook map

| Notebook                                                 | Input representation                                  | Tuned model                                                                | Outputs                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `hyper parameter/linear_regression_tuning.ipynb`       | Collision-level road/environment inputs               | Linear Regression risk regressor                                           | OOF/test risk scores, regression CV rows,`linear_regression_1.pkl`       |
| `hyper parameter/decision_tree_regressor_tuning.ipynb` | Same collision-level inputs with nonlinear splits     | Decision Tree Regressor risk model                                         | OOF/test risk scores, regression CV rows,`decision_tree_regressor_1.pkl` |
| `hyper parameter/logistic_regression_tuning.ipynb`     | Broad one-hot domain plus scaled numeric values       | Logistic Regression, class weights, penalty strength                       | All CV rows, best params, test metrics,`logistic_regression_best.pkl`    |
| `hyper parameter/knn_tuning.ipynb`                     | Compact one-hot distance space plus scaling           | KNN neighborhood size, weights, Minkowski distance                         | All CV rows, best params, test metrics,`knn_best.pkl`                    |
| `hyper parameter/decision_tree_tuning.ipynb`           | Interpretable one-hot road/vehicle domain             | Depth, leaves, splits, pruning, criterion, feature sampling, class weights | All CV rows, best params, test metrics,`decision_tree_best.pkl`          |
| `hyper parameter/meta_ensemble_tuning.ipynb`           | Nine OOF probabilities plus each candidate risk score | Final Logistic Regression ensemble and risk-source choice                  | Candidate comparison, final metrics,`meta_ensemble_3.pkl`                |

Each notebook is intentionally independent so a model can be changed or
re-run manually without changing the other two. Each saves a complete pipeline
that includes preprocessing, so the application never applies a different
scaler or encoder.

## Final implementation status for handoff

The tuning notebooks have **not** been executed in this review. This is
intentional: the full grids are prepared for the user's Windows overnight run
after the code is pushed. Validation performed here covers notebook JSON/code
syntax, model-pipeline construction, prepared-column availability, grouped
split integrity, and class-rate auditing only.

The final active contract supersedes the earlier exploratory feature tables in
Part 1:

- prediction timing is immediately after collision;
- raw `number_of_casualties` and `did_police_officer_attend_scene_of_accident`
  are excluded from classification inputs; the regressors create a derived
  `risk_score` candidate for the final ensemble;
- all three classes are audited in each grouped fold;
- Fatal is measured at the actual **1.20%** data rate, not assumed to be 1.7%;
- no collision group is shared between train/test or an OOF fit/validation
  fold;
- each model receives its own feature list and encoder from the active
  notebooks;
- the final artifacts are expected only after the tuning notebooks and then
  the five-stage manual-parameter pipeline have been run.

## Complete manual-parameter demonstration pipeline

The faculty demonstration path is now controlled by
`models/utils/manual_parameters.py`. Copy the selected values from the tuning
notebooks into that one file. Then run:

```powershell
python pipeline.py --clean
```

The complete five-stage run is:

1. `models/stage1_merge/build_dataset.py` — raw joins, leakage removal,
   grouped split, collision-level regression tables.
2. `models/stage1_regression/train_risk_models.py` — Linear Regression and
   Decision Tree Regressor, OOF risk scores, test risk scores, and regression
   metrics.
3. `models/stage3a_base_classifiers/train_base_models.py` — Logistic
   Regression, KNN, and Decision Tree OOF probabilities, base-model metrics,
   and meta input tables.
4. `models/stage4_meta_classifier/train_meta_ensemble.py` — final ensemble
   using the selected risk score plus nine base probabilities.
5. `models/evaluation/evaluate_pipeline.py` — common metric table, reports,
   and confusion matrices for every base model and the ensemble.

The Streamlit pages are **Home**, **Workflow**, **Model metrics**, **Single
prediction**, and **CSV testing**. Workflow displays the complete Mermaid data
flow. Model metrics reads `model_comparison.csv`, shows Accuracy, Precision,
Recall, F1, balanced accuracy, per-class recall, classification reports, and
confusion matrices for all three base models and the final ensemble.

The simple stage-numbered artifact names are:

```text
models/saved_models/linear_regression_1.pkl
models/saved_models/decision_tree_regressor_1.pkl
models/saved_models/logistic_regression_2.pkl
models/saved_models/knn_2.pkl
models/saved_models/decision_tree_2.pkl
models/saved_models/meta_ensemble_3.pkl
```

`reset_generated.py --dry-run` previews generated files. `reset_generated.py --yes` or `delete_generated.bat` deletes generated data, models, metrics,
reports, and tuning outputs while preserving raw CSVs, source code, notebooks,
and documentation. `pipeline.py --clean` performs that reset before rebuilding
everything.

`run_full_pipeline.bat` is the Windows one-click equivalent of
`python pipeline.py --clean`.
