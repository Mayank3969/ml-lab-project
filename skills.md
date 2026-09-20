---
name: road-accident-severity-pipeline
description: Stacked ensemble pipeline predicting casualty severity (Fatal/Serious/Slight) from UK STATS19-style road accident data. Use this as ground truth to generate or cross-check code, stage by stage.
---

# Road Accident Severity Prediction — Stacked Pipeline Spec

## Goal
Predict `casualty_severity` (Fatal / Serious / Slight) from collision/vehicle/casualty
data. Not a naive "train 6 models, compare accuracy" project — a staged pipeline where
a regression stage engineers a feature, base classifiers consume it, and a
meta-classifier combines base-classifier outputs + that feature for the final call.

## Dataset
- 3 raw tables: `collision.csv`, `vehicle.csv`, `casualty.csv`
- Join key: `collision_index` (+ `vehicle_reference` for casualty↔vehicle)
- Target: `casualty_severity` ∈ {Fatal, Serious, Slight}
- Known class imbalance: ~76% Slight / ~22.7% Serious / ~1.2% Fatal → use macro-F1
  as the primary metric, not plain accuracy.

## Directory structure
```
project-root/
├── dataset/
│   ├── raw/                        # collision.csv, vehicle.csv, casualty.csv
│   ├── processed/
│   │   ├── merged_train.csv        # stage 1 output
│   │   ├── merged_test.csv
│   │   ├── meta_train.csv          # stage 3A+2 output, train side
│   │   └── meta_test.csv           # stage 3A+2 output, test side
│   └── README.md
├── models/
│   ├── utils/
│   │   ├── preprocessing.py
│   │   └── config.py               # fold count=5, seed, column lists
│   ├── stage1_merge/
│   ├── stage2_regression/
│   ├── stage3a_base_classifiers/
│   │   ├── oof_generator.py        # Phase A
│   │   └── final_models_trainer.py # Phase B
│   ├── stage4_meta_classifier/
│   ├── saved_models/                # all .pkl artifacts, incl. encoders/scaler
│   └── evaluation/
├── streamlit/
├── docs/
│   └── architecture_diagram.png
├── requirements.txt
└── README.md
```

---

## Stage 1 — EDA, merge, feature selection

**Input:** 3 raw tables.
**Processing:**
1. Force `collision_index` (and any other ID column) to string dtype on load. Pandas
   parses 13-digit IDs as float/scientific notation by default — this silently
   collapses distinct accidents into the same ID and inflates joins with duplicates.
2. Merge at casualty level: `casualty ← vehicle ← collision`, joined on
   `collision_index` (+ `vehicle_reference` for the vehicle join).
3. Drop leakage columns — anything derived from or duplicating the target:
   `enhanced_casualty_severity`, `casualty_adjusted_severity_serious`,
   `casualty_adjusted_severity_slight`, `collision_severity`, `casualty_injury_based`,
   `enhanced_severity_collision`, `collision_injury_based`,
   `collision_adjusted_severity_serious`, `collision_adjusted_severity_slight`
   (confirm final count — should land near 9 columns dropped for this reason).
4. Drop irrelevant/leaky-by-memorization columns: GPS coordinates
   (`location_easting_osgr`, `location_northing_osgr`, `longitude`, `latitude`),
   road numbers, vehicle make/model, driver demographics, LSOA codes.
5. Encode remaining categoricals (weather, road_type, light_conditions,
   junction_detail, etc.) — save the encoder object, streamlit needs it later.
6. **Split train/test using `GroupShuffleSplit` grouped on `collision_index`**,
   80/20. Do NOT use a plain random split — casualties from the same crash share
   near-identical features; a random split would let some land in train and others
   in test, leaking information across the split.

**Model:** none.
**Output:** `merged_train.csv`, `merged_test.csv`.
**Output columns:** ~19 features + `y` (casualty_severity). Save the fitted
encoder(s) to `models/saved_models/`.

---

## Stage 2 — Regression risk_score

**Input:** Stage 1's ~19 features, collapsed to one row per collision (dedupe —
`number_of_casualties` is identical for every casualty in the same crash).
**Processing:**
1. Train Linear Regression and Decision Tree Regressor:
   features → `number_of_casualties`.
2. Compare via RMSE / R². (Reference result: Linear Regression R²=0.0232 beat
   Decision Tree R²=-0.0283 — expect a low R², casualty count is noisy;
   the feature is still worth keeping, just don't expect it to carry much signal.)
3. Generate the TRAIN-side risk_score via 5-fold OOF (same leak-avoidance logic as
   stage 3A below) — do not let a row's risk_score come from a model that saw
   that row's true casualty count during training.
4. Generate the TEST-side risk_score from the model retrained on full train data.
5. Map the predicted number back onto every individual casualty row via
   `collision_index`.

**Model:** Linear Regression (winner) — save it to `saved_models/`.
**Output:** one new column, `risk_score`, on every row of both train and test sets.
**Output columns:** 1.

---

## Stage 3A — Base classifiers (Phase A + Phase B)

**Input:** Stage 1's ~19 features + `risk_score` = ~20 columns.
**Models:** Logistic Regression, KNN, Decision Tree. **No Perceptron.**

### Phase A — builds the TRAIN-side 9 columns
1. Split TRAIN rows into 5 folds (D1..D5).
2. For each of 5 rounds: train fresh copies of all 3 models on the other 4 folds,
   predict (`predict_proba`) on the held-out fold.
3. Stitch the 5 rounds' held-out predictions back into original row order.
4. **Delete all 15 temporary models (3 algorithms × 5 rounds)** — they only exist to
   produce this table.

### Phase B — builds the TEST-side 9 columns
1. Train 3 fresh, final models on the FULL train set, no folding.
2. Predict (`predict_proba`) on the TEST set directly.
3. **Save these 3 final models** to `saved_models/` — these are the ones the
   Streamlit app will load.

**Output columns (9, same shape for train and test):**
```
P_logreg_fatal, P_logreg_serious, P_logreg_slight,
P_knn_fatal,    P_knn_serious,    P_knn_slight,
P_dt_fatal,     P_dt_serious,     P_dt_slight
```

⚠️ Do not build a "Stage 3B" that trains a new classifier on these 9 columns
without its own OOF treatment — that reintroduces the same leakage Phase A/B was
built to avoid, for uncertain benefit. Go straight to Stage 4.

---

## Stage 4 — Meta-classifier

**Input:** Stage 3A's 9 columns + `risk_score` = **10 columns.**
**Model:** Logistic Regression (meta-classifier).
**Processing:**
1. Train on the 10-column TRAIN table (`X_meta_train`) + true labels.
2. Predict on the 10-column TEST table (`X_meta_test`).

**Output:** final severity class + (optionally) its 3 class probabilities.
**Output columns:** 1 label (+3 probabilities if reported). Save this model to
`saved_models/`.

---

## Stage 5 — Evaluation

Compare, in this order:
1. Regression models against each other — RMSE, R².
2. The 3 base classifiers standalone (Phase B's direct predictions vs true test
   labels) — accuracy + macro-F1 each.
3. Best standalone classifier vs Stage 4's final output — accuracy + macro-F1.
   **This delta is the headline result — the whole reason the pipeline exists.**
4. Confusion matrix on the final model — check Fatal-class recall specifically,
   it's the rarest and most safety-critical class.

**Model:** none — metrics/plots only.

---

## Stage 6 — Streamlit app

**Input:** saved `.pkl` artifacts from every stage (encoder, stage 2 regressor,
stage 3A's 3 final classifiers, stage 4 meta-classifier) + live user form input.

**Inference order at request time (must match training order exactly):**
1. Raw user input → apply saved encoder (same one from stage 1).
2. Encoded features → stage 2 model → `risk_score`.
3. Encoded features + `risk_score` → each of stage 3A's 3 final models →
   9 probability columns.
4. 9 columns + `risk_score` (10 total) → stage 4 meta-model → final prediction.

**Display:** final severity class, probability breakdown per class, optionally
each base model's individual vote/probabilities and the risk_score value, so a
viewer can see *why* the final answer came out the way it did.

---

## Do-not-break checklist (use this to cross-check generated code)
- [ ] `collision_index` loaded as string, not float/int
- [ ] Split is `GroupShuffleSplit` on `collision_index`, not random
- [ ] All leakage columns dropped before any model sees the data
- [ ] Only 3 base classifiers in stage 3A (no Perceptron)
- [ ] Phase A's 15 temporary models are never reused after building `meta_train`
- [ ] No row's Phase-A prediction comes from a model trained on that row's fold
- [ ] Stage 3A output = exactly 9 columns, both train and test
- [ ] Stage 4 input = exactly 10 columns (9 + risk_score)
- [ ] No "Stage 3B" — 3A feeds Stage 4 directly
- [ ] Stage 5 reports macro-F1, not just accuracy
- [ ] Streamlit's inference order matches training order exactly (encode → stage2
      → stage3A → stage4)

## How to use this with Antigravity
1. Paste this whole file as project context/spec.
2. Ask it to scaffold the directory structure first, before writing any logic.
3. Generate code one stage at a time, not the whole pipeline in one shot.
4. After each stage, assert the output shape matches this doc
   (e.g. `assert X_meta_train.shape[1] == 10`) before moving to the next stage.
5. Run the checklist above against the generated code before accepting it.

---

## AI-Learning-Toolkit Integration

The `AI-Learning-Toolkit/` folder is a local learning companion (git-ignored).
Before generating or reviewing code for any stage, **read the relevant files below**
to catch common mistakes and reinforce concepts.

### Folder structure
```
AI-Learning-Toolkit/
└── learning/
    ├── README.md        # Master reference: pipeline theory, stage-by-stage
    │                      explanations, sklearn API patterns, and gotchas
    ├── concepts.md      # Key ML concepts used in this project (OOF, stacking,
    │                      GroupShuffleSplit, macro-F1, leakage prevention)
    ├── mistakes.md      # Common mistakes made during development and their
    │                      fixes (float collision_index, random split leakage,
    │                      Perceptron inclusion, wrong column counts)
    ├── questions.md     # Conceptual questions to self-test understanding
    │                      before moving to the next stage
    └── review.md        # Post-stage review checklist — run after each stage
                           to verify correctness before proceeding
```

### When to read what
| Before building…       | Read these first                                  |
|------------------------|---------------------------------------------------|
| Stage 3A (base classifiers) | `concepts.md` (OOF, stacking), `mistakes.md` |
| Stage 4 (meta-classifier)   | `concepts.md` (meta-learner), `review.md`    |
| Stage 5 (evaluation)        | `concepts.md` (macro-F1), `questions.md`     |
| Stage 6 (Streamlit)         | `mistakes.md` (inference order), `review.md` |

### Workflow
1. **Read** the relevant toolkit files for the current stage.
2. **Generate** code following skills.md spec + toolkit guidance.
3. **Run** the code and verify output shapes/metrics.
4. **Review** using `review.md` checklist.
5. **Update** `mistakes.md` with any new issues encountered.
6. **Update** `concepts.md` if new concepts were learned.

> If the `AI-Learning-Toolkit/` folder doesn't exist yet, create it with
> the structure above and populate files as you learn from each stage.

---

## Progress tracker
- [x] Stage 1 — EDA, merge, feature selection (merged_train.csv + merged_test.csv)
- [x] Stage 2 — Regression risk_score (LR R²=0.023, risk_score in both CSVs)
- [x] Stage 3A — Base classifiers Phase A (oof_generator.py → meta_train.csv)
- [x] Stage 3A — Base classifiers Phase B (final_models_trainer.py → meta_test.csv + 3 .pkl)
- [x] Stage 4 — Meta-classifier (train_meta_classifier.py → final prediction)
- [x] Stage 5 — Evaluation (baseline_vs_pipeline.py → metrics + confusion matrix)
- [x] Stage 6 — Streamlit app (inference pipeline matching training order)

