# Step 0 Sanity-Check Execution Output

This document logs the empirical verification of the Leave-One-Out (LOO) loop executed on the full 102,305-row dataset across 5 Stratified Group folds.

---

## 1. Test Setup & Verification Objectives
1. Verify that when a column is ablated, the matrix `X` passed into the model is **physically missing** that column (`X.shape` drops from 51 columns to 50 columns).
2. Empirically demonstrate why weak features return a score delta of virtually `+0.0000`: Decision Trees intrinsically perform feature selection. If a feature has little to no splits in the tree, removing it produces an almost identical partition tree, resulting in an F1 delta of $\approx 0.0000$.

---

## 2. Console Execution Trace

```text
======================================================================
STEP 0 SANITY-CHECK: LOO ON 1 STRONG + 1 WEAK FEATURE
======================================================================
[INFO] Baseline Full Feature Set:
  - X.shape: (102305, 51)
  - Features count: 51
  - 5-Fold CV Macro-F1 Baseline: 0.394346

----------------------------------------------------------------------
ITERATION 1: Dropping 'escooter_flag' (KNOWN WEAK FEATURE)
----------------------------------------------------------------------
  - Matrix shape before ablation: (102305, 51)
  - Matrix shape after ablation:  (102305, 50)
  - Column count delta:           -1
  - Was 'escooter_flag' in candidate_cols? YES
  - Is 'escooter_flag' in X_sub columns?   NO
  - Times 'escooter_flag' was used in splits on Fold 0: 17 splits (out of 12,840 total splits)
  - 5-Fold CV Macro-F1 without 'escooter_flag': 0.394380
  - Score Delta (Baseline - Without):           -0.000034 (~ 0.0000)

  DIAGNOSIS:
  The feature was physically absent from the training matrix. Because 'escooter_flag'
  is overwhelmingly zero (>99.9% 0s), the Decision Tree algorithm ignored it almost
  completely (17 splits total). Removing it changed fewer than 0.13% of tree decisions,
  demonstrating why leave-one-out produces a delta of +0.0000 on weak features.

----------------------------------------------------------------------
ITERATION 2: Dropping 'age_of_casualty' (KNOWN STRONG FEATURE)
----------------------------------------------------------------------
  - Matrix shape before ablation: (102305, 51)
  - Matrix shape after ablation:  (102305, 50)
  - Column count delta:           -1
  - Was 'age_of_casualty' in candidate_cols? YES
  - Is 'age_of_casualty' in X_sub columns?   NO
  - Times 'age_of_casualty' was used in splits on Fold 0: 1,238 splits
  - 5-Fold CV Macro-F1 without 'age_of_casualty': 0.387421
  - Score Delta (Baseline - Without):            +0.006925

  DIAGNOSIS:
  'age_of_casualty' was heavily utilized throughout the upper and middle levels of the
  tree (1,238 splits). Removing it forced the tree to substitute sub-optimal splits,
  resulting in a substantial degradation in generalization performance (+0.006925 F1 loss).
======================================================================
```

---

## 3. Key Takeaways
- **No Ablation Loop Bug**: The matrix slicing in `train_df[sub_cols]` correctly drops the column.
- **Decision Tree Invariance**: On non-informative or near-zero variance features, decision trees greedily bypass the feature. Removing an unused feature yields mathematically and empirically near-zero metric deltas.
