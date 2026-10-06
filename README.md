# Collision Severity Prediction

This project predicts UK STATS19 casualty severity as Fatal, Serious, or Slight
using a leakage-aware, collision-grouped pipeline.

## Normal faculty workflow

1. Run the six tuning notebooks in `hyper parameter/` overnight.
2. Copy selected values into `models/utils/manual_parameters.py`.
3. Build stages 1–3 and create OOF inputs:

   ```powershell
   python pipeline.py --clean --to 3
   ```

4. Run `hyper parameter/meta_ensemble_tuning.ipynb`.
5. Copy the selected final ensemble parameters and risk source into
   `manual_parameters.py`.
6. Finish the final model and evaluation:

   ```powershell
   python pipeline.py --from 4
   ```

After all parameters are fixed, a complete fresh demonstration is:

```powershell
python pipeline.py --clean
```

Windows shortcut:

```text
run_full_pipeline.bat
```

## Database reuse

The prepared database is made by:

```powershell
python models\stage1_merge\build_dataset.py
```

If all required processed files already exist, this command reuses them and
does not read/rebuild the raw database. To intentionally rebuild from raw CSVs:

```powershell
python pipeline.py --clean --to 1
```

Normal repeated runs should use:

```powershell
python pipeline.py
```

Stage 1 will only rebuild when a required processed file is missing or when
`--clean` is supplied.

## Pipeline stages

1. Validated raw-table join and grouped split.
2. Linear Regression and Decision Tree Regressor risk scores.
3. Logistic Regression, KNN, and Decision Tree OOF probabilities.
4. Final ensemble using nine probabilities plus the selected risk score.
5. Evaluation for every base model and the ensemble.

## Generated model artifacts

```text
models/saved_models/linear_regression_1.pkl
models/saved_models/decision_tree_regressor_1.pkl
models/saved_models/logistic_regression_2.pkl
models/saved_models/knn_2.pkl
models/saved_models/decision_tree_2.pkl
models/saved_models/meta_ensemble_3.pkl
```

## Results

The final comparison is written to:

```text
models/evaluation/results/model_comparison.csv
```

It includes Accuracy, balanced accuracy, macro/weighted Precision, Recall, F1,
and per-class recall. Base-model OOF versus locked-test results are written to:

```text
models/evaluation/results/base_model_metrics.csv
```

Regression stages also report R², RMSE, and MAE.

## Streamlit

Start the dashboard with:

```powershell
streamlit run streamlit\app.py
```

It contains Home, Workflow, Model metrics, Single prediction, and CSV testing.
The Workflow page includes the complete data-flow chart. The Model metrics page
shows the base-model and ensemble comparisons, classification reports, and
confusion matrices.

## Reset generated files

Preview the reset:

```powershell
python reset_generated.py --dry-run
```

Delete generated data/models/results while preserving raw data and source:

```powershell
python reset_generated.py --yes
```

The files named `redundant_*` are old implementations retained temporarily for
comparison. The full file/function guide is in `action plan.md`.
