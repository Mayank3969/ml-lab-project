# Collision Severity Prediction

An end-to-end machine-learning product for predicting UK road-collision
casualty severity from STATS19 collision data.

The project combines:

- a reproducible training pipeline;
- leakage-aware feature selection and grouped train/test splitting;
- a risk-score regression model;
- three complementary base classifiers;
- an out-of-fold stacking meta-classifier;
- a Streamlit dashboard for single-incident and batch prediction;
- evaluation reports, confusion matrices, and a complete execution log.

The model predicts one of three classes:

| Code | Prediction |
| ---: | --- |
| 1 | Fatal |
| 2 | Serious |
| 3 | Slight |

> **Important:** This is a research and decision-support system. It is not a
> substitute for emergency services, professional road-safety analysis, or
> operational risk assessment. Predictions are statistical estimates and can
> be wrong.

## Product workflow

```text
Raw STATS19 CSV files
        |
        v
1. Join, clean, select features, and create grouped train/test data
        |
        v
2. Train a casualty-count regressor and create risk_score
        |
        v
3. Train Logistic Regression, KNN, and Decision Tree base classifiers
        |
        v
4. Train the stacking meta-classifier
        |
        v
5. Evaluate baselines against the stacked pipeline
        |
        v
Streamlit dashboard -> single prediction or batch prediction
```

The dashboard uses the saved model artifacts in `models/saved_models/`; it
does not retrain models when it starts.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- The raw UK STATS19 CSV files:
  - `collision.csv`
  - `vehicle.csv`
  - `casualty.csv`
- Enough disk space for processed datasets and serialized models
- Optional: `openpyxl` for Excel upload/download support in the dashboard

The project uses `pandas`, `NumPy`, `SciPy`, `scikit-learn`, `joblib`,
`matplotlib`, `seaborn`, and `streamlit`.

## Installation

From the repository root:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install openpyxl
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install openpyxl
```

Verify the environment:

```bash
python --version
python -c "import pandas, sklearn, streamlit; print('Environment ready')"
```

## Data layout

Place the source files here:

```text
dataset/
├── raw/
│   ├── collision.csv
│   ├── vehicle.csv
│   └── casualty.csv
└── processed/
```

The source data should use the UK STATS19 column names expected by the
preprocessing scripts. The pipeline joins the tables using the collision
identifier and operates at casualty level for classification while using a
collision-level grouping strategy to reduce train/test leakage.

Do not commit restricted, personally identifiable, or otherwise sensitive
source data to a public repository.

## Train the complete pipeline

Run all pipeline stages:

```bash
python pipeline.py
```

The master runner executes the stages in order, streams each child script's
output, records input/output metadata, and writes `pipeline_log.json`.

The pipeline currently performs these operational steps:

1. Feature selection, table joining, target preparation, and grouped split.
2. Risk-score regression and out-of-fold risk-score generation.
3. Feature profiling.
4. Base-model training and out-of-fold meta-feature generation.
5. Meta-classifier training.
6. Baseline-versus-stacked-pipeline evaluation.

The terminal banner in older generated logs may refer to nine steps; the
current `STEPS` definition in `pipeline.py` is the source of truth.

### Resume after a failure

Resume from a specific step:

```bash
python pipeline.py --from 4
```

Stop after a specific step:

```bash
python pipeline.py --from 1 --to 3
```

Re-run feature selection instead of using its existing cache:

```bash
python pipeline.py --refresh-features
```

When a step fails, inspect `pipeline_log.json` before retrying. It contains the
command stage, status, return code, duration, input snapshots, output
snapshots, and captured console output.

## Start the product dashboard

After the model artifacts exist, start Streamlit:

```bash
streamlit run streamlit/app.py
```

Streamlit prints a local URL, normally:

```text
http://localhost:8501
```

Open that URL in a browser. To use a different port:

```bash
streamlit run streamlit/app.py --server.port 8502
```

The application loads the following five artifacts:

```text
models/saved_models/
├── stage1_best_regressor.pkl
├── stage2_logreg_final.pkl
├── stage2_knn_final.pkl
├── stage2_dt_final.pkl
└── stage3_meta_classifier.pkl
```

If the dashboard reports that models cannot be loaded, run the training
pipeline first and confirm that all five files exist.

## Using the dashboard

The sidebar contains these sections:

### System Architecture & Flow

Explains the data flow, grouping strategy, leakage prevention, out-of-fold
stacking design, and model stages.

### Model Theory & Loss Functions

Provides the mathematical and modeling background for the regression,
classifier, and meta-classifier stages.

### Model Comparisons & Metrics

Reads evaluation artifacts from:

```text
models/evaluation/results/
```

This includes comparison tables and the meta-classifier confusion matrix when
those files have been generated.

### Single Incident Prediction

Select the incident conditions from the form and click **Run Full Pipeline
Inference**. The dashboard displays:

- predicted severity: Fatal, Serious, or Slight;
- meta-classifier confidence;
- predicted continuous `risk_score`;
- probability for each severity class;
- each base model's probability distribution and vote;
- the ten-feature vector passed to the meta-classifier.

The form uses the human-readable mappings in `streamlit/mappings.py` and
converts them to the numeric codes expected by the saved models.

### Batch CSV / Excel Testing

You can either load the built-in ten-row sample from the held-out test set or
upload a `.csv`, `.xlsx`, or `.xls` file.

The uploaded file must contain these 16 columns:

```text
casualty_class
casualty_type
number_of_vehicles
number_of_casualties
day_of_week
first_road_class
road_type
speed_limit
junction_detail
junction_control
light_conditions
weather_conditions
road_surface_conditions
special_conditions_at_site
carriageway_hazards
urban_or_rural_area
```

After clicking **Process Batch Through Pipeline**, the app adds:

```text
Predicted_Severity_Code
Predicted_Severity
P_Fatal
P_Serious
P_Slight
Predicted_Risk_Score
```

The enriched results can be downloaded as CSV or Excel. The input file is
read into memory for the current Streamlit session; it is not uploaded to an
external service by this application.

### Pipeline Audit Logs

This view displays `pipeline_log.json`, including the status and captured
output for each completed step.

## Model inputs and interpretation

The inference engine in `streamlit/inference.py` performs four prediction
operations:

1. The regression model estimates `risk_score` from:
   `speed_limit`, `road_type`, `junction_detail`, `light_conditions`, and
   `weather_conditions`.
2. Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers
   produce class probabilities.
3. Those probabilities are combined with `risk_score` into ten meta-features.
4. The meta-classifier produces the final severity class and probabilities.

The `risk_score` is a model-generated signal, not a guaranteed count of
casualties. A high confidence value means the model's probability distribution
is concentrated; it does not guarantee that the prediction is correct.

## Generated files

| Location | Purpose |
| --- | --- |
| `dataset/processed/merged_train.csv` | Training data after joining and feature preparation |
| `dataset/processed/merged_test.csv` | Held-out test data |
| `dataset/processed/meta_train.csv` | Out-of-fold base-model probabilities |
| `dataset/processed/meta_test.csv` | Test-set base-model probabilities |
| `dataset/processed/selected_features.json` | Selected feature metadata |
| `models/saved_models/*.pkl` | Serialized regression and classifier artifacts |
| `models/evaluation/results/` | Metrics, comparisons, and confusion matrices |
| `docs/ablation_log.md` | Feature-ablation results |
| `docs/data_profile.md` | Feature suitability profile |
| `docs/stage5_evaluation_report.md` | Final evaluation narrative |
| `pipeline_log.json` | Pipeline execution audit trail |

Generated datasets, models, reports, and logs may be large. Review `.gitignore`
before committing them.

## Project structure

```text
.
├── dataset/
│   ├── raw/
│   └── processed/
├── models/
│   ├── stage1_merge/
│   ├── stage2_regression/
│   ├── stage3a_base_classifiers/
│   ├── stage4_meta_classifier/
│   ├── evaluation/
│   ├── saved_models/
│   └── utils/
├── streamlit/
│   ├── app.py
│   ├── inference.py
│   ├── mappings.py
│   └── styles.py
├── docs/
├── pipeline.py
├── pipeline_log.json
├── requirements.txt
└── README.md
```

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

For Excel support, also install:

```bash
python -m pip install openpyxl
```

### Models are not found

Run:

```bash
python pipeline.py
```

Then verify that all five `.pkl` files listed above exist under
`models/saved_models/`.

### Raw files are missing

Confirm that the three required files are under `dataset/raw/` and that their
names match exactly:

```text
collision.csv
vehicle.csv
casualty.csv
```

### Batch validation errors

Check that every required column is present and spelled exactly as documented.
The batch form expects numeric STATS19 codes, not the human-readable labels
shown in the single-incident form.

### Dashboard starts but evaluation panels are empty

Run the final evaluation stage or the complete pipeline:

```bash
python pipeline.py --from 6 --to 6
```

Then confirm that `models/evaluation/results/` contains the comparison and
confusion-matrix files.

### Windows encoding or console display issues

The master pipeline forces UTF-8 output for child processes and replaces
unsupported console characters when streaming logs. If a terminal still
displays unusual symbols, set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## Reproducibility and safety notes

- The pipeline uses grouped splitting so rows from the same collision are not
  deliberately distributed across train and test.
- Out-of-fold predictions are used to create training-side stacking features,
  reducing direct target leakage into the meta-classifier.
- Re-running the pipeline can overwrite processed datasets, model artifacts,
  reports, and `pipeline_log.json`; copy important outputs before retraining.
- Serialized `joblib`/pickle model files should only be loaded from trusted
  sources because deserialization can execute code.
- The project is intended for offline/local execution. The Streamlit app
  reads local files and does not provide authentication or multi-user access
  control.

## Development checks

There is no separate package build step. Use the active environment to run
the pipeline and dashboard. Before sharing changes, at minimum verify:

```bash
python -m compileall pipeline.py streamlit models
python -m pip check
```

## License and data provenance

Check the repository's license and the license/terms of the UK STATS19 data
before redistribution. The project documentation does not override the terms
of the source dataset.
