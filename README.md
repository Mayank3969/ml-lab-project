# Collision Severity Prediction

An end-to-end machine-learning mini project for predicting road-collision
casualty severity from UK STATS19 collision, vehicle, and casualty data.

The system predicts one of three outcomes:

| Code | Severity |
| ---: | --- |
| 1 | Fatal |
| 2 | Serious |
| 3 | Slight |

The project is designed around a practical question: **can casualty severity
be estimated using information available soon after a collision, without
leaking the outcome into the features?**

## Highlights

- Joins collision, vehicle, and casualty records using the collision identifier.
- Keeps all casualties from the same collision in the same train or test split.
- Removes outcome-proxy and target-leakage fields before modelling.
- Uses model-specific feature contracts selected during hyperparameter tuning.
- Produces out-of-fold predictions for a leakage-aware stacking model.
- Provides a Streamlit dashboard for single-record and CSV predictions.
- Reports accuracy, balanced accuracy, macro and weighted metrics, and
  per-class recall.

## Model architecture

The final prediction is produced by a stacking pipeline:

1. Two collision-level regressors estimate the number of casualties:
   - **Linear Regression**
   - **Decision Tree Regressor**
2. Three casualty-severity classifiers estimate Fatal, Serious, and Slight
   probabilities:
   - **Logistic Regression**
   - **K-Nearest Neighbours (KNN)**
   - **Decision Tree Classifier**
3. A final **Logistic Regression meta-classifier** combines the nine base-model
   probabilities with the selected linear-regression risk score.

The regressors are not additional severity predictions. They provide
collision-level risk signals to the final severity classifier.

## Project structure

```text
.
├── dataset/
│   ├── raw/                         # Source collision, vehicle, casualty CSVs
│   └── processed/                   # Generated train/test and OOF data
├── docs/                            # Pipeline documentation and diagrams
├── hyper parameter/                # Tuning notebooks and selected contracts
├── models/
│   ├── stage1_merge/                # Join, clean, split, and graph generation
│   ├── stage1_regression/           # Collision-level risk scores
│   ├── stage3a_base_classifiers/   # Three severity base classifiers
│   ├── stage4_meta_classifier/     # Final stacking classifier
│   ├── evaluation/                 # Test-set comparison and reports
│   └── utils/                      # Configuration and preprocessing
├── streamlit/
│   ├── app.py                       # Dashboard
│   └── inference.py                 # Saved-model inference
├── pipeline.py                      # End-to-end pipeline runner
├── requirements.txt
└── run_full_pipeline.bat            # Windows pipeline shortcut
```

Generated datasets, saved models, evaluation outputs, local environments, and
temporary files are intentionally excluded from version control. They are
created locally when the pipeline is run.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- The raw CSV files in `dataset/raw/`

Install the dependencies from the project root:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Run the training pipeline

The normal pipeline runs five stages:

| Stage | Purpose |
| ---: | --- |
| 1 | Join and validate the raw tables, remove leakage, and create grouped train/test data |
| 2 | Train the Linear Regression and Decision Tree Regressor risk models |
| 3 | Train the Logistic Regression, KNN, and Decision Tree classifiers and create out-of-fold features |
| 4 | Train the final Logistic Regression stacking model |
| 5 | Evaluate the base classifiers and final ensemble on the locked test set |

Run every stage from a clean state:

```powershell
python pipeline.py --clean
```

The Windows shortcut runs the same complete demonstration:

```powershell
.\run_full_pipeline.bat
```

Run only a selected range when intermediate artifacts already exist:

```powershell
python pipeline.py --to 3
python pipeline.py --from 4
```

Stage 1 reuses existing processed data when it is complete. To force a fresh
data build:

```powershell
python models\stage1_merge\build_dataset.py --force
```

The pipeline reads the selected feature contracts from
`hyper parameter/best/` and the final parameter choices from
`models/utils/manual_parameters.py`. If the best-parameter files are not
available, run the tuning notebooks first.

## Hyperparameter tuning

The notebooks in `hyper parameter/` are optional for using the already selected
configuration, but they document how the active feature sets and parameters
were selected:

1. `linear_regression_tuning.ipynb`
2. `decision_tree_regressor_tuning.ipynb`
3. `logistic_regression_tuning.ipynb`
4. `knn_tuning.ipynb`
5. `decision_tree_tuning.ipynb`
6. `meta_ensemble_tuning.ipynb`

The final ensemble notebook should be run after the base-classifier stage has
created out-of-fold features. The selected values are then recorded in
`models/utils/manual_parameters.py`.

## Run the dashboard

After the saved model artifacts have been generated:

```powershell
streamlit run streamlit\app.py
```

The dashboard includes:

- Project overview and workflow explanation
- Model metric comparison
- Saved data and model graphs
- Single-record severity prediction
- CSV batch prediction
- Fatal, Serious, and Slight probability outputs
- Downloadable prediction results

For CSV prediction, upload a file containing the raw feature columns required
by the active model contracts. The target column and collision identifier are
not required for prediction.

## Evaluation outputs

The evaluation stage writes generated files to
`models/evaluation/results/`, including:

- `model_comparison.csv`
- `base_model_metrics.csv`
- Classification reports
- Confusion matrices
- Final ensemble metrics

The main comparison includes accuracy, balanced accuracy, macro and weighted
precision, recall and F1, together with Fatal, Serious, and Slight recall.

## Generate graphs

Graphs are generated separately from model training:

```powershell
python models\stage1_merge\create_graphs.py
```

The dashboard reads the saved images from `docs/graphs`; it does not create
plots while serving the application.

## Reset generated artifacts

Preview the cleanup first:

```powershell
python reset_generated.py --dry-run
```

Delete generated datasets, models, and evaluation outputs:

```powershell
python reset_generated.py --yes
```

Raw source data and project source code are preserved.

## Data and modelling notes

- `collision_id`/`accident_index` links the source tables.
- The split is collision-grouped so records from one collision cannot appear in
  both training and test data.
- The target is `casualty_severity`.
- Regression models use `number_of_casualties` as their collision-level target.
- Class labels are kept in the fixed order Fatal, Serious, Slight throughout
  training, evaluation, and inference.
- All models are CPU-based scikit-learn estimators; a GPU is not required.

## Disclaimer

This project is intended for academic demonstration and exploratory analysis.
Predictions should not be used as the sole basis for emergency response,
insurance, legal, or safety-critical decisions.
