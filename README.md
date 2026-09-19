<!-- README.md: Project overview, run instructions, architecture summary, and results. -->
# Collision Severity Prediction Pipeline

A multi-stage machine learning system designed to predict traffic collision severity levels.

## Pipeline Architecture
1. **Data Preprocessing & Joining**:
   - Merge `collision.csv`, `vehicle.csv`, and `casualty.csv`.
   - Prevent data leakage by dropping future/post-accident features.
   - Encode categorical variables and split into `merged_train.csv` (64k rows) and `merged_test.csv` (16k rows).

2. **Stage 1: Continuous Risk Scoring (Regression)**:
   - Train Linear Regression and Decision Tree Regressor.
   - Compare RMSE / R² metrics via `compare_regressors.py`.
   - Select best model (`stage1_best_regressor.pkl`) and append `risk_score` feature.

3. **Stage 2: Base Classifiers**:
   - **Phase A (`oof_generator.py`)**: 5-fold cross validation generating out-of-fold probability distributions -> `meta_train.csv`.
   - **Phase B (`final_models_trainer.py`)**: Fit 4 models (Logistic Regression, KNN, Decision Tree, Perceptron) on full training set -> generate `meta_test.csv`.

4. **Stage 3: Meta-Classifier**:
   - Train stacking meta-classifier (`train_meta_classifier.py`) on `meta_train.csv`.
   - Evaluate on `meta_test.csv` and save `stage3_meta_classifier.pkl`.

5. **Concept Learning Demonstrations**:
   - `find_s.py`: Find-S algorithm on categorical subset.
   - `candidate_elimination.py`: Candidate Elimination algorithm.

6. **Evaluation**:
   - Benchmark individual standalone base models against stacked pipeline (`baseline_vs_pipeline.py`).
   - Store metrics and confusion matrices in `models/evaluation/results/`.

## Running the Pipeline
```bash
pip install -r requirements.txt
python models/utils/preprocessing.py
python models/stage1_regression/compare_regressors.py
python models/stage2_classifiers/oof_generator.py
python models/stage2_classifiers/final_models_trainer.py
python models/stage3_meta_classifier/train_meta_classifier.py
python models/evaluation/baseline_vs_pipeline.py
```
