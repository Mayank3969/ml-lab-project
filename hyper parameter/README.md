# Hyperparameter notebooks

Run these notebooks manually on Windows after running
`models/stage1_merge/build_dataset.py`:

1. `linear_regression_tuning.ipynb`
2. `decision_tree_regressor_tuning.ipynb`
3. `logistic_regression_tuning.ipynb`
4. `knn_tuning.ipynb`
5. `decision_tree_tuning.ipynb`
6. `meta_ensemble_tuning.ipynb` after OOF features exist

They are independent. Each notebook contains its own time-bounded
`RandomizedSearchCV` tuning code and uses a model-specific input contract,
collision-group-aware validation, a complete preprocessing/model pipeline, and
one CPU worker to avoid the KNN memory failure. Every candidate result is saved
in that model's `results/` CSV. The best fitted pipeline and a JSON summary are
saved in `best/` and copied to `models/saved_models/`.

The active models are scikit-learn CPU estimators. The Windows GPU is not
required for these models. Search sizes are deliberately bounded so the six
notebooks fit the planned total runtime. The time budget controls admission of
new candidates; one already-running candidate and final refit can finish after
the budget.

Approximate screening budgets:

| Notebook | Candidates | CV folds | Budget |
|---|---:|---:|---:|
| Linear Regression | 4 | 5 | short |
| Decision Tree Regressor | 16 | 5 | medium |
| Logistic Regression | 12 | 5 | medium |
| KNN | 6 | 5 | medium |
| Decision Tree | 24 | 5 | longest |
| Final ensemble | 12 per risk source | 5 | medium |

The exact values can be changed directly in the notebook's `n_iter` and
`n_jobs` cells. Tuning logic is contained in the notebooks; no external tuning
Python file is required.
