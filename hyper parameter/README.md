# Hyperparameter notebooks

Run these notebooks manually on Windows after running
`models/stage1_merge/build_dataset.py`:

1. `linear_regression_tuning.ipynb`
2. `decision_tree_regressor_tuning.ipynb`
3. `logistic_regression_tuning.ipynb`
4. `knn_tuning.ipynb`
5. `decision_tree_tuning.ipynb`
6. `meta_ensemble_tuning.ipynb` after OOF features exist

They are independent. Each notebook uses a model-specific input contract,
collision-group-aware five-fold validation, macro-F1 selection, and a complete
preprocessing/model pipeline. Every grid candidate is saved in that model's
`results/` CSV. The best fitted pipeline and a JSON summary are saved in
`best/` and copied to `models/saved_models/`.

The active models are scikit-learn CPU estimators. The Windows GPU is not
required for these models. The notebooks intentionally use full grids so
they can run unattended overnight.
