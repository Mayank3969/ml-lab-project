"""Parameters used by the complete demonstration pipeline.

After the tuning notebooks finish, copy the selected values into this file.
The normal pipeline never performs a search; it trains exactly these values.
"""

# Stage 1: collision-level risk-score regressors
LINEAR_REGRESSION_PARAMS = {
    "fit_intercept": True,
    "positive": False,
}

DECISION_TREE_REGRESSOR_PARAMS = {
    "max_depth": 8,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": None,
    "criterion": "squared_error",
    "ccp_alpha": 0.0,
}

# Stage 2: casualty-severity base classifiers
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 2500,
}

KNN_PARAMS = {
    "n_neighbors": 15,
    "weights": "distance",
    "p": 2,
    "algorithm": "auto",
    "n_jobs": -1,
}

DECISION_TREE_PARAMS = {
    "max_depth": 12,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": None,
    "criterion": "gini",
    "ccp_alpha": 0.0,
    "class_weight": "balanced",
}

# Stage 3: final ensemble. The notebook compares both risk sources; this is
# the manually selected source used by the faculty demonstration pipeline.
META_ENSEMBLE_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 2500,
}
META_RISK_FEATURE = "risk_score_linear_regression"
