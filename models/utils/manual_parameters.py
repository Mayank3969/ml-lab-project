"""Parameters used by the complete demonstration pipeline.

After the tuning notebooks finish, copy the selected values into this file.
The normal pipeline never performs a search; it trains exactly these values.
"""

# Stage 1: collision-level risk-score regressors
LINEAR_REGRESSION_PARAMS = {
    "regressor__positive": True,
    "regressor__fit_intercept": False
}

DECISION_TREE_REGRESSOR_PARAMS = {
   "regressor__min_samples_split": 2,
    "regressor__min_samples_leaf": 10,
    "regressor__max_features": None,
    "regressor__max_depth": 3,
    "regressor__criterion": "squared_error",
    "regressor__ccp_alpha": 0.001
}

# Stage 2: casualty-severity base classifiers
LOGISTIC_REGRESSION_PARAMS = {
    "classifier__solver": "lbfgs",
    "classifier__l1_ratio": 0,
    "classifier__class_weight": {
      "1": 5,
      "2": 2,
      "3": 1
    },
    "classifier__C": 100.0
}

KNN_PARAMS = {
    "classifier__weights": "distance",
    "classifier__p": 2,
    "classifier__n_neighbors": 15,
    "classifier__algorithm": "auto"
}

DECISION_TREE_PARAMS = {
    "classifier__min_samples_split": 50,
    "classifier__min_samples_leaf": 5,
    "classifier__max_features": 0.7,
    "classifier__max_depth": 24,
    "classifier__criterion": "gini",
    "classifier__class_weight": {
      "1": 3,
      "2": 1.5,
      "3": 1
    },
    "classifier__ccp_alpha": 0.0001
}

# Stage 3: final ensemble. The notebook compares both risk sources; this is
# the manually selected source used by the faculty demonstration pipeline.
META_ENSEMBLE_PARAMS = {
    "classifier__solver": "lbfgs",
    "classifier__class_weight": {
        "1": 5,
        "2": 2,
        "3": 1
    },
    "classifier__C": 0.02682695795279726
}
META_RISK_FEATURE = "risk_score_linear_regression"
