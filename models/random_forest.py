from sklearn.ensemble import RandomForestRegressor


NAME = "Random Forest"


DEFAULT_PARAMS = {
    "n_estimators": 25,          # paper value
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 25,      # paper value
    "max_features": 1.0,         # paper value
    "bootstrap": True,
    "oob_score": False,
    "ccp_alpha": 0.0,
    "random_state": 1234,
    "n_jobs": 1
}


def build_model(params=None):
    model_params = DEFAULT_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return RandomForestRegressor(**model_params)


def get_param_grid():
    return {
        "n_estimators": [25, 100, 300],
        "criterion": ["squared_error"],
        "max_depth": [None, 15],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [5, 25],
        "max_features": [1.0, "sqrt"],
        "ccp_alpha": [0.0]
    }
