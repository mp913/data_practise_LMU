from sklearn.ensemble import GradientBoostingRegressor


NAME = "Gradient Boosting"


DEFAULT_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_depth": 3,
    "random_state": 1234
}


def build_model(params=None):
    model_params = DEFAULT_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return GradientBoostingRegressor(**model_params)


def get_param_grid():
    return {
        "n_estimators": [100, 250],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
        "min_samples_leaf": [1, 25]
    }
