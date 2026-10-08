from xgboost import XGBRegressor


NAME = "XGBoost"


DEFAULT_PARAMS = {
    "objective": "reg:squarederror",

    # Paper values
    "n_estimators": 250,
    "learning_rate": 0.07,
    "gamma": 0,
    "max_depth": 3,
    "min_child_weight": 1.5,
    "reg_alpha": 0.01,
    "reg_lambda": 0.45,
    "subsample": 0.8,
    "colsample_bytree": 0.4,

    # Reproducibility / execution
    "random_state": 1234,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda"
}


def build_model(params=None):
    model_params = DEFAULT_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return XGBRegressor(**model_params)


def get_param_grid():
    return {
        # Every dimension contains the paper value
        "n_estimators": [100, 250, 500],
        "learning_rate": [0.03, 0.07, 0.1],
        "max_depth": [3, 6],
        "min_child_weight": [1.0, 1.5, 3.0],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [0.45, 1.0],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.4, 0.8, 1.0]
    }
