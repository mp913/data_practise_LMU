from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


NAME = "K-Nearest Neighbours"


def build_model(params=None):
    """
    Build KNN regression pipeline.

    StandardScaler is our preprocessing addition.
    KNN parameters reproduce the paper configuration.
    """

    model_params = {
        # Paper values
        "n_neighbors": 25,
        "weights": "uniform",
        "algorithm": "brute",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "n_jobs": -1
    }

    if params is not None:
        model_params.update(params)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(**model_params))
    ])


def get_param_grid():
    return {
        "model__n_neighbors": [5, 10, 25, 50],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__algorithm": ["brute"]
    }
