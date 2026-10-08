from sklearn.tree import DecisionTreeRegressor


NAME = "Regression Tree"


DEFAULT_PARAMS = {
    "criterion": "squared_error",
    "random_state": 1234
}


def build_model(params=None):
    """
    Create a DecisionTreeRegressor model.

    Parameters
    ----------
    params : dict, optional
        Parameters overriding the default configuration.

    Returns
    -------
    DecisionTreeRegressor
        Untrained regression tree.
    """

    model_params = DEFAULT_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return DecisionTreeRegressor(**model_params)


def get_param_grid():
    """
    Default hyperparameter grid for regression tree tuning.
    """

    return {
        "criterion": ["squared_error"],
        "splitter": ["best"],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf": [1, 5, 20]
    }
