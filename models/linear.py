# models/linear.py

from sklearn.linear_model import LinearRegression


NAME = "Linear Regression"


DEFAULT_PARAMS = {
    "fit_intercept": True
}


def build_model(params=None):
    """
    Create a LinearRegression model.

    Parameters
    ----------
    params : dict, optional
        Parameters overriding the default model configuration.

    Returns
    -------
    LinearRegression
        Untrained model.
    """

    model_params = DEFAULT_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return LinearRegression(**model_params)


def get_param_grid():
    """
    Return a default hyperparameter grid.

    Linear regression has very few meaningful hyperparameters.
    """

    return {
        "fit_intercept": [True, False]
    }
