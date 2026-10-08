import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}


def train_model(
    model,
    X_train,
    y_train,
    param_grid=None,
    cv=5,
    n_jobs=-1,
):
    """
    Train a model.

    If param_grid is provided, perform GridSearchCV using RMSE
    as the primary metric for selecting the best model.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Model to train.

    X_train : array-like
        Training features.

    y_train : array-like
        Training target.

    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV.
        If None or empty, the model is trained directly.

    cv : int or CV splitter, default=5
        Cross-validation strategy used during hyperparameter search.

    Returns
    -------
    dict
        Contains trained model and, if tuning was performed,
        best parameters and CV metrics.
    """

    # No hyperparameter tuning
    if not param_grid:
        model.fit(X_train, y_train)

        return {
            "model": model,
            "best_params": None,
            "cv_metrics": None,
            "search": None
        }

    # Hyperparameter tuning
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=SCORING,
        refit="rmse",
        cv=cv,
        n_jobs=n_jobs,
        verbose=2
    )

    search.fit(X_train, y_train)

    best_idx = search.best_index_

    cv_metrics = {
        "rmse": -search.cv_results_["mean_test_rmse"][best_idx],
        "rmse_std": search.cv_results_["std_test_rmse"][best_idx],

        "mae": -search.cv_results_["mean_test_mae"][best_idx],
        "mae_std": search.cv_results_["std_test_mae"][best_idx],

        "r2": search.cv_results_["mean_test_r2"][best_idx],
        "r2_std": search.cv_results_["std_test_r2"][best_idx]
    }

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_metrics": cv_metrics,
        "search": search
    }


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate an already trained model on an independent test set.

    Returns RMSE, MAE and R² together with predictions.
    """

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions
    }
