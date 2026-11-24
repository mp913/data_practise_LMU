import numpy as np
from threading import Event, Thread
from time import perf_counter

import pandas as pd
from IPython.display import display

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


def run_grid_search(
    X_train,
    y_train,
    *,
    imputers,
    models,
    cv,
    prepare_features,
    fit_and_score,
):
    folds = list(cv.split(X_train, y_train))
    n_folds = len(folds)
    total = len(imputers) * n_folds * (1 + len(models))
    completed = 0
    results = []
    status = display("Starting grid search", display_id=True)

    def run_step(label, operation, *args):
        nonlocal completed

        started = perf_counter()
        stop = Event()

        def show(state):
            elapsed = perf_counter() - started
            status.update(
                f"{completed}/{total} ({completed / total:.0%}) | "
                f"{state}: {label} | {elapsed:.1f}s"
            )

        def refresh():
            while not stop.wait(1):
                show("Running")

        show("Running")
        timer = Thread(target=refresh, daemon=True)
        timer.start()

        state = "Interrupted"
        try:
            result = operation(*args)
            completed += 1
            state = "Completed"
            return result
        except Exception:
            state = "Failed"
            raise
        finally:
            stop.set()
            timer.join()
            show(state)

    for imputer_name, imputer in imputers:
        scores = {spec[0]: [] for spec in models}

        for fold_no, (fit_idx, val_idx) in enumerate(folds, start=1):
            label = f"{imputer_name} | Fold {fold_no}/{n_folds}"
            X_fit, X_val = run_step(
                f"{label} | Imputation",
                prepare_features,
                imputer,
                X_train.iloc[fit_idx],
                X_train.iloc[val_idx],
            )
            y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]

            for spec in models:
                model_name = spec[0]
                metrics = run_step(
                    f"{label} | {model_name}",
                    fit_and_score,
                    spec,
                    X_fit,
                    X_val,
                    y_fit,
                    y_val,
                )
                scores[model_name].append(metrics)

        for model_name, fold_scores in scores.items():
            means = pd.DataFrame(fold_scores).mean().to_dict()
            results.append({"imputer": imputer_name, "model": model_name, **means})

    status.update(f"Grid search complete | {completed}/{total} (100%)")
    return pd.DataFrame(results).sort_values("rmse", kind="stable", ignore_index=True)
