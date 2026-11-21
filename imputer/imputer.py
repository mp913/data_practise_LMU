"""Gaussian Expectation-Maximization (EM) imputer compatible with scikit-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class EMImputer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Impute missing values using EM under a multivariate Gaussian model.

    ``fit()`` estimates the latent standardized mean and covariance from data
    containing missing values. ``transform()`` replaces each missing entry with
    its conditional expectation given the observed values in the same row.

    Features are centered and scaled internally using training statistics, and a
    diagonal ridge term keeps covariance sub-blocks invertible when columns are
    constant, sparse, or collinear (such as one-hot encoded indicators).

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of EM iterations during ``fit``.
    tol : float, default=1e-4
        Convergence threshold on the maximum absolute change in ``mean_`` or
        ``covariance_`` between consecutive iterations.
    ridge : float, default=1e-6
        Non-negative diagonal regularization added to covariance matrices for
        numerical stability.

    Attributes
    ----------
    center_ : ndarray of shape (n_features,)
        Per-feature mean learned on non-missing training entries.
    scale_ : ndarray of shape (n_features,)
        Per-feature standard deviation learned on non-missing training entries
        (with zero or all-NaN scales replaced by ``1.0``).
    mean_ : ndarray of shape (n_features,)
        Estimated standardized mean at convergence.
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated standardized covariance matrix at convergence.
    n_iter_ : int
        Number of EM iterations executed.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit`` (only when ``X`` has feature
        names such as a pandas DataFrame).
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-4, ridge: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    @staticmethod
    def _to_float_array(X) -> np.ndarray:
        """Convert input (including nullable pandas dtypes like ``<NA>``) to float64."""
        if isinstance(X, pd.DataFrame):
            # Convert nullable extension dtypes (e.g., Int64, Float64, boolean)
            # before handing off to validate_data so <NA> becomes np.nan cleanly.
            arr = X.to_numpy(dtype=np.float64, na_value=np.nan)
            return arr
        return np.asarray(X, dtype=np.float64)

    def _standardize(self, X: np.ndarray, *, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        if fit:
            center = np.nanmean(X, axis=0)
            # If an entire column is NaN during fit, fall back to 0.0 center and 1.0 scale.
            self.center_ = np.where(np.isnan(center), 0.0, center)

            scale = np.nanstd(X, axis=0)
            invalid_scale = (~np.isfinite(scale)) | (scale <= 0.0)
            self.scale_ = np.where(invalid_scale, 1.0, scale)

        Z = (X - self.center_) / self.scale_
        mask = np.isnan(Z)
        return Z, mask

    def _e_step(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute conditional expectations E[z_m | z_o] and covariance term C."""
        d = Z.shape[1]
        Z_hat = np.where(mask, 0.0, Z)
        C = np.zeros((d, d), dtype=np.float64)
        ridge_eye = self.ridge * np.eye(d, dtype=np.float64)
        sigma_reg = sigma + ridge_eye

        patterns, inverse = np.unique(mask, axis=0, return_inverse=True)
        inverse = inverse.ravel()

        for p, miss in enumerate(patterns):
            if not miss.any():
                continue

            rows = inverse == p
            n_rows = int(rows.sum())
            obs = ~miss

            if not obs.any():
                # Entire row is missing: conditional mean is mu[miss],
                # conditional covariance is the prior covariance block.
                Z_hat[np.ix_(rows, miss)] = mu[miss]
                cond_cov = sigma[np.ix_(miss, miss)] + self.ridge * np.eye(
                    int(miss.sum()), dtype=np.float64
                )
                C[np.ix_(miss, miss)] += n_rows * cond_cov
                continue

            S_oo = sigma_reg[np.ix_(obs, obs)]
            S_mo = sigma_reg[np.ix_(miss, obs)]
            B = np.linalg.solve(S_oo, S_mo.T).T

            obs_centered = Z[np.ix_(rows, obs)] - mu[obs]
            Z_hat[np.ix_(rows, miss)] = mu[miss] + obs_centered @ B.T

            cond_cov = sigma[np.ix_(miss, miss)] - B @ S_mo.T
            cond_cov = 0.5 * (cond_cov + cond_cov.T)
            C[np.ix_(miss, miss)] += n_rows * cond_cov

        return Z_hat, C

    def fit(self, X, y=None):
        """Fit the EM Gaussian model on ``X``."""
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}")
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")

        X_raw = self._to_float_array(X)
        X_arr = validate_data(
            self,
            X_raw,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            reset=True,
        )

        # Preserve feature_names_in_ when given a pandas DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        Z, mask = self._standardize(X_arr, fit=True)
        n, d = Z.shape

        mu = np.zeros(d, dtype=np.float64)
        obs_init = np.where(mask, 0.0, Z)
        if n > 1:
            cov = np.cov(obs_init, rowvar=False, bias=True)
            cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        else:
            cov = np.zeros((d, d), dtype=np.float64)

        cov = cov + self.ridge * np.eye(d, dtype=np.float64)

        n_iter = 0
        for it in range(self.max_iter):
            n_iter = it + 1
            Z_hat, C = self._e_step(Z, mask, mu, sigma=cov)

            mu_new = Z_hat.mean(axis=0)
            diff = Z_hat - mu_new
            sigma_new = (diff.T @ diff + C) / n
            sigma_new = 0.5 * (sigma_new + sigma_new.T)

            change = max(
                float(np.abs(mu_new - mu).max()),
                float(np.abs(sigma_new - cov).max()),
            )
            mu, cov = mu_new, sigma_new

            if change < self.tol:
                break

        self.mean_ = mu
        self.covariance_ = cov
        self.n_iter_ = n_iter
        return self

    def transform(self, X):
        """Impute missing values in ``X`` using learned EM parameters."""
        check_is_fitted(self)

        X_raw = self._to_float_array(X)
        X_arr = validate_data(
            self,
            X_raw,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            reset=False,
            copy=True,
        )

        Z, mask = self._standardize(X_arr, fit=False)
        Z_hat, _ = self._e_step(Z, mask, self.mean_, self.covariance_)
        X_hat = Z_hat * self.scale_ + self.center_

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_hat, index=X.index, columns=X.columns)

        return X_hat