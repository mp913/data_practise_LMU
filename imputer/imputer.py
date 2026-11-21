"""Gaussian and GMM Expectation-Maximization (EM) imputers compatible with scikit-learn."""

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


class GMMImputer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Impute missing values using EM under a Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int, default=3
        Number of Gaussian mixture components.
    max_iter : int, default=100
        Maximum number of EM iterations during ``fit``.
    tol : float, default=1e-4
        Convergence threshold on the maximum absolute change in ``weights_``,
        ``means_``, or ``covariances_`` between consecutive iterations.
    ridge : float, default=1e-6
        Non-negative diagonal regularization added to covariance sub-blocks
        for numerical stability.
    reg_covar : float, default=1e-6
        Minimum diagonal regularization added to component covariances during
        the M-step to keep matrices positive definite.
    random_state : int or np.random.Generator, default=None
        Seed or NumPy Generator used for initializing component parameters.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        ridge: float = 1e-6,
        reg_covar: float = 1e-6,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.reg_covar = reg_covar
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    @staticmethod
    def _to_float_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64, na_value=np.nan)
        return np.asarray(X, dtype=np.float64)

    def _standardize(self, X: np.ndarray, *, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        if fit:
            center = np.nanmean(X, axis=0)
            self.center_ = np.where(np.isnan(center), 0.0, center)

            scale = np.nanstd(X, axis=0)
            invalid_scale = (~np.isfinite(scale)) | (scale <= 0.0)
            self.scale_ = np.where(invalid_scale, 1.0, scale)

        Z = (X - self.center_) / self.scale_
        mask = np.isnan(Z)
        return Z, mask

    @staticmethod
    def _log_gaussian_obs(obs_centered: np.ndarray, cov_oo: np.ndarray) -> np.ndarray:
        d_obs = obs_centered.shape[1]
        if d_obs == 0:
            return np.zeros(obs_centered.shape[0], dtype=np.float64)

        try:
            L = np.linalg.cholesky(cov_oo)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            sol = np.linalg.solve(L, obs_centered.T)
            maha = np.sum(sol * sol, axis=0)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_oo)
            eigvals = np.maximum(eigvals, 1e-12)
            log_det = float(np.sum(np.log(eigvals)))
            proj = obs_centered @ eigvecs
            maha = np.sum((proj * proj) / eigvals, axis=1)

        return -0.5 * (d_obs * np.log(2.0 * np.pi) + log_det + maha)

    @staticmethod
    def _responsibilities(log_joint: np.ndarray) -> np.ndarray:
        max_lj = np.max(log_joint, axis=1, keepdims=True)
        safe_max = np.where(np.isfinite(max_lj), max_lj, 0.0)
        shifted = np.exp(log_joint - safe_max)
        denom = np.sum(shifted, axis=1, keepdims=True)
        resp = shifted / np.maximum(denom, 1e-300)

        bad = ~np.isfinite(resp).all(axis=1)
        if np.any(bad):
            resp[bad] = 1.0 / log_joint.shape[1]
        return resp

    def _e_step(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, d = Z.shape
        K = self.n_components

        log_joint = np.broadcast_to(
            np.log(np.maximum(weights, 1e-300))[None, :], (n, K)
        ).copy()
        cond_means = np.broadcast_to(means[None, :, :], (n, K, d)).copy()
        cond_cov_weighted = np.zeros((K, d, d), dtype=np.float64)

        eye_d = np.eye(d, dtype=np.float64)
        cov_reg_all = np.empty_like(covariances)
        for k in range(K):
            cov_k = 0.5 * (covariances[k] + covariances[k].T)
            cov_reg_all[k] = cov_k + self.ridge * eye_d

        patterns, inverse = np.unique(mask, axis=0, return_inverse=True)
        inverse = inverse.ravel()

        for p, miss in enumerate(patterns):
            rows = np.flatnonzero(inverse == p)
            obs = ~miss
            n_miss = int(miss.sum())

            if obs.any():
                cond_means[np.ix_(rows, np.arange(K), np.flatnonzero(obs))] = Z[
                    np.ix_(rows, obs)
                ][:, None, :]
                Z_obs = Z[np.ix_(rows, obs)]

                for k in range(K):
                    mu_k = means[k]
                    cov_k = cov_reg_all[k]
                    S_oo = cov_k[np.ix_(obs, obs)]
                    obs_centered = Z_obs - mu_k[obs]
                    log_joint[rows, k] += self._log_gaussian_obs(obs_centered, S_oo)

                    if n_miss > 0:
                        S_mo = cov_k[np.ix_(miss, obs)]
                        try:
                            B = np.linalg.solve(S_oo, S_mo.T).T
                        except np.linalg.LinAlgError:
                            B = (np.linalg.pinv(S_oo) @ S_mo.T).T
                        pred_miss = mu_k[miss] + obs_centered @ B.T
                        cond_means[np.ix_(rows, [k], np.flatnonzero(miss))] = pred_miss[
                            :, None, :
                        ]

        resp = self._responsibilities(log_joint)

        for p, miss in enumerate(patterns):
            if not miss.any():
                continue
            rows = np.flatnonzero(inverse == p)
            obs = ~miss
            n_miss = int(miss.sum())
            miss_idx = np.flatnonzero(miss)
            eye_m = np.eye(n_miss, dtype=np.float64)

            for k in range(K):
                gamma_sum = float(np.sum(resp[rows, k]))
                if gamma_sum <= 0.0:
                    continue

                prior_mm = (
                    0.5
                    * (
                        covariances[k][np.ix_(miss, miss)]
                        + covariances[k][np.ix_(miss, miss)].T
                    )
                    + self.ridge * eye_m
                )
                if not obs.any():
                    cond_cov = prior_mm
                else:
                    S_oo = cov_reg_all[k][np.ix_(obs, obs)]
                    S_mo = cov_reg_all[k][np.ix_(miss, obs)]
                    try:
                        B = np.linalg.solve(S_oo, S_mo.T).T
                    except np.linalg.LinAlgError:
                        B = (np.linalg.pinv(S_oo) @ S_mo.T).T
                    cond_cov = prior_mm - B @ S_mo.T
                    cond_cov = 0.5 * (cond_cov + cond_cov.T)

                cond_cov_weighted[k][np.ix_(miss_idx, miss_idx)] += gamma_sum * cond_cov

        if mask.any():
            weighted_means = np.einsum("nk,nkd->nd", resp, cond_means)
            Z_hat = np.where(mask, weighted_means, Z)
        else:
            Z_hat = Z.copy()

        return Z_hat, resp, cond_means, cond_cov_weighted

    def _m_step(
        self,
        resp: np.ndarray,
        cond_means: np.ndarray,
        cond_cov_weighted: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, K, d = cond_means.shape
        nk = np.sum(resp, axis=0)
        weights = np.maximum(nk, 1e-12)
        weights /= np.sum(weights)

        means = np.zeros((K, d), dtype=np.float64)
        covariances = np.zeros((K, d, d), dtype=np.float64)
        reg_eye = self.reg_covar * np.eye(d, dtype=np.float64)

        for k in range(K):
            if nk[k] > 1e-12:
                mu_k = np.sum(resp[:, k : k + 1] * cond_means[:, k, :], axis=0) / nk[k]
                diff = cond_means[:, k, :] - mu_k[None, :]
                second_moment = (
                    (resp[:, k : k + 1] * diff).T @ diff + cond_cov_weighted[k]
                ) / nk[k]
                cov_k = 0.5 * (second_moment + second_moment.T) + reg_eye
            else:
                mu_k = rng.normal(loc=0.0, scale=0.1, size=d)
                cov_k = np.eye(d, dtype=np.float64) + reg_eye

            means[k] = mu_k
            covariances[k] = cov_k

        return weights, means, covariances

    def _initialize_params(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, d = Z.shape
        K = self.n_components

        weights = np.full(K, 1.0 / K, dtype=np.float64)
        Z_init = np.where(mask, 0.0, Z)

        chosen_idx = rng.choice(n, size=min(n, K), replace=(n < K))
        means = np.zeros((K, d), dtype=np.float64)
        means[: len(chosen_idx)] = Z_init[chosen_idx]
        if len(chosen_idx) < K:
            means[len(chosen_idx) :] = rng.normal(
                loc=0.0, scale=0.5, size=(K - len(chosen_idx), d)
            )
        means += rng.normal(loc=0.0, scale=0.05, size=(K, d))

        if n > 1:
            base_cov = np.atleast_2d(
                np.asarray(np.cov(Z_init, rowvar=False, bias=True), dtype=np.float64)
            )
        else:
            base_cov = np.eye(d, dtype=np.float64)

        base_cov = (
            0.5 * (base_cov + base_cov.T)
            + (self.ridge + self.reg_covar) * np.eye(d, dtype=np.float64)
        )
        covariances = np.broadcast_to(base_cov[None, :, :], (K, d, d)).copy()

        return weights, means, covariances

    def fit(self, X, y=None):
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}")
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")
        if self.reg_covar < 0:
            raise ValueError(f"reg_covar must be >= 0, got {self.reg_covar}")

        X_raw = self._to_float_array(X)
        X_arr = validate_data(
            self,
            X_raw,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            reset=True,
        )

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        rng = np.random.default_rng(self.random_state)
        Z, mask = self._standardize(X_arr, fit=True)
        weights, means, covariances = self._initialize_params(Z, mask, rng)

        n_iter = 0
        converged = False
        for it in range(self.max_iter):
            n_iter = it + 1
            _, resp, cond_means, cond_cov_weighted = self._e_step(
                Z, mask, weights, means, covariances
            )
            weights_new, means_new, covariances_new = self._m_step(
                resp, cond_means, cond_cov_weighted, rng
            )

            change = max(
                float(np.abs(weights_new - weights).max()),
                float(np.abs(means_new - means).max()),
                float(np.abs(covariances_new - covariances).max()),
            )
            weights, means, covariances = weights_new, means_new, covariances_new

            if change < self.tol:
                converged = True
                break

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.n_iter_ = n_iter
        self.converged_ = converged
        return self

    def transform(self, X):
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
        Z_hat, _, _, _ = self._e_step(
            Z, mask, self.weights_, self.means_, self.covariances_
        )
        X_hat = Z_hat * self.scale_ + self.center_

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_hat, index=X.index, columns=X.columns)

        return X_hat