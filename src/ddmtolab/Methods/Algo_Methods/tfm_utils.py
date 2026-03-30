"""
TabPFN utility functions for surrogate-based Bayesian Optimization.

The predictive std is derived from the 80% CI (10th–90th percentile) under a
Gaussian assumption:  std = (q90 - q10) / (2 * z_{0.90})  where z_{0.90} ≈ 1.2816.
"""
import numpy as np
from scipy.stats import norm as scipy_norm

_Z_80 = scipy_norm.ppf(0.90)   # 1.2816


def tabpfn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    return_std: bool = False,
    n_estimators: int = 8,
    random_state: int = 42,
) -> 'np.ndarray | tuple[np.ndarray, np.ndarray]':
    """Fit TabPFN on (X_train, y_train) and predict on X_test.

    Parameters
    ----------
    X_train, y_train : training data
    X_test           : test features
    return_std       : if True, also return GP-equivalent predictive std
                       computed as (q90 - q10) / (2 * 1.2816)
    n_estimators     : TabPFN ensemble size
    random_state     : RNG seed

    Returns
    -------
    y_pred           : shape (n_test,)
    std              : shape (n_test,)  — only when return_std=True
    """
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
    )
    model.fit(X_train, y_train)

    if not return_std:
        return model.predict(X_test)

    output = model.predict(X_test, output_type="main")
    y_pred = output["mean"]
    quantiles = np.array(output["quantiles"])   # (n_quantiles, n_test)
    q10, q90 = quantiles[0], quantiles[-1]
    std = (q90 - q10) / (2.0 * _Z_80)          # GP-equivalent σ
    return y_pred, std


def lcb(mean: np.ndarray, std: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Lower Confidence Bound for minimization: mean - beta * std."""
    return mean - beta * std


def append_task_id(X: np.ndarray, task_id: int) -> np.ndarray:
    """Append a constant task-ID column to a feature matrix."""
    id_col = np.full((len(X), 1), task_id, dtype=X.dtype)
    return np.hstack([X, id_col])


def pad_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad X columns to target_dim if X.shape[1] < target_dim."""
    if X.shape[1] < target_dim:
        pad = np.zeros((len(X), target_dim - X.shape[1]), dtype=X.dtype)
        return np.hstack([X, pad])
    return X
