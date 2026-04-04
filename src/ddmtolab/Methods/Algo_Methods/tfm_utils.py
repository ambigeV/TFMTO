"""
TabPFN utility functions for surrogate-based Bayesian Optimization.

Model version: TabPFN v2.5 (loaded via ModelVersion.V2_5).

This module supports both:
  - Regression: TabPFNRegressor via tabpfn_predict()
  - Classification: TabPFNClassifier via tabpfn_predict_proba()

The predictive std is derived from the 80% CI (10th–90th percentile) under a
Gaussian assumption:  std = (q90 - q10) / (2 * z_{0.90})  where z_{0.90} ≈ 1.2816.

Batch prediction advice:
  - Optimal chunk size equals n_train: minimises total attention ops
    (n_train + c)^2 * (n_test / c), solved at c* = n_train.
  - tabpfn_predict() sets chunk = max(len(X_train), 20) automatically.
  - Never use a fixed large chunk (e.g. 500) — it is suboptimal for
    small training sets and wastes compute throughout the BO run.

Acquisition optimiser:
  - optimize_acq_cmaes() uses LM-CMA-ES (Limited-Memory CMA).
  - No d×d covariance matrix; no eigendecomposition.
  - Cost per generation: O(m × d × λ),  m = max(4, ceil(log2(d))).
  - Full CMA-ES was O(d² × λ + d³) — 6× cheaper at d=50, identical at d=10.
"""
import numpy as np
from scipy.stats import norm as scipy_norm

_Z_80 = scipy_norm.ppf(0.90)   # 1.2816


def _build_model(n_estimators: int, random_state: int, device: str = 'cpu'):
    """Instantiate a TabPFN v2.5 regressor with the given settings."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
    model.set_params(
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
        device=device,
    )
    return model


def _build_classifier_model(n_estimators: int, random_state: int, device: str = 'cpu'):
    """Instantiate a TabPFN v2.5 classifier with the given settings."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
    model.set_params(
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
        device=device,
    )
    return model


def tabpfn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    return_std: bool = False,
    n_estimators: int = 8,
    random_state: int = 42,
    device: str = None,
) -> 'np.ndarray | tuple[np.ndarray, np.ndarray]':
    """Fit TabPFN v2.5 on (X_train, y_train) and predict on X_test.

    Parameters
    ----------
    X_train, y_train : training data
    X_test           : test features
    return_std       : if True, also return GP-equivalent predictive std
                       computed as (q90 - q10) / (2 * 1.2816)
    n_estimators     : TabPFN ensemble size
    random_state     : RNG seed
    device           : 'cuda' or 'cpu'.  None = 'cuda' if available else 'cpu'.

    Returns
    -------
    y_pred           : shape (n_test,)
    std              : shape (n_test,)  — only when return_std=True

    Notes
    -----
    X_test is chunked at chunk_size = max(len(X_train), 20) — the optimal
    split that minimises total attention ops (n_train + c)^2 * (n_test / c).
    """
    import torch as _torch
    if device is None:
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    model = _build_model(n_estimators, random_state, device=device)
    model.fit(X_train, y_train)

    n_test  = len(X_test)
    chunk   = max(len(X_train), 20)   # optimal: c* = n_train

    if not return_std:
        if n_test <= chunk:
            return model.predict(X_test)
        parts = [
            model.predict(X_test[i:i + chunk])
            for i in range(0, n_test, chunk)
        ]
        return np.concatenate(parts)

    # --- with std ---
    if n_test <= chunk:
        output = model.predict(X_test, output_type="main")
        y_pred = output["mean"]
        quantiles = np.array(output["quantiles"])   # (n_quantiles, n_test)
        q10, q90 = quantiles[0], quantiles[-1]
        std = (q90 - q10) / (2.0 * _Z_80)
        return y_pred, std

    y_parts, std_parts = [], []
    for i in range(0, n_test, chunk):
        block  = X_test[i:i + chunk]
        output = model.predict(block, output_type="main")
        y_parts.append(output["mean"])
        quantiles = np.array(output["quantiles"])
        q10, q90  = quantiles[0], quantiles[-1]
        std_parts.append((q90 - q10) / (2.0 * _Z_80))
    return np.concatenate(y_parts), np.concatenate(std_parts)


def tabpfn_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_estimators: int = 8,
    random_state: int = 42,
    device: str = None,
) -> np.ndarray:
    """Fit TabPFN classifier on (X_train, y_train) and return class probabilities.

    Parameters
    ----------
    X_train, y_train : training data with integer labels in [0, n_classes-1]
    X_test           : test features
    n_estimators     : TabPFN ensemble size
    random_state     : RNG seed
    device           : 'cuda' or 'cpu'. None = auto

    Returns
    -------
    proba : np.ndarray, shape (n_test, n_classes)
    """
    import torch as _torch
    if device is None:
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'

    model = _build_classifier_model(n_estimators, random_state, device=device)
    model.fit(X_train, y_train.astype(int))

    n_test = len(X_test)
    chunk = max(len(X_train), 20)
    if n_test <= chunk:
        return model.predict_proba(X_test)

    parts = [
        model.predict_proba(X_test[i:i + chunk])
        for i in range(0, n_test, chunk)
    ]
    return np.vstack(parts)


def lcb(mean: np.ndarray, std: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Lower Confidence Bound for minimization: mean - beta * std."""
    return mean - beta * std


def append_task_id(X: np.ndarray, task_id: int) -> np.ndarray:
    """Append a constant task-ID column to a feature matrix."""
    id_col = np.full((len(X), 1), task_id, dtype=X.dtype)
    return np.hstack([X, id_col])


def append_task_id_onehot(X: np.ndarray, task_id: int, n_tasks: int) -> np.ndarray:
    """Append a one-hot task encoding (n_tasks columns) to a feature matrix.

    Avoids the false ordinal relationship introduced by a scalar task ID.
    For task_id=1 with n_tasks=3 the appended columns are [0, 1, 0].
    """
    onehot = np.zeros((len(X), n_tasks), dtype=X.dtype)
    onehot[:, task_id] = 1.0
    return np.hstack([X, onehot])


def optimize_acq_cmaes(
    score_fn,
    dim: int,
    population_size: int = 20,
    max_iter: int = 50,
) -> np.ndarray:
    """Optimise an acquisition function over [0,1]^dim using LM-CMA-ES.

    Limited-Memory CMA-ES: replaces the full d×d covariance matrix with a
    rolling buffer of m unit-direction vectors derived from the rank-1
    evolution path.  No eigendecomposition is required.

    Parameters
    ----------
    score_fn       : callable (n, dim) → (n,)  lower = better (LCB).
                     Called once per generation with the full population batch.
    dim            : decision variable dimensionality
    population_size: λ — candidates sampled per generation (default 20)
    max_iter       : number of generations (default 50)

    Returns
    -------
    best_x : np.ndarray, shape (1, dim), clipped to [0, 1]

    Complexity
    ----------
    Per generation: O(m × d × λ)  vs  O(d² × λ + d³) for full CMA-ES.
    Memory:         O(m × d)       vs  O(d²).
    m = max(4, ceil(log2(dim))):   m=4 at d=10,  m=6 at d=50,  m=7 at d=100.

    Method
    ------
    The implicit covariance is C ≈ I + Σ_k c_k · v_k · v_k^T where {v_k} are
    unit vectors stored in the buffer.  Sampling applies sequential rank-1
    forward transforms; the CSA step-size path uses the corresponding reverse
    inverse transforms — both are O(m × d) per generation.

    The forward coefficient for direction v derived from evolution path pc is:
        c_fwd = sqrt(1 + c1 · ‖pc‖²) − 1
    which is the exact linearised square-root of the rank-1 covariance update
    c1 · pc · pc^T (unit-norm direction, scaled by ‖pc‖).
    The inverse coefficient follows from (I + c·vv^T)^{-1} = I − c/(1+c) · vv^T.
    """
    m_lim = max(4, int(np.ceil(np.log2(max(dim, 2)))))   # buffer size
    lam   = population_size
    mu    = lam // 2

    # --- recombination weights ---
    raw_w   = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw_w / raw_w.sum()
    mueff   = 1.0 / np.sum(weights ** 2)

    # --- learning rates (identical to full CMA-ES) ---
    cc    = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs    = (mueff + 2.0) / (dim + mueff + 5.0)
    c1    = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu   = min(1.0 - c1,
                2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chiN  = dim ** 0.5 * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    # --- state ---
    mean  = np.random.rand(dim)
    sigma = 0.3
    pc    = np.zeros(dim)
    ps    = np.zeros(dim)

    # directions: list of (v, c_fwd) — unit vector + forward coefficient, oldest first
    directions = []

    best_x = mean.copy()
    best_f = np.inf

    for gen in range(max_iter):
        # --- sample λ candidates via sequential rank-1 forward transforms ---
        z = np.random.randn(lam, dim)                  # (lam, d)  isotropic base
        for v, c in directions:
            proj = z @ v                               # (lam,)
            z   += c * np.outer(proj, v)               # (lam, d)

        x_raw  = mean + sigma * z
        x_cand = np.clip(x_raw, 0.0, 1.0)

        # --- evaluate entire population in ONE batch call ---
        scores = score_fn(x_cand)                      # (lam,)

        # --- track global best ---
        idx = int(np.argmin(scores))
        if scores[idx] < best_f:
            best_f = float(scores[idx])
            best_x = x_cand[idx].copy()

        # --- selection + mean update ---
        order    = np.argsort(scores)
        x_sel    = x_raw[order[:mu]]                   # unclipped elites
        mean_old = mean.copy()
        mean     = np.clip(weights @ x_sel, 0.0, 1.0)
        step     = (mean - mean_old) / sigma

        # --- approximate invsqrtC @ step via reverse inverse transforms ---
        # (I + c·vv^T)^{-1} x = x − c/(1+c) · (v^T x) · v, applied oldest→newest reversed
        s = step.copy()
        for v, c in reversed(directions):
            s -= (c / (1.0 + c)) * np.dot(v, s) * v

        # --- update evolution paths ---
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * s
        hsig = float(
            np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (gen + 1))) / chiN
            < 1.4 + 2.0 / (dim + 1.0)
        )
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * step

        # --- add new direction from rank-1 evolution path ---
        pc_norm = float(np.linalg.norm(pc))
        if pc_norm > 1e-10:
            v_new = pc / pc_norm
            c_fwd = float(np.sqrt(1.0 + c1 * pc_norm ** 2) - 1.0)
            if len(directions) >= m_lim:
                directions.pop(0)                      # evict oldest
            directions.append((v_new, c_fwd))

        # --- cumulative step-size adaptation ---
        sigma = float(sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0)))
        sigma = float(np.clip(sigma, 1e-10, 2.0))

    return best_x.reshape(1, -1)


def pad_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad X columns to target_dim if X.shape[1] < target_dim."""
    if X.shape[1] < target_dim:
        pad = np.zeros((len(X), target_dim - X.shape[1]), dtype=X.dtype)
        return np.hstack([X, pad])
    return X
