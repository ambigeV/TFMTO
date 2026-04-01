"""
TabPFN utility functions for surrogate-based Bayesian Optimization.

Model version: TabPFN v2.5 (loaded via ModelVersion.V2_5).

The predictive std is derived from the 80% CI (10th–90th percentile) under a
Gaussian assumption:  std = (q90 - q10) / (2 * z_{0.90})  where z_{0.90} ≈ 1.2816.

Batch prediction advice:
  - Always pass the full candidate pool in a single predict() call.
  - Each predict() recomputes the training context, so N separate calls
    is ~N× slower than one batched call.
  - If n_test > 1000, split into chunks of 1000 and concatenate results.
"""
import numpy as np
from scipy.stats import norm as scipy_norm

_Z_80 = scipy_norm.ppf(0.90)   # 1.2816
_CHUNK = 500                    # max test points per predict() call


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
    X_test is scored in a single batched predict() call (or in chunks of
    _CHUNK if n_test > _CHUNK) to avoid redundant context recomputation.
    """
    import torch as _torch
    if device is None:
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    model = _build_model(n_estimators, random_state, device=device)
    model.fit(X_train, y_train)

    n_test = len(X_test)

    if not return_std:
        if n_test <= _CHUNK:
            return model.predict(X_test)
        # chunk for large test sets
        parts = [
            model.predict(X_test[i:i + _CHUNK])
            for i in range(0, n_test, _CHUNK)
        ]
        return np.concatenate(parts)

    # --- with std ---
    if n_test <= _CHUNK:
        output = model.predict(X_test, output_type="main")
        y_pred = output["mean"]
        quantiles = np.array(output["quantiles"])   # (n_quantiles, n_test)
        q10, q90 = quantiles[0], quantiles[-1]
        std = (q90 - q10) / (2.0 * _Z_80)
        return y_pred, std

    # chunk for large test sets
    y_parts, std_parts = [], []
    for i in range(0, n_test, _CHUNK):
        chunk = X_test[i:i + _CHUNK]
        output = model.predict(chunk, output_type="main")
        y_parts.append(output["mean"])
        quantiles = np.array(output["quantiles"])
        q10, q90 = quantiles[0], quantiles[-1]
        std_parts.append((q90 - q10) / (2.0 * _Z_80))
    return np.concatenate(y_parts), np.concatenate(std_parts)


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
    """Optimise an acquisition function over [0,1]^dim using (μ/μ_w, λ)-CMA-ES.

    Parameters
    ----------
    score_fn       : callable (n, dim) → (n,)  lower = better (LCB)
                     Called once per generation with the full population batch,
                     so TabPFN's training context is recomputed only max_iter times.
    dim            : decision variable dimensionality
    population_size: λ — number of candidates sampled per generation (default 20)
    max_iter       : number of CMA-ES generations (default 50)

    Returns
    -------
    best_x : np.ndarray, shape (1, dim), clipped to [0, 1]
    """
    lam = population_size
    mu  = lam // 2

    # --- recombination weights ---
    raw_w   = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw_w / raw_w.sum()
    mueff   = 1.0 / np.sum(weights ** 2)

    # --- learning rates ---
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
    C     = np.eye(dim)
    B     = np.eye(dim)          # eigenvectors of C
    D     = np.ones(dim)         # sqrt of eigenvalues of C
    invsqrtC = np.eye(dim)
    eigeneval = 0

    best_x = mean.copy()
    best_f = np.inf

    for gen in range(max_iter):
        # --- lazy eigendecomposition (amortised cost) ---
        if gen - eigeneval > lam / (c1 + cmu) / dim / 10.0:
            eigeneval = gen
            C = np.triu(C) + np.triu(C, 1).T          # enforce symmetry
            eigvals, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-20))
            invsqrtC = B @ np.diag(1.0 / D) @ B.T

        # --- sample λ candidates ---
        z         = np.random.randn(lam, dim)
        y         = z @ (B * D).T                      # shape (lam, dim)
        x_raw     = mean + sigma * y                   # unclipped (for CMA updates)
        x_cand    = np.clip(x_raw, 0.0, 1.0)          # clipped (for evaluation)

        # --- evaluate entire population in ONE batch call ---
        scores = score_fn(x_cand)                      # (lam,)

        # --- track global best ---
        best_gen = int(np.argmin(scores))
        if scores[best_gen] < best_f:
            best_f = float(scores[best_gen])
            best_x = x_cand[best_gen].copy()

        # --- selection: top-μ by fitness ---
        order   = np.argsort(scores)
        x_sel   = x_raw[order[:mu]]                    # unclipped elites for update

        # --- update mean ---
        mean_old = mean.copy()
        mean     = weights @ x_sel
        mean     = np.clip(mean, 0.0, 1.0)

        # --- update evolution paths ---
        step     = (mean - mean_old) / sigma
        ps       = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (invsqrtC @ step)
        hsig     = float(
            np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (gen + 1))) / chiN
            < 1.4 + 2.0 / (dim + 1.0)
        )
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * step

        # --- update covariance ---
        artmp = (x_sel - mean_old) / sigma             # shape (mu, dim)
        C = (
            (1.0 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
            + cmu * (weights * artmp.T) @ artmp
        )

        # --- update step size ---
        sigma = float(sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0)))
        sigma = float(np.clip(sigma, 1e-10, 2.0))

    return best_x.reshape(1, -1)


def pad_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad X columns to target_dim if X.shape[1] < target_dim."""
    if X.shape[1] < target_dim:
        pad = np.zeros((len(X), target_dim - X.shape[1]), dtype=X.dtype)
        return np.hstack([X, pad])
    return X
