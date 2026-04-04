"""
Utilities for BO_TFM_ResGP.

Pipeline
--------
  TabPFN  →  μ_TFM(X_train)  →  residuals r = y - μ_TFM
  GP (RBF ARD) fitted on (X_train, r) in original x-space
  rolling cache of (x_cand, μ_TFM(x_cand)) pairs  →  PredMLP distils x → μ_hat
  gradient-based LCB optimisation:
      x  →  PredMLP  →  μ_hat(x)
      x  →  GP posterior (residuals)  →  μ_GP(x),  σ_GP(x)
      LCB = μ_hat(x) + μ_GP(x) - β · σ_GP(x)
      gradient flows through both branches simultaneously via autograd.

Shared GP components (kernel, posterior, fitting) are imported from
tfm_gp_embed_utils to avoid duplication.

Public API
----------
  PredMLP               MLP: x_raw → scalar μ̂  (distils TabPFN prediction map)
  fit_pred_mlp          weighted MSE training for PredMLP
  lbfgs_optimize_res_gp multi-start L-BFGS-B minimising the combined LCB
  PredictionCache       alias of EmbeddingCache — stores (x, μ_TFM) pairs

Shared from tfm_gp_embed_utils (re-exported for convenience):
  fit_gp_ard_rbf        train GP on (X_train, r), return precomputed param dict
  gp_posterior_torch    differentiable GP posterior mean & std
  EmbeddingCache        rolling cache with exp-decay weights
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from ddmtolab.Methods.Algo_Methods.tfm_gp_embed_utils import (
    fit_gp_ard_rbf,
    gp_posterior_torch,
    EmbeddingCache,
)

# Re-export for callers that import everything from this module
__all__ = [
    'PredMLP',
    'fit_pred_mlp',
    'lbfgs_optimize_res_gp',
    'PredictionCache',
    'fit_gp_ard_rbf',
    'gp_posterior_torch',
]

# PredictionCache is the same rolling-cache structure; rename for clarity
PredictionCache = EmbeddingCache


# =============================================================================
# 1.  PredMLP  — distils x_raw → scalar μ̂_TFM(x)
# =============================================================================

class PredMLP(nn.Module):
    """
    MLP that approximates the TabPFN prediction map:  x_raw → scalar μ̂(x).

    Architecture:  [Linear → SiLU] × depth  →  Linear(hidden, 1)

    The scalar output has no activation so the network can represent any
    real-valued function without range compression.
    """

    def __init__(self, input_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., input_dim)  →  μ̂ : (...,)  (squeezed scalar per sample)"""
        return self.net(x).squeeze(-1)


# =============================================================================
# 2.  fit_pred_mlp  — weighted MSE training
# =============================================================================

def fit_pred_mlp(
    X_cache: np.ndarray,
    mu_cache: np.ndarray,
    weights: np.ndarray,
    input_dim: int,
    *,
    hidden: int = 128,
    depth: int = 3,
    n_epochs: int = 300,
    finetune_epochs: int = 50,
    lr: float = 3e-3,
    init_model=None,
    device: torch.device = None,
) -> PredMLP:
    """
    Train PredMLP on (X_cache, mu_cache) with per-sample weights.

    Weighted MSE:  L = Σ_i  w_i · (μ̂(x_i) - μ_TFM_i)²

    Parameters
    ----------
    X_cache    : (N, d)   raw decision vectors from rolling cache
    mu_cache   : (N,)     TabPFN mean predictions (targets)
    weights    : (N,)     non-negative sample weights (will be normalised)
    input_dim  : d
    n_epochs   : epochs for cold-start training
    finetune_epochs : epochs when warm-starting from init_model
    lr         : Adam learning rate
    init_model : optional PredMLP to warm-start from
    device     : torch.device  (None → auto-detect)

    Returns
    -------
    mlp : trained PredMLP in eval mode on `device`
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_t  = torch.tensor(X_cache,  dtype=torch.float32, device=device)
    mu_t = torch.tensor(mu_cache, dtype=torch.float32, device=device)
    w_t  = torch.tensor(weights,  dtype=torch.float32, device=device)
    w_t  = w_t / w_t.sum()

    if init_model is not None:
        mlp    = copy.deepcopy(init_model).to(device)
        epochs = finetune_epochs
    else:
        mlp    = PredMLP(input_dim, hidden=hidden, depth=depth).to(device)
        epochs = n_epochs

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        mu_pred = mlp(X_t)                          # (N,)
        sq_err  = (mu_pred - mu_t) ** 2             # (N,)
        loss    = (w_t * sq_err).sum()
        loss.backward()
        opt.step()

    mlp.eval()
    return mlp


# =============================================================================
# 3.  lbfgs_optimize_res_gp  — multi-start L-BFGS-B on combined LCB
# =============================================================================

def lbfgs_optimize_res_gp(
    mlp: PredMLP,
    gp_params: dict,
    opt_dim: int,
    beta: float = 3.0,
    x0_points: np.ndarray = None,
    n_restarts: int = 5,
    device: torch.device = None,
) -> np.ndarray:
    """
    Minimise the combined LCB:

        LCB(x) = μ̂_MLP(x)  +  μ_GP(x)  −  β · σ_GP(x)

    where μ̂_MLP approximates the TabPFN prior mean and the GP models
    residuals in the original x-space.

    Gradient chain:
        ∂LCB/∂x = ∂μ̂_MLP/∂x  +  ∂μ_GP/∂x  −  β · ∂σ_GP/∂x

    Both terms are differentiable w.r.t. x:
      - MLP branch: standard backprop through PredMLP.
      - GP branch:  the RBF ARD kernel is differentiable w.r.t. x_new
                    (see gp_posterior_torch in tfm_gp_embed_utils).

    Parameters
    ----------
    mlp        : trained PredMLP (eval mode), maps x → scalar
    gp_params  : dict from fit_gp_ard_rbf fitted on (X_train, residuals)
                 — GP operates in the same x-space as PredMLP
    opt_dim    : raw decision variable dimension
    beta       : LCB exploration weight
    x0_points  : (K, opt_dim) warm-start points; if None uses n_restarts uniform randoms
    n_restarts : fallback restart count when x0_points is None
    device     : torch.device (None → inferred from mlp parameters)

    Returns
    -------
    best_x : np.ndarray (1, opt_dim) clipped to [0, 1]
    """
    from scipy.optimize import minimize as sp_minimize

    if device is None:
        try:
            device = next(mlp.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    mlp.eval()
    bounds = [(0.0, 1.0)] * opt_dim

    gp_p = {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in gp_params.items()}

    def _obj_and_grad(x_flat: np.ndarray):
        x_t = torch.tensor(
            x_flat.reshape(1, -1), dtype=torch.float32, device=device,
            requires_grad=True,
        )

        # TabPFN prior mean from distilled MLP
        mu_hat = mlp(x_t)                                  # (1,) → scalar after squeeze

        # GP residual posterior — GP lives in x-space, pass x_t directly
        mu_gp, std_gp = gp_posterior_torch(x_t, gp_p)    # (1,), (1,)

        acq = (mu_hat + mu_gp - beta * std_gp).sum()
        acq.backward()

        val  = acq.item()
        grad = x_t.grad.detach().cpu().numpy().ravel().astype(np.float64)
        return val, grad

    if x0_points is None:
        x0_points = np.random.rand(n_restarts, opt_dim)

    best_x   = None
    best_val = np.inf

    for x0 in x0_points:
        res = sp_minimize(
            _obj_and_grad,
            x0.astype(np.float64),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x   = res.x.reshape(1, -1)

    return np.clip(best_x, 0.0, 1.0)
