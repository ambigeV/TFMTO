"""
Utilities for BO_TFM_GPEmbed.

Pipeline
--------
  TabPFN  →  embedding φ(x)  →  PCA reduction  →  GP (RBF ARD) fit on φ(X_train)
  rolling cache of (x, φ_pca) pairs  →  EmbedMLP distils x → φ_hat
  gradient-based LCB optimisation:  x → MLP → φ_hat → GP posterior → LCB
  gradient flows all the way back to x via autograd.

Public API
----------
  tabpfn_get_embeddings     fit TabPFN, return test-role embeddings averaged over estimators
  fit_pca / apply_pca       lightweight numpy PCA (no sklearn dependency)
  fit_gp_ard_rbf            train ExactGP with RBF ARD, return precomputed param dict (f32)
  rbf_ard_kernel            differentiable RBF ARD kernel matrix (torch)
  gp_posterior_torch        differentiable GP posterior mean & variance
  EmbedMLP                  MLP: x_raw → φ_hat  (distils the embedding map)
  fit_embed_mlp             weighted MSE training for EmbedMLP
  lbfgs_optimize_gp_embed   multi-start L-BFGS-B through MLP + GP to minimise LCB
  EmbeddingCache            rolling window (last K iters) with exp-decay sample weights
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import gpytorch
from collections import deque


# =============================================================================
# 1.  TabPFN embedding extraction
# =============================================================================

def tabpfn_get_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    *,
    n_estimators: int = 1,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Fit TabPFN v2.5 on (X_train, y_train) and return test-role embeddings for X_query.

    Uses data_source='test' for both the training-point embeddings (when X_query
    overlaps with X_train) and for new candidates, keeping the embedding space
    consistent across GP training and acquisition optimisation.

    Parameters
    ----------
    X_train, y_train : context data  (already encoded if multi-task)
    X_query          : points to embed  (N, d_enc)
    n_estimators     : TabPFN ensemble size  (1 = fastest; matches TFM defaults)
    device           : 'cuda' or 'cpu'

    Returns
    -------
    phi : np.ndarray (N, D_emb) — mean over the estimator ensemble axis
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
    model.set_params(
        n_estimators=n_estimators,
        random_state=42,
        ignore_pretraining_limits=True,
        device=device,
    )
    model.fit(X_train, y_train)
    emb = model.get_embeddings(X_query, data_source='test')  # (n_est, N, D)
    return emb.mean(axis=0)                                  # (N, D)


# =============================================================================
# 2.  PCA  (numpy, no external dependency)
# =============================================================================

def fit_pca(phi: np.ndarray, n_components: int):
    """
    Fit PCA on phi (N, D).

    Returns
    -------
    pca_mean       : (D,)
    pca_components : (n_components, D) — rows are principal directions (unit vectors)

    Notes
    -----
    Uses full SVD of the centred matrix.  For large N this is O(N·D·min(N,D)),
    but typical cache sizes (≤10K points) make this negligible.
    """
    mean = phi.mean(axis=0)
    _, _, Vt = np.linalg.svd(phi - mean, full_matrices=False)
    return mean, Vt[:n_components]


def apply_pca(
    phi: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Project phi (N, D) → (N, n_components)."""
    return (phi - pca_mean) @ pca_components.T


# =============================================================================
# 3.  Differentiable RBF ARD kernel & GP posterior  (torch)
# =============================================================================

def rbf_ard_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
) -> torch.Tensor:
    """
    RBF ARD kernel matrix K(A, B).

    k(a, b) = outputscale * exp(-0.5 * Σ_d ((a_d - b_d) / ls_d)²)

    Parameters
    ----------
    A            : (n, D)
    B            : (m, D)
    lengthscale  : (D,)  — per-dimension length scales
    outputscale  : scalar

    Returns
    -------
    K : (n, m)   fully differentiable w.r.t. A (and B)
    """
    diff = (A.unsqueeze(1) - B.unsqueeze(0)) / lengthscale   # (n, m, D)
    sq_dist = (diff ** 2).sum(-1)                             # (n, m)
    return outputscale * torch.exp(-0.5 * sq_dist)


def gp_posterior_torch(
    phi_new: torch.Tensor,
    gp_params: dict,
) -> tuple:
    """
    Differentiable GP posterior mean and standard deviation at phi_new.

    Uses precomputed Cholesky factor L and weight vector alpha stored in
    gp_params (computed once by fit_gp_ard_rbf) to avoid O(n³) work per call.

    Gradient flows through phi_new  →  useful when phi_new = MLP(x).

    Parameters
    ----------
    phi_new   : (n*, D_pca) — may require grad
    gp_params : dict returned by fit_gp_ard_rbf  (all float32, same device)
        keys: lengthscale (D,), outputscale (), noise (),
              phi_train (n, D), L (n, n), alpha (n,)

    Returns
    -------
    mean : (n*,)  — differentiable w.r.t. phi_new
    std  : (n*,)  — differentiable w.r.t. phi_new, always > 0
    """
    ls    = gp_params['lengthscale']   # (D,)
    os2   = gp_params['outputscale']   # scalar
    phi_tr = gp_params['phi_train']    # (n, D)  — detached
    L      = gp_params['L']            # (n, n)  — detached lower-triangular
    alpha  = gp_params['alpha']        # (n,)    — detached

    K_nt = rbf_ard_kernel(phi_new, phi_tr, ls, os2)          # (n*, n)
    mean = K_nt @ alpha                                        # (n*,)

    # var(φ*) = k(φ*,φ*) - k(φ*,Φ) (K+σ²I)⁻¹ k(Φ,φ*)
    #         = k_diag  - ||L⁻¹ k(Φ, φ*)||²
    k_diag = os2 * torch.ones(
        phi_new.shape[0], dtype=phi_new.dtype, device=phi_new.device
    )
    v   = torch.linalg.solve_triangular(L, K_nt.T, upper=False)  # (n, n*)
    var = (k_diag - (v * v).sum(0)).clamp(min=1e-8)               # (n*,)
    std = torch.sqrt(var)

    return mean, std


# =============================================================================
# 4.  GP fitting  (GPyTorch ExactGP + RBF ARD, Adam MLL)
# =============================================================================

class _GPModel(gpytorch.models.ExactGP):
    """Internal ExactGP with Zero mean and ScaleKernel(RBFKernel(ard))."""

    def __init__(self, X, y, likelihood, D: int):
        super().__init__(X, y, likelihood)
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_gp_ard_rbf(
    phi_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    n_iter: int = 100,
    lr: float = 0.1,
) -> dict:
    """
    Train an ExactGP with RBF ARD kernel on (phi_train, y_train).

    y_train is z-scored before fitting so the GP operates in a well-scaled
    space regardless of objective magnitude.

    After training, precomputes:
      • Cholesky factor  L  of  K(Φ,Φ) + noise·I
      • Weight vector    α  =  L⁻ᵀ L⁻¹ y_norm
    These are stored in float32 for use in gp_posterior_torch during
    gradient-based acquisition optimisation.

    Parameters
    ----------
    phi_train   : (n, D_pca)  PCA-reduced embeddings of training data
    y_train     : (n,)        raw objective values
    device      : torch.device
    n_iter      : Adam iterations for MLL maximisation  (default 100)
    lr          : Adam learning rate  (default 0.1)

    Returns
    -------
    gp_params : dict of float32 tensors on `device`
        lengthscale : (D_pca,)
        outputscale : scalar
        noise       : scalar
        phi_train   : (n, D_pca)   — fixed reference embeddings
        L           : (n, n)       — lower Cholesky of (K + noise·I)
        alpha       : (n,)         — (K + noise·I)⁻¹ y_norm
    """
    D = phi_train.shape[1]

    # z-score y — keeps outputscale and noise in interpretable range
    y_mean = float(y_train.mean())
    y_std  = max(float(y_train.std()), 1e-8)
    y_norm = (y_train - y_mean) / y_std

    phi_t = torch.tensor(phi_train, dtype=torch.float64, device=device)
    y_t   = torch.tensor(y_norm,    dtype=torch.float64, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model      = _GPModel(phi_t, y_t, likelihood, D).to(device)
    model.train(); likelihood.train()

    optimiser = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimiser.zero_grad()
        loss = -mll(model(phi_t), y_t)
        loss.backward()
        optimiser.step()

    model.eval(); likelihood.eval()

    # --- extract hyperparams (float32) ---
    with torch.no_grad():
        ls    = model.covar_module.base_kernel.lengthscale \
                     .detach().squeeze(0).float().to(device)     # (D,)
        os2   = model.covar_module.outputscale.detach().float().to(device)   # scalar
        noise = likelihood.noise.detach().squeeze().float().to(device)       # scalar

        phi32 = phi_t.float()
        y32   = y_t.float()

        # Precompute K + noise·I and its Cholesky
        K_tt = rbf_ard_kernel(phi32, phi32, ls, os2)
        n    = K_tt.shape[0]
        K_tt_n = K_tt + (noise + 1e-6) * torch.eye(n, dtype=torch.float32, device=device)
        L     = torch.linalg.cholesky(K_tt_n)               # (n, n)
        alpha = torch.cholesky_solve(
            y32.unsqueeze(-1), L
        ).squeeze(-1)                                         # (n,)

    return {
        'lengthscale': ls,
        'outputscale': os2,
        'noise':       noise,
        'phi_train':   phi32,
        'L':           L,
        'alpha':       alpha,
    }


# =============================================================================
# 5.  EmbedMLP  — distils x_raw → φ_hat
# =============================================================================

class EmbedMLP(nn.Module):
    """
    MLP that approximates the TabPFN embedding map:  x_raw → φ_hat(x).

    Architecture:  [Linear → SiLU] × depth  →  Linear  (linear output head)

    The linear output head allows the network to represent any affine
    transformation of the embedding space without saturation artefacts.
    """

    def __init__(self, input_dim: int, embed_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., input_dim)  →  φ_hat : (..., embed_dim)"""
        return self.net(x)


def fit_embed_mlp(
    X_cache: np.ndarray,
    phi_pca_cache: np.ndarray,
    weights: np.ndarray,
    input_dim: int,
    embed_dim: int,
    *,
    hidden: int = 128,
    depth: int = 3,
    n_epochs: int = 300,
    finetune_epochs: int = 50,
    lr: float = 3e-3,
    init_model=None,
    device: torch.device = None,
) -> EmbedMLP:
    """
    Train EmbedMLP on (X_cache, phi_pca_cache) with per-sample weights.

    Weighted MSE:  L = Σ_i  w_i · ||φ_hat(x_i) - φ_i||²

    Parameters
    ----------
    X_cache       : (N, d)      raw decision vectors from rolling cache
    phi_pca_cache : (N, D_pca)  PCA-reduced TabPFN embeddings (targets)
    weights       : (N,)        non-negative sample weights (will be normalised)
    input_dim     : d   — raw decision variable dimension
    embed_dim     : D_pca
    n_epochs      : epochs for cold-start training
    finetune_epochs: epochs when warm-starting from init_model
    lr            : Adam learning rate
    init_model    : optional EmbedMLP to warm-start from
    device        : torch.device  (None → auto-detect)

    Returns
    -------
    mlp : trained EmbedMLP in eval mode on `device`
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_t   = torch.tensor(X_cache,       dtype=torch.float32, device=device)
    phi_t = torch.tensor(phi_pca_cache, dtype=torch.float32, device=device)
    w_t   = torch.tensor(weights,       dtype=torch.float32, device=device)
    w_t   = w_t / w_t.sum()

    if init_model is not None:
        mlp    = copy.deepcopy(init_model).to(device)
        epochs = finetune_epochs
    else:
        mlp    = EmbedMLP(input_dim, embed_dim, hidden=hidden, depth=depth).to(device)
        epochs = n_epochs

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        phi_pred = mlp(X_t)                                   # (N, D_pca)
        sq_err   = ((phi_pred - phi_t) ** 2).mean(dim=-1)    # (N,)  per-sample MSE
        loss     = (w_t * sq_err).sum()
        loss.backward()
        opt.step()

    mlp.eval()
    return mlp


# =============================================================================
# 6.  L-BFGS-B acquisition optimisation  (MLP → GP posterior → LCB)
# =============================================================================

def lbfgs_optimize_gp_embed(
    mlp: EmbedMLP,
    gp_params: dict,
    opt_dim: int,
    beta: float = 3.0,
    x0_points: np.ndarray = None,
    n_restarts: int = 5,
    device: torch.device = None,
) -> np.ndarray:
    """
    Minimise LCB(x) = mean(GP(MLP(x))) - beta * std(GP(MLP(x)))
    over [0, 1]^opt_dim via multi-start L-BFGS-B.

    Gradient chain:  x  →  MLP  →  φ_hat  →  GP posterior  →  LCB
    The GP posterior uses precomputed L and alpha from gp_params, so
    each objective evaluation costs one MLP forward pass + one
    RBF kernel row + a triangular solve — all O(n·D).

    Parameters
    ----------
    mlp        : fitted EmbedMLP (eval mode)
    gp_params  : dict from fit_gp_ard_rbf  (float32, on device)
    opt_dim    : raw decision variable dimension
    beta       : LCB exploration weight
    x0_points  : (K, opt_dim) warm-start points.  If None, uses n_restarts
                 uniform-random initialisations.
    n_restarts : fallback restart count when x0_points is None
    device     : torch.device  (None → inferred from mlp parameters)

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

    # Move gp_params to device (already there in normal usage, but be safe)
    gp_p = {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in gp_params.items()}

    def _obj_and_grad(x_flat: np.ndarray):
        x_t = torch.tensor(
            x_flat.reshape(1, -1), dtype=torch.float32, device=device,
            requires_grad=True,
        )
        phi_hat       = mlp(x_t)                              # (1, D_pca)
        mean, std     = gp_posterior_torch(phi_hat, gp_p)    # (1,), (1,)
        acq           = (mean - beta * std).sum()
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


# =============================================================================
# 7.  EmbeddingCache  — rolling window with exp-decay weights
# =============================================================================

class EmbeddingCache:
    """
    Rolling cache of raw (x, φ) pairs from the last `max_iters` BO iterations.

    Storage
    -------
    Each entry (one per BO iteration) holds:
        x   : np.ndarray (N_cand, d)      — raw decision vectors
        phi : np.ndarray (N_cand, D_emb)  — full-dim TabPFN embeddings
        iter: int                          — BO iteration index when added

    Entries older than `max_iters` iterations are dropped automatically on add().

    Weight decay
    ------------
    At retrieval time each sample's weight is:

        w_i = exp(-lambda_ * (current_iter - entry_iter))

    so the newest samples receive weight 1 and older batches decay smoothly.
    Weights are normalised to sum to 1 before being returned.

    Parameters
    ----------
    max_iters : int   rolling window size in iterations  (default 10)
    lambda_   : float exponential decay rate             (default 0.5)
    """

    def __init__(self, max_iters: int = 10, lambda_: float = 0.5):
        self.max_iters = max_iters
        self.lambda_   = lambda_
        self._entries: deque = deque()

    def add(self, x: np.ndarray, phi: np.ndarray, iteration: int):
        """
        Add a batch of (x, φ) pairs recorded at `iteration`.
        Drops all entries whose iteration ≤ (current - max_iters).
        """
        cutoff = iteration - self.max_iters
        while self._entries and self._entries[0]['iter'] <= cutoff:
            self._entries.popleft()
        self._entries.append({'x': x, 'phi': phi, 'iter': iteration})

    def retrieve(self, current_iter: int):
        """
        Return all cached (x, φ) pairs with their decay weights.

        Returns
        -------
        X_all   : np.ndarray (N_total, d)
        phi_all : np.ndarray (N_total, D_emb)
        weights : np.ndarray (N_total,)   — normalised, exp-decay
        None, None, None if the cache is empty.
        """
        if not self._entries:
            return None, None, None

        X_parts, phi_parts, w_parts = [], [], []
        for entry in self._entries:
            n_i = len(entry['x'])
            age = current_iter - entry['iter']
            w_i = float(np.exp(-self.lambda_ * age))
            X_parts.append(entry['x'])
            phi_parts.append(entry['phi'])
            w_parts.append(np.full(n_i, w_i, dtype=np.float32))

        X_all   = np.vstack(X_parts)
        phi_all = np.vstack(phi_parts)
        w_all   = np.concatenate(w_parts)
        w_all  /= w_all.sum()

        return X_all, phi_all, w_all

    def __len__(self) -> int:
        return sum(len(e['x']) for e in self._entries)

    @property
    def n_iters_stored(self) -> int:
        return len(self._entries)
