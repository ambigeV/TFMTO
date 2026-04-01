"""
Distilled-MLP utilities for gradient-based acquisition optimisation.

Workflow per BO step
--------------------
1. Fit TabPFN on (X_train, y_train)  — one call.
2. Sample N_distill points, query TabPFN once  → (mean, std).
3. Fit a small dual-head MLP (DistillMLP) on those (X_enc, mean, std) pairs.
4. Minimise LCB = mean(x) - beta * std(x) over [0,1]^d via multi-start
   Adam through the differentiable MLP.

GPU support
-----------
Pass device='cuda' (or torch.device('cuda')) to fit_distill_mlp and
adam_optimize_lcb to move the MLP and all tensors to GPU.  If device=None,
both functions auto-detect: GPU if torch.cuda.is_available(), else CPU.
tabpfn_predict honours its own device param to run TabPFN inference on GPU.
"""

import copy
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device) -> torch.device:
    """Return a torch.device, defaulting to CUDA if available."""
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


# ---------------------------------------------------------------------------
# MLP surrogate
# ---------------------------------------------------------------------------

class DistillMLP(nn.Module):
    """
    Dual-head MLP: encoded_x → (mean, std).

    Architecture:
      input → [Linear → SiLU] × depth → trunk
      trunk → head_mean  (linear)
      trunk → head_std   (linear → exp(clamp(log_std, -10, 6)))

    Why exp(log_std) over softplus(pre_act):
      - head_std outputs log_std directly (unbounded), so the MSE loss
        can supervise in log space: MSE(log_std_pred, log(std_target)).
        This is scale-invariant — a factor-of-2 error at std=0.01 is
        penalised the same as at std=1.0, unlike linear MSE.
      - For NLL loss: log(σ²) = 2·log_std, computed exactly without
        approximation artefacts.
      - clamp(-10, 6) keeps std in [4.5e-5, 403], covering any realistic
        TabPFN output range while preventing exp overflow/underflow.
      - Gradient of LCB w.r.t. x scales as exp(log_std)·∂log_std/∂x,
        i.e. proportional to the local uncertainty — high-std regions
        produce larger acquisition gradients, which is the right inductive
        bias for exploration-driven Adam optimisation.
    """

    def __init__(self, input_dim: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk     = nn.Sequential(*layers)
        self.head_mean = nn.Linear(hidden, 1)
        self.head_std  = nn.Linear(hidden, 1)   # outputs log_std

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor, shape (..., input_dim)

        Returns
        -------
        mean    : Tensor, shape (...,)
        std     : Tensor, shape (...,)  — always positive via exp(log_std)
        """
        h       = self.trunk(x)
        mean    = self.head_mean(h).squeeze(-1)
        log_std = self.head_std(h).squeeze(-1).clamp(-10, 6)
        std     = torch.exp(log_std)
        return mean, std


# ---------------------------------------------------------------------------
# MC-Dropout surrogate
# ---------------------------------------------------------------------------

class MCDropoutDistillMLP(nn.Module):
    """
    MC-Dropout dual-head MLP: encoded_x → (mean, std).

    Architecture:
      input → [Linear → SiLU → Dropout(p)] × depth → trunk
      trunk → head_mean  (linear)
      trunk → head_std   (linear → exp(clamp(log_std, -10, 6)))

    Two distinct inference modes
    ----------------------------
    forward(x)  [eval mode, dropout OFF]:
        Deterministic output from the std head only.
        Used by Adam — smooth, differentiable, no gradient noise.

    mc_predict(x, n_samples=20)  [dropout FORCED ON]:
        Runs T stochastic forward passes and combines aleatoric uncertainty
        (std head) with epistemic uncertainty (variance of mean across passes)
        via the law of total variance:

            total_var(x) = E_mask[σ²(x)]  +  Var_mask[μ(x)]
                           └── aleatoric ──┘  └─ epistemic ─┘
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        depth: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(p=dropout_p)]
            d = hidden
        self.trunk     = nn.Sequential(*layers)
        self.head_mean = nn.Linear(hidden, 1)
        self.head_std  = nn.Linear(hidden, 1)   # outputs log_std
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor):
        """Deterministic forward (dropout OFF in eval mode). Used for Adam."""
        h       = self.trunk(x)
        mean    = self.head_mean(h).squeeze(-1)
        log_std = self.head_std(h).squeeze(-1).clamp(-10, 6)
        return mean, torch.exp(log_std)

    def mc_predict(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ):
        """
        MC-Dropout inference with T stochastic forward passes.

        Returns
        -------
        mean_mc : Tensor (N,)  — E[μ(x)] over dropout masks
        std_mc  : Tensor (N,)  — sqrt(E[σ²] + Var[μ])  total uncertainty
        """
        was_training = self.training
        self.train()   # force dropout ON

        means, vars_al = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                h       = self.trunk(x)
                mu      = self.head_mean(h).squeeze(-1)
                log_std = self.head_std(h).squeeze(-1).clamp(-10, 6)
                means.append(mu)
                vars_al.append(torch.exp(2.0 * log_std))   # σ²

        means_t  = torch.stack(means,  dim=0)   # (T, N)
        vars_t   = torch.stack(vars_al, dim=0)  # (T, N)

        mean_mc = means_t.mean(dim=0)
        var_ep  = means_t.var(dim=0, unbiased=False)        # epistemic
        var_al  = vars_t.mean(dim=0)                        # mean aleatoric
        std_mc  = torch.sqrt(var_al + var_ep + 1e-12)

        if not was_training:
            self.eval()

        return mean_mc, std_mc


# ---------------------------------------------------------------------------
# Distillation fitting
# ---------------------------------------------------------------------------

def fit_distill_mlp(
    X_enc: np.ndarray,
    mean_targets: np.ndarray,
    std_targets: np.ndarray,
    hidden: int = 64,
    depth: int = 2,
    n_epochs: int = 300,
    lr: float = 3e-3,
    loss: str = 'mse',
    model_type: str = 'mlp',
    dropout_p: float = 0.1,
    init_model=None,
    device=None,
):
    """
    Fit a distilled MLP surrogate to approximate TabPFN's (mean, std) output.

    Parameters
    ----------
    X_enc        : (N, d_enc)  encoded candidate features (padded + task id)
    mean_targets : (N,)        TabPFN predicted means
    std_targets  : (N,)        TabPFN predicted stds
    hidden       : hidden layer width
    depth        : number of hidden layers
    n_epochs     : Adam steps
    lr           : Adam learning rate
    loss         : 'mse' — MSE(mean) + MSE(log_std)  scale-invariant  [default]
                   'nll' — Gaussian NLL, jointly calibrates mean & std
    model_type   : 'mlp'        — plain heteroscedastic DistillMLP      [default]
                   'mc_dropout' — MCDropoutDistillMLP
    dropout_p    : dropout probability (only used when model_type='mc_dropout')
    init_model   : optional pre-trained model to warm-start from
    device       : torch.device or str ('cuda'/'cpu').  None = auto-detect.

    Returns
    -------
    mlp : fitted model in eval mode, on the resolved device
    """
    if loss not in ('mse', 'nll'):
        raise ValueError(f"loss must be 'mse' or 'nll', got '{loss}'")
    if model_type not in ('mlp', 'mc_dropout'):
        raise ValueError(f"model_type must be 'mlp' or 'mc_dropout', got '{model_type}'")

    device = _resolve_device(device)

    X_t   = torch.tensor(X_enc,        dtype=torch.float32, device=device)
    mu_t  = torch.tensor(mean_targets, dtype=torch.float32, device=device)
    sig_t = torch.tensor(std_targets,  dtype=torch.float32, device=device)

    if init_model is not None:
        # Warm-start: deep-copy the previous model, move to device
        mlp = copy.deepcopy(init_model).to(device)
    else:
        input_dim = X_enc.shape[1]
        if model_type == 'mc_dropout':
            mlp = MCDropoutDistillMLP(input_dim, hidden=hidden, depth=depth,
                                      dropout_p=dropout_p).to(device)
        else:
            mlp = DistillMLP(input_dim, hidden=hidden, depth=depth).to(device)

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    log_sig_t = torch.log(sig_t.clamp(min=1e-8))   # target in log space for MSE

    mlp.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        # Access trunk + heads directly to obtain log_std without a second
        # log() call — avoids log(exp(x)) round-trip and keeps gradients exact.
        h         = mlp.trunk(X_t)
        mean_p    = mlp.head_mean(h).squeeze(-1)
        log_std_p = mlp.head_std(h).squeeze(-1).clamp(-10, 6)
        std_p     = torch.exp(log_std_p)

        if loss == 'nll':
            loss_val = F.gaussian_nll_loss(mean_p, mu_t, std_p ** 2, eps=1e-6)
        else:
            # MSE in log space for std: scale-invariant supervision.
            loss_val = F.mse_loss(mean_p, mu_t) + F.mse_loss(log_std_p, log_sig_t)
        loss_val.backward()
        opt.step()

    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# Adam-based acquisition optimisation
# ---------------------------------------------------------------------------

def adam_optimize_lcb(
    mlp,
    opt_dim: int,
    encode_torch_fn,
    beta: float = 1.0,
    n_restarts: int = 5,
    n_steps: int = 200,
    lr: float = 1e-2,
    device=None,
) -> np.ndarray:
    """
    Minimise LCB(x) = mean(x) - beta * std(x) over [0,1]^opt_dim
    using multi-start Adam with sigmoid reparameterisation.

    Reparameterisation:  x = sigmoid(theta),  theta ∈ R^opt_dim

    Adam is stochastic-friendly: unlike L-BFGS it does not require a
    consistent line-search objective, so it works naturally with both
    deterministic DistillMLP and MCDropoutDistillMLP (dropout in eval mode).
    Each restart uses a fresh random initialisation for ensemble-like
    diversity across the decision space.

    Parameters
    ----------
    mlp             : fitted DistillMLP or MCDropoutDistillMLP (in eval mode)
    opt_dim         : dimensionality of the raw decision variables (dims[i])
    encode_torch_fn : callable (Tensor[n, opt_dim]) -> Tensor[n, d_enc]
    beta            : LCB exploration weight
    n_restarts      : independent random initialisations
    n_steps         : Adam gradient steps per restart
    lr              : Adam learning rate
    device          : torch.device or str.  None = infer from mlp parameters.

    Returns
    -------
    best_x : np.ndarray, shape (1, opt_dim), clipped to [0, 1]
    """
    if device is None:
        try:
            device = next(mlp.parameters()).device
        except StopIteration:
            device = _resolve_device(None)

    mlp.eval()

    best_x   = None
    best_val = np.inf

    for _ in range(n_restarts):
        x0    = torch.empty(opt_dim, device=device).uniform_(0.1, 0.9)
        theta = torch.logit(x0).detach().requires_grad_(True)

        optimizer = torch.optim.Adam([theta], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            x     = torch.sigmoid(theta).unsqueeze(0)
            x_enc = encode_torch_fn(x)
            mean, std = mlp(x_enc)
            acq   = mean - beta * std
            acq.backward()
            optimizer.step()

        # --- evaluate candidate ---
        with torch.no_grad():
            x_fin     = torch.sigmoid(theta).unsqueeze(0)
            x_enc_fin = encode_torch_fn(x_fin)
            mean_ev, std_ev = mlp(x_enc_fin)
            val = (mean_ev - beta * std_ev).item()

        if val < best_val:
            best_val = val
            best_x   = torch.sigmoid(theta).detach().cpu().numpy().reshape(1, -1).copy()

    return np.clip(best_x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# L-BFGS-B acquisition optimisation
# ---------------------------------------------------------------------------

def lbfgs_optimize_lcb(
    mlp,
    opt_dim: int,
    encode_torch_fn,
    beta: float = 1.0,
    x0_points: np.ndarray = None,
    n_restarts: int = 5,
    device=None,
) -> np.ndarray:
    """
    Minimise LCB(x) = mean(x) - beta * std(x) over [0,1]^opt_dim
    using multi-start L-BFGS-B with box constraints.

    Starting points are selected externally (e.g. top-k LHS points ranked
    by LCB) and passed via ``x0_points``.  If not provided, falls back to
    ``n_restarts`` uniform-random initialisations.

    Parameters
    ----------
    mlp             : fitted DistillMLP or MCDropoutDistillMLP (in eval mode)
    opt_dim         : dimensionality of the raw decision variables (dims[i])
    encode_torch_fn : callable (Tensor[n, opt_dim]) -> Tensor[n, d_enc]
    beta            : LCB exploration weight
    x0_points       : (K, opt_dim) starting points for L-BFGS-B restarts.
                      When provided, ``n_restarts`` is ignored.
    n_restarts      : fallback number of random initialisations when
                      ``x0_points`` is None.
    device          : torch.device or str.  None = infer from mlp parameters.

    Returns
    -------
    best_x : np.ndarray, shape (1, opt_dim), clipped to [0, 1]
    """
    from scipy.optimize import minimize as sp_minimize

    if device is None:
        try:
            device = next(mlp.parameters()).device
        except StopIteration:
            device = _resolve_device(None)

    mlp.eval()

    bounds = [(0.0, 1.0)] * opt_dim

    def _obj_and_grad(x_flat):
        x_t = torch.tensor(
            x_flat.reshape(1, -1), dtype=torch.float32, device=device,
        )
        x_t.requires_grad_(True)
        x_enc = encode_torch_fn(x_t)
        mean, std = mlp(x_enc)
        acq = (mean - beta * std).sum()
        acq.backward()
        val  = acq.item()
        grad = x_t.grad.detach().cpu().numpy().ravel().astype(np.float64)
        return val, grad

    if x0_points is None:
        x0_points = np.random.rand(n_restarts, opt_dim)

    best_x   = None
    best_val = np.inf

    for k in range(len(x0_points)):
        x0 = x0_points[k].astype(np.float64)
        result = sp_minimize(
            _obj_and_grad, x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.reshape(1, -1)

    return np.clip(best_x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Combined distill + Adam helper (for non-distill TFM methods)
# ---------------------------------------------------------------------------

def adam_optimize_acq_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    opt_dim: int,
    encode_np_fn,
    encode_torch_fn,
    *,
    beta: float = 1.0,
    n_estimators: int = 8,
    n_distill: int = 200,
    mlp_hidden: int = 32,
    mlp_epochs: int = 100,
    adam_restarts: int = 3,
    adam_steps: int = 200,
    adam_lr: float = 1e-2,
    device=None,
) -> np.ndarray:
    """
    Adam-based acquisition optimisation for non-distill TFM variants.

    Workflow
    --------
    1. Sample n_distill random points from [0,1]^opt_dim; encode with
       encode_np_fn (handles padding + task-ID appending).
    2. Call TabPFN once on (X_train, y_train, X_enc) → (mean, std).
    3. Fit a lightweight DistillMLP on those predictions.
    4. Minimise LCB via Adam through the differentiable MLP.

    This replaces random-pool and CMA-ES acquisition with a single TabPFN
    call + MLP training + gradient-based inner loop — same speed class as
    the random baseline but with gradient information.

    Parameters
    ----------
    X_train, y_train : training data already assembled (with task encoding)
    opt_dim          : raw decision variable dimension (before encoding)
    encode_np_fn     : X_raw (n, opt_dim) → X_enc (n, d_enc)  [numpy]
    encode_torch_fn  : Tensor(n, opt_dim) → Tensor(n, d_enc)  [torch, for Adam]
    beta             : LCB weight
    n_estimators     : TabPFN ensemble size
    n_distill        : number of distillation points
    mlp_hidden       : MLP hidden width
    mlp_epochs       : MLP training epochs
    adam_restarts    : Adam restarts
    adam_steps       : Adam steps per restart
    adam_lr          : Adam learning rate
    device           : torch.device or str.  None = auto-detect.

    Returns
    -------
    best_x : np.ndarray, shape (1, opt_dim)
    """
    from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict

    device = _resolve_device(device)
    device_str = str(device).split(':')[0]   # 'cuda' or 'cpu' for TabPFN

    # 1. Distillation query
    X_raw = np.random.rand(n_distill, opt_dim)
    X_enc = encode_np_fn(X_raw)
    mean_d, std_d = tabpfn_predict(
        X_train, y_train, X_enc,
        return_std=True,
        n_estimators=n_estimators,
        device=device_str,
    )

    # 2. Fit lightweight MLP
    mlp = fit_distill_mlp(
        X_enc, mean_d, std_d,
        hidden=mlp_hidden,
        depth=2,
        n_epochs=mlp_epochs,
        lr=3e-3,
        loss='mse',
        device=device,
    )

    # 3. Adam optimisation
    return adam_optimize_lcb(
        mlp, opt_dim, encode_torch_fn,
        beta=beta,
        n_restarts=adam_restarts,
        n_steps=adam_steps,
        lr=adam_lr,
        device=device,
    )


# ---------------------------------------------------------------------------
# PyTorch-native task encoding (keeps gradient graph intact)
# ---------------------------------------------------------------------------

def encode_torch_scalar(
    x: torch.Tensor,
    max_dim: int,
    task_id: int,
) -> torch.Tensor:
    """
    Pad x to max_dim columns, then append a scalar task-ID column.
    Feature layout: [x_0, ..., x_{max_dim-1}, task_id]
    Gradient flows through x (not through the constant padding/task columns).
    """
    n, d = x.shape
    if d < max_dim:
        pad   = torch.zeros(n, max_dim - d, dtype=x.dtype, device=x.device)
        x_pad = torch.cat([x, pad], dim=1)
    else:
        x_pad = x
    tid = torch.full((n, 1), float(task_id), dtype=x.dtype, device=x.device)
    return torch.cat([x_pad, tid], dim=1)


def encode_torch_onehot(
    x: torch.Tensor,
    max_dim: int,
    task_id: int,
    n_tasks: int,
) -> torch.Tensor:
    """
    Pad x to max_dim columns, then append a one-hot task encoding.
    Feature layout: [x_0, ..., x_{max_dim-1}, oh_0, ..., oh_{n_tasks-1}]
    """
    n, d = x.shape
    if d < max_dim:
        pad   = torch.zeros(n, max_dim - d, dtype=x.dtype, device=x.device)
        x_pad = torch.cat([x, pad], dim=1)
    else:
        x_pad = x
    oh = torch.zeros(n, n_tasks, dtype=x.dtype, device=x.device)
    oh[:, task_id] = 1.0
    return torch.cat([x_pad, oh], dim=1)
