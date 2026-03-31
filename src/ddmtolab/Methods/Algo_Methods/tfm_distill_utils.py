"""
Distilled-MLP utilities for gradient-based acquisition optimisation.

Workflow per BO step
--------------------
1. Fit TabPFN on (X_train, y_train)  — one call.
2. Sample N_distill points, query TabPFN once  → (mean, std).
3. Fit a small dual-head MLP (DistillMLP) on those (X_enc, mean, std) pairs.
4. Minimise LCB = mean(x) - beta * std(x) over [0,1]^d via multi-start
   L-BFGS through the differentiable MLP.

Compared to CMA-ES (50 TabPFN calls per BO step), this approach calls
TabPFN exactly once for distillation and then uses fast MLP gradient
evaluations for the inner optimisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        bias for exploration-driven L-BFGS.
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
        Used by L-BFGS — smooth, differentiable, no gradient noise.

    mc_predict(x, n_samples=20)  [dropout FORCED ON]:
        Runs T stochastic forward passes and combines aleatoric uncertainty
        (std head) with epistemic uncertainty (variance of mean across passes)
        via the law of total variance:

            total_var(x) = E_mask[σ²(x)]  +  Var_mask[μ(x)]
                           └── aleatoric ──┘  └─ epistemic ─┘

        This is richer than the plain DistillMLP: regions poorly covered by
        distillation points get higher epistemic variance, which widens the
        LCB and promotes exploration there.

    Design note — why keep the std head alongside dropout
    -----------------------------------------------------
    Pure MC-Dropout (mean head only) discards the aleatoric structure that
    TabPFN explicitly predicts.  Keeping the std head lets the model
    distinguish between:
      • inherent objective noise  (std head ↑, epistemic ≈ 0)
      • under-explored regions    (std head moderate, epistemic ↑)
    Both matter for a well-calibrated LCB.
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
        """
        Deterministic forward (dropout OFF in eval mode).
        Returns (mean, std_aleatoric). Used for L-BFGS.
        """
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

        Parameters
        ----------
        x        : Tensor (N, d_enc)
        n_samples: T — number of MC samples

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
    init_model   : optional pre-trained DistillMLP / MCDropoutDistillMLP.
                   When provided, weights are deep-copied and fine-tuned rather
                   than training from a fresh random initialisation.
                   Use with a reduced n_epochs (e.g. 50) for warm-starting:
                   the previous model already fits the old TabPFN landscape;
                   a handful of gradient steps adapts it to the one new point
                   that was added this iteration.

    Returns
    -------
    mlp : fitted model in eval mode
    """
    import copy

    if loss not in ('mse', 'nll'):
        raise ValueError(f"loss must be 'mse' or 'nll', got '{loss}'")
    if model_type not in ('mlp', 'mc_dropout'):
        raise ValueError(f"model_type must be 'mlp' or 'mc_dropout', got '{model_type}'")

    X_t   = torch.tensor(X_enc,        dtype=torch.float32)
    mu_t  = torch.tensor(mean_targets, dtype=torch.float32)
    sig_t = torch.tensor(std_targets,  dtype=torch.float32)

    if init_model is not None:
        # Warm-start: deep-copy the previous model so the cache is not mutated
        # in-place by subsequent training steps.
        mlp = copy.deepcopy(init_model)
    else:
        input_dim = X_enc.shape[1]
        if model_type == 'mc_dropout':
            mlp = MCDropoutDistillMLP(input_dim, hidden=hidden, depth=depth, dropout_p=dropout_p)
        else:
            mlp = DistillMLP(input_dim, hidden=hidden, depth=depth)

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
            # Gaussian NLL: jointly calibrates mean and std.
            # F.gaussian_nll_loss expects var = std².
            loss_val = F.gaussian_nll_loss(mean_p, mu_t, std_p ** 2, eps=1e-6)
        else:
            # MSE in log space for std: scale-invariant supervision.
            loss_val = F.mse_loss(mean_p, mu_t) + F.mse_loss(log_std_p, log_sig_t)
        loss_val.backward()
        opt.step()

    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# Gradient-based acquisition optimisation
# ---------------------------------------------------------------------------

def lbfgs_optimize_lcb(
    mlp,
    opt_dim: int,
    encode_torch_fn,
    beta: float = 1.0,
    n_restarts: int = 5,
    max_iter: int = 100,
    mc_lbfgs: bool = False,
    mc_samples_eval: int = 20,
) -> np.ndarray:
    """
    Minimise LCB(x) = mean(x) - beta * std(x) over [0,1]^opt_dim
    using multi-start L-BFGS with sigmoid reparameterisation.

    Reparameterisation:  x = sigmoid(theta),  theta ∈ R^opt_dim

    Parameters
    ----------
    mlp             : fitted DistillMLP or MCDropoutDistillMLP
    opt_dim         : dimensionality of the raw decision variables (dims[i])
    encode_torch_fn : callable (Tensor[n, opt_dim]) -> Tensor[n, d_enc]
    beta            : LCB exploration weight
    n_restarts      : independent random initialisations
    max_iter        : L-BFGS iterations per restart
    mc_lbfgs        : bool, default False
        False (default) — eval mode, dropout OFF.
            Plain heteroscedastic std head drives the acquisition.
            Works for both DistillMLP and MCDropoutDistillMLP.

        True — fixed-mask L-BFGS (only meaningful for MCDropoutDistillMLP).
            Before each restart the PyTorch RNG state is captured and restored
            at the start of every closure call, so every evaluation within a
            restart uses the IDENTICAL dropout mask.  This makes the objective
            deterministic for the line search while each restart samples a
            DIFFERENT mask, giving ensemble-like diversity across restarts.

            Winner selection uses mc_predict (full MC mean) rather than the
            single-mask LCB, so the returned x* is the one with the best
            combined (aleatoric + epistemic) acquisition value.

            Why this works
            --------------
            Standard L-BFGS breaks with stochastic dropout because the Wolfe
            line search evaluates f(x + α·d) multiple times and requires
            consistent values.  Pinning the RNG state makes f deterministic
            within a restart: the closure always computes the same function,
            strong Wolfe is satisfied, and the gradient through the fixed
            sparse network is valid.  Across restarts, each mask corresponds
            to a different sampled sub-network from the weight-uncertainty
            posterior — n_restarts restarts now serve double duty as both
            multi-start diversity AND Monte Carlo integration over masks.
    mc_samples_eval : MC samples used for winner evaluation when mc_lbfgs=True

    Returns
    -------
    best_x : np.ndarray, shape (1, opt_dim), clipped to [0, 1]
    """
    use_fixed_mask = mc_lbfgs and isinstance(mlp, MCDropoutDistillMLP)

    if use_fixed_mask:
        mlp.train()   # dropout ON — mask is controlled via RNG state pinning
    else:
        mlp.eval()    # dropout OFF — deterministic forward for standard L-BFGS

    best_x   = None
    best_val = np.inf

    for _ in range(n_restarts):
        x0    = torch.empty(opt_dim).uniform_(0.1, 0.9)
        theta = torch.logit(x0).detach().requires_grad_(True)

        if use_fixed_mask:
            # Capture RNG state once per restart — every closure call will
            # restore it, guaranteeing the same dropout mask throughout.
            restart_rng = torch.get_rng_state()

        optimizer = torch.optim.LBFGS(
            [theta],
            max_iter=max_iter,
            line_search_fn='strong_wolfe',
        )

        def closure():
            if use_fixed_mask:
                torch.set_rng_state(restart_rng)   # pin mask for this restart
            optimizer.zero_grad()
            x     = torch.sigmoid(theta).unsqueeze(0)
            x_enc = encode_torch_fn(x)
            mean, std = mlp(x_enc)
            acq   = mean - beta * std
            acq.backward()
            return acq

        optimizer.step(closure)

        # --- evaluate the candidate ---
        with torch.no_grad():
            x_fin = torch.sigmoid(theta).unsqueeze(0)
            x_enc = encode_torch_fn(x_fin)
            if use_fixed_mask:
                # score with full MC (not the single-mask value used during optim)
                mean_ev, std_ev = mlp.mc_predict(x_fin, n_samples=mc_samples_eval)
            else:
                mean_ev, std_ev = mlp(x_enc)
            val = (mean_ev - beta * std_ev).item()

        if val < best_val:
            best_val = val
            best_x   = torch.sigmoid(theta).detach().numpy().reshape(1, -1).copy()

    if use_fixed_mask:
        mlp.eval()   # restore eval mode after optimisation

    return np.clip(best_x, 0.0, 1.0)


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
