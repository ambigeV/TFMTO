"""
MTBO-TFM-MAP-Asym: MAP-regularised MTBO with per-target asymmetric Cls-derived B prior.

Algorithm per BO step
---------------------
1.  Compute directed similarity S via TabPFN binary classification (Cls):
        delta      = clip(log(K) - CE(t1 -> t2), 0, log(K))
        S[t1, t2]  = (delta / log(K)) ** (1/tau)    ∈ [0, 1]
    S = 0 when CE = log(K) (random classifier = independent tasks, ρ = 0).
    S = 1 when CE = 0 (perfect classification = full transfer, ρ = 1).

2.  For each TARGET task i, build a per-target B prior:
        B_prior_i  s.t.  off-diagonal (j, i) = S[j -> i]
    (same construction as MTBO-TFM-Covar-Cls, see _build_directed_corr_for_target)

3.  For each active task i:
      a. Build a fresh MultiTaskGP (same training data for all).
      b. Warm-start its IndexKernel from B_prior_i.
      c. MAP fitting toward B_prior_i with decaying λ:
             L_i(B) = -MLL_i(B)  +  λ(step) · ||B - B_prior_i||²_F
      d. Acquire next point for task i via LogEI Adam.

This is the asymmetric counterpart of MTBO-TFM-MAP-Sym:
  - Each task's GP is regularised toward a *different* B_prior that captures
    the directional transfer utility into that specific target.
  - At small n: the prior dominates → behaviour similar to MTBO-TFM-Covar-Cls.
  - At large n: λ → 0 → each per-task GP converges to its own MLL solution,
    which is standard MTBO applied independently per task (graceful fallback).

Cost: T MLL fits per BO step (same as MTBO-TFM-Covar-Cls) + TabPFN overhead.

Parameters
----------
lambda_0       initial regularisation weight  (default 5.0)
lambda_decay   exponential decay rate per BO step  (default 0.05)
               at step k:  λ = lambda_0 * exp(-k * lambda_decay)
tau            sharpness of CE→S mapping  (default 1.0)
               τ=1 gives a linear map; τ<1 pushes scores toward 1; τ>1 toward 0
n_classes      number of quantile bins for classification  (default 2)
n_estimators   TabPFN ensemble size  (default 1)
lbfgs_iter     L-BFGS iterations per MAP fit  (default 100)
"""

import time
import warnings

import numpy as np
import torch
import gpytorch
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results, normalize,
)
from ddmtolab.Methods.Algo_Methods.bo_utils import mtbo_next_point
from ddmtolab.Methods.Algo_Methods.tfm_task_covar_utils import (
    compute_task_similarity_matrix_directed_classification,
    compute_task_similarity_matrix_directed_classification_ranked,
    make_psd,
    RhoRecorder,
)

warnings.filterwarnings('ignore')


# =============================================================================
# Helpers
# =============================================================================

def _build_mtgp_data(decs, objs_neg_norm, dims, data_type=torch.double):
    nt = len(decs)
    max_dim = max(dims)
    task_range = torch.linspace(0, 1, nt)
    X_parts, Y_parts = [], []
    for i in range(nt):
        x_i = torch.tensor(decs[i], dtype=data_type)
        y_i = torch.tensor(objs_neg_norm[i], dtype=data_type)
        if y_i.dim() == 1:
            y_i = y_i.unsqueeze(-1)
        if dims[i] < max_dim:
            pad = torch.zeros(x_i.shape[0], max_dim - dims[i], dtype=data_type)
            x_i = torch.cat([x_i, pad], dim=1)
        task_col = torch.full((x_i.shape[0], 1), task_range[i].item(), dtype=data_type)
        X_parts.append(torch.cat([x_i, task_col], dim=1))
        Y_parts.append(y_i)
    return torch.cat(X_parts, dim=0), torch.cat(Y_parts, dim=0)


def _build_directed_corr_for_target(S_np: np.ndarray, target: int) -> np.ndarray:
    """
    Build a symmetric PSD correlation matrix for TARGET task using directed
    scores into that target: R[j, target] = R[target, j] = S[j -> target].
    Source-source pairs use geometric mean. Final PSD projection applied.
    """
    T = S_np.shape[0]
    R = np.eye(T, dtype=np.float64)
    for j in range(T):
        if j == target:
            continue
        R[j, target] = S_np[j, target]
        R[target, j] = S_np[j, target]
    for j in range(T):
        for k in range(j + 1, T):
            if j == target or k == target:
                continue
            R[j, k] = np.sqrt(S_np[j, k] * S_np[k, j])
            R[k, j] = R[j, k]
    return make_psd(R)


def _find_index_kernel(model: MultiTaskGP) -> gpytorch.kernels.IndexKernel:
    kc = model.covar_module
    if not hasattr(kc, 'kernels') and hasattr(kc, 'base_kernel'):
        kc = kc.base_kernel
    for k in kc.kernels:
        if isinstance(k, gpytorch.kernels.IndexKernel):
            return k
    raise RuntimeError('IndexKernel not found in MultiTaskGP.')


def _get_b_matrix(model: MultiTaskGP) -> torch.Tensor:
    """B = W W^T + diag(var)  from the IndexKernel."""
    ik = _find_index_kernel(model)
    return ik.covar_factor @ ik.covar_factor.T + torch.diag(ik.var)


def _warm_start_from_R(model: MultiTaskGP, R_np: np.ndarray) -> None:
    """
    Initialise IndexKernel so that W W^T + diag(v) ≈ R via rank-r eigendecomposition.
    Residual diagonal assigned to var.
    """
    ik = _find_index_kernel(model)
    rank = ik.covar_factor.shape[1]
    dtype = ik.covar_factor.dtype
    device = ik.covar_factor.device

    R_t = torch.tensor(R_np, dtype=torch.float64)
    eigvals, V = torch.linalg.eigh(R_t)
    eigvals = eigvals.flip(0).clamp(min=1e-6)
    V = V.flip(1)

    W_init = (V[:, :rank] * eigvals[:rank].sqrt()).to(dtype=dtype, device=device)
    diag_R = torch.tensor(np.diag(R_np), dtype=dtype, device=device)
    var_init = (diag_R - (W_init @ W_init.T).diag()).clamp(min=1e-6)
    raw_var_init = torch.log(torch.exp(var_init.double()) - 1.0 + 1e-9).to(dtype=dtype)

    with torch.no_grad():
        ik.covar_factor.copy_(W_init)
        ik.raw_var.copy_(raw_var_init)


def _map_fit(
    mll: ExactMarginalLogLikelihood,
    B_prior: torch.Tensor,
    lambda_reg: float,
    lbfgs_iter: int = 100,
) -> None:
    """MAP fitting: minimise  -MLL(B) + λ · ||B − B_prior||²_F."""
    model = mll.model
    model.train()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
    )

    def closure():
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        try:
            loss = -mll(output, model.train_targets, *model.train_inputs)
        except ValueError:
            # HadamardGaussianLikelihood noise is not softplus-reparameterised, so
            # LBFGS can drive raw_noise negative during line search, tripping the
            # LogNormalPrior support check.  Return a barrier penalty that pushes
            # noise back into the valid region (>1e-4) with a meaningful gradient.
            noise_val = model.likelihood.noise_covar.raw_noise
            penalty = (torch.relu(1e-4 - noise_val) * 1e6).sum()
            penalty.backward()
            return penalty
        except RuntimeError:
            # Cholesky failure (NotPSDError): covariance is ill-conditioned at
            # this search point.  Return a large constant so LBFGS backtracks.
            return torch.tensor(1e10, dtype=B_prior.dtype)
        if lambda_reg > 1e-10:
            B_curr = _get_b_matrix(model)
            loss = loss + lambda_reg * ((B_curr - B_prior) ** 2).sum()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        # LBFGS could not converge (all trial points numerically invalid).
        # Fall back to standard MTBO MLL fitting without MAP regularisation.
        model.train()
        fit_gpytorch_mll(mll)
    model.eval()


# =============================================================================
# Algorithm
# =============================================================================

class MTBO_TFM_MAP_Asym:
    """
    MAP-regularised MTBO with per-target asymmetric TabPFN Cls prior.

    Each task gets its own GP regularised toward a directed B_prior that captures
    S[j -> i] only — the transfer utility INTO that task from each source.
    See module docstring for full description.
    """

    algorithm_information = {
        'n_tasks':      '[2, K]',
        'n_objectives': 1,
        'surrogate':    'Per-target MultiTaskGP — ARD Matern-5/2 × IndexKernel (MAP)',
        'task_prior':   'Directed Cls CE → S=(clip(log K-CE,0,logK)/logK)^(1/τ) → B_prior_i per target i',
        'acquisition':  'LogEI (Adam, same as MTBO)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        lambda_0: float = 5.0,
        lambda_decay: float = 0.05,
        tau: float = 1.0,
        n_classes: int = 2,
        n_estimators: int = 1,
        lbfgs_iter: int = 100,
        adam_restarts: int = 5,
        adam_steps: int = 200,
        adam_lr: float = 1e-2,
        use_ranked: bool = False,
        rank_alpha: float = 5.0,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-MAP-Asym',
        disable_tqdm: bool = True,
    ):
        self.problem       = problem
        self.n_initial     = n_initial if n_initial is not None else 50
        self.max_nfes      = max_nfes  if max_nfes  is not None else 100
        self.lambda_0      = lambda_0
        self.lambda_decay  = lambda_decay
        self.tau           = tau
        self.n_classes     = n_classes
        self.n_estimators  = n_estimators
        self.lbfgs_iter    = lbfgs_iter
        self.adam_restarts = adam_restarts
        self.adam_steps    = adam_steps
        self.adam_lr       = adam_lr
        self.use_ranked    = use_ranked
        self.rank_alpha    = rank_alpha
        self.save_data     = save_data
        self.save_path     = save_path
        self.name          = name
        self.disable_tqdm  = disable_tqdm

        self.rho_recorder    = RhoRecorder(asymmetric=True)
        self.lambda_history: list = []   # λ per step

    # ------------------------------------------------------------------

    def optimize(self):
        start_time = time.time()
        problem    = self.problem
        nt         = problem.n_tasks
        dims       = problem.dims
        data_type  = torch.double
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task  = par_list(self.max_nfes,  nt)

        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(
            total=sum(max_nfes_per_task),
            initial=sum(n_initial_per_task),
            desc=self.name,
            disable=self.disable_tqdm,
        )

        bo_step = 0

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # ----------------------------------------------------------
            # λ schedule
            # ----------------------------------------------------------
            lambda_reg = self.lambda_0 * np.exp(-bo_step * self.lambda_decay)
            self.lambda_history.append(lambda_reg)

            # ----------------------------------------------------------
            # Step 1: normalise objectives
            # ----------------------------------------------------------
            objs_norm, _, _ = normalize(objs, axis=0, method='minmax')
            objs_neg_norm   = [-o for o in objs_norm]

            # ----------------------------------------------------------
            # Step 2: compute directed S from Cls
            # ----------------------------------------------------------
            if self.use_ranked:
                S_np = compute_task_similarity_matrix_directed_classification_ranked(
                    decs, objs_norm,
                    n_classes=self.n_classes,
                    n_estimators=self.n_estimators,
                    device=device_str,
                    tau=self.tau,
                    alpha=self.rank_alpha,
                )
            else:
                S_np = compute_task_similarity_matrix_directed_classification(
                    decs, objs_norm,
                    n_classes=self.n_classes,
                    n_estimators=self.n_estimators,
                    device=device_str,
                    tau=self.tau,
                )

            # Shared training tensors (same data for all per-target GPs)
            train_X, train_Y = _build_mtgp_data(decs, objs_neg_norm, dims, data_type)

            step_rho = {}

            # ----------------------------------------------------------
            # Steps 3–6: per target task — separate GP, MAP, acquire
            # ----------------------------------------------------------
            for i in active_tasks:
                # Per-target B prior: off-diagonal = S[j -> i]
                R_i = _build_directed_corr_for_target(S_np, target=i)
                step_rho[i] = R_i

                # Fresh unfitted GP (same data, different B prior)
                mtgp = MultiTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    task_feature=-1,
                ).to(data_type)

                # Warm-start from R_i
                _warm_start_from_R(mtgp, R_i)

                # MAP fit toward B_prior_i
                mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
                B_prior_i = _get_b_matrix(mtgp).detach().clone()
                _map_fit(mll, B_prior_i, lambda_reg, self.lbfgs_iter)

                # Acquire for task i
                candidate_np = mtbo_next_point(
                    mtgp=mtgp,
                    task_id=i,
                    objs=objs_norm,
                    dims=dims,
                    nt=nt,
                    data_type=data_type,
                    adam_restarts=self.adam_restarts,
                    adam_steps=self.adam_steps,
                    adam_lr=self.adam_lr,
                )

                obj, _ = evaluation_single(problem, candidate_np, i)
                decs[i], objs[i] = vstack_groups(
                    (decs[i], candidate_np), (objs[i], obj)
                )
                nfes_per_task[i] += 1

                if nt >= 2:
                    pbar.set_postfix_str(
                        f'task={i} best={objs[i].min():.4f} '
                        f'λ={lambda_reg:.3f} '
                        f'S(0->1)={S_np[0,1]:.3f} S(1->0)={S_np[1,0]:.3f} '
                        f'R_i({i})={R_i[0,1]:.3f}'
                    )
                else:
                    pbar.set_postfix_str(
                        f'task={i} best={objs[i].min():.4f} λ={lambda_reg:.3f}'
                    )
                pbar.update(1)

            self.rho_recorder.record(S_np, step_rho)
            bo_step += 1

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data,
        )
        if self.save_data:
            self.rho_recorder.save(self.save_path, self.name)
        return results
