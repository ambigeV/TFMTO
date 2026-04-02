"""
MTBO-TFM-Covar: Multi-Task BO with TabPFN-derived Task Covariance.

Motivation (Research Proposal §4)
----------------------------------
BoTorch's standard MTBO fits B = WW^T + diag(v) (IndexKernel) by MLL from ~40
observations per step.  With small data in high-dimensional spaces this estimate is
unreliable — MLL may converge to degenerate solutions (all-correlated or all-independent).

This method replaces the IndexKernel with a data-driven task correlation matrix R
derived from TabPFN's cross-predictive NLL:

    NLL(t1 → t2) ≈ how well task t1's data predicts task t2's values (one TabPFN call)
    ρ_t1t2       = symmetrised exp(-NLL / τ)  ∈ (0, 1]

The MultiTaskGP kernel becomes:
    k_MT((x,t),(x',t')) = k_ARD(x, x') × [diag(σ) · R · diag(σ)]_tt'

where R is FIXED each BO step (computed from TabPFN) and only σ (per-task scale)
is fitted by MLL.  Negative transfer protection is automatic: when tasks are
dissimilar, NLL is high → ρ → 0 → GP degenerates to T independent GPs.

Per-iteration steps
-------------------
1.  Normalise objectives (min-max per task, same as MTBO).
2.  Compute task correlation matrix R:
    - T(T-1) TabPFN forward passes (2 for the common T=2 case)
    - Symmetrised NLL similarity → R
    - PSD projection
3.  Build MultiTaskGP (same data format as standard MTBO).
4.  Replace IndexKernel with FixedCorrelationTaskKernel(R).
5.  Fit MLL (now only fits lengthscales + noise + σ²; R is frozen).
6.  Acquire next point via Adam-based LogEI (identical to MTBO).
7.  Evaluate and update.

Diagnostics
-----------
The ρ trajectory (per-step correlation matrix) is stored in self.rho_history
as a list of (T, T) numpy arrays — one entry per BO step.  This lets you plot
how task correlation evolves over BO iterations, which is a strong diagnostic for
CI (ρ → 1) vs NI (ρ → 0) problem pairs.

Parameters
----------
n_initial        initial LHS samples per task
max_nfes         total function evaluations (including initial samples)
tau              temperature for NLL → similarity conversion  (default 1.0)
                 smaller → sharper discrimination between tasks
n_estimators     TabPFN ensemble size for NLL computation  (default 1)
adam_restarts    Adam restarts for LogEI acquisition  (same as MTBO default: 5)
adam_steps       Adam steps per restart               (same as MTBO default: 200)
adam_lr          Adam learning rate                   (same as MTBO default: 1e-2)
name             algorithm name key  (default 'MTBO-TFM-Covar')
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
    compute_task_similarity_matrix,
    FixedCorrelationTaskKernel,
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Data builder — mirrors mtgp_build but returns unfitted model for injection
# ---------------------------------------------------------------------------

def _build_mtgp_data(
    decs: list,
    objs_neg_norm: list,
    dims: list,
    data_type: torch.dtype = torch.double,
):
    """
    Assemble training tensors in the format expected by BoTorch's MultiTaskGP.

    Uses the same linspace(0, 1, T) task index convention as mtgp_build so
    the acquisition code (mtbo_next_point) stays fully compatible.

    Returns (train_X_with_task, train_Y) as double tensors.
    """
    nt = len(decs)
    max_dim = max(dims)
    task_range = torch.linspace(0, 1, nt)

    X_parts, Y_parts = [], []
    for i in range(nt):
        x_i = torch.tensor(decs[i], dtype=data_type)
        y_i = torch.tensor(objs_neg_norm[i], dtype=data_type)
        if y_i.dim() == 1:
            y_i = y_i.unsqueeze(-1)

        # Zero-pad to max_dim if needed
        if dims[i] < max_dim:
            pad = torch.zeros(x_i.shape[0], max_dim - dims[i], dtype=data_type)
            x_i = torch.cat([x_i, pad], dim=1)

        # Append task index column (linspace convention)
        task_col = torch.full((x_i.shape[0], 1), task_range[i].item(), dtype=data_type)
        X_parts.append(torch.cat([x_i, task_col], dim=1))
        Y_parts.append(y_i)

    train_X = torch.cat(X_parts, dim=0)
    train_Y = torch.cat(Y_parts, dim=0)
    return train_X, train_Y


def _inject_fixed_corr_kernel(model: MultiTaskGP, R_np: np.ndarray) -> None:
    """
    Replace the IndexKernel inside model.covar_module.kernels with a
    FixedCorrelationTaskKernel built from R_np.

    The replacement is done in-place so the model reference stays valid.
    All other sub-kernels (spatial ARD Matern) are left untouched.
    """
    R_t = torch.tensor(R_np, dtype=torch.float32)
    fixed_kernel = FixedCorrelationTaskKernel(R_t)

    replaced = False
    for idx, kernel in enumerate(model.covar_module.kernels):
        if isinstance(kernel, gpytorch.kernels.IndexKernel):
            model.covar_module.kernels[idx] = fixed_kernel
            replaced = True
            break

    if not replaced:
        raise RuntimeError(
            "Could not find an IndexKernel in model.covar_module.kernels. "
            "BoTorch MultiTaskGP internal structure may have changed."
        )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

class MTBO_TFM_Covar:
    """
    Multi-Task BO with TabPFN-derived fixed task correlation matrix.

    See module docstring for full description and motivation.
    """

    algorithm_information = {
        'n_tasks':            '[2, K]',
        'n_objectives':       1,
        'surrogate':          'MultiTaskGP — ARD Matern-5/2 × FixedCorrelationTaskKernel',
        'task_covar':         'TabPFN cross-predictive NLL → symmetrised ρ',
        'acquisition':        'LogEI (Adam, same as MTBO)',
        'negative_transfer':  'automatic (ρ → 0 when tasks dissimilar)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        tau: float = 1.0,
        n_estimators: int = 1,
        adam_restarts: int = 5,
        adam_steps: int = 200,
        adam_lr: float = 1e-2,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-Covar',
        disable_tqdm: bool = True,
    ):
        self.problem       = problem
        self.n_initial     = n_initial if n_initial is not None else 50
        self.max_nfes      = max_nfes  if max_nfes  is not None else 100
        self.tau           = tau
        self.n_estimators  = n_estimators
        self.adam_restarts = adam_restarts
        self.adam_steps    = adam_steps
        self.adam_lr       = adam_lr
        self.save_data     = save_data
        self.save_path     = save_path
        self.name          = name
        self.disable_tqdm  = disable_tqdm

        # Diagnostic: stores (T, T) R matrix for each BO step
        self.rho_history: list = []

    # ------------------------------------------------------------------

    def optimize(self):
        start_time = time.time()
        problem    = self.problem
        nt         = problem.n_tasks
        dims       = problem.dims

        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)
        data_type  = torch.double

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

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [
                i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]
            ]
            if not active_tasks:
                break

            # ----------------------------------------------------------
            # Step 1: normalise objectives (same as MTBO)
            # ----------------------------------------------------------
            objs_norm, _, _ = normalize(objs, axis=0, method='minmax')
            # Negate for BoTorch maximisation convention
            objs_neg_norm = [-o for o in objs_norm]

            # ----------------------------------------------------------
            # Step 2: compute task correlation matrix R via TabPFN NLL
            # T(T-1) TabPFN calls — for T=2: exactly 2 calls
            # ----------------------------------------------------------
            R_np = compute_task_similarity_matrix(
                decs, objs_norm,
                n_estimators=self.n_estimators,
                device=device_str,
                tau=self.tau,
            )
            self.rho_history.append(R_np.copy())

            # ----------------------------------------------------------
            # Step 3: build MultiTaskGP (unfitted)
            # ----------------------------------------------------------
            train_X, train_Y = _build_mtgp_data(decs, objs_neg_norm, dims, data_type)
            mtgp = MultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                task_feature=-1,
            ).to(data_type)

            # ----------------------------------------------------------
            # Step 4: inject FixedCorrelationTaskKernel (replaces IndexKernel)
            # ----------------------------------------------------------
            _inject_fixed_corr_kernel(mtgp, R_np)

            # ----------------------------------------------------------
            # Step 5: fit MLL — only σ² + lengthscales + noise are free;
            #         R is frozen inside FixedCorrelationTaskKernel
            # ----------------------------------------------------------
            mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
            fit_gpytorch_mll(mll)

            # ----------------------------------------------------------
            # Step 6: acquire next point per active task (same as MTBO)
            # ----------------------------------------------------------
            for i in active_tasks:
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

                pbar.set_postfix_str(
                    f'task={i} best={objs[i].min():.4f} new={float(obj):.4f} '
                    + ' '.join(
                        f'ρ({a},{b})={R_np[a,b]:.3f}'
                        for a in range(nt) for b in range(a + 1, nt)
                    )
                )
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data,
        )
        return results
