"""
MTBO-TFM-MAP-Sym: MAP-regularised MTBO with symmetric Cls-derived B prior.

Algorithm per BO step
---------------------
1.  Compute directed similarity S via TabPFN binary classification (Cls):
        delta      = clip(log(K) - CE(t1 -> t2), 0, log(K))
        S[t1, t2]  = (delta / log(K)) ** (1/tau)    ∈ [0, 1]
    S = 0 when CE = log(K) (random classifier = independent tasks, ρ = 0).
    S = 1 when CE = 0 (perfect classification = full transfer, ρ = 1).
    Symmetrise:  R = make_psd( (S + S^T) / 2 )

2.  Build standard MultiTaskGP (unfitted).

3.  Warm-start the IndexKernel from R:
        W_init, v_init  s.t.  W W^T + diag(v)  ≈  R

4.  MAP fitting — replace fit_gpytorch_mll with a regularised L-BFGS loop:
        L(B) = -MLL(B)  +  λ(step) · ||B - R||²_F
    λ decays exponentially so the prior fades as the data grows:
        λ(step) = lambda_0 · exp(-step · lambda_decay)

5.  Acquire via standard LogEI Adam optimiser (same as MTBO).

One GP per BO step (same cost as standard MTBO + Cls TabPFN overhead).

Motivation
----------
Freezing R (MTBO-TFM-Covar) prevents MLL from correcting a bad prior at large n.
This variant lets B move freely but anchors it toward R at small n — MAP gives
standard MTBO asymptotically (as λ → 0) while exploiting the TabPFN prior early.

Parameters
----------
lambda_0       initial regularisation weight  (default 5.0)
lambda_decay   exponential decay rate per BO step  (default 0.05)
               at step k:  λ = lambda_0 * exp(-k * lambda_decay)
tau            sharpness of CE→S mapping  (default 1.0)
               τ=1 gives a linear map; τ<1 pushes scores toward 1; τ>1 toward 0
n_classes      number of quantile bins for classification  (default 2)
n_estimators   TabPFN ensemble size  (default 1)
lbfgs_iter     L-BFGS iterations per BO step  (default 100)
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
    W = ik.covar_factor          # (T, rank)
    v = ik.var                   # (T,)  — softplus-constrained
    return W @ W.T + torch.diag(v)


def _warm_start_from_R(model: MultiTaskGP, R_np: np.ndarray) -> None:
    """
    Initialise the IndexKernel's W and raw_var so that W W^T + diag(v) ≈ R.

    Uses a rank-r eigendecomposition of R (r = covar_factor rank).
    The residual diagonal is assigned to var.
    """
    ik = _find_index_kernel(model)
    rank = ik.covar_factor.shape[1]
    dtype = ik.covar_factor.dtype
    device = ik.covar_factor.device

    R_t = torch.tensor(R_np, dtype=torch.float64)
    eigvals, V = torch.linalg.eigh(R_t)   # ascending
    eigvals = eigvals.flip(0).clamp(min=1e-6)
    V = V.flip(1)

    W_init = (V[:, :rank] * eigvals[:rank].sqrt()).to(dtype=dtype, device=device)

    # Residual: what the low-rank part doesn't capture
    diag_R = torch.tensor(np.diag(R_np), dtype=dtype, device=device)
    var_init = (diag_R - (W_init @ W_init.T).diag()).clamp(min=1e-6)

    # Inverse softplus for raw_var
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
    """
    MAP fitting:  minimise  -MLL(B)  +  λ · ||B − B_prior||²_F
    Falls back to a single standard MLL step when lambda_reg ≈ 0.
    """
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

class MTBO_TFM_MAP_Sym:
    """
    MAP-regularised MTBO with symmetric TabPFN Cls prior on the task covariance.

    See module docstring for full description.
    """

    algorithm_information = {
        'n_tasks':      '[2, K]',
        'n_objectives': 1,
        'surrogate':    'MultiTaskGP — ARD Matern-5/2 × IndexKernel (MAP)',
        'task_prior':   'Symmetric Cls CE → S=(clip(log K-CE,0,logK)/logK)^(1/τ) → R=(S+Sᵀ)/2 → B_prior',
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
        name: str = 'MTBO-TFM-MAP-Sym',
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

        self.rho_recorder    = RhoRecorder(asymmetric=False)
        self.lambda_history: list = []   # λ per step

    # ------------------------------------------------------------------

    def optimize(self):
        start_time = time.time()
        problem   = self.problem
        nt        = problem.n_tasks
        dims      = problem.dims
        data_type = torch.double
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        bo_step = 0   # counts outer loop iterations for λ schedule

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
            # Step 2: compute symmetric R from Cls
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
            R_np = make_psd((S_np + S_np.T) / 2.0)
            self.rho_recorder.record(S_np, R_np)

            # ----------------------------------------------------------
            # Step 3: build unfitted MultiTaskGP
            # ----------------------------------------------------------
            train_X, train_Y = _build_mtgp_data(decs, objs_neg_norm, dims, data_type)
            mtgp = MultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                task_feature=-1,
            ).to(data_type)

            # ----------------------------------------------------------
            # Step 4: warm-start IndexKernel from R
            # ----------------------------------------------------------
            _warm_start_from_R(mtgp, R_np)

            # ----------------------------------------------------------
            # Step 5: MAP fitting
            # ----------------------------------------------------------
            mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
            B_prior = _get_b_matrix(mtgp).detach().clone()   # R in B-space after warm-start
            _map_fit(mll, B_prior, lambda_reg, self.lbfgs_iter)

            # ----------------------------------------------------------
            # Step 6: acquire per active task
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
                    f'task={i} best={objs[i].min():.4f} '
                    f'λ={lambda_reg:.3f} '
                    + ' '.join(
                        f'R({a},{b})={R_np[a,b]:.3f}'
                        for a in range(nt) for b in range(a + 1, nt)
                    )
                )
                pbar.update(1)

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
