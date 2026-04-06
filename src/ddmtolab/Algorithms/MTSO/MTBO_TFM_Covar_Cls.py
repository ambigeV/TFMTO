"""
MTBO-TFM-Covar-Cls: MTBO with TabPFN binary-classifier asymmetric task covariance.

Workflow per BO iteration:
1) Convert each task objective into binary quality labels via median split:
       label = 1  (top 50% of task's y values — "good region")
       label = 0  (bottom 50%               — "bad region")
   Labels are defined per-task so each task's own notion of "good" is used.

2) Build directed transfer matrix S using TabPFN binary classification:
   For each ordered pair (t1 -> t2):
       - Fit TabPFN binary classifier on (X_t1, binary_label_t1).
       - Query classifier on X_t2 -> P(good | X_t2).
       - Binary CE of t2's actual labels under these probabilities:
             CE(t1->t2) = -mean [ y_t2 * log P(1|x) + (1-y_t2) * log P(0|x) ]
       - S[t1, t2] = exp(-CE(t1->t2) / tau)  in (0, 1]  (CE >= 0 always)

3) For each TARGET task i, build a SEPARATE MultiTaskGP whose task covariance
   uses the DIRECTED score into task i:

       B_i[j, i] = B_i[i, j] = sigma_i * sigma_j * S[j -> i]   for j != i
       B_i[k, k] = sigma_k^2

   This is symmetric (valid GP) but differs across target tasks:
       B_0 uses S[1->0]  (how well does task 1 predict task 0's good regions?)
       B_1 uses S[0->1]  (how well does task 0 predict task 1's good regions?)

   Directionality is preserved — no symmetrisation step needed.
   S[j->i] in (0,1] guarantees B_i is PSD by construction.

4) Fit each MultiTaskGP_i independently and optimise LogEI for task i.
   Cost: T MLL fits per BO step (vs 1 for symmetric variants). For T=2: 2x.

Why asymmetric?
---------------
The symmetric approach (avg/gram) averages S[t1->t2] and S[t2->t1], losing
directional information. For BO on task i, only the direction "j helps i"
(S[j->i]) is relevant — what j tells about i's good regions. The reverse
direction S[i->j] is irrelevant when searching for task i.

Example: task 1 strongly informs task 0 but not vice versa.
  Symmetric R[0,1] = 0.5*(S[0,1]+S[1,0]) = moderate correlation for BOTH tasks.
  Asymmetric: B_0 uses high S[1->0] (correct for task 0 search).
              B_1 uses low  S[0->1] (correct for task 1 search — task 0 unhelpful).
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
    make_psd,
    FixedCorrelationTaskKernel,
)

warnings.filterwarnings('ignore')


def _build_mtgp_data(
    decs: list,
    objs_neg_norm: list,
    dims: list,
    data_type: torch.dtype = torch.double,
):
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

    train_X = torch.cat(X_parts, dim=0)
    train_Y = torch.cat(Y_parts, dim=0)
    return train_X, train_Y


def _build_directed_corr_for_target(
    S_np: np.ndarray,
    target: int,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Build a symmetric PSD correlation matrix for a specific TARGET task,
    using the directed scores into that target.

    For T tasks, R_target[j, target] = R_target[target, j] = S[j -> target]
    for j != target. Diagonal is 1. Source-source pairs (j,k both != target)
    are set to the geometric mean of their mutual directed scores to maintain
    PSD, then projected via make_psd as a safety check.

    Parameters
    ----------
    S_np   : (T, T) directed similarity matrix, S[i,j] = similarity i->j
    target : index of the target task
    eps    : minimum eigenvalue for PSD projection

    Returns
    -------
    R : (T, T) symmetric PSD correlation matrix tuned to target task
    """
    T = S_np.shape[0]
    R = np.eye(T, dtype=np.float64)

    for j in range(T):
        if j == target:
            continue
        # Directed score: how well does source j predict target task
        R[j, target] = S_np[j, target]
        R[target, j] = S_np[j, target]

    # For T>2: source-source pairs not involving target.
    # Use geometric mean of mutual directed scores as a neutral estimate.
    for j in range(T):
        for k in range(j + 1, T):
            if j == target or k == target:
                continue
            R[j, k] = np.sqrt(S_np[j, k] * S_np[k, j])
            R[k, j] = R[j, k]

    return make_psd(R, eps=eps)


def _inject_fixed_corr_kernel(model: MultiTaskGP, R_np: np.ndarray) -> None:
    model_param = next(model.parameters())
    model_dtype = model_param.dtype
    model_device = model_param.device
    R_t = torch.tensor(R_np, dtype=model_dtype, device=model_device)

    replaced = False
    kernel_container = model.covar_module
    if not hasattr(kernel_container, "kernels") and hasattr(kernel_container, "base_kernel"):
        kernel_container = kernel_container.base_kernel

    if hasattr(kernel_container, "kernels"):
        for idx, kernel in enumerate(kernel_container.kernels):
            if isinstance(kernel, gpytorch.kernels.IndexKernel):
                fixed_kernel = FixedCorrelationTaskKernel(
                    R_t,
                    active_dims=kernel.active_dims,
                ).to(device=model_device, dtype=model_dtype)
                kernel_container.kernels[idx] = fixed_kernel
                replaced = True
                break

    if not replaced:
        raise RuntimeError(
            "Could not find an IndexKernel in model.covar_module.kernels. "
            "BoTorch MultiTaskGP internal structure may have changed."
        )


class MTBO_TFM_Covar_Cls:
    """
    MTBO with per-target asymmetric TabPFN binary-classifier task covariance.

    See module docstring for full algorithm description.
    """

    algorithm_information = {
        'n_tasks':      '[2, K]',
        'n_objectives': 1,
        'surrogate':    'Per-target MultiTaskGP — ARD Matern-5/2 × FixedCorrelationTaskKernel',
        'task_covar':   'Directed binary CE S[j->i] injected per target task i',
        'acquisition':  'LogEI (Adam, same as MTBO)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        n_classes: int = 2,
        tau: float = 1.0,
        n_estimators: int = 1,
        adam_restarts: int = 5,
        adam_steps: int = 200,
        adam_lr: float = 1e-2,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-Covar-Cls',
        disable_tqdm: bool = True,
    ):
        self.problem          = problem
        self.n_initial        = n_initial if n_initial is not None else 50
        self.max_nfes         = max_nfes  if max_nfes  is not None else 100
        self.n_classes        = n_classes
        self.tau              = tau
        self.n_estimators     = n_estimators
        self.adam_restarts    = adam_restarts
        self.adam_steps       = adam_steps
        self.adam_lr          = adam_lr
        self.save_data        = save_data
        self.save_path        = save_path
        self.name             = name
        self.disable_tqdm     = disable_tqdm

        # Diagnostics
        self.s_history: list   = []   # directed S per step
        self.rho_history: list = []   # per-target R matrices per step

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
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            objs_norm, _, _ = normalize(objs, axis=0, method='minmax')
            objs_neg_norm   = [-o for o in objs_norm]

            # Binary median split (n_classes=2): directed S[j->i] in (0,1]
            S_np = compute_task_similarity_matrix_directed_classification(
                decs, objs_norm,
                n_classes=self.n_classes,
                n_estimators=self.n_estimators,
                device=device_str,
                tau=self.tau,
            )
            self.s_history.append(S_np.copy())

            step_rho = {}
            train_X, train_Y = _build_mtgp_data(decs, objs_neg_norm, dims, data_type)

            for i in active_tasks:
                # Per-target correlation: off-diagonal = S[j -> i] (directed into i)
                R_i = _build_directed_corr_for_target(S_np, target=i)
                step_rho[i] = R_i

                # Separate GP fitted for this target task
                mtgp = MultiTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    task_feature=-1,
                ).to(data_type)

                _inject_fixed_corr_kernel(mtgp, R_i)

                mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
                fit_gpytorch_mll(mll)

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
                decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))
                nfes_per_task[i] += 1

                if nt >= 2:
                    pbar.set_postfix_str(
                        f'task={i} best={objs[i].min():.4f} new={float(obj):.4f} '
                        f'S(0->1)={S_np[0,1]:.3f} S(1->0)={S_np[1,0]:.3f} '
                        f'R_i(0,1)={R_i[0,1]:.3f}'
                    )
                else:
                    pbar.set_postfix_str(
                        f'task={i} best={objs[i].min():.4f} new={float(obj):.4f}'
                    )
                pbar.update(1)

            self.rho_history.append(step_rho)

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
