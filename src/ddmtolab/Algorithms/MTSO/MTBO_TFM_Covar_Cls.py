"""
MTBO-TFM-Covar-Cls: MTBO with TFN classifier-based inter-task covariance.

Workflow per BO iteration:
1) Convert each task objective into ordinal classes via rank-quantile binning.
2) Build directed transfer matrix S using TabPFN classification cross-entropy:
       S[i, j] = exp(-CE(i -> j) / tau)
3) Map S to symmetric PSD correlation R (gram/avg mapping).
4) Inject R into MultiTaskGP task kernel and continue standard MTBO steps.
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
    directed_similarity_to_correlation,
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


def _inject_fixed_corr_kernel(model: MultiTaskGP, R_np: np.ndarray) -> None:
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


class MTBO_TFM_Covar_Cls:
    """
    MTBO variant that derives task transfer from TFN classifier probabilities.
    """

    algorithm_information = {
        'n_tasks':            '[2, K]',
        'n_objectives':       1,
        'surrogate':          'MultiTaskGP — ARD Matern-5/2 × FixedCorrelationTaskKernel',
        'task_covar':         'Directed TFN CE -> S, then S->R via gram/avg',
        'acquisition':        'LogEI (Adam, same as MTBO)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        n_classes: int = 3,
        tau: float = 1.0,
        n_estimators: int = 1,
        directed_to_corr: str = 'gram',
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
        self.max_nfes         = max_nfes if max_nfes is not None else 100
        self.n_classes        = n_classes
        self.tau              = tau
        self.n_estimators     = n_estimators
        self.directed_to_corr = directed_to_corr
        self.adam_restarts    = adam_restarts
        self.adam_steps       = adam_steps
        self.adam_lr          = adam_lr
        self.save_data        = save_data
        self.save_path        = save_path
        self.name             = name
        self.disable_tqdm     = disable_tqdm

        # Diagnostics:
        # - s_history: directed classifier-transfer matrix S
        # - rho_history: mapped GP-valid correlation matrix R
        self.s_history: list = []
        self.rho_history: list = []

    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)
        data_type = torch.double

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

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
            objs_neg_norm = [-o for o in objs_norm]

            S_np = compute_task_similarity_matrix_directed_classification(
                decs, objs_norm,
                n_classes=self.n_classes,
                n_estimators=self.n_estimators,
                device=device_str,
                tau=self.tau,
            )
            R_np = directed_similarity_to_correlation(
                S_np,
                method=self.directed_to_corr,
            )
            self.s_history.append(S_np.copy())
            self.rho_history.append(R_np.copy())

            train_X, train_Y = _build_mtgp_data(decs, objs_neg_norm, dims, data_type)
            mtgp = MultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                task_feature=-1,
            ).to(data_type)

            _inject_fixed_corr_kernel(mtgp, R_np)

            mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
            fit_gpytorch_mll(mll)

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
                decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))
                nfes_per_task[i] += 1

                if nt >= 2:
                    pbar.set_postfix_str(
                        f'task={i} best={objs[i].min():.4f} new={float(obj):.4f} '
                        f'S(0,1)={S_np[0,1]:.3f} S(1,0)={S_np[1,0]:.3f} '
                        f'R(0,1)={R_np[0,1]:.3f}'
                    )
                else:
                    pbar.set_postfix_str(
                        f'task={i} best={objs[i].min():.4f} new={float(obj):.4f}'
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
