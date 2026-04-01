"""
MTBO-TFM-Uni: Multi-Task BO with a single shared TabPFN surrogate.

All tasks are pooled into one training set.  Each task is distinguished by
appending its integer task ID (0, 1, 2, …) as an extra feature column.
Tasks with fewer dimensions are zero-padded to max_dim before the ID column.

Feature layout:  [x_0, ..., x_{max_dim-1}, task_id]

For tasks with different objectives scales, each task's y is min-max
normalised to [0, 1] before pooling so that the shared surrogate is not
dominated by tasks with larger absolute values.

LCB (minimisation) = mean - beta * std,  default beta = 1.0

GPU support
-----------
TabPFN inference runs on GPU when CUDA is available.  For acq_optimizer='adam',
the distillation MLP and Adam inner loop also run on GPU.
"""
import time
import warnings
import functools

import numpy as np
import torch
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import (
    tabpfn_predict, lcb, append_task_id, pad_to_dim, optimize_acq_cmaes,
)
from ddmtolab.Methods.Algo_Methods.tfm_distill_utils import (
    adam_optimize_acq_tabpfn, encode_torch_scalar,
)

warnings.filterwarnings("ignore")


def _normalize_y(y: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns y unchanged if range is zero."""
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class MTBO_TFM_Uniform:
    """
    Multi-Task Bayesian Optimisation with a uniform shared TabPFN surrogate.

    All tasks' data are combined into a single training set (with task ID as
    an extra feature).  A fresh TabPFN is fitted every iteration using the
    full pooled dataset.

    Each BO iteration:
      1. Normalise each task's objectives independently to [0, 1].
      2. Pool all tasks into one (X_all, y_all) with task-ID feature appended.
      3. Fit a single TabPFN on the pooled data.
      4. For each active task i, optimise LCB with the selected acq_optimizer.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN (uniform multi-task)',
        'acquisition': 'LCB',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        beta: float = 1.0,
        n_candidates: int = 500,
        n_estimators: int = 8,
        acq_optimizer: str = 'random',
        cmaes_popsize: int = 20,
        cmaes_maxiter: int = 50,
        # Adam optimizer params
        adam_n_distill: int = 200,
        adam_hidden: int = 32,
        adam_epochs: int = 100,
        adam_restarts: int = 3,
        adam_steps: int = 200,
        adam_lr: float = 1e-2,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-Uni',
        disable_tqdm: bool = True,
    ):
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.beta = beta
        self.n_candidates = n_candidates
        self.n_estimators = n_estimators
        self.acq_optimizer = acq_optimizer
        self.cmaes_popsize = cmaes_popsize
        self.cmaes_maxiter = cmaes_maxiter
        self.adam_n_distill  = adam_n_distill
        self.adam_hidden     = adam_hidden
        self.adam_epochs     = adam_epochs
        self.adam_restarts   = adam_restarts
        self.adam_steps      = adam_steps
        self.adam_lr         = adam_lr
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    # ------------------------------------------------------------------
    def _build_pooled_dataset(self, decs, objs, dims, max_dim):
        """Combine all tasks into (X_all, y_all) with task-ID feature."""
        X_parts, y_parts = [], []
        for j, (X_j, y_j) in enumerate(zip(decs, objs)):
            X_padded = pad_to_dim(X_j, max_dim)
            X_with_id = append_task_id(X_padded, j)
            X_parts.append(X_with_id)
            y_parts.append(_normalize_y(y_j.ravel()))
        return np.vstack(X_parts), np.concatenate(y_parts)

    # ------------------------------------------------------------------
    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        max_dim = max(dims)
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)

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

            # ---------- build shared surrogate once per round ----------
            X_all, y_all = self._build_pooled_dataset(decs, objs, dims, max_dim)

            for i in active_tasks:
                if self.acq_optimizer == 'cmaes':
                    def _score(cands, _Xa=X_all, _ya=y_all, _i=i):
                        cands_padded = pad_to_dim(cands, max_dim)
                        X_test = append_task_id(cands_padded, _i)
                        m, s = tabpfn_predict(_Xa, _ya, X_test, return_std=True,
                                              n_estimators=self.n_estimators,
                                              device=device_str)
                        return lcb(m, s, self.beta)
                    candidate_np = optimize_acq_cmaes(
                        _score, dims[i], self.cmaes_popsize, self.cmaes_maxiter
                    )

                elif self.acq_optimizer == 'adam':
                    encode_np   = lambda X, _i=i: append_task_id(pad_to_dim(X, max_dim), _i)
                    encode_t    = functools.partial(
                        encode_torch_scalar, max_dim=max_dim, task_id=i
                    )
                    candidate_np = adam_optimize_acq_tabpfn(
                        X_all, y_all,
                        opt_dim=dims[i],
                        encode_np_fn=encode_np,
                        encode_torch_fn=encode_t,
                        beta=self.beta,
                        n_estimators=self.n_estimators,
                        n_distill=self.adam_n_distill,
                        mlp_hidden=self.adam_hidden,
                        mlp_epochs=self.adam_epochs,
                        adam_restarts=self.adam_restarts,
                        adam_steps=self.adam_steps,
                        adam_lr=self.adam_lr,
                        device=device,
                    )

                else:   # 'random'
                    candidates = np.random.rand(self.n_candidates, dims[i])
                    candidates_padded = pad_to_dim(candidates, max_dim)
                    X_test = append_task_id(candidates_padded, i)
                    mean, std = tabpfn_predict(
                        X_all, y_all, X_test,
                        return_std=True,
                        n_estimators=self.n_estimators,
                        device=device_str,
                    )
                    acq = lcb(mean, std, self.beta)
                    best_idx = int(np.argmin(acq))
                    candidate_np = candidates[best_idx:best_idx + 1]

                obj, _ = evaluation_single(problem, candidate_np, i)
                decs[i], objs[i] = vstack_groups(
                    (decs[i], candidate_np), (objs[i], obj)
                )

                nfes_per_task[i] += 1
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
