"""
MTBO-TFM-Elite-B: MTBO-TFM-Elite variant that selects candidates by predicted mean only.

Identical to MTBO-TFM-Elite except the acquisition criterion is:
    argmin  μ_TFM(x)     (pure exploitation of the surrogate mean)
instead of:
    argmin  μ_TFM(x) - β·σ_TFM(x)    (LCB)

Motivation: TabPFN's uncertainty estimate is non-smooth and unreliable in sparse
high-dimensional regimes, causing the LCB's std term to introduce noise rather than
useful exploration.  Dropping the std term makes the criterion smoother and less
sensitive to wiggly uncertainty estimates, at the cost of pure exploitation behaviour.

Elite transfer and all other hyperparameters are identical to MTBO-TFM-Elite.
"""
import time
import warnings
import functools

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import (
    tabpfn_predict, append_task_id, pad_to_dim, optimize_acq_cmaes,
)
from ddmtolab.Methods.Algo_Methods.tfm_distill_utils import (
    adam_optimize_acq_tabpfn, encode_torch_scalar,
)

warnings.filterwarnings("ignore")


def _normalize_y(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class MTBO_TFM_Elite_B:
    """
    MTBO-TFM-Elite-B: elite multi-task TabPFN with mean-only acquisition.

    For each active task i the training set contains all of task i's data plus
    the top *elite_ratio* fraction of each other task's data (by objective value).
    At each step the candidate with the lowest predicted mean is selected,
    bypassing the noisy std term entirely.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN (elite multi-task transfer)',
        'acquisition': 'Mean (no uncertainty)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        n_candidates: int = 1000,
        n_estimators: int = 8,
        elite_ratio: float = 0.1,
        acq_optimizer: str = 'random',
        cmaes_popsize: int = 20,
        cmaes_maxiter: int = 50,
        adam_n_distill: int = 200,
        adam_hidden: int = 32,
        adam_epochs: int = 100,
        adam_restarts: int = 3,
        adam_steps: int = 200,
        adam_lr: float = 1e-2,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-Elite-B',
        disable_tqdm: bool = True,
    ):
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.n_candidates = n_candidates
        self.n_estimators = n_estimators
        self.elite_ratio = elite_ratio
        self.acq_optimizer = acq_optimizer
        self.cmaes_popsize = cmaes_popsize
        self.cmaes_maxiter = cmaes_maxiter
        self.adam_n_distill = adam_n_distill
        self.adam_hidden    = adam_hidden
        self.adam_epochs    = adam_epochs
        self.adam_restarts  = adam_restarts
        self.adam_steps     = adam_steps
        self.adam_lr        = adam_lr
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    # ------------------------------------------------------------------
    def _build_elite_dataset(self, task_i: int, decs, objs, dims, max_dim):
        X_parts, y_parts = [], []
        for j, (X_j, y_j) in enumerate(zip(decs, objs)):
            y_norm = _normalize_y(y_j.ravel())
            X_padded = pad_to_dim(X_j, max_dim)
            if j == task_i:
                X_parts.append(append_task_id(X_padded, j))
                y_parts.append(y_norm)
            else:
                n_j = len(y_norm)
                n_elite = max(1, int(np.ceil(self.elite_ratio * n_j)))
                elite_idx = np.argsort(y_norm)[:n_elite]
                X_parts.append(append_task_id(X_padded[elite_idx], j))
                y_parts.append(y_norm[elite_idx])
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

            for i in active_tasks:
                X_train, y_train = self._build_elite_dataset(
                    i, decs, objs, dims, max_dim
                )

                if self.acq_optimizer == 'cmaes':
                    def _score(cands, _Xtr=X_train, _ytr=y_train, _i=i):
                        cands_padded = pad_to_dim(cands, max_dim)
                        X_test = append_task_id(cands_padded, _i)
                        m = tabpfn_predict(_Xtr, _ytr, X_test, return_std=False,
                                           n_estimators=1, device=device_str)
                        return m
                    candidate_np = optimize_acq_cmaes(
                        _score, dims[i], self.cmaes_popsize, self.cmaes_maxiter
                    )

                elif self.acq_optimizer == 'adam':
                    encode_np = lambda X, _i=i: append_task_id(pad_to_dim(X, max_dim), _i)
                    encode_t  = functools.partial(
                        encode_torch_scalar, max_dim=max_dim, task_id=i
                    )
                    # beta=0 reduces adam acquisition to pure mean minimisation
                    candidate_np = adam_optimize_acq_tabpfn(
                        X_train, y_train,
                        opt_dim=dims[i],
                        encode_np_fn=encode_np,
                        encode_torch_fn=encode_t,
                        beta=0.0,
                        n_estimators=self.n_estimators,
                        n_distill=self.adam_n_distill,
                        mlp_hidden=self.adam_hidden,
                        mlp_epochs=self.adam_epochs,
                        adam_restarts=self.adam_restarts,
                        adam_steps=self.adam_steps,
                        adam_lr=self.adam_lr,
                        device=device,
                    )

                else:   # 'random' — select by predicted mean only
                    candidates = LatinHypercube(d=dims[i]).random(n=self.n_candidates)
                    candidates_padded = pad_to_dim(candidates, max_dim)
                    X_test = append_task_id(candidates_padded, i)
                    mean = tabpfn_predict(
                        X_train, y_train, X_test,
                        return_std=False,
                        n_estimators=self.n_estimators,
                        device=device_str,
                    )
                    best_idx = int(np.argmin(mean))
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
