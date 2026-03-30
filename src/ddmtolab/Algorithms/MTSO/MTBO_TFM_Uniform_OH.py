"""
MTBO-TFM-Uni-OH: Multi-Task BO with shared TabPFN and one-hot task encoding.

Identical to MTBO_TFM_Uniform except the task identifier is encoded as a
one-hot binary vector (n_tasks columns) instead of a scalar integer.
This removes the false ordinal relationship between task IDs that the scalar
encoding implies (e.g. task 0 and task 1 are NOT inherently "closer" than
task 0 and task 3).

Feature layout:  [x_0, ..., x_{max_dim-1}, oh_0, ..., oh_{n_tasks-1}]
"""
import time
import warnings

import numpy as np
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import (
    tabpfn_predict, lcb, append_task_id_onehot, pad_to_dim, optimize_acq_cmaes,
)

warnings.filterwarnings("ignore")


def _normalize_y(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class MTBO_TFM_Uniform_OH:
    """
    Multi-Task BO with a uniform shared TabPFN surrogate and one-hot task encoding.

    All tasks' data are pooled into a single training set. Each task is
    identified by a one-hot binary vector appended to the decision variables,
    avoiding the misleading ordinal relationship of a scalar task ID.

    Each BO iteration:
      1. Normalise each task's objectives independently to [0, 1].
      2. Pool all tasks into (X_all, y_all) with one-hot task columns appended.
      3. Fit a single TabPFN v2.5 on the pooled data.
      4. For each active task i, draw n_candidates random points, append
         one-hot for task i, score with LCB, evaluate argmin on true objective.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN v2.5 (uniform multi-task, one-hot encoding)',
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
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-Uni-OH',
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
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def _build_pooled_dataset(self, decs, objs, dims, max_dim, nt):
        X_parts, y_parts = [], []
        for j, (X_j, y_j) in enumerate(zip(decs, objs)):
            X_padded = pad_to_dim(X_j, max_dim)
            X_with_oh = append_task_id_onehot(X_padded, j, nt)
            X_parts.append(X_with_oh)
            y_parts.append(_normalize_y(y_j.ravel()))
        return np.vstack(X_parts), np.concatenate(y_parts)

    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        max_dim = max(dims)
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

            X_all, y_all = self._build_pooled_dataset(decs, objs, dims, max_dim, nt)

            for i in active_tasks:
                if self.acq_optimizer == 'cmaes':
                    def _score(cands, _Xa=X_all, _ya=y_all, _i=i):
                        cands_padded = pad_to_dim(cands, max_dim)
                        X_test = append_task_id_onehot(cands_padded, _i, nt)
                        m, s = tabpfn_predict(_Xa, _ya, X_test, return_std=True,
                                              n_estimators=self.n_estimators)
                        return lcb(m, s, self.beta)
                    candidate_np = optimize_acq_cmaes(
                        _score, dims[i], self.cmaes_popsize, self.cmaes_maxiter
                    )
                else:
                    candidates = np.random.rand(self.n_candidates, dims[i])
                    candidates_padded = pad_to_dim(candidates, max_dim)
                    X_test = append_task_id_onehot(candidates_padded, i, nt)

                    mean, std = tabpfn_predict(
                        X_all, y_all, X_test,
                        return_std=True,
                        n_estimators=self.n_estimators,
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
