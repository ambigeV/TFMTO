"""
MTBO-TFM-Elite: Multi-Task BO with selective elite knowledge transfer via TabPFN.

Training set per task i:
  • ALL observed data from task i  (with task_id = i)
  • Top *elite_ratio* (default 10 %) of observed data from every other task j
    (ranked by objective value, lowest = best), with task_id = j

This gives a lower-budget transfer signal than the uniform variant: only the
highest-quality samples from auxiliary tasks are shared, reducing noise from
poorly performing auxiliary regions.

Feature layout:  [x_0, ..., x_{max_dim-1}, task_id]

Each task's objectives are min-max normalised independently before training.

LCB (minimisation) = mean - beta * std,  default beta = 1.0
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
    tabpfn_predict, lcb, append_task_id, pad_to_dim,
)

warnings.filterwarnings("ignore")


def _normalize_y(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class MTBO_TFM_Elite:
    """
    Multi-Task Bayesian Optimisation with elite transfer via a shared TabPFN.

    For each active task i a task-specific training set is assembled:
      - All samples from task i (normalised).
      - The top *elite_ratio* fraction of samples from every other task j,
        selected by lowest (best) normalised objective value.

    A fresh TabPFN is fitted on this per-task dataset each iteration.

    Each BO iteration:
      1. For each active task i, build the elite-transfer training set.
      2. Fit TabPFN on that set.
      3. Draw n_candidates random points, append task_id=i, score with LCB.
      4. Evaluate argmin on the true objective.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN (elite multi-task transfer)',
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
        elite_ratio: float = 0.1,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'MTBO-TFM-Elite',
        disable_tqdm: bool = True,
    ):
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.beta = beta
        self.n_candidates = n_candidates
        self.n_estimators = n_estimators
        self.elite_ratio = elite_ratio
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    # ------------------------------------------------------------------
    def _build_elite_dataset(self, task_i: int, decs, objs, dims, max_dim):
        """
        Build training set for task i:
          - all data from task i
          - top elite_ratio from each other task j
        Returns (X_train, y_train).
        """
        X_parts, y_parts = [], []

        for j, (X_j, y_j) in enumerate(zip(decs, objs)):
            y_norm = _normalize_y(y_j.ravel())
            X_padded = pad_to_dim(X_j, max_dim)

            if j == task_i:
                # include all samples from current task
                X_parts.append(append_task_id(X_padded, j))
                y_parts.append(y_norm)
            else:
                # include only the top elite_ratio (lowest normalised obj)
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
                # ---------- task-specific elite training set ----------
                X_train, y_train = self._build_elite_dataset(
                    i, decs, objs, dims, max_dim
                )

                # ---------- random candidate pool for task i ----------
                candidates = np.random.rand(self.n_candidates, dims[i])
                candidates_padded = pad_to_dim(candidates, max_dim)
                X_test = append_task_id(candidates_padded, i)

                mean, std = tabpfn_predict(
                    X_train, y_train, X_test,
                    return_std=True,
                    n_estimators=self.n_estimators,
                )

                # ---------- LCB selection ----------
                acq = lcb(mean, std, self.beta)
                best_idx = int(np.argmin(acq))
                candidate_np = candidates[best_idx:best_idx + 1]   # (1, d_i)

                # ---------- real evaluation ----------
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
