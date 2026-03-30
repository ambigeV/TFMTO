"""
BO-TFM: Bayesian Optimization with TabPFN surrogate (independent per task).

Each task is modelled by its own TabPFN regressor.  The next candidate is
selected by evaluating LCB over a random pool of n_candidates points and
picking the argmin — no gradient-based inner optimisation.

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
from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict, lcb, optimize_acq_cmaes

warnings.filterwarnings("ignore")


class BO_TFM:
    """
    Bayesian Optimisation with an independent TabPFN surrogate per task.

    Each BO iteration:
      1. Fit a fresh TabPFN on the current task's observed data.
      2. Draw *n_candidates* random points uniformly from [0,1]^d.
      3. Score them with LCB = mean - beta * std.
      4. Evaluate the argmin on the true objective.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN (independent)',
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
        name: str = 'BO-TFM',
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

    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
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
                # ---------- surrogate ----------
                X_train = decs[i]                       # (n, d)
                y_train = objs[i].ravel()               # (n,)

                # ---------- acquisition optimisation ----------
                if self.acq_optimizer == 'cmaes':
                    def _score(cands, _Xtr=X_train, _ytr=y_train):
                        m, s = tabpfn_predict(_Xtr, _ytr, cands, return_std=True,
                                              n_estimators=self.n_estimators)
                        return lcb(m, s, self.beta)
                    candidate_np = optimize_acq_cmaes(
                        _score, dims[i], self.cmaes_popsize, self.cmaes_maxiter
                    )
                else:
                    candidates = np.random.rand(self.n_candidates, dims[i])
                    mean, std = tabpfn_predict(
                        X_train, y_train, candidates,
                        return_std=True, n_estimators=self.n_estimators,
                    )
                    acq = lcb(mean, std, self.beta)
                    candidate_np = candidates[int(np.argmin(acq)):int(np.argmin(acq)) + 1]

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
