"""
BO-TFM: Bayesian Optimization with TabPFN surrogate (independent per task).

Each task is modelled by its own TabPFN regressor.  The next candidate is
selected by one of three acquisition optimisers:

  'random' (default) — evaluate LCB on n_candidates random points, take argmin.
  'adam'             — distil TabPFN into a small MLP, optimise LCB via Adam.
  'cmaes'            — optimise LCB via CMA-ES (calls TabPFN once per generation).

LCB (minimisation) = mean - beta * std,  default beta = 1.0

GPU support
-----------
When a CUDA device is available, TabPFN inference runs on GPU automatically.
For 'adam', the distillation MLP and Adam optimisation also run on GPU.
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
from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict, lcb, optimize_acq_cmaes
from ddmtolab.Methods.Algo_Methods.tfm_distill_utils import (
    adam_optimize_acq_tabpfn,
)

warnings.filterwarnings("ignore")


class BO_TFM:
    """
    Bayesian Optimisation with an independent TabPFN surrogate per task.

    Each BO iteration:
      1. Fit a fresh TabPFN on the current task's observed data.
      2. Optimise LCB with the selected acq_optimizer.
      3. Evaluate the best candidate on the true objective.
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
        # Adam optimizer params (used when acq_optimizer='adam')
        adam_n_distill: int = 200,
        adam_hidden: int = 32,
        adam_epochs: int = 100,
        adam_restarts: int = 3,
        adam_steps: int = 200,
        adam_lr: float = 1e-2,
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

    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)   # 'cuda' or 'cpu' for TabPFN

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
                X_train = decs[i]                       # (n, d)
                y_train = objs[i].ravel()               # (n,)

                if self.acq_optimizer == 'cmaes':
                    def _score(cands, _Xtr=X_train, _ytr=y_train):
                        m, s = tabpfn_predict(_Xtr, _ytr, cands, return_std=True,
                                              n_estimators=self.n_estimators,
                                              device=device_str)
                        return lcb(m, s, self.beta)
                    candidate_np = optimize_acq_cmaes(
                        _score, dims[i], self.cmaes_popsize, self.cmaes_maxiter
                    )

                elif self.acq_optimizer == 'adam':
                    # Identity encode (no task ID for independent BO)
                    candidate_np = adam_optimize_acq_tabpfn(
                        X_train, y_train,
                        opt_dim=dims[i],
                        encode_np_fn=lambda X: X,
                        encode_torch_fn=lambda x: x,
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
                    mean, std = tabpfn_predict(
                        X_train, y_train, candidates,
                        return_std=True, n_estimators=self.n_estimators,
                        device=device_str,
                    )
                    acq = lcb(mean, std, self.beta)
                    candidate_np = candidates[int(np.argmin(acq)):int(np.argmin(acq)) + 1]

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
