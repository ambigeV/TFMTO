"""
BO-TFM-ResGP: Bayesian Optimisation with TabPFN Prior Mean + GP-on-Residuals.

Motivation
----------
TabPFN-as-surrogate fails at high dimensions due to a non-smooth, wiggly LCB
landscape — the ensemble of transformer predictors disagrees in sparse regimes,
producing high-frequency noise that defeats both gradient-based and gradient-free
acquisition optimisers.

This variant exploits TabPFN's genuine strength (accurately predicting function
values from in-context data) as a PRIOR MEAN for a GP, while delegating
uncertainty quantification entirely to the GP:

    μ_hybrid(x) = μ̂_MLP(x)  +  μ_GP_residual(x)
    σ_hybrid(x) = σ_GP_residual(x)          ← smooth, kernel-defined, differentiable

where:
  • μ̂_MLP(x)  — a lightweight MLP that distils the TabPFN prediction map
                 x → μ_TFM(x) so it is cheap and differentiable.
  • GP_residual — an ExactGP with RBF ARD kernel fitted on the residuals
                  r_i = y_i − μ_TFM(x_i) in the original x-space.

The combined LCB is:

    LCB(x) = μ̂_MLP(x)  +  μ_GP(x)  −  β · σ_GP(x)

with gradient ∂LCB/∂x available via autograd through both branches simultaneously,
enabling full multi-start L-BFGS-B acquisition optimisation.

Distillation cache
------------------
Mirrors the rolling cache in BO_TFM_GPEmbed:
  - Each iteration: sample N_candidates LHS points, evaluate TabPFN mean
    predictions μ_TFM(x_cand) conditioned on (X_train, y_train).
  - Store (x_cand, μ_TFM(x_cand)) in a rolling deque (last cache_max_iters iters).
  - Per-sample weight: w_i = exp(−λ · (current_iter − stored_iter)).
  - PredMLP is warm-started from the previous iteration's weights.

Note on dimensionality
----------------------
Unlike BO_TFM_GPEmbed, no PCA step is needed here — the GP operates in the
original d-dimensional x-space, not in a high-dimensional embedding space.
The residuals have reduced variance (TabPFN explains part of the signal), which
improves ARD kernel fitting in sparse regimes compared to fitting a GP on raw y.

Per-iteration steps
-------------------
1.  Fit TabPFN on (X_train, y_train).
2.  Predict μ_TFM(X_train); compute residuals r = y_train − μ_TFM(X_train).
3.  Fit ExactGP with RBF ARD kernel on (X_train, r).
4.  Sample N_candidates LHS points; evaluate μ_TFM(X_cand) from TabPFN.
5.  Add (X_cand, μ_TFM(X_cand)) to rolling prediction cache.
6.  Fit / warm-start PredMLP on weighted cache: x → μ_TFM(x).
7.  Score cached candidates via combined LCB; pick top-k as L-BFGS-B starts.
8.  Minimise LCB via multi-start L-BFGS-B through PredMLP + GP residual.
9.  Evaluate best candidate on the true objective.

Parameters
----------
n_initial         initial LHS samples per task
max_nfes          total function evaluations (including initial samples)
beta              LCB exploration weight (default 3.0)
n_candidates      LHS candidates queried for TabPFN predictions each iter (default 500)
n_estimators      TabPFN ensemble size (default 1)
gp_n_iter         Adam iterations for GP MLL maximisation (default 100)
cache_max_iters   rolling window size in BO iterations (default 10)
cache_lambda      exp-decay rate for cache weights (default 0.5)
mlp_hidden        PredMLP hidden width (default 128)
mlp_depth         PredMLP hidden layers (default 3)
mlp_epochs        cold-start epochs (default 300)
mlp_finetune_epochs warm-start epochs (default 50)
mlp_lr            Adam learning rate for PredMLP (default 3e-3)
warm_start        reuse PredMLP from previous iteration (default True)
lbfgs_restarts    L-BFGS-B restart count (= top-k candidates used as x0)
name              algorithm name key (default 'BO-TFM-ResGP')
"""

import csv
import os
import time
import warnings

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict
from ddmtolab.Methods.Algo_Methods.tfm_res_gp_utils import (
    PredictionCache,
    fit_gp_ard_rbf,
    gp_posterior_torch,
    fit_pred_mlp,
    lbfgs_optimize_res_gp,
)

warnings.filterwarnings('ignore')


def _normalize_y(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class BO_TFM_ResGP:
    """
    Single-task Bayesian Optimisation with a TabPFN prior mean + GP-on-residuals.

    See module docstring for full algorithm description.
    """

    algorithm_information = {
        'n_tasks':      '[1, K]',
        'n_objectives': 1,
        'surrogate':    'TabPFN prior mean (MLP-distilled) + GP on residuals (RBF ARD)',
        'acquisition':  'LCB (L-BFGS-B, gradient through PredMLP + GP)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        beta: float = 3.0,
        n_candidates: int = 1000,
        n_estimators: int = 1,
        gp_n_iter: int = 100,
        cache_max_iters: int = 10,
        cache_lambda: float = 1.0,
        mlp_hidden: int = 128,
        mlp_depth: int = 3,
        mlp_epochs: int = 300,
        mlp_finetune_epochs: int = 50,
        mlp_lr: float = 3e-3,
        warm_start: bool = False,
        lbfgs_restarts: int = 5,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'BO-TFM-ResGP',
        disable_tqdm: bool = True,
    ):
        self.problem              = problem
        self.n_initial            = n_initial if n_initial is not None else 50
        self.max_nfes             = max_nfes  if max_nfes  is not None else 100
        self.beta                 = beta
        self.n_candidates         = n_candidates
        self.n_estimators         = n_estimators
        self.gp_n_iter            = gp_n_iter
        self.cache_max_iters      = cache_max_iters
        self.cache_lambda         = cache_lambda
        self.mlp_hidden           = mlp_hidden
        self.mlp_depth            = mlp_depth
        self.mlp_epochs           = mlp_epochs
        self.mlp_finetune_epochs  = mlp_finetune_epochs
        self.mlp_lr               = mlp_lr
        self.warm_start           = warm_start
        self.lbfgs_restarts       = lbfgs_restarts
        self.save_data            = save_data
        self.save_path            = save_path
        self.name                 = name
        self.disable_tqdm         = disable_tqdm

    # ------------------------------------------------------------------

    def optimize(self):
        start_time = time.time()
        problem    = self.problem
        nt         = problem.n_tasks
        dims       = problem.dims

        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task  = par_list(self.max_nfes,  nt)

        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Distillation accuracy log
        os.makedirs(self.save_path, exist_ok=True)
        log_path = os.path.join(self.save_path, f'{self.name}_distill_log.csv')
        log_file = open(log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            'iteration', 'task_id',
            'mlp_train_mse',   # MSE of PredMLP(X_train) vs μ_TFM(X_train)
            'gp_snr',          # outputscale / (outputscale + noise) on residual GP
        ])

        # Per-task state
        caches:    list = [
            PredictionCache(
                max_iters=self.cache_max_iters,
                lambda_=self.cache_lambda,
            )
            for _ in range(nt)
        ]
        mlp_state: list = [None] * nt
        bo_iter:   list = [0]    * nt

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

            for i in active_tasks:
                bo_iter[i] += 1
                X_train = decs[i]
                y_train = _normalize_y(objs[i].ravel())

                # ----------------------------------------------------------
                # Step 1-3: TabPFN predictions on X_train → residuals → GP
                # ----------------------------------------------------------
                mu_train = tabpfn_predict(
                    X_train, y_train, X_train,
                    return_std=False,
                    n_estimators=self.n_estimators,
                    device=device_str,
                )                                       # (n_train,)

                residuals = y_train - mu_train          # (n_train,)

                gp_params = fit_gp_ard_rbf(
                    X_train, residuals, device, n_iter=self.gp_n_iter
                )

                # ----------------------------------------------------------
                # Step 4-5: TabPFN predictions on candidates → cache
                # ----------------------------------------------------------
                X_cand = LatinHypercube(d=dims[i]).random(n=self.n_candidates)

                mu_cand = tabpfn_predict(
                    X_train, y_train, X_cand,
                    return_std=False,
                    n_estimators=self.n_estimators,
                    device=device_str,
                )                                       # (n_candidates,)

                caches[i].add(X_cand, mu_cand[:, None], iteration=bo_iter[i])

                # ----------------------------------------------------------
                # Step 6: fit / warm-start PredMLP on weighted cache
                # ----------------------------------------------------------
                X_cache, mu_cache_2d, weights = caches[i].retrieve(bo_iter[i])
                mu_cache = mu_cache_2d.ravel()          # (N_cache,)

                mlp = fit_pred_mlp(
                    X_cache, mu_cache, weights,
                    input_dim=dims[i],
                    hidden=self.mlp_hidden,
                    depth=self.mlp_depth,
                    n_epochs=self.mlp_epochs,
                    finetune_epochs=self.mlp_finetune_epochs,
                    lr=self.mlp_lr,
                    init_model=mlp_state[i] if self.warm_start else None,
                    device=device,
                )
                if self.warm_start:
                    mlp_state[i] = mlp

                # ----------------------------------------------------------
                # Distillation accuracy log
                # ----------------------------------------------------------
                with torch.no_grad():
                    X_train_t = torch.tensor(
                        X_train, dtype=torch.float32, device=device
                    )
                    mu_mlp_train = mlp(X_train_t).cpu().numpy()    # (n_train,)
                mlp_train_mse = float(np.mean((mu_mlp_train - mu_train) ** 2))
                gp_snr = float(
                    gp_params['outputscale'] /
                    (gp_params['outputscale'] + gp_params['noise'] + 1e-12)
                )
                log_writer.writerow([
                    bo_iter[i], i,
                    f'{mlp_train_mse:.6f}',
                    f'{gp_snr:.4f}',
                ])
                log_file.flush()

                # ----------------------------------------------------------
                # Step 7: score current candidates to pick L-BFGS-B starts
                # ----------------------------------------------------------
                n_cur     = min(self.n_candidates, len(X_cache))
                X_cur     = X_cache[-n_cur:]
                mu_cur    = mu_cache[-n_cur:]

                X_cur_t   = torch.tensor(X_cur, dtype=torch.float32, device=device)
                mu_cur_t  = torch.tensor(mu_cur, dtype=torch.float32, device=device)

                with torch.no_grad():
                    mu_gp_cur, std_gp_cur = gp_posterior_torch(X_cur_t, gp_params)
                    lcb_cur = (
                        mu_cur_t + mu_gp_cur - self.beta * std_gp_cur
                    ).cpu().numpy()

                top_k     = min(self.lbfgs_restarts, n_cur)
                top_idx   = np.argsort(lcb_cur)[:top_k]
                x0_points = X_cur[top_idx]

                # ----------------------------------------------------------
                # Step 8: L-BFGS-B through PredMLP + GP residual → LCB
                # ----------------------------------------------------------
                candidate_np = lbfgs_optimize_res_gp(
                    mlp, gp_params,
                    opt_dim=dims[i],
                    beta=self.beta,
                    x0_points=x0_points,
                    device=device,
                )

                # ----------------------------------------------------------
                # Step 9: evaluate on true objective
                # ----------------------------------------------------------
                obj, _ = evaluation_single(problem, candidate_np, i)
                decs[i], objs[i] = vstack_groups(
                    (decs[i], candidate_np), (objs[i], obj)
                )
                nfes_per_task[i] += 1

                pbar.set_postfix_str(
                    f'task={i} best={objs[i].min():.4f} new={float(obj):.4f} '
                    f'cache={len(caches[i])}'
                )
                pbar.update(1)

        pbar.close()
        log_file.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data,
        )
        return results
