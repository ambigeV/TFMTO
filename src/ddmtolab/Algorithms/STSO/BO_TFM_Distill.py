"""
BO-TFM-Distill: Single-task BO with TabPFN distillation into a dual-head MLP.

Each task is modelled independently (no knowledge transfer), mirroring BO_TFM.

Per BO step for task i
-----------------------
1. Fit TabPFN on task i's observed data  (ONE fit call).
2. Sample N_distill random points from [0,1]^d; query TabPFN ONCE
   to get (mean, std)  (ONE predict call).
3. Fit a small dual-head MLP (DistillMLP) on those predictions.
4. Minimise LCB = mean(x) - beta * std(x) over [0,1]^d via multi-start
   L-BFGS through the differentiable MLP.
5. Evaluate the best candidate on the true objective.

Loss options
------------
mlp_loss='mse'  (default)
    L = MSE(mean_pred, mean_tfpfn) + MSE(std_pred, std_tfpfn)
    Independent supervision for mean and std; fast and stable.

mlp_loss='nll'
    L = -log N(mean_tfpfn | mean_pred, std_pred²)
      = 0.5 * (log(std²) + (mean_tfpfn - mean_pred)² / std²)
    Probabilistic objective: std is pushed to be calibrated rather than
    just close in L2; typically improves LCB quality when the TabPFN
    uncertainty landscape is heteroscedastic.
"""

import time
import warnings

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict
from ddmtolab.Methods.Algo_Methods.tfm_distill_utils import (
    fit_distill_mlp, lbfgs_optimize_lcb, MCDropoutDistillMLP,
)

warnings.filterwarnings("ignore")


class BO_TFM_Distill:
    """
    Bayesian Optimisation with an independent TabPFN-distilled MLP per task.

    Each BO iteration:
      1. Fit TabPFN on the task's observed data.
      2. Query TabPFN on N_distill random points → (mean, std).
      3. Fit a dual-head MLP on those predictions (MSE or NLL loss).
      4. Minimise LCB via multi-start L-BFGS through the MLP.
      5. Evaluate the argmin on the true objective.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN → DistillMLP (dual-head, independent)',
        'acquisition': 'LCB (L-BFGS via MLP gradients)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        beta: float = 1.0,
        n_distill: int = 1000,
        n_estimators: int = 8,
        mlp_hidden: int = 64,
        mlp_depth: int = 2,
        mlp_epochs: int = 300,
        mlp_finetune_epochs: int = 50,
        mlp_lr: float = 3e-3,
        mlp_loss: str = 'mse',
        distill_model: str = 'mlp',
        dropout_p: float = 0.1,
        mc_samples: int = 20,
        mc_lbfgs: bool = False,
        warm_start: bool = False,
        lbfgs_restarts: int = 5,
        lbfgs_maxiter: int = 100,
        save_data: bool = True,
        save_path: str = './Data',
        name: str = 'BO-TFM-Distill',
        disable_tqdm: bool = True,
        wandb_run=None,
    ):
        self.problem        = problem
        self.n_initial      = n_initial if n_initial is not None else 50
        self.max_nfes       = max_nfes  if max_nfes  is not None else 100
        self.beta           = beta
        self.n_distill      = n_distill
        self.n_estimators   = n_estimators
        self.mlp_hidden     = mlp_hidden
        self.mlp_depth      = mlp_depth
        self.mlp_epochs     = mlp_epochs
        self.mlp_lr         = mlp_lr
        self.mlp_loss            = mlp_loss
        self.distill_model       = distill_model
        self.dropout_p           = dropout_p
        self.mc_samples          = mc_samples
        self.mc_lbfgs            = mc_lbfgs
        self.warm_start          = warm_start
        self.mlp_finetune_epochs = mlp_finetune_epochs
        self.lbfgs_restarts      = lbfgs_restarts
        self.lbfgs_maxiter  = lbfgs_maxiter
        self.save_data      = save_data
        self.save_path      = save_path
        self.name           = name
        self.disable_tqdm   = disable_tqdm
        self.wandb_run      = wandb_run

    def optimize(self):
        start_time = time.time()
        problem    = self.problem
        nt         = problem.n_tasks
        dims       = problem.dims

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task  = par_list(self.max_nfes,  nt)

        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        _mlp_cache: dict = {}   # task_id → previous fitted MLP for warm-starting

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
                X_train = decs[i]           # (n, dims[i])
                y_train = objs[i].ravel()   # (n,)

                # --- distillation: sample train + holdout in a single TabPFN call ---
                n_holdout = min(200, self.n_distill // 5)
                X_all     = np.random.rand(self.n_distill + n_holdout, dims[i])

                t0 = time.time()
                mean_all, std_all = tabpfn_predict(
                    X_train, y_train, X_all,
                    return_std=True,
                    n_estimators=self.n_estimators,
                )
                t_tabpfn = time.time() - t0

                X_dist, X_ho       = X_all[:self.n_distill],    X_all[self.n_distill:]
                mean_d, std_d      = mean_all[:self.n_distill],  std_all[:self.n_distill]
                mean_ho, std_ho    = mean_all[self.n_distill:],  std_all[self.n_distill:]

                # --- fit MLP (cold-start or warm-start from cache) ---
                init_model = _mlp_cache.get(i) if self.warm_start else None
                epochs     = self.mlp_finetune_epochs if init_model is not None \
                             else self.mlp_epochs

                t0 = time.time()
                mlp = fit_distill_mlp(
                    X_dist, mean_d, std_d,
                    hidden=self.mlp_hidden,
                    depth=self.mlp_depth,
                    n_epochs=epochs,
                    lr=self.mlp_lr,
                    loss=self.mlp_loss,
                    model_type=self.distill_model,
                    dropout_p=self.dropout_p,
                    init_model=init_model,
                )
                t_mlp = time.time() - t0

                if self.warm_start:
                    _mlp_cache[i] = mlp

                # --- fidelity check ---
                X_ho_t = torch.tensor(X_ho, dtype=torch.float32)
                if isinstance(mlp, MCDropoutDistillMLP):
                    mean_mlp, std_mlp = mlp.mc_predict(X_ho_t, n_samples=self.mc_samples)
                else:
                    with torch.no_grad():
                        mean_mlp, std_mlp = mlp(X_ho_t)
                mean_mlp = mean_mlp.detach().numpy()
                std_mlp  = std_mlp.detach().numpy()

                lcb_ho        = mean_ho  - self.beta * std_ho
                lcb_mlp       = mean_mlp - self.beta * std_mlp
                lcb_rmse      = float(np.sqrt(np.mean((lcb_mlp  - lcb_ho ) ** 2)))
                mean_rmse     = float(np.sqrt(np.mean((mean_mlp  - mean_ho) ** 2)))
                std_rmse      = float(np.sqrt(np.mean((std_mlp   - std_ho ) ** 2)))
                lcb_rank_corr = float(spearmanr(lcb_mlp, lcb_ho).statistic)

                # --- L-BFGS ---
                t0 = time.time()
                candidate_np = lbfgs_optimize_lcb(
                    mlp,
                    opt_dim=dims[i],
                    encode_torch_fn=lambda x: x,
                    beta=self.beta,
                    n_restarts=self.lbfgs_restarts,
                    max_iter=self.lbfgs_maxiter,
                    mc_lbfgs=self.mc_lbfgs,
                    mc_samples_eval=self.mc_samples,
                )
                t_lbfgs = time.time() - t0

                # --- MLP's LCB at selected candidate ---
                with torch.no_grad():
                    cand_t = torch.tensor(candidate_np, dtype=torch.float32)
                    m_sel, s_sel = mlp(cand_t)
                    acq_val = float((m_sel - self.beta * s_sel).item())

                # --- evaluate on true objective ---
                obj, _ = evaluation_single(problem, candidate_np, i)
                decs[i], objs[i] = vstack_groups(
                    (decs[i], candidate_np), (objs[i], obj)
                )
                nfes_per_task[i] += 1

                # --- W&B logging ---
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        'algo':           self.name,
                        'task':           i,
                        'step':           nfes_per_task[i],
                        'global_step':    sum(nfes_per_task),
                        # convergence
                        'best_obj':       float(objs[i].min()),
                        'new_obj':        float(obj),
                        # acquisition quality
                        'acq_val':        acq_val,
                        # distillation fidelity
                        'lcb_rank_corr':  lcb_rank_corr,
                        'lcb_rmse':       lcb_rmse,
                        'mean_rmse':      mean_rmse,
                        'std_rmse':       std_rmse,
                        # timing
                        't_tabpfn':       t_tabpfn,
                        't_mlp':          t_mlp,
                        't_lbfgs':        t_lbfgs,
                        't_step':         t_tabpfn + t_mlp + t_lbfgs,
                    })

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
