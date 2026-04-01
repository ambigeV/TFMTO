"""
BO-TFM-Distill: Single-task BO with TabPFN distillation into a dual-head MLP.

Each task is modelled independently (no knowledge transfer), mirroring BO_TFM.

Per BO step for task i
-----------------------
1. Fit TabPFN on task i's observed data  (ONE fit call).
2. Generate distillation dataset: N_distill LHS samples + observed training
   points; query TabPFN ONCE to get (mean, std)  (ONE predict call).
3. Fit a small dual-head MLP (DistillMLP) on those predictions.
4. Minimise LCB = mean(x) - beta * std(x) over [0,1]^d via multi-start
   L-BFGS-B through the differentiable MLP.
5. Evaluate the best candidate on the true objective.

Loss options
------------
mlp_loss='mse'  (default)
    L = MSE(mean_pred, mean_tfpfn) + MSE(log_std_pred, log(std_tfpfn))
    Scale-invariant supervision in log space for std.

mlp_loss='nll'
    L = -log N(mean_tfpfn | mean_pred, std_pred²)
    Probabilistic objective; calibrates std jointly with mean.

GPU support
-----------
When CUDA is available, TabPFN inference, MLP training, and L-BFGS-B
optimisation all run on GPU automatically.
"""

import time
import warnings

import numpy as np
import torch
from scipy.stats import qmc, spearmanr
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict
from ddmtolab.Methods.Algo_Methods.tfm_distill_utils import (
    fit_distill_mlp, lbfgs_optimize_lcb,
)

warnings.filterwarnings("ignore")


class BO_TFM_Distill:
    """
    Bayesian Optimisation with an independent TabPFN-distilled MLP per task.

    Each BO iteration:
      1. Fit TabPFN on the task's observed data.
      2. Generate LHS samples + include training points; query TabPFN for (mean, std).
      3. Fit a dual-head MLP on those predictions (MSE or NLL loss).
      4. Minimise LCB via multi-start L-BFGS-B through the MLP.
      5. Evaluate the argmin on the true objective.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN → DistillMLP (dual-head, independent)',
        'acquisition': 'LCB (L-BFGS-B via MLP gradients)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        beta: float = 1.0,
        n_distill: int = 500,
        n_estimators: int = 8,
        mlp_hidden: int = 64,
        mlp_depth: int = 2,
        mlp_epochs: int = 300,
        mlp_finetune_epochs: int = 50,
        mlp_lr: float = 3e-3,
        mlp_loss: str = 'mse',
        distill_model: str = 'mlp',
        dropout_p: float = 0.1,
        warm_start: bool = False,
        lbfgs_restarts: int = 5,
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
        self.warm_start          = warm_start
        self.mlp_finetune_epochs = mlp_finetune_epochs
        self.lbfgs_restarts      = lbfgs_restarts
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

        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = str(device)

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task  = par_list(self.max_nfes,  nt)

        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        _mlp_cache: dict = {}

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

                # --- distillation: LHS samples + holdout + training points ---
                n_holdout = min(100, self.n_distill // 5)
                X_lhs_all = qmc.LatinHypercube(d=dims[i]).random(
                    n=self.n_distill + n_holdout
                )
                X_lhs     = X_lhs_all[:self.n_distill]
                X_holdout = X_lhs_all[self.n_distill:]

                X_distill = np.vstack([X_lhs, X_train])
                X_query   = np.vstack([X_distill, X_holdout])

                t0 = time.time()
                mean_all, std_all = tabpfn_predict(
                    X_train, y_train, X_query,
                    return_std=True,
                    n_estimators=self.n_estimators,
                    device=device_str,
                )
                t_tabpfn = time.time() - t0

                n_dist = len(X_distill)
                mean_d,  std_d  = mean_all[:n_dist], std_all[:n_dist]
                mean_ho, std_ho = mean_all[n_dist:], std_all[n_dist:]

                # --- fit MLP (cold-start or warm-start from cache) ---
                init_model = _mlp_cache.get(i) if self.warm_start else None
                epochs     = self.mlp_finetune_epochs if init_model is not None \
                             else self.mlp_epochs

                t0 = time.time()
                mlp = fit_distill_mlp(
                    X_distill, mean_d, std_d,
                    hidden=self.mlp_hidden,
                    depth=self.mlp_depth,
                    n_epochs=epochs,
                    lr=self.mlp_lr,
                    loss=self.mlp_loss,
                    model_type=self.distill_model,
                    dropout_p=self.dropout_p,
                    init_model=init_model,
                    device=device,
                )
                t_mlp = time.time() - t0

                if self.warm_start:
                    _mlp_cache[i] = mlp

                # --- fidelity check on holdout ---
                X_ho_t = torch.tensor(X_holdout, dtype=torch.float32, device=device)
                with torch.no_grad():
                    mean_mlp, std_mlp = mlp(X_ho_t)
                mean_mlp = mean_mlp.cpu().numpy()
                std_mlp  = std_mlp.cpu().numpy()

                lcb_ho        = mean_ho  - self.beta * std_ho
                lcb_mlp       = mean_mlp - self.beta * std_mlp
                lcb_rmse      = float(np.sqrt(np.mean((lcb_mlp  - lcb_ho ) ** 2)))
                mean_rmse     = float(np.sqrt(np.mean((mean_mlp - mean_ho) ** 2)))
                std_rmse      = float(np.sqrt(np.mean((std_mlp  - std_ho ) ** 2)))
                lcb_rank_corr = float(spearmanr(lcb_mlp, lcb_ho).statistic)

                # --- select top-k LHS points by LCB as L-BFGS-B starts ---
                lcb_lhs = mean_d[:self.n_distill] - self.beta * std_d[:self.n_distill]
                top_k = min(self.lbfgs_restarts, len(lcb_lhs))
                best_idx = np.argsort(lcb_lhs)[:top_k]
                x0_points = X_lhs[best_idx]

                t0 = time.time()
                candidate_np = lbfgs_optimize_lcb(
                    mlp,
                    opt_dim=dims[i],
                    encode_torch_fn=lambda x: x,
                    beta=self.beta,
                    x0_points=x0_points,
                    device=device,
                )
                t_lbfgs = time.time() - t0

                # --- MLP's LCB at selected candidate ---
                with torch.no_grad():
                    cand_t = torch.tensor(candidate_np, dtype=torch.float32, device=device)
                    m_sel, s_sel = mlp(cand_t)
                    acq_val = float((m_sel - self.beta * s_sel).item())

                # --- evaluate on true objective ---
                obj, _ = evaluation_single(problem, candidate_np, i)
                decs[i], objs[i] = vstack_groups(
                    (decs[i], candidate_np), (objs[i], obj)
                )
                nfes_per_task[i] += 1

                pbar.set_postfix_str(
                    f"task={i} best={objs[i].min():.4f} "
                    f"new={float(obj):.4f} acq={acq_val:.4f} "
                    f"\u03c1={lcb_rank_corr:.3f} "
                    f"tfn={t_tabpfn:.1f}s mlp={t_mlp:.1f}s lb={t_lbfgs:.1f}s"
                )

                # --- W&B logging ---
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        'algo':           self.name,
                        'task':           i,
                        'step':           nfes_per_task[i],
                        'global_step':    sum(nfes_per_task),
                        'best_obj':       float(objs[i].min()),
                        'new_obj':        float(obj),
                        'acq_val':        acq_val,
                        'lcb_rank_corr':  lcb_rank_corr,
                        'lcb_rmse':       lcb_rmse,
                        'mean_rmse':      mean_rmse,
                        'std_rmse':       std_rmse,
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
