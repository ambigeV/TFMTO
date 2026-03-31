"""
MTBO-TFM-Distill: Multi-Task BO with TabPFN distillation into a dual-head MLP.

Per BO step for task i
-----------------------
1. Build training set (pooled uniform or elite-transfer, scalar or one-hot
   task encoding) — identical to the corresponding MTBO-TFM-* variants.
2. Fit TabPFN on that training set  (ONE fit call).
3. Sample N_distill random points encoded for task i; query TabPFN ONCE
   to get (mean, std) predictions  (ONE predict call).
4. Fit a small dual-head MLP (DistillMLP) on those (X_enc, mean, std) pairs.
5. Minimise LCB = mean(x) - beta * std(x) over [0,1]^dims[i] via
   multi-start L-BFGS through the differentiable MLP.
6. Evaluate the best candidate on the true objective.

Speed comparison per BO step
-----------------------------
  Random   : 1 TabPFN fit  +  1 predict(N_candidates)
  CMA-ES   : 1 TabPFN fit  +  popsize × maxiter predict calls  (sequential)
  Distill  : 1 TabPFN fit  +  1 predict(N_distill)  +  MLP train  +  L-BFGS

With default settings the distill variant is ~10–30× faster than CMA-ES while
producing candidates comparable to or better than a large random pool.

Parameters
----------
transfer  : 'uniform'  — pool ALL data from every task
            'elite'    — pool ALL from task i + top elite_ratio from others
encoding  : 'scalar'   — append integer task ID
            'onehot'   — append one-hot binary vector (removes false ordinal)

Name convention: MTBO-TFM-{Uni|Elite}-{OH-}Distill
  e.g. transfer='uniform', encoding='scalar'  →  'MTBO-TFM-Uni-Distill'
       transfer='elite',   encoding='onehot'  →  'MTBO-TFM-Elite-OH-Distill'
"""

import time
import warnings
import functools

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    par_list, initialization, evaluation, evaluation_single,
    vstack_groups, build_staircase_history, build_save_results,
)
from ddmtolab.Methods.Algo_Methods.tfm_utils import (
    tabpfn_predict, append_task_id, append_task_id_onehot, pad_to_dim,
)
from ddmtolab.Methods.Algo_Methods.tfm_distill_utils import (
    fit_distill_mlp, lbfgs_optimize_lcb,
    encode_torch_scalar, encode_torch_onehot,
    MCDropoutDistillMLP,
)

warnings.filterwarnings("ignore")


def _normalize_y(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class MTBO_TFM_Distill:
    """
    Multi-Task BO with TabPFN distillation into a differentiable MLP surrogate.

    Supports all four dataset-building strategies via `transfer` and `encoding`:

      transfer='uniform', encoding='scalar'  ≈  MTBO-TFM-Uni   + distill
      transfer='uniform', encoding='onehot'  ≈  MTBO-TFM-Uni-OH + distill
      transfer='elite',   encoding='scalar'  ≈  MTBO-TFM-Elite  + distill
      transfer='elite',   encoding='onehot'  ≈  MTBO-TFM-Elite-OH + distill
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'n_objectives': 1,
        'surrogate': 'TabPFN → DistillMLP (dual-head)',
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
        transfer: str = 'uniform',
        elite_ratio: float = 0.1,
        encoding: str = 'scalar',
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
        name: str = None,
        disable_tqdm: bool = True,
        wandb_run=None,
    ):
        if transfer not in ('uniform', 'elite'):
            raise ValueError(f"transfer must be 'uniform' or 'elite', got '{transfer}'")
        if encoding not in ('scalar', 'onehot'):
            raise ValueError(f"encoding must be 'scalar' or 'onehot', got '{encoding}'")

        self.problem       = problem
        self.n_initial     = n_initial if n_initial is not None else 50
        self.max_nfes      = max_nfes  if max_nfes  is not None else 100
        self.beta          = beta
        self.n_distill     = n_distill
        self.n_estimators  = n_estimators
        self.transfer      = transfer
        self.elite_ratio   = elite_ratio
        self.encoding      = encoding
        self.mlp_hidden          = mlp_hidden
        self.mlp_depth           = mlp_depth
        self.mlp_epochs          = mlp_epochs
        self.mlp_finetune_epochs = mlp_finetune_epochs
        self.mlp_lr              = mlp_lr
        self.mlp_loss            = mlp_loss
        self.distill_model       = distill_model
        self.dropout_p           = dropout_p
        self.mc_samples          = mc_samples
        self.mc_lbfgs            = mc_lbfgs
        self.warm_start          = warm_start
        self.lbfgs_restarts      = lbfgs_restarts
        self.lbfgs_maxiter  = lbfgs_maxiter
        self.save_data     = save_data
        self.save_path     = save_path
        self.disable_tqdm  = disable_tqdm
        self.wandb_run     = wandb_run

        # Auto-generate name if not provided
        if name is None:
            enc_tag = '-OH' if encoding == 'onehot' else ''
            tr_tag  = 'Uni' if transfer == 'uniform' else 'Elite'
            self.name = f'MTBO-TFM-{tr_tag}{enc_tag}-Distill'
        else:
            self.name = name

    # ------------------------------------------------------------------
    # Dataset builders (mirror MTBO-TFM-Uni / MTBO-TFM-Elite)
    # ------------------------------------------------------------------

    def _encode_np(self, X: np.ndarray, task_id: int, n_tasks: int) -> np.ndarray:
        """Append task encoding (numpy) used for TabPFN distillation query."""
        if self.encoding == 'scalar':
            return append_task_id(X, task_id)
        return append_task_id_onehot(X, task_id, n_tasks)

    def _build_uniform_dataset(self, decs, objs, max_dim, n_tasks):
        """Pool all tasks into a single (X_all, y_all)."""
        X_parts, y_parts = [], []
        for j in range(n_tasks):
            X_j = pad_to_dim(decs[j], max_dim)
            X_j = self._encode_np(X_j, j, n_tasks)
            y_j = _normalize_y(objs[j].ravel())
            X_parts.append(X_j)
            y_parts.append(y_j)
        return np.vstack(X_parts), np.concatenate(y_parts)

    def _build_elite_dataset(self, task_i, decs, objs, max_dim, n_tasks):
        """All from task i  +  top elite_ratio from each other task."""
        X_parts, y_parts = [], []
        for j in range(n_tasks):
            y_norm = _normalize_y(objs[j].ravel())
            X_j    = pad_to_dim(decs[j], max_dim)
            if j == task_i:
                X_sel, y_sel = X_j, y_norm
            else:
                n_elite  = max(1, int(np.ceil(self.elite_ratio * len(y_norm))))
                idx      = np.argsort(y_norm)[:n_elite]
                X_sel, y_sel = X_j[idx], y_norm[idx]
            X_parts.append(self._encode_np(X_sel, j, n_tasks))
            y_parts.append(y_sel)
        return np.vstack(X_parts), np.concatenate(y_parts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(self):
        start_time = time.time()
        problem    = self.problem
        nt         = problem.n_tasks
        dims       = problem.dims
        max_dim    = max(dims)

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task  = par_list(self.max_nfes,  nt)

        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Per-task MLP cache for warm-starting.
        # Key: task index i.  Value: the fitted model from the previous BO step.
        # Reset each optimize() call so independent runs don't bleed into each other.
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

            # ---------- For uniform transfer: build shared dataset once ----------
            if self.transfer == 'uniform':
                X_shared, y_shared = self._build_uniform_dataset(decs, objs, max_dim, nt)

            for i in active_tasks:
                # --- select training set ---
                if self.transfer == 'uniform':
                    X_train, y_train = X_shared, y_shared
                else:
                    X_train, y_train = self._build_elite_dataset(i, decs, objs, max_dim, nt)

                # --- distillation: sample train + holdout in a single TabPFN call ---
                # Last n_holdout points are held out for fidelity measurement;
                # TabPFN is queried on all of them at once so there is no extra cost.
                n_holdout  = min(200, self.n_distill // 5)
                X_raw_all  = np.random.rand(self.n_distill + n_holdout, dims[i])
                X_enc_all  = self._encode_np(pad_to_dim(X_raw_all, max_dim), i, nt)

                t0 = time.time()
                mean_all, std_all = tabpfn_predict(
                    X_train, y_train, X_enc_all,
                    return_std=True,
                    n_estimators=self.n_estimators,
                )
                t_tabpfn = time.time() - t0

                X_dist_enc = X_enc_all[:self.n_distill]
                mean_d, std_d = mean_all[:self.n_distill], std_all[:self.n_distill]
                X_ho_enc   = X_enc_all[self.n_distill:]
                mean_ho, std_ho = mean_all[self.n_distill:], std_all[self.n_distill:]

                # --- fit MLP (cold-start or warm-start from cache) ---
                init_model = _mlp_cache.get(i) if self.warm_start else None
                epochs     = self.mlp_finetune_epochs if init_model is not None \
                             else self.mlp_epochs

                t0 = time.time()
                mlp = fit_distill_mlp(
                    X_dist_enc, mean_d, std_d,
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
                    _mlp_cache[i] = mlp   # store for next iteration

                # --- fidelity check on holdout ---
                # mc_dropout: use mc_predict (combined aleatoric+epistemic uncertainty)
                # mlp:        use deterministic forward
                X_ho_t = torch.tensor(X_ho_enc, dtype=torch.float32)
                if isinstance(mlp, MCDropoutDistillMLP):
                    mean_mlp, std_mlp = mlp.mc_predict(X_ho_t, n_samples=self.mc_samples)
                else:
                    with torch.no_grad():
                        mean_mlp, std_mlp = mlp(X_ho_t)
                mean_mlp = mean_mlp.detach().numpy()
                std_mlp  = std_mlp.detach().numpy()

                lcb_ho   = mean_ho  - self.beta * std_ho
                lcb_mlp  = mean_mlp - self.beta * std_mlp
                lcb_rmse      = float(np.sqrt(np.mean((lcb_mlp  - lcb_ho ) ** 2)))
                mean_rmse     = float(np.sqrt(np.mean((mean_mlp  - mean_ho) ** 2)))
                std_rmse      = float(np.sqrt(np.mean((std_mlp   - std_ho ) ** 2)))
                lcb_rank_corr = float(spearmanr(lcb_mlp, lcb_ho).statistic)

                # --- build PyTorch encode fn for L-BFGS (gradient graph intact) ---
                if self.encoding == 'scalar':
                    encode_fn = functools.partial(
                        encode_torch_scalar, max_dim=max_dim, task_id=i
                    )
                else:
                    encode_fn = functools.partial(
                        encode_torch_onehot, max_dim=max_dim, task_id=i, n_tasks=nt
                    )

                # --- minimise LCB via L-BFGS through MLP ---
                t0 = time.time()
                candidate_np = lbfgs_optimize_lcb(
                    mlp, dims[i], encode_fn,
                    beta=self.beta,
                    n_restarts=self.lbfgs_restarts,
                    max_iter=self.lbfgs_maxiter,
                    mc_lbfgs=self.mc_lbfgs,
                    mc_samples_eval=self.mc_samples,
                )
                t_lbfgs = time.time() - t0

                # --- MLP's LCB at the selected candidate ---
                with torch.no_grad():
                    cand_enc_np = self._encode_np(pad_to_dim(candidate_np, max_dim), i, nt)
                    cand_t = torch.tensor(cand_enc_np, dtype=torch.float32)
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
                        # acquisition quality (MLP landscape)
                        'acq_val':        acq_val,
                        # distillation fidelity
                        'lcb_rank_corr':  lcb_rank_corr,
                        'lcb_rmse':       lcb_rmse,
                        'mean_rmse':      mean_rmse,
                        'std_rmse':       std_rmse,
                        # timing (seconds)
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
