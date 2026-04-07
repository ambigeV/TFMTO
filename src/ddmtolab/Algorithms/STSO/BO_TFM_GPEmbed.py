"""
BO-TFM-GPEmbed: Bayesian Optimisation with TabPFN embedding → GP surrogate.

Motivation
----------
Standard TabPFN-BO (BO_TFM) uses TabPFN directly for prediction and optimises
acquisition via random sampling or CMA-ES.  This variant instead uses TabPFN
as a learned feature extractor: its internal test-token embeddings are fed into
a GP with an RBF ARD kernel, providing principled Gaussian uncertainty while
leveraging the transformer's in-context representations.

A differentiable MLP distils the embedding map  x → φ_hat(x), enabling
gradient-based (L-BFGS-B) acquisition optimisation through the full chain

    x  →  EmbedMLP  →  φ_hat  →  GP posterior  →  LCB

Rolling embedding cache
-----------------------
Rather than querying TabPFN for fresh embeddings every iteration, each
iteration adds N_candidates (x, φ) pairs to a rolling deque.  Entries from
the last `cache_max_iters` iterations are retained; older ones are dropped.
When training the EmbedMLP, samples are weighted by
    w_i = exp(-lambda_ · (current_iter - stored_iter))
so recent embeddings dominate but older data still contribute ("don't waste
food").  This amortises the TabPFN embedding cost across iterations.

Per-iteration steps
-------------------
1.  Fit TabPFN on (X_train, y_train).
2.  Sample N_candidates LHS points; extract embeddings  φ(x_cand) from TabPFN.
3.  Add (x_cand, φ_cand) to the rolling cache.
4.  Also extract embeddings of X_train (φ_train) and apply PCA to all cached
    embeddings → φ_pca.  PCA is re-fit each iteration from the full cache.
5.  Fit ExactGP with RBF ARD kernel on (φ_pca(X_train), y_train).
6.  Retrieve full cache; fit / warm-start EmbedMLP on
    (x_cache, φ_pca_cache) with exp-decay weights.
7.  Score each cached candidate via GP(φ_pca(x)); pick top-k as L-BFGS-B starts.
8.  Minimise LCB(x) via multi-start L-BFGS-B through MLP → GP.
9.  Evaluate best candidate on the true objective.

Parameters
----------
n_initial         initial LHS samples per task
max_nfes          total function evaluations (including initial samples)
beta              LCB exploration weight  (default 3.0)
n_candidates      LHS candidates queried for embeddings each iteration (default 500)
n_estimators      TabPFN ensemble size (default 1; n_estimators>1 gives richer
                  embeddings at higher cost)
embed_pca_dim     PCA target dimension for embeddings before GP fitting
                  (default 32; None = use full embedding, not recommended for
                  ARD with large D_emb)
gp_n_iter         Adam iterations for MLL maximisation  (default 100)
cache_max_iters   rolling window size in BO iterations  (default 10)
cache_lambda      exp-decay rate for cache weights       (default 0.5)
mlp_hidden        EmbedMLP hidden width  (default 128)
mlp_depth         EmbedMLP hidden layers (default 3)
mlp_epochs        cold-start epochs     (default 300)
mlp_finetune_epochs warm-start epochs   (default 50)
mlp_lr            Adam learning rate    (default 3e-3)
warm_start        reuse MLP from previous iteration as warm-start (default True)
lbfgs_restarts    L-BFGS-B restart count (also = top-k cache points used as x0)
name              algorithm name key   (default 'BO-TFM-GPEmbed')
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
from ddmtolab.Methods.Algo_Methods.tfm_gp_embed_utils import (
    tabpfn_get_embeddings,
    fit_pca, apply_pca,
    fit_gp_ard_rbf,
    gp_posterior_torch,
    fit_embed_mlp,
    lbfgs_optimize_gp_embed,
    EmbeddingCache,
)

warnings.filterwarnings('ignore')


def _normalize_y(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


class BO_TFM_GPEmbed:
    """
    Single-task Bayesian Optimisation with a TabPFN embedding → GP surrogate.

    See module docstring for full algorithm description.
    """

    algorithm_information = {
        'n_tasks':    '[1, K]',
        'n_objectives': 1,
        'surrogate':  'TabPFN embedding → GP (RBF ARD)',
        'acquisition': 'LCB (L-BFGS-B via EmbedMLP gradients)',
    }

    def __init__(
        self,
        problem,
        n_initial: int = None,
        max_nfes: int = None,
        beta: float = 3.0,
        n_candidates: int = 1000,
        n_estimators: int = 1,
        embed_pca_dim: int = 8,
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
        name: str = 'BO-TFM-GPEmbed',
        disable_tqdm: bool = True,
    ):
        self.problem             = problem
        self.n_initial           = n_initial if n_initial is not None else 50
        self.max_nfes            = max_nfes  if max_nfes  is not None else 100
        self.beta                = beta
        self.n_candidates        = n_candidates
        self.n_estimators        = n_estimators
        self.embed_pca_dim       = embed_pca_dim
        self.gp_n_iter           = gp_n_iter
        self.cache_max_iters     = cache_max_iters
        self.cache_lambda        = cache_lambda
        self.mlp_hidden          = mlp_hidden
        self.mlp_depth           = mlp_depth
        self.mlp_epochs          = mlp_epochs
        self.mlp_finetune_epochs = mlp_finetune_epochs
        self.mlp_lr              = mlp_lr
        self.warm_start          = warm_start
        self.lbfgs_restarts      = lbfgs_restarts
        self.save_data           = save_data
        self.save_path           = save_path
        self.name                = name
        self.disable_tqdm        = disable_tqdm

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
            'mlp_train_mse',       # MSE of MLP embeddings vs true TabPFN embeddings on X_train
            'gp_snr',              # outputscale / (outputscale + noise) — GP signal quality
            'pca_var_explained',   # fraction of total embedding variance captured by top-D_pca PCs
        ])

        # Per-task state
        caches:    list = [
            EmbeddingCache(
                max_iters=self.cache_max_iters,
                lambda_=self.cache_lambda,
            )
            for _ in range(nt)
        ]
        mlp_state: list = [None] * nt   # warm-start cache
        bo_iter:   list = [0]    * nt   # per-task iteration counter

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
                X_train = decs[i]                  # (n, d)
                y_train = _normalize_y(objs[i].ravel())

                # ----------------------------------------------------------
                # Step 1-3: fit TabPFN, extract + cache embeddings
                # ----------------------------------------------------------
                X_cand = LatinHypercube(d=dims[i]).random(n=self.n_candidates)

                # Embeddings for candidates AND training points in one call each
                phi_cand  = tabpfn_get_embeddings(
                    X_train, y_train, X_cand,
                    n_estimators=self.n_estimators, device=device_str,
                )
                phi_train_full = tabpfn_get_embeddings(
                    X_train, y_train, X_train,
                    n_estimators=self.n_estimators, device=device_str,
                )

                caches[i].add(X_cand, phi_cand, iteration=bo_iter[i])

                # ----------------------------------------------------------
                # Step 4: PCA on full cache + training embeddings
                # ----------------------------------------------------------
                X_cache, phi_cache_full, weights = caches[i].retrieve(bo_iter[i])

                # Fit PCA on all cached embeddings (cache + training points)
                phi_all_for_pca = np.vstack([phi_cache_full, phi_train_full])
                D_pca = (
                    min(self.embed_pca_dim, phi_all_for_pca.shape[1])
                    if self.embed_pca_dim is not None
                    else phi_all_for_pca.shape[1]
                )
                pca_mean, pca_comp = fit_pca(phi_all_for_pca, D_pca)

                # Track fraction of variance explained by top-D_pca PCs
                phi_centered = phi_all_for_pca - phi_all_for_pca.mean(axis=0)
                total_var = float(np.sum(phi_centered ** 2))
                proj = phi_centered @ pca_comp.T
                explained_var = float(np.sum(proj ** 2))
                pca_var_explained = explained_var / total_var if total_var > 1e-12 else 0.0

                phi_cache_pca = apply_pca(phi_cache_full, pca_mean, pca_comp)  # (N_cache, D_pca)
                phi_train_pca = apply_pca(phi_train_full, pca_mean, pca_comp)  # (n_train, D_pca)

                # ----------------------------------------------------------
                # Step 5: fit GP (RBF ARD) on embedding space
                # ----------------------------------------------------------
                gp_params = fit_gp_ard_rbf(
                    phi_train_pca, y_train, device, n_iter=self.gp_n_iter
                )

                # ----------------------------------------------------------
                # Step 6: fit / warm-start EmbedMLP  x_raw → φ_pca
                # ----------------------------------------------------------
                mlp = fit_embed_mlp(
                    X_cache, phi_cache_pca, weights,
                    input_dim=dims[i],
                    embed_dim=D_pca,
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
                    phi_mlp_train = mlp(X_train_t).cpu().numpy()   # (n_train, D_pca)
                mlp_train_mse = float(
                    np.mean((phi_mlp_train - phi_train_pca) ** 2)
                )
                gp_snr = float(
                    gp_params['outputscale'] /
                    (gp_params['outputscale'] + gp_params['noise'] + 1e-12)
                )
                log_writer.writerow([
                    bo_iter[i], i,
                    f'{mlp_train_mse:.6f}',
                    f'{gp_snr:.4f}',
                    f'{pca_var_explained:.4f}',
                ])
                log_file.flush()

                # ----------------------------------------------------------
                # Step 7: score cached candidates to pick L-BFGS-B warm starts
                # ----------------------------------------------------------
                # Only score the CURRENT iteration's candidates (first N_candidates rows)
                n_cur = min(self.n_candidates, len(X_cache))
                X_cur  = X_cache[-n_cur:]
                phi_cur_pca = phi_cache_pca[-n_cur:]

                phi_cur_t = torch.tensor(
                    phi_cur_pca, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    mean_cur, std_cur = gp_posterior_torch(phi_cur_t, gp_params)
                    lcb_cur = (mean_cur - self.beta * std_cur).cpu().numpy()

                top_k    = min(self.lbfgs_restarts, n_cur)
                top_idx  = np.argsort(lcb_cur)[:top_k]
                x0_points = X_cur[top_idx]

                # ----------------------------------------------------------
                # Step 8: L-BFGS-B through MLP → GP posterior → LCB
                # ----------------------------------------------------------
                candidate_np = lbfgs_optimize_gp_embed(
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
                    f'cache={len(caches[i])} D_pca={D_pca}'
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
