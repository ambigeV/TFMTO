"""
Scenario-based Self-Learning Transfer Differential Evolution (SSLT-DE)

This module implements SSLT-DE for multi-task optimization using a DQN-based
reinforcement learning framework to adaptively select among four knowledge
transfer scenarios.

References
----------
    [1] Z. Yuan, G. Dai, L. Peng, M. Wang, Z. Song, and X. Chen, "Scenario-based
        self-learning transfer framework for multi-task optimization problems,"
        Knowledge-Based Systems, vol. 325, p. 113824, 2025.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


# ============================================================================
# Helper Functions
# ============================================================================

def _wasserstein_1d(u_decs, v_decs):
    """Compute 1D Wasserstein distance between two flattened population arrays."""
    u = np.sort(u_decs.ravel())
    v = np.sort(v_decs.ravel())
    all_vals = np.unique(np.concatenate([u, v]))
    if len(all_vals) < 2:
        return 0.0
    u_cdf = np.searchsorted(u, all_vals[:-1], side='right') / len(u)
    v_cdf = np.searchsorted(v, all_vals[:-1], side='right') / len(v)
    return float(np.sum(np.abs(u_cdf - v_cdf) * np.diff(all_vals)))


def _dispersion_metric(decs, objs):
    """Mean pairwise squared distance among top 10% individuals."""
    M = max(int(0.1 * len(objs)), 1)
    rank = np.argsort(objs.flatten())
    top = decs[rank[:M]]
    if M <= 1:
        return 0.0
    total = 0.0
    for i in range(M - 1):
        diff = top[i + 1:] - top[i]
        total += np.sum(diff ** 2)
    return total / (M * (M - 1))


def _dispersion_type(decs, objs, decs_old, objs_old):
    """Compare dispersion: 1=decreasing, 2=same, 3=increasing."""
    dm = _dispersion_metric(decs, objs)
    dm_old = _dispersion_metric(decs_old, objs_old)
    if dm < dm_old:
        return 1
    elif dm == dm_old:
        return 2
    else:
        return 3


def _convergence_dist(decs_old, decs_new):
    """Euclidean distance between old and new population centers."""
    c_old = np.mean(decs_old, axis=0)
    c_new = np.mean(decs_new, axis=0)
    return float(np.sqrt(np.sum((c_old - c_new) ** 2)))


def _smooth(decs, objs):
    """Keep the best individual from each consecutive triple."""
    keep = []
    n = len(decs)
    for i in range(0, n - 2, 3):
        triple_objs = objs[i:i + 3].flatten()
        best = np.argmin(triple_objs)
        keep.append(i + best)
    if len(keep) == 0:
        keep = [np.argmin(objs.flatten())]
    return decs[keep], objs[keep]


def _de_crossover_single(trial, target, CR):
    """Binomial crossover for a single individual pair."""
    d = len(trial)
    mask = np.random.rand(d) < CR
    j_rand = np.random.randint(d)
    mask[j_rand] = True
    offspring = target.copy()
    offspring[mask] = trial[mask]
    return offspring


def _normalize(X):
    """Min-max normalize columns to [-1, 1], matching MATLAB mapminmax."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    return 2.0 * (X - mins) / rng - 1.0, mins, maxs


def _normalize_apply(x, mins, maxs):
    """Apply saved min-max normalization."""
    rng = maxs - mins
    rng[rng == 0] = 1.0
    return 2.0 * (x - mins) / rng - 1.0


# ============================================================================
# Q-Network for DQN
# ============================================================================

class _QNet(nn.Module):
    """Simple MLP for Q-value prediction."""

    def __init__(self, input_dim=7, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_qnet(model, X, y, epochs=200, lr=0.005):
    """Train Q-network on normalized data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        pred = model(X_t)
        loss = nn.functional.mse_loss(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ============================================================================
# SSLT-DE Algorithm
# ============================================================================

class SSLT_DE:
    """
    Scenario-based Self-Learning Transfer Differential Evolution.

    Uses a DQN-based reinforcement learning framework to adaptively select
    among four knowledge transfer scenarios:
    1. No transfer (standard DE/rand/1/bin)
    2. Shape transfer (shift smoothed source toward target center)
    3. Bi-directional transfer (DE on merged populations)
    4. Domain transfer (direction-guided from best source-target difference)

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None,
                 threshold=150, gap=50, gamma=0.9, epsilon=0.8,
                 F=0.5, CR=0.9,
                 save_data=True, save_path='./Data', name='SSLT-DE',
                 disable_tqdm=True):
        """
        Initialize SSLT-DE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        threshold : int, optional
            Number of generations before building DQN (default: 150)
        gap : int, optional
            DQN update interval in generations (default: 50)
        gamma : float, optional
            Discount factor for Q-learning (default: 0.9)
        epsilon : float, optional
            Epsilon-greedy exploration rate (default: 0.8)
        F : float, optional
            DE mutation scale factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SSLT-DE')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.threshold = threshold
        self.gap = gap
        self.gamma = gamma
        self.epsilon = epsilon
        self.F = F
        self.CR = CR
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SSLT-DE algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt
        eps = 1e-30

        # Initialize and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Convert to unified space for cross-task operations
        pop_decs, pop_cons = space_transfer(
            problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = objs
        maxD = pop_decs[0].shape[1]
        maxC = pop_cons[0].shape[1]

        # Per-task DQN state
        data_task = [[] for _ in range(nt)]       # Experience replay buffer
        model_built = [False] * nt                 # Whether DQN is built
        count_task = [0] * nt                      # Update counter
        q_model = [None] * nt                      # Q-network per task
        norm_params = [None] * nt                  # Normalization parameters

        # Store previous generation populations
        pop_decs_old = [d.copy() for d in pop_decs]
        pop_objs_old = [o.copy() for o in pop_objs]

        gen = 0
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            for t in range(nt):
                if nfes >= max_nfes:
                    break

                # Random source task
                s = np.random.randint(nt)
                while s == t and nt > 1:
                    s = np.random.randint(nt)

                # ============================================================
                # Compute state features
                # ============================================================
                min_old_t = np.min(pop_objs_old[t]) + eps
                min_cur_t = np.min(pop_objs[t]) + eps
                min_old_s = np.min(pop_objs_old[s]) + eps
                min_cur_s = np.min(pop_objs[s]) + eps

                conv_target = (min_old_t - min_cur_t) / abs(min_old_t)
                conv_source = (min_old_s - min_cur_s) / abs(min_old_s)
                wsd = _wasserstein_1d(pop_decs[t], pop_decs[s])
                ls_target = _dispersion_type(pop_decs[t], pop_objs[t],
                                             pop_decs_old[t], pop_objs_old[t])
                ls_source = _dispersion_type(pop_decs[s], pop_objs[s],
                                             pop_decs_old[s], pop_objs_old[s])
                pha = nfes / max_nfes

                state = np.array([conv_source, conv_target, wsd,
                                  ls_target, ls_source, pha])

                # ============================================================
                # Action selection
                # ============================================================
                if gen <= self.threshold:
                    action = np.random.randint(1, 5)
                elif not model_built[t]:
                    # Build DQN model
                    exp = np.array(data_task[t])
                    X_raw = exp[:, :7]
                    y_raw = exp[:, 7]
                    X_norm, x_min, x_max = _normalize(X_raw)
                    y_norm, y_min, y_max = _normalize(y_raw.reshape(-1, 1))
                    y_norm = y_norm.flatten()

                    q_model[t] = _QNet(input_dim=7, hidden_dim=32)
                    _train_qnet(q_model[t], X_norm, y_norm)
                    norm_params[t] = (x_min, x_max, y_min, y_max)
                    model_built[t] = True
                    action = np.random.randint(1, 5)
                else:
                    # Epsilon-greedy
                    if np.random.rand() > self.epsilon:
                        action = np.random.randint(1, 5)
                    else:
                        x_min, x_max, y_min, y_max = norm_params[t]
                        q_vals = []
                        q_model[t].eval()
                        with torch.no_grad():
                            for a in range(1, 5):
                                x_raw = np.append(state, a).reshape(1, -1)
                                x_n = _normalize_apply(x_raw, x_min, x_max)
                                x_t = torch.tensor(x_n, dtype=torch.float32)
                                q_vals.append(q_model[t](x_t).item())
                        action = np.argmax(q_vals) + 1

                # ============================================================
                # Execute action
                # ============================================================
                pop_decs_old[t] = pop_decs[t].copy()
                pop_objs_old[t] = pop_objs[t].copy()

                if action == 1:
                    # No KT: standard DE/rand/1/bin
                    off_decs = de_generation(pop_decs[t], F=self.F, CR=self.CR)
                    off_objs, off_cons_real = evaluation_single(
                        problem, off_decs[:, :dims[t]], t)
                    nfes += n
                    pbar.update(n)

                    # 1-to-1 DE selection
                    better = off_objs.flatten() <= pop_objs[t].flatten()
                    pop_decs[t][better] = off_decs[better]
                    pop_objs[t][better] = off_objs[better]

                elif action == 2:
                    # Shape KT: shift smoothed source toward target center
                    sm_s_decs, sm_s_objs = _smooth(pop_decs[s], pop_objs[s])
                    sm_t_decs, _ = _smooth(pop_decs[t], pop_objs[t])

                    center_t = np.mean(sm_t_decs, axis=0)
                    center_s = np.mean(sm_s_decs, axis=0)
                    shifted = sm_s_decs + (center_t - center_s)
                    shifted = np.clip(shifted, 0, 1)

                    n_shifted = len(shifted)
                    sh_objs, sh_cons_real = evaluation_single(
                        problem, shifted[:, :dims[t]], t)
                    nfes += n_shifted
                    pbar.update(n_shifted)

                    # Elite selection (target ∪ shifted)
                    merged_decs = np.vstack([pop_decs[t], shifted])
                    merged_objs = np.vstack([pop_objs[t], sh_objs])
                    sel = selection_elit(objs=merged_objs, n=n)
                    pop_decs[t] = merged_decs[sel]
                    pop_objs[t] = merged_objs[sel]

                elif action == 3:
                    # Bi-KT: DE on merged populations
                    merged = np.vstack([pop_decs[t], pop_decs[s]])
                    n_merged = len(merged)

                    off_decs = np.zeros_like(merged)
                    for i in range(n_merged):
                        # DE/current-to-rand/1
                        idxs = list(range(n_merged))
                        idxs.remove(i)
                        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                        v = merged[i] + self.F * (merged[r1] - merged[i]) \
                            + 0.5 * (merged[r2] - merged[r3])
                        off_decs[i] = _de_crossover_single(v, merged[i], self.CR)
                    off_decs = np.clip(off_decs, 0, 1)

                    off_objs, off_cons_real = evaluation_single(
                        problem, off_decs[:, :dims[t]], t)
                    nfes += n_merged
                    pbar.update(n_merged)

                    # Elite selection (target ∪ offspring)
                    merged_sel = np.vstack([pop_decs[t], off_decs])
                    merged_sel_objs = np.vstack([pop_objs[t], off_objs])
                    sel = selection_elit(objs=merged_sel_objs, n=n)
                    pop_decs[t] = merged_sel[sel]
                    pop_objs[t] = merged_sel_objs[sel]

                elif action == 4:
                    # Domain KT: direction-guided transfer
                    best_s = np.argmin(pop_objs[s].flatten())
                    best_t = np.argmin(pop_objs[t].flatten())
                    direction = pop_decs[s][best_s] - pop_decs[t][best_t]

                    num = max(1, round(pha * 10))
                    perm = np.random.permutation(n)

                    off_decs = np.zeros((num, maxD))
                    for i in range(num):
                        idx = perm[i % n]
                        off_decs[i] = _de_crossover_single(
                            pop_decs[t][idx], direction, self.CR)
                    off_decs = np.clip(off_decs, 0, 1)

                    off_objs, off_cons_real = evaluation_single(
                        problem, off_decs[:, :dims[t]], t)
                    nfes += num
                    pbar.update(num)

                    # Elite selection (target ∪ offspring)
                    merged_decs = np.vstack([pop_decs[t], off_decs])
                    merged_objs = np.vstack([pop_objs[t], off_objs])
                    sel = selection_elit(objs=merged_objs, n=n)
                    pop_decs[t] = merged_decs[sel]
                    pop_objs[t] = merged_objs[sel]

                # ============================================================
                # Compute reward and store experience
                # ============================================================
                fold = np.min(pop_decs_old[t] @ np.ones((maxD, 1)))  # dummy
                fold = np.min(pop_objs_old[t])
                f = np.min(pop_objs[t])
                fold_mean = np.mean(pop_objs_old[t])
                f_mean = np.mean(pop_objs[t])

                imp_rate = (fold - f) / (abs(fold) + eps)
                pop_rate = (fold_mean - f_mean) / (abs(fold_mean) + eps)
                move_dis = _convergence_dist(pop_decs_old[t], pop_decs[t])

                vals = np.array([imp_rate, pop_rate, move_dis])
                max_val, min_val = vals.max(), vals.min()
                rng = max_val - min_val
                if rng > eps:
                    imp_rate_n = (imp_rate - min_val) / rng
                    pop_rate_n = (pop_rate - min_val) / rng
                    move_dis_n = (move_dis - min_val) / rng
                else:
                    imp_rate_n = pop_rate_n = move_dis_n = 0.0

                pha_new = nfes / max_nfes
                reward = (imp_rate_n + pop_rate_n + move_dis_n) * pha_new

                # New state features
                min_new_t = np.min(pop_objs[t]) + eps
                min_new_s = np.min(pop_objs[s]) + eps
                conv_new_target = (np.min(pop_objs_old[t]) - min_new_t) / abs(np.min(pop_objs_old[t]) + eps)
                conv_new_source = (np.min(pop_objs_old[s]) - min_new_s) / abs(np.min(pop_objs_old[s]) + eps)
                wsd_new = _wasserstein_1d(pop_decs[s], pop_decs[t])
                ls_new_target = _dispersion_type(pop_decs[t], pop_objs[t],
                                                 pop_decs_old[t], pop_objs_old[t])
                ls_new_source = _dispersion_type(pop_decs[s], pop_objs[s],
                                                 pop_decs_old[s], pop_objs_old[s])

                record = np.array([
                    conv_source, conv_target, wsd, ls_target, ls_source, pha, action,
                    reward, conv_new_source, conv_new_target, wsd_new,
                    ls_new_target, ls_new_source, pha_new
                ])
                data_task[t].append(record)
                if len(data_task[t]) > 500:
                    data_task[t].pop(0)

                # ============================================================
                # Update DQN periodically
                # ============================================================
                if model_built[t]:
                    count_task[t] += 1
                    if count_task[t] > self.gap:
                        exp = np.array(data_task[t])
                        X_raw = exp[:, :7]
                        rewards_raw = exp[:, 7]

                        X_norm, x_min, x_max = _normalize(X_raw)

                        # Compute max Q across all experiences
                        q_model[t].eval()
                        with torch.no_grad():
                            X_t = torch.tensor(X_norm, dtype=torch.float32)
                            q_preds = q_model[t](X_t).numpy()
                        max_q = np.max(q_preds)

                        # Q-learning target: R + gamma * max(Q)
                        target_q = rewards_raw + self.gamma * max_q

                        y_norm, y_min, y_max = _normalize(target_q.reshape(-1, 1))
                        y_norm = y_norm.flatten()

                        norm_params[t] = (x_min, x_max, y_min, y_max)
                        q_model[t] = _QNet(input_dim=7, hidden_dim=32)
                        _train_qnet(q_model[t], X_norm, y_norm)
                        count_task[t] = 0

            # Record history in real space
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results
