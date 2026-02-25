"""
Multifactorial Evolutionary Algorithm with OT-Based Cross-Task Generation (MFEA-FM)

This module implements MFEA-FM for expensive multi-task optimization using
Optimal Transport theory for principled cross-task knowledge transfer:

1. **Gaussian OT Maps**: Each task's elite population is modeled as a Gaussian.
   The OT map between Gaussians has a closed-form solution.

2. **Wasserstein Barycenter**: The W2 barycenter captures shared structure
   across tasks. New solutions are sampled from barycenter-interpolated
   distributions, achieving principled knowledge transfer.

3. **Surrogate Pre-Screening**: RBF surrogates rank batch-generated candidates
   before expensive evaluation.

4. **UCB Operator Selection**: Adaptively balances OT generation vs GA crossover.

References
----------
    [1] M. Agueh and G. Carlier, "Barycenters in the Wasserstein Space,"
        SIAM J. Math. Anal., 43(2), 904-924, 2011.

Notes
-----
Author: XiaoYu (AI Assistant)
Date: 2026.02.23
Version: 4.0
"""
import time
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.MTSO.MFEA import mfea_selection


# ============================================================================
# RBF Surrogate
# ============================================================================

class RBFSurrogate:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.X_train = None
        self.weights = None
        self.fitted = False
        self.y_mean = 0.0
        self.y_std = 1.0

    def _kernel(self, r):
        return np.sqrt(1 + (self.epsilon * r) ** 2)

    def fit(self, X, y):
        self.X_train = X.copy()
        y = y.flatten()
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-10
        y_norm = (y - self.y_mean) / self.y_std
        dist = cdist(X, X, 'euclidean')
        Phi = self._kernel(dist) + 1e-6 * np.eye(len(X))
        try:
            self.weights = np.linalg.solve(Phi, y_norm)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(Phi, y_norm, rcond=None)[0]
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            return np.zeros(len(X))
        dist = cdist(X, self.X_train, 'euclidean')
        return (self._kernel(dist) @ self.weights) * self.y_std + self.y_mean


# ============================================================================
# Gaussian Optimal Transport
# ============================================================================

def _matrix_sqrt(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_sqrt_inv(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class GaussianOT:
    """Gaussian OT for cross-task transfer via Wasserstein barycenter."""

    def __init__(self, reg=1e-3):
        self.reg = reg
        self.means = {}
        self.covs = {}
        self.bary_mean = None
        self.bary_cov = None
        self.dim = 0

    def fit(self, populations):
        """
        Fit Gaussian per task and compute Wasserstein-2 barycenter.

        Parameters
        ----------
        populations : dict {task_id: (data, objs)} where data is (n, d)
        """
        task_ids = sorted(populations.keys())
        nt = len(task_ids)

        for tid in task_ids:
            data, objs = populations[tid]
            self.dim = data.shape[1]

            # Fitness-weighted statistics (better solutions = higher weight)
            ranks = np.argsort(np.argsort(objs.flatten())) + 1.0
            w = 1.0 / ranks
            w = w / w.sum()

            mean = np.average(data, weights=w, axis=0)
            centered = data - mean
            cov = (centered * w[:, None]).T @ centered + self.reg * np.eye(self.dim)

            self.means[tid] = mean
            self.covs[tid] = cov

        # Wasserstein-2 barycenter (uniform weights)
        weights = np.ones(nt) / nt
        self.bary_mean = sum(weights[i] * self.means[tid] for i, tid in enumerate(task_ids))

        # Fixed-point iteration for barycenter covariance
        S = sum(weights[i] * self.covs[tid] for i, tid in enumerate(task_ids))
        for _ in range(50):
            S_sqrt = _matrix_sqrt(S)
            S_sqrt_inv = _matrix_sqrt_inv(S)
            T_sum = np.zeros_like(S)
            for i, tid in enumerate(task_ids):
                inner = S_sqrt @ self.covs[tid] @ S_sqrt
                T_sum += weights[i] * _matrix_sqrt(inner)
            S_new = S_sqrt_inv @ T_sum @ T_sum @ S_sqrt_inv
            if np.linalg.norm(S_new - S, 'fro') < 1e-6:
                break
            S = S_new
        self.bary_cov = S

    def generate(self, task_id, n_samples, alpha=0.5):
        """
        Generate candidates via barycenter-interpolated distribution.

        alpha=0: task-only, alpha=1: barycenter-only.
        """
        if task_id not in self.means:
            return np.random.rand(n_samples, self.dim)

        mixed_mean = (1 - alpha) * self.means[task_id] + alpha * self.bary_mean
        mixed_cov = (1 - alpha) * self.covs[task_id] + alpha * self.bary_cov
        cov_sqrt = _matrix_sqrt(mixed_cov)

        z = np.random.randn(n_samples, self.dim)
        return np.clip(mixed_mean + z @ cov_sqrt.T, 0.0, 1.0)

    def transport(self, x, from_task, to_task):
        """Transport solutions from one task to another via closed-form OT map."""
        if from_task not in self.means or to_task not in self.means:
            return x
        x_c = x - self.means[from_task]
        cov1_si = _matrix_sqrt_inv(self.covs[from_task])
        cov2_s = _matrix_sqrt(self.covs[to_task])
        return np.clip(self.means[to_task] + x_c @ cov1_si.T @ cov2_s.T, 0.0, 1.0)


# ============================================================================
# UCB Selector
# ============================================================================

class UCBSelector:
    def __init__(self, n_ops=2, c=1.0, window=30):
        self.n_ops = n_ops
        self.c = c
        self.window = window
        self.rewards = [[] for _ in range(n_ops)]
        self.counts = [0] * n_ops
        self.total = 0

    def select(self):
        for i in range(self.n_ops):
            if self.counts[i] < 2:
                return i
        ucbs = []
        for i in range(self.n_ops):
            r = self.rewards[i][-self.window:]
            ucbs.append(np.mean(r) + self.c * np.sqrt(np.log(self.total + 1) / (self.counts[i] + 1)))
        return int(np.argmax(ucbs))

    def update(self, op, reward):
        self.rewards[op].append(reward)
        self.counts[op] += 1
        self.total += 1


# ============================================================================
# MFEA-FM Algorithm
# ============================================================================

class MFEA_FM:
    """
    Multifactorial EA with OT-Based Cross-Task Generation.

    OT branch: generate K candidates from barycenter-interpolated Gaussian →
    surrogate pre-screen → evaluate top-1.
    GA branch: standard SBX crossover + mutation.
    UCB1 selects between branches adaptively.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2, mum=5,
                 refine_freq=3, n_candidates=30, transfer_strength=0.5,
                 ot_reg=1e-3, ucb_c=1.0, top_ratio=0.5,
                 save_data=True, save_path='./Data', name='MFEA-FM', disable_tqdm=True):
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.refine_freq = refine_freq
        self.n_candidates = n_candidates
        self.transfer_strength = transfer_strength
        self.ot_reg = ot_reg
        self.ucb_c = ucb_c
        self.top_ratio = top_ratio
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def _fit_models(self, pop_decs, pop_objs, nt, uni_dim):
        populations = {}
        for i in range(nt):
            decs = pop_decs[i]
            objs = pop_objs[i]
            n_elite = max(int(len(objs) * self.top_ratio), min(len(objs), 3))
            idx = np.argsort(objs.flatten())[:n_elite]
            ed = decs[idx]
            if ed.shape[1] < uni_dim:
                ed = np.hstack([ed, np.random.rand(len(ed), uni_dim - ed.shape[1])])
            elif ed.shape[1] > uni_dim:
                ed = ed[:, :uni_dim]
            populations[i] = (ed, objs[idx])

        ot = GaussianOT(reg=self.ot_reg)
        ot.fit(populations)

        surrogates = []
        for i in range(nt):
            s = RBFSurrogate(epsilon=1.0)
            if len(pop_decs[i]) >= 3:
                d = pop_decs[i]
                if d.shape[1] < uni_dim:
                    d = np.hstack([d, np.random.rand(len(d), uni_dim - d.shape[1])])
                s.fit(d[:, :uni_dim], pop_objs[i])
            surrogates.append(s)
        return ot, surrogates

    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]
        uni_dim = pop_decs[0].shape[1]

        ot, surrogates = self._fit_models(pop_decs, pop_objs, nt, uni_dim)
        ucb = UCBSelector(n_ops=2, c=self.ucb_c)
        gen = 0
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)
            n_cons_uni = pop_cons.shape[1]
            off_decs_list, off_objs_list, off_sfs_list = [], [], []

            best_per_task = {}
            for t in range(nt):
                mask = pop_sfs.flatten() == t
                if mask.sum() > 0:
                    best_per_task[t] = pop_objs[mask].min()

            shuffled = np.random.permutation(pop_decs.shape[0])

            for i in range(0, len(shuffled), 2):
                if nfes >= max_nfes:
                    break
                p1, p2 = shuffled[i], shuffled[i + 1]
                sf1, sf2 = int(pop_sfs[p1].item()), int(pop_sfs[p2].item())
                op = ucb.select()

                if op == 0 and (sf1 == sf2 or np.random.rand() < self.rmp):
                    assigned_sf = np.random.choice([sf1, sf2])

                    # Generate from OT model + cross-task transport
                    candidates = ot.generate(assigned_sf, self.n_candidates, alpha=self.transfer_strength)
                    other = 1 - assigned_sf if nt == 2 else np.random.choice([t for t in range(nt) if t != assigned_sf])
                    cross = ot.generate(other, max(self.n_candidates // 2, 1), alpha=0.3)
                    cross_t = ot.transport(cross, other, assigned_sf)
                    candidates = np.vstack([candidates, cross_t])

                    # Surrogate pre-screening
                    if surrogates[assigned_sf].fitted:
                        pred = surrogates[assigned_sf].predict(candidates[:, :uni_dim])
                        best_idx = np.argmin(pred)
                    else:
                        best_idx = 0

                    dec = mutation(candidates[best_idx], mu=self.mum)
                    off_obj, off_con = evaluation_single(problem, dec[:dims[assigned_sf]], assigned_sf)
                    nfes += 1
                    pbar.update(1)
                    off_decs_list.append(dec.reshape(1, -1))
                    off_objs_list.append(off_obj)
                    off_sfs_list.append(np.array([[assigned_sf]]))

                    cur = best_per_task.get(assigned_sf, off_obj.flatten()[0])
                    ucb.update(0, max(0, cur - off_obj.flatten()[0]) / (abs(cur) + 1e-10))
                else:
                    off_dec1, off_dec2 = crossover(pop_decs[p1], pop_decs[p2], mu=self.muc)
                    sf_o1 = np.random.choice([sf1, sf2])
                    sf_o2 = sf1 if sf_o1 == sf2 else sf2
                    for od, sf in [(off_dec1, sf_o1), (off_dec2, sf_o2)]:
                        if nfes >= max_nfes:
                            break
                        off_obj, off_con = evaluation_single(problem, od[:dims[sf]], sf)
                        nfes += 1
                        pbar.update(1)
                        off_decs_list.append(od.reshape(1, -1))
                        off_objs_list.append(off_obj)
                        off_sfs_list.append(np.array([[sf]]))
                        cur = best_per_task.get(sf, off_obj.flatten()[0])
                        ucb.update(1, max(0, cur - off_obj.flatten()[0]) / (abs(cur) + 1e-10))

            if not off_decs_list:
                break

            off_decs = np.vstack(off_decs_list)
            off_objs = np.vstack(off_objs_list)
            off_cons = np.zeros((len(off_decs_list), n_cons_uni))
            off_sfs = np.vstack(off_sfs_list)

            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(
                (pop_decs, off_decs), (pop_objs, off_objs), (pop_cons, off_cons), (pop_sfs, off_sfs))
            pop_decs, pop_objs, pop_cons, pop_sfs = mfea_selection(
                pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            decs, cons = space_transfer(problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)

            if gen % self.refine_freq == 0:
                ot, surrogates = self._fit_models(pop_decs, pop_objs, nt, uni_dim)
            gen += 1

        pbar.close()
        runtime = time.time() - start_time
        return build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name, save_data=self.save_data)