"""
Two-phase Evolutionary Algorithm (TEA)

This module implements TEA for computationally expensive multi-objective optimization.
It uses Probabilistic Dominant Product Dominance (PDPD) sorting that groups objectives
by uncertainty level and applies product dominance for uncertain objectives, with
Kriging surrogate models to guide the search.

References
----------
    [1] Z. Zhang, Y. Wang, J. Liu, G. Sun, and K. Tang. A two-phase Kriging-
        assisted evolutionary algorithm for expensive constrained multiobjective
        optimization problems. IEEE Transactions on Systems, Man, and Cybernetics:
        Systems, 2024, 54(8): 4579-4591.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, gp_predict
import warnings

warnings.filterwarnings("ignore")


class TEA:
    """
    Two-phase Evolutionary Algorithm for expensive multi-objective optimization
    using PDPD sorting and Kriging surrogates.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'equal',
        'n_cons': '[0, C]',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100,
                 wmax=20, mu=5,
                 save_data=True, save_path='./Data', name='TEA', disable_tqdm=True):
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.wmax = wmax
        self.mu = mu
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, interval=self.mu)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, interval=self.mu)

        pop_indices = []
        for i in range(nt):
            N = n_per_task[i]
            if decs[i].shape[0] <= N:
                pop_indices.append(np.arange(decs[i].shape[0]))
            else:
                idx = _pop_reselect(objs[i], N)
                pop_indices.append(np.where(idx)[0])

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                N = n_per_task[i]

                try:
                    obj_models = mo_gp_build(decs[i], objs[i], data_type)
                except Exception:
                    continue

                P_decs = decs[i][pop_indices[i]]
                P_objs = objs[i][pop_indices[i]]

                pop_decs, pop_objs, pop_mse = _evo_search(
                    P_decs, P_objs, obj_models, M, N, self.wmax, data_type
                )

                candidates = _candi_select(
                    pop_decs, pop_objs, pop_mse,
                    decs[i], objs[i], self.mu
                )

                if candidates is None or candidates.shape[0] == 0:
                    continue

                cand_objs, _ = evaluation_single(problem, candidates, i)

                decs[i] = np.vstack([decs[i], candidates])
                objs[i] = np.vstack([objs[i], cand_objs])

                idx = _pop_reselect(objs[i], N)
                pop_indices[i] = np.where(idx)[0]

                nfes_per_task[i] += candidates.shape[0]
                pbar.update(candidates.shape[0])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# PDPD Non-dominated Sorting
# =============================================================================

def _ndsort_pdpd(pop_objs, obj_mse, n_sort, epsilon=0.75):
    """
    Non-dominated sorting based on Probabilistic Dominant Product Dominance.

    For each pair (i, j), objectives are split into "certain" (|Pi-Pj| > epsilon)
    and "uncertain" (|Pi-Pj| <= epsilon) groups. Uncertain objectives are aggregated
    via product dominance. Solution i dominates j if it has higher probability on
    all certain objectives AND higher product probability on uncertain objectives.

    Parameters
    ----------
    pop_objs : np.ndarray, shape (N, M)
        Predicted objectives (normalized).
    obj_mse : np.ndarray, shape (N, M)
        Prediction MSE (normalized).
    n_sort : int
        Number of solutions to sort.
    epsilon : float
        Threshold for uncertain/certain objective classification (default: 0.75).

    Returns
    -------
    front_no : np.ndarray, shape (N,)
        Front assignment. inf = unassigned.
    max_fno : int
        Last assigned front number.
    """
    N, M = pop_objs.shape

    # Vectorized computation of pairwise sigma and mean
    # sigma[i,j,k] = sqrt(mse_i_k + mse_j_k)
    sigma = np.sqrt(np.maximum(
        obj_mse[:, np.newaxis, :] + obj_mse[np.newaxis, :, :], 1e-40
    ))  # (N, N, M)

    # mean_diff[i,j,k] = obj_i_k - obj_j_k
    mean_diff = pop_objs[:, np.newaxis, :] - pop_objs[np.newaxis, :, :]  # (N, N, M)

    # Pi[i,j,k] = P(obj_i_k <= obj_j_k) = Phi(-mean_diff / sigma)
    Pi_all = norm.cdf(-mean_diff / sigma)  # (N, N, M)
    Pj_all = 1.0 - Pi_all  # (N, N, M)

    # Build dominance matrix (loop needed due to per-pair uncertain/certain split)
    dominate = np.zeros((N, N), dtype=bool)
    TOL = 1e-12

    for i in range(N - 1):
        for j in range(i + 1, N):
            pi = Pi_all[i, j, :]
            pj = Pj_all[i, j, :]

            diff = np.abs(pi - pj)
            idx_uncertain = diff <= epsilon
            idx_certain = ~idx_uncertain

            PDi = np.prod(pi[idx_uncertain]) if np.any(idx_uncertain) else 1.0
            PDj = np.prod(pj[idx_uncertain]) if np.any(idx_uncertain) else 1.0

            vals_i = np.append(pi[idx_certain], PDi)
            vals_j = np.append(pj[idx_certain], PDj)

            # i dominates j: all(vals_i >= vals_j) with strict somewhere
            if np.all(vals_i >= vals_j - TOL) and np.any(vals_i > vals_j + TOL):
                dominate[i, j] = True
            elif np.all(vals_j >= vals_i - TOL) and np.any(vals_j > vals_i + TOL):
                dominate[j, i] = True

    # Modified NDS with minimum domination count (handles cycles)
    front_no = np.full(N, np.inf)
    max_fno = 0

    while np.sum(np.isfinite(front_no)) < min(n_sort, N):
        max_fno += 1
        current = np.where(~np.isfinite(front_no))[0]
        if len(current) == 0:
            break

        dom_count = np.sum(dominate[np.ix_(current, current)], axis=0)
        min_count = np.min(dom_count)
        index = current[dom_count == min_count]

        front_no[index] = max_fno
        dominate[index, :] = False

    return front_no, max_fno


# =============================================================================
# Evolutionary Search
# =============================================================================

def _evo_search(P_decs, P_objs, obj_models, M, N, wmax, data_type):
    """
    Surrogate-based evolutionary search using PDPD environmental selection.

    Parameters
    ----------
    P_decs : np.ndarray
        Current population decisions, shape (N, D).
    P_objs : np.ndarray
        Current population real objectives, shape (N, M).
    obj_models : list
        List of trained GP models (one per objective).
    M : int
        Number of objectives.
    N : int
        Population size.
    wmax : int
        Number of generations.
    data_type : torch.dtype
        Data type for GP prediction.

    Returns
    -------
    pop_decs, pop_objs, pop_mse : np.ndarray
        Final population with predicted objectives and MSE.
    """
    pop_decs = P_decs.copy()
    pop_objs = P_objs.copy()
    pop_mse = np.zeros_like(pop_objs)

    for w in range(wmax):
        off_decs = ga_generation(pop_decs, muc=20, mum=20)
        off_objs, off_mse = _predict_with_mse(off_decs, obj_models, M, data_type)

        merged_decs = np.vstack([pop_decs, off_decs])
        merged_objs = np.vstack([pop_objs, off_objs])
        merged_mse = np.vstack([pop_mse, off_mse])

        selected = _env_selection(merged_objs, merged_mse, N)

        pop_decs = merged_decs[selected]
        pop_objs = merged_objs[selected]
        pop_mse = merged_mse[selected]

    return pop_decs, pop_objs, pop_mse


def _predict_with_mse(pop_decs, obj_models, M, data_type):
    """Predict objectives and MSE using per-objective GP models."""
    N = pop_decs.shape[0]
    pop_objs = np.zeros((N, M))
    pop_mse = np.zeros((N, M))

    for j in range(M):
        pred, std = gp_predict(obj_models[j], pop_decs, data_type)
        pop_objs[:, j] = pred.flatten()
        pop_mse[:, j] = std.flatten() ** 2

    return pop_objs, pop_mse


# =============================================================================
# Environmental Selection (for evolutionary search)
# =============================================================================

def _env_selection(pop_objs, pop_mse, N):
    """
    Environmental selection using PDPD sorting + diversity.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (n_total, M).
    pop_mse : np.ndarray
        MSE, shape (n_total, M).
    N : int
        Target population size.

    Returns
    -------
    selected : np.ndarray
        Indices of selected solutions.
    """
    n_total = pop_objs.shape[0]

    # Normalize objectives and MSE
    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-9)
    norm_objs = (pop_objs - zmin) / obj_range
    norm_mse = pop_mse / (obj_range ** 2)

    # PDPD sorting
    front_no, max_fno = _ndsort_pdpd(norm_objs, norm_mse, N)

    next_mask = np.zeros(n_total, dtype=bool)
    next_mask[front_no < max_fno] = True
    last_front = np.where(front_no == max_fno)[0]
    n_remaining = N - np.sum(next_mask)

    if n_remaining <= 0:
        return np.where(next_mask)[0][:N]

    if len(last_front) <= n_remaining:
        next_mask[last_front] = True
    elif max_fno == 1:
        chosen = spea2_truncation(norm_objs[last_front], N)
        next_mask[last_front[chosen]] = True
    else:
        chosen = _dist_selection(norm_objs, last_front, n_remaining)
        next_mask[last_front[chosen]] = True

    return np.where(next_mask)[0]


# =============================================================================
# Candidate Selection
# =============================================================================

def _candi_select(pop_decs, pop_objs, pop_mse, db_decs, db_objs, mu):
    """
    Select mu candidate solutions for expensive re-evaluation.

    Uses PDPD sorting on novel solutions, then greedy max-min diversity
    selection from the last front using the database ND front as reference.

    Parameters
    ----------
    pop_decs : np.ndarray
        Evolved population decisions.
    pop_objs : np.ndarray
        Evolved population predicted objectives.
    pop_mse : np.ndarray
        Evolved population prediction MSE.
    db_decs : np.ndarray
        All evaluated decisions (database).
    db_objs : np.ndarray
        All evaluated objectives (database).
    mu : int
        Number of candidates to select.

    Returns
    -------
    candidates : np.ndarray or None
        Selected candidate decisions.
    """
    # Filter duplicates with database
    if db_decs.shape[0] > 0:
        dist_to_db = cdist(pop_decs, db_decs)
        novel_mask = np.min(dist_to_db, axis=1) > 1e-5
    else:
        novel_mask = np.ones(pop_decs.shape[0], dtype=bool)

    if np.sum(novel_mask) == 0:
        return None

    novel_decs = pop_decs[novel_mask]
    novel_objs = pop_objs[novel_mask]
    novel_mse = pop_mse[novel_mask]
    n_novel = novel_decs.shape[0]

    # Normalize objectives
    all_objs_combined = np.vstack([db_objs, novel_objs])
    zmin = np.min(all_objs_combined, axis=0)
    zmax = np.max(all_objs_combined, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-9)

    norm_novel_objs = (novel_objs - zmin) / obj_range
    norm_novel_mse = novel_mse / (obj_range ** 2)
    norm_db_objs = (db_objs - zmin) / obj_range

    # Reference set: ND front from database
    front_no_db, _ = nd_sort(db_objs, db_objs.shape[0])
    ref_objs = norm_db_objs[front_no_db == 1]

    if n_novel <= mu:
        C_ = novel_decs
    else:
        # PDPD sort on novel solutions
        front_no, max_fno = _ndsort_pdpd(norm_novel_objs, norm_novel_mse, mu)

        selected = list(np.where(front_no < max_fno)[0])
        last_front = list(np.where(front_no == max_fno)[0])
        n_remaining = mu - len(selected)

        if len(last_front) <= n_remaining:
            selected.extend(last_front)
        elif len(last_front) > n_remaining and n_remaining > 0:
            # Greedy max-min diversity selection against reference
            current_ref = ref_objs.copy()
            if selected:
                current_ref = np.vstack([current_ref, norm_novel_objs[selected]])

            remaining = list(range(len(last_front)))
            for _ in range(n_remaining):
                if not remaining:
                    break
                cand = norm_novel_objs[[last_front[r] for r in remaining]]
                dists = cdist(cand, current_ref)
                min_dists = np.min(dists, axis=1)
                best = np.argmax(min_dists)

                selected.append(last_front[remaining[best]])
                current_ref = np.vstack([
                    current_ref,
                    norm_novel_objs[last_front[remaining[best]]].reshape(1, -1)
                ])
                remaining.pop(best)

        C_ = novel_decs[selected] if selected else None

    if C_ is None or len(C_) == 0:
        return None

    # Final distance check against database
    C = []
    for k in range(C_.shape[0]):
        d = cdist(C_[k:k + 1], db_decs)
        if np.min(d) > 1e-5:
            C.append(C_[k])

    return np.array(C) if C else None


# =============================================================================
# Population Reselection (standard NDS on real evaluations)
# =============================================================================

def _pop_reselect(pop_objs, N):
    """
    Reselect N solutions using standard non-dominated sorting.

    Parameters
    ----------
    pop_objs : np.ndarray
        All evaluated objectives.
    N : int
        Target population size.

    Returns
    -------
    next_mask : np.ndarray
        Boolean mask for selected solutions.
    """
    n_total = pop_objs.shape[0]
    if n_total <= N:
        return np.ones(n_total, dtype=bool)

    # Normalize for distance computations
    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-9)
    norm_objs = (pop_objs - zmin) / obj_range

    front_no, max_fno = nd_sort(pop_objs, N)

    next_mask = front_no < max_fno
    last_front = np.where(front_no == max_fno)[0]
    n_remaining = N - np.sum(next_mask)

    if n_remaining <= 0:
        return next_mask

    if len(last_front) <= n_remaining:
        next_mask[last_front] = True
    elif max_fno == 1:
        chosen = spea2_truncation(norm_objs[last_front], n_remaining)
        next_mask[last_front[chosen]] = True
    else:
        chosen = _dist_selection(norm_objs, last_front, n_remaining)
        next_mask[last_front[chosen]] = True

    return next_mask


# =============================================================================
# Diversity Helper Functions
# =============================================================================

def _dist_selection(pop_objs, last_indices, mu):
    """
    Select mu solutions from last front using nearest-neighbor density.
    Solutions with smallest density (most isolated) are preferred.

    Parameters
    ----------
    pop_objs : np.ndarray, shape (N_all, M)
        All population objectives (normalized).
    last_indices : np.ndarray
        Indices of solutions in the last front.
    mu : int
        Number to select.

    Returns
    -------
    chosen : np.ndarray
        Indices into last_indices of selected solutions.
    """
    N = pop_objs.shape[0]

    dist = cdist(pop_objs, pop_objs)
    np.fill_diagonal(dist, np.inf)

    # Nearest neighbor distance
    min_dist = np.min(dist, axis=1)

    # D = 1/(min_dist + 2) for solutions in last front
    D = 1.0 / (min_dist[last_indices] + 2.0)

    # Sort ascending (smallest D = most isolated = most diverse)
    sorted_idx = np.argsort(D)

    return sorted_idx[:mu]
