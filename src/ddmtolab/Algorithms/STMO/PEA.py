"""
Pareto-based Efficient Algorithm (PEA)

This module implements PEA for computationally expensive multi-objective optimization.
It uses Constrained Probabilistic Pareto Dominance (CPPD) sorting that accounts for
prediction uncertainty from Kriging models during evolutionary search, and selects
promising candidates for expensive re-evaluation using diversity-based strategies.

References
----------
    [1] T. Sonoda and M. Nakata. Multiple Objective Optimization Based on Kriging Surrogate Model with Constrained Probabilistic Pareto Dominance. IEEE Congress on Evolutionary Computation (CEC), 2020.

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
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
import warnings

warnings.filterwarnings("ignore")


class PEA:
    """
    Pareto-based Efficient Algorithm for expensive multi-objective optimization
    using Constrained Probabilistic Pareto Dominance and Kriging surrogates.
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
                 save_data=True, save_path='./Data', name='PEA', disable_tqdm=True):
        """
        Initialize PEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population/archive size per task (default: 100)
        wmax : int, optional
            Number of inner surrogate evolution generations (default: 20)
        mu : int, optional
            Number of candidate solutions per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'PEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
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
        """
        Execute the PEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
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

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize working populations (indices into database)
        pop_indices = []
        for i in range(nt):
            N = n_per_task[i]
            if decs[i].shape[0] <= N:
                pop_indices.append(np.arange(decs[i].shape[0]))
            else:
                idx = _pop_update(objs[i], None, N)
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

                # ===== Build GP models =====
                try:
                    obj_models = mo_gp_build(decs[i], objs[i], data_type)
                except Exception:
                    continue

                # ===== Evolutionary Search =====
                P_decs = decs[i][pop_indices[i]]
                P_objs = objs[i][pop_indices[i]]

                pop_decs, pop_objs, pop_mse = _evo_search(
                    P_decs, P_objs, obj_models, M, N, self.wmax, data_type
                )

                # ===== Candidate Selection =====
                candidates = _candi_select(
                    pop_decs, pop_objs, pop_mse,
                    decs[i], objs[i], self.mu, N
                )

                if candidates is None or candidates.shape[0] == 0:
                    continue

                # ===== Evaluate Candidates =====
                cand_objs, _ = evaluation_single(problem, candidates, i)

                # Update database
                decs[i] = np.vstack([decs[i], candidates])
                objs[i] = np.vstack([objs[i], cand_objs])

                # Update working population
                idx = _pop_update(objs[i], None, N)
                pop_indices[i] = np.where(idx)[0]

                nfes_per_task[i] += candidates.shape[0]
                pbar.update(candidates.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# Evolutionary Search
# =============================================================================

def _evo_search(P_decs, P_objs, obj_models, M, N, wmax, data_type):
    """
    Surrogate-based evolutionary search using CPPD environmental selection.

    Parameters
    ----------
    P_decs : np.ndarray
        Current population decisions, shape (N, D)
    P_objs : np.ndarray
        Current population objectives, shape (N, M)
    obj_models : list
        List of trained GP models
    M : int
        Number of objectives
    N : int
        Population size
    wmax : int
        Number of generations
    data_type : torch.dtype
        Data type for GP

    Returns
    -------
    pop_decs, pop_objs, pop_mse : np.ndarray
        Final population with predicted objectives and MSE
    """
    pop_decs = P_decs.copy()
    pop_objs = P_objs.copy()
    pop_mse = np.zeros_like(pop_objs)

    for w in range(wmax):
        # Generate offspring via GA
        off_decs = ga_generation(pop_decs, muc=20, mum=20)

        # Predict objectives and MSE for offspring
        off_objs, off_mse = _predict_with_mse(off_decs, obj_models, M, data_type)

        # Merge parent + offspring
        merged_decs = np.vstack([pop_decs, off_decs])
        merged_objs = np.vstack([pop_objs, off_objs])
        merged_mse = np.vstack([pop_mse, off_mse])

        # Environmental selection using CPPD
        selected = _environmental_selection(merged_objs, merged_mse, N)

        pop_decs = merged_decs[selected]
        pop_objs = merged_objs[selected]
        pop_mse = merged_mse[selected]

    return pop_decs, pop_objs, pop_mse


def _predict_with_mse(pop_decs, obj_models, M, data_type):
    """Predict objectives and MSE using GP models."""
    N = pop_decs.shape[0]
    pop_objs = np.zeros((N, M))
    pop_mse = np.zeros((N, M))

    for j in range(M):
        pred, std = gp_predict_single(obj_models[j], pop_decs, data_type)
        pop_objs[:, j] = pred.flatten()
        pop_mse[:, j] = (std.flatten()) ** 2

    return pop_objs, pop_mse


def gp_predict_single(gp, test_X, data_type=torch.float):
    """Predict using a single GP model. Wrapper around bo_utils gp_predict."""
    from ddmtolab.Methods.Algo_Methods.bo_utils import gp_predict
    return gp_predict(gp, test_X, data_type)


# =============================================================================
# Constrained Probabilistic Pareto Dominance (CPPD) Sorting
# =============================================================================

def _ndsort_cppd(pop_objs, obj_mse, n_sort):
    """
    Non-dominated sorting based on Constrained Probabilistic Pareto Dominance.

    For unconstrained problems, feasibility probability = 1 for all solutions.

    Parameters
    ----------
    pop_objs : np.ndarray
        Predicted objectives, shape (N, M)
    obj_mse : np.ndarray
        Prediction MSE, shape (N, M)
    n_sort : int
        Number of solutions to sort

    Returns
    -------
    front_no : np.ndarray
        Front assignment, shape (N,). inf = unassigned.
    max_fno : int
        Last assigned front number
    """
    N, M = pop_objs.shape

    # Build probabilistic dominance matrix
    # For each pair (i, j), check if i probabilistically dominates j
    # x_PD[i,j,k] = P(obj_j_k <= obj_i_k) -> probability j is at least as good as i on obj k
    # y_PD[i,j,k] = P(obj_i_k <= obj_j_k) -> probability i is at least as good as j on obj k

    # Pairwise differences and combined std
    # mean_diff[i,j,k] = obj_i_k - obj_j_k
    mean_diff = pop_objs[:, np.newaxis, :] - pop_objs[np.newaxis, :, :]  # (N, N, M)
    sigma_sum = obj_mse[:, np.newaxis, :] + obj_mse[np.newaxis, :, :]    # (N, N, M)
    sigma = np.sqrt(np.maximum(sigma_sum, 1e-20))

    # P(obj_i_k - obj_j_k <= 0) = Phi(-mean_diff / sigma) = Phi((obj_j - obj_i) / sigma)
    # This is probability i is better than or equal to j on objective k
    z = -mean_diff / sigma
    y_PD = norm.cdf(z)  # P(i <= j) on each objective
    x_PD = 1.0 - y_PD  # P(j <= i) on each objective

    # For unconstrained: feasibility = 1, so no modification needed
    # x_PD and y_PD already represent dominance probabilities

    # Dominance: i dominates j iff all(x_PD[i,j,:] <= y_PD[i,j,:]) and not all equal
    # This means: for all k, P(j <= i) <= P(i <= j), with strict somewhere
    # i.e., i is probabilistically at least as good as j on all objectives
    dominates = np.all(x_PD <= y_PD, axis=2) & ~np.all(
        np.abs(x_PD - y_PD) < 1e-10, axis=2)  # (N, N)
    np.fill_diagonal(dominates, False)

    # Non-dominated sorting
    front_no = np.full(N, np.inf)
    max_fno = 0
    n_assigned = 0

    # Count how many solutions dominate each solution
    dom_count = np.sum(dominates, axis=0)  # dom_count[j] = how many dominate j

    remaining = np.ones(N, dtype=bool)

    while n_assigned < min(n_sort, N):
        max_fno += 1
        current = np.where(remaining)[0]
        if len(current) == 0:
            break

        # Compute domination count among remaining
        sub_dom_count = np.sum(dominates[np.ix_(current, current)], axis=0)

        # Find minimum domination count
        min_count = np.min(sub_dom_count)
        frontal = current[sub_dom_count == min_count]

        front_no[frontal] = max_fno
        remaining[frontal] = False
        n_assigned += len(frontal)

    return front_no, max_fno


def _environmental_selection(pop_objs, pop_mse, N):
    """
    Environmental selection using CPPD sorting + diversity truncation.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (n_total, M)
    pop_mse : np.ndarray
        MSE, shape (n_total, M)
    N : int
        Target size

    Returns
    -------
    selected : np.ndarray
        Indices of selected solutions
    """
    n_total = pop_objs.shape[0]
    front_no, max_fno = _ndsort_cppd(pop_objs, pop_mse, N)

    # Select all solutions from fronts before last
    selected = np.where(front_no < max_fno)[0]
    last_front = np.where(front_no == max_fno)[0]

    n_remaining = N - len(selected)

    if n_remaining <= 0:
        return selected[:N]

    if len(last_front) <= n_remaining:
        selected = np.concatenate([selected, last_front])
    elif max_fno == 1:
        # Only one front: truncation by distance
        chosen = spea2_truncation_fast(pop_objs[last_front], n_remaining)
        selected = np.concatenate([selected, last_front[chosen]])
    else:
        # Multiple fronts: diversity selection from last front
        chosen = _dist_selection(pop_objs[selected], pop_objs[last_front], n_remaining)
        selected = np.concatenate([selected, last_front[chosen]])

    return selected


# =============================================================================
# Candidate Selection
# =============================================================================

def _candi_select(pop_decs, pop_objs, pop_mse, db_decs, db_objs, mu, N):
    """
    Select mu candidate solutions for expensive re-evaluation.

    Parameters
    ----------
    pop_decs : np.ndarray
        Evolved population decisions
    pop_objs : np.ndarray
        Evolved population predicted objectives
    pop_mse : np.ndarray
        Evolved population prediction MSE
    db_decs : np.ndarray
        All evaluated decisions (database)
    db_objs : np.ndarray
        All evaluated objectives (database)
    mu : int
        Number of candidates to select
    N : int
        Population size

    Returns
    -------
    candidates : np.ndarray or None
        Selected candidate decisions, shape (n_selected, D)
    """
    n_pop = pop_decs.shape[0]

    # Check for duplicates against database
    if db_decs.shape[0] > 0:
        dist_to_db = cdist(pop_decs, db_decs)
        min_dist = np.min(dist_to_db, axis=1)
        novel_mask = min_dist > 1e-5
    else:
        novel_mask = np.ones(n_pop, dtype=bool)

    n_novel = np.sum(novel_mask)
    if n_novel == 0:
        return None

    if n_novel <= mu:
        return pop_decs[novel_mask]

    # Keep only novel solutions
    novel_decs = pop_decs[novel_mask]
    novel_objs = pop_objs[novel_mask]
    novel_mse = pop_mse[novel_mask]

    # Normalize objectives
    all_objs = np.vstack([db_objs, novel_objs])
    zmin = np.min(all_objs, axis=0)
    zmax = np.max(all_objs, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-10)

    norm_objs = (novel_objs - zmin) / obj_range
    norm_mse = novel_mse / (obj_range ** 2)
    norm_db_objs = (db_objs - zmin) / obj_range

    # Build reference set from database (non-dominated front)
    n_db = db_objs.shape[0]
    front_no_db, _ = nd_sort(db_objs, n_db)
    ref_objs = norm_db_objs[front_no_db == 1]

    # CPPD sorting on novel solutions
    front_no, max_fno = _ndsort_cppd(norm_objs, norm_mse, mu)

    # Select from best fronts
    selected = []
    for fno in range(1, max_fno + 1):
        front_idx = np.where(front_no == fno)[0]

        if len(selected) + len(front_idx) <= mu:
            selected.extend(front_idx.tolist())
        else:
            # Need to select from this front with diversity
            n_need = mu - len(selected)
            if len(selected) == 0:
                # First front only, use truncation
                chosen = spea2_truncation_fast(norm_objs[front_idx], n_need)
                selected.extend(front_idx[chosen].tolist())
            else:
                # Use distance-based selection
                already_objs = norm_objs[np.array(selected)]
                chosen = _dist_selection(already_objs, norm_objs[front_idx], n_need)
                selected.extend(front_idx[chosen].tolist())
            break

    if len(selected) == 0:
        return None

    # Final duplicate check
    candidates = []
    for idx in selected:
        c = novel_decs[idx]
        if db_decs.shape[0] > 0:
            d = cdist(c.reshape(1, -1), db_decs)
            if np.min(d) > 1e-5:
                candidates.append(c)
        else:
            candidates.append(c)

    if len(candidates) == 0:
        return None

    return np.array(candidates)


# =============================================================================
# Population Update (for real evaluated solutions)
# =============================================================================

def _pop_update(pop_objs, pop_cons, N):
    """
    Standard non-dominated sorting based population update.

    Parameters
    ----------
    pop_objs : np.ndarray
        All evaluated objectives
    pop_cons : np.ndarray or None
        All evaluated constraints
    N : int
        Target population size

    Returns
    -------
    next_mask : np.ndarray
        Boolean mask for selected solutions
    """
    n_total = pop_objs.shape[0]
    if n_total <= N:
        return np.ones(n_total, dtype=bool)

    front_no, max_fno = nd_sort(pop_objs, N)

    next_mask = front_no < max_fno
    last_front = np.where(front_no == max_fno)[0]
    n_remaining = N - np.sum(next_mask)

    if n_remaining <= 0:
        return next_mask

    if len(last_front) <= n_remaining:
        next_mask[last_front] = True
    elif max_fno == 1:
        chosen = spea2_truncation_fast(pop_objs[last_front], n_remaining)
        next_mask[last_front[chosen]] = True
    else:
        selected_objs = pop_objs[next_mask]
        chosen = _dist_selection(selected_objs, pop_objs[last_front], n_remaining)
        next_mask[last_front[chosen]] = True

    return next_mask


# =============================================================================
# Diversity Helper Functions
# =============================================================================

def _dist_selection(selected_objs, candidate_objs, n_select):
    """
    Select n_select solutions from candidates maximizing min distance to selected.

    Parameters
    ----------
    selected_objs : np.ndarray
        Already selected objectives, shape (N1, M)
    candidate_objs : np.ndarray
        Candidate objectives, shape (N2, M)
    n_select : int
        Number to select

    Returns
    -------
    chosen : list
        Indices into candidate_objs
    """
    N2 = candidate_objs.shape[0]
    if N2 <= n_select:
        return list(range(N2))

    # Combined distance matrix
    all_objs = np.vstack([selected_objs, candidate_objs])
    dist = cdist(all_objs, all_objs)
    np.fill_diagonal(dist, np.inf)

    N1 = selected_objs.shape[0]
    chosen_set = set(range(N1))
    remaining = set(range(N1, N1 + N2))
    chosen = []

    for _ in range(n_select):
        if not remaining:
            break
        best_idx = None
        best_min_dist = -1.0

        for idx in remaining:
            min_d = min(dist[idx, c] for c in chosen_set)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx

        chosen_set.add(best_idx)
        remaining.remove(best_idx)
        chosen.append(best_idx - N1)

    return chosen
