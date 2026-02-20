"""
Dual-Surrogate Assisted Evolutionary Algorithm with Portfolio Strategy (DSAEA-PS)

This module implements DSAEA-PS for computationally expensive multi/many-objective optimization.
It uses two types of surrogates (Kriging for objective prediction and RBF for dominance relation
prediction) combined with a portfolio of three environmental selection strategies (IBEA, RVEA,
NSGA-II/CSDR) to balance convergence and diversity.

References
----------
    [1] J. Shen, P. Wang, Y. Tian, and H. Dong. A dual surrogate assisted evolutionary algorithm
        based on parallel search for expensive multi/many-objective optimization. Applied Soft
        Computing, 2023, 148: 110879.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.18
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Algorithms.STMO.RVEA import rvea_selection
import warnings

warnings.filterwarnings("ignore")


class DSAEA_PS:
    """
    Dual-Surrogate Assisted Evolutionary Algorithm with Portfolio Strategy for expensive
    multi/many-objective optimization.

    Uses Kriging models for objective prediction and an RBF model for dominance relation
    prediction, combined with three environmental selection strategies (IBEA, RVEA, CSDR).

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, wmax=20, mu=5,
                 save_data=True, save_path='./Data', name='DSAEA-PS', disable_tqdm=True):
        """
        Initialize DSAEA-PS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size (number of reference vectors) per task (default: 100)
        wmax : int, optional
            Number of inner surrogate evolution generations (default: 20)
        mu : int, optional
            Number of re-evaluated solutions per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'DSAEA-PS')
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
        Execute the DSAEA-PS algorithm.

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

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate uniformly distributed reference vectors for each task
        V0 = []
        for i in range(nt):
            v_i, actual_n = uniform_point(n_per_task[i], n_objs[i])
            V0.append(v_i)
            n_per_task[i] = actual_n

        # Generate identity matrix reference vectors (for diversity checking)
        v0 = [np.eye(n_objs[i]) for i in range(nt)]

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # A1: archive of all evaluated solutions
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]

        # History tracking
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, interval=self.mu)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, interval=self.mu)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                dim = dims[i]
                N = n_per_task[i]

                A1Dec = arc_decs[i].copy()
                A1Obj = arc_objs[i].copy()

                # Scale reference vectors by objective range
                obj_range = A1Obj.max(axis=0) - A1Obj.min(axis=0)
                obj_range = np.maximum(obj_range, 1e-10)
                V = V0[i] * obj_range
                v = v0[i] * obj_range

                # === Build dominance relation model (RBF on CSDR front numbers) ===
                DA1Obj = _normalize_csdr(A1Obj)
                front_no_dom, _ = _nd_sort_csdr(DA1Obj, len(A1Dec))
                dmodel = rbf_build(A1Dec, front_no_dom.astype(float))

                # === Build Kriging (GP) models for each objective ===
                models = mo_gp_build(A1Dec, A1Obj, data_type)

                # === Inner optimization: 3 parallel MOEAs on surrogates ===
                pop_dec_1, pop_obj_1 = _moea_inner(
                    A1Dec.copy(), models, self.wmax, N, M, data_type,
                    strategy='ibea'
                )
                pop_dec_2, pop_obj_2 = _moea_inner(
                    A1Dec.copy(), models, self.wmax, N, M, data_type,
                    strategy='rvea', V=V
                )
                pop_dec_3, pop_obj_3 = _moea_inner(
                    A1Dec.copy(), models, self.wmax, N, M, data_type,
                    strategy='csdr'
                )

                # === Combine results from all 3 MOEAs ===
                CPopDec = np.vstack([pop_dec_1, pop_dec_2, pop_dec_3])
                CPopObj = np.vstack([pop_obj_1, pop_obj_2, pop_obj_3])

                # Normalize combined objectives using CSDR
                CPopObj_norm = _normalize_csdr(CPopObj)
                FN, max_FN = _nd_sort_csdr(CPopObj_norm, np.inf)

                # === Predict dominance front numbers via RBF model ===
                NP = CPopDec.shape[0]
                FNO = np.zeros(NP)
                for j in range(NP):
                    FNO[j] = rbf_predict(dmodel, A1Dec, CPopDec[j:j + 1, :])

                # Map predicted front numbers to discrete levels using kmeans
                if max_FN > 1 and NP > max_FN:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=max_FN, n_init=10, random_state=0)
                    labels = km.fit_predict(FNO.reshape(-1, 1))
                    centers = km.cluster_centers_.flatten()
                    # Assign front number: cluster with smallest center gets front 1, etc.
                    sorted_center_idx = np.argsort(centers)
                    rank_map = np.zeros(max_FN, dtype=int)
                    for r, idx in enumerate(sorted_center_idx):
                        rank_map[idx] = r + 1
                    FNO_mapped = rank_map[labels].astype(float)
                else:
                    FNO_mapped = np.ones(NP)

                # === Combine FN and FNO into 2-objective problem ===
                # Row 0: FN + FNO_mapped (sum), Row 1: |FN - FNO_mapped| (difference)
                ss = np.column_stack([FN + FNO_mapped, np.abs(FN - FNO_mapped)])
                front_no_ss, _ = nd_sort(ss, NP)
                indexF1 = np.where(front_no_ss == 1)[0]

                # === Select mu infill points ===
                if len(indexF1) > self.mu:
                    PopNew = _se_ibea(CPopDec, CPopObj, A1Obj, self.mu)
                else:
                    PopNew = CPopDec[indexF1]

                # Remove duplicates
                PopNew = remove_duplicates(PopNew, decs[i])

                if PopNew.shape[0] > 0:
                    # Limit to remaining budget
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if PopNew.shape[0] > remaining:
                        PopNew = PopNew[:remaining]

                    # Re-evaluate with expensive function
                    new_objs, _ = evaluation_single(problem, PopNew, i)

                    # Update archive (merge and deduplicate)
                    arc_decs[i], arc_objs[i] = merge_archive(
                        arc_decs[i], arc_objs[i], PopNew, new_objs
                    )

                    # Update cumulative dataset
                    decs[i] = np.vstack([decs[i], PopNew])
                    objs[i] = np.vstack([objs[i], new_objs])

                    nfes_per_task[i] += PopNew.shape[0]
                    pbar.update(PopNew.shape[0])

                    append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results


# =============================================================================
# CSDR-based Non-Dominated Sorting (NDSort_CSDR)
# =============================================================================

def _normalize_csdr(objs):
    """
    Normalize objectives for CSDR-based sorting.

    Translates to origin and optionally scales by range if the range ratio
    is not too extreme (matching the MATLAB 0.05*max(range) < min(range) check).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (N, M)

    Returns
    -------
    norm_objs : np.ndarray
        Normalized objective values
    """
    zmin = objs.min(axis=0)
    zmax = objs.max(axis=0)
    norm_objs = objs - zmin
    obj_range = zmax - zmin
    if obj_range.max() > 0 and 0.05 * obj_range.max() < obj_range.min():
        norm_objs = norm_objs / np.maximum(obj_range, 1e-10)
    return norm_objs


def _nd_sort_csdr(objs, n_sort):
    """
    CSDR-based non-dominated sorting.

    Uses both Pareto dominance and angle-based dominance relation to sort
    solutions into fronts, matching the MATLAB NDSort_CSDR implementation.

    Parameters
    ----------
    objs : np.ndarray
        Objective values (already normalized), shape (N, M)
    n_sort : float
        Number of solutions to sort (use np.inf for all)

    Returns
    -------
    front_no : np.ndarray
        Front number for each solution, shape (N,)
    max_fno : int
        Maximum front number assigned
    """
    N, M = objs.shape
    if N == 0:
        return np.array([]), 0

    # Compute pairwise cosine similarity (matching MATLAB: 1 - pdist2(...,'cosine'))
    cosine = 1 - cdist(objs, objs, metric='cosine')
    np.fill_diagonal(cosine, 0)
    angle = np.arccos(np.clip(cosine, -1, 1))

    # Compute threshold angle: 50th percentile of minimum angles
    min_angles = np.sort(np.unique(np.min(angle, axis=1)))
    idx = min(int(np.ceil(0.5 * N)) - 1, len(min_angles) - 1)
    min_a = min_angles[max(0, idx)]
    min_a = max(min_a, 1e-10)

    # Theta matrix for angle-based dominance
    theta = np.maximum(1.0, angle / min_a)

    # NormP: sum of objectives for each solution
    norm_p = np.sum(objs, axis=1)

    # Build dominance matrix
    dominate = np.zeros((N, N), dtype=bool)

    for ii in range(N - 1):
        for jj in range(ii + 1, N):
            # Check Pareto dominance (matching MATLAB IfDominate)
            a_leq_b = np.all(objs[ii] <= objs[jj])
            b_leq_a = np.all(objs[jj] <= objs[ii])
            are_equal = np.all(objs[ii] == objs[jj])

            if a_leq_b and not are_equal:
                dominate[ii, jj] = True
            elif b_leq_a and not are_equal:
                dominate[jj, ii] = True

            # Check angle-based dominance
            if norm_p[ii] * theta[ii, jj] < norm_p[jj]:
                dominate[ii, jj] = True
            elif norm_p[jj] * theta[jj, ii] < norm_p[ii]:
                dominate[jj, ii] = True

    # Assign front numbers
    front_no = np.full(N, np.inf)
    max_fno = 0
    n_target = min(n_sort, N) if np.isfinite(n_sort) else N

    while np.sum(front_no != np.inf) < n_target:
        max_fno += 1
        # Solutions not dominated by any unassigned solution
        current = np.where(
            (~np.any(dominate, axis=0)) & (front_no == np.inf)
        )[0]
        if len(current) == 0:
            break
        front_no[current] = max_fno
        dominate[current, :] = False

    return front_no, max_fno


# =============================================================================
# Inner MOEA on Surrogates (MOEAK)
# =============================================================================

def _moea_inner(pop_decs, models, wmax, N, M, data_type, strategy='ibea', V=None):
    """
    Run inner surrogate-based MOEA for wmax generations.

    Parameters
    ----------
    pop_decs : np.ndarray
        Initial population decisions, shape (n, d)
    models : list
        List of M GP models for objective prediction
    wmax : int
        Number of inner generations
    N : int
        Population size
    M : int
        Number of objectives
    data_type : torch.dtype
        Data type for GP prediction
    strategy : str
        Selection strategy: 'ibea', 'rvea', or 'csdr'
    V : np.ndarray, optional
        Reference vectors (required for 'rvea')

    Returns
    -------
    pop_decs : np.ndarray
        Final population decisions
    pop_objs : np.ndarray
        Final predicted objectives
    """
    for w in range(1, wmax + 1):
        # Generate offspring via GA operators
        off_decs = ga_generation(pop_decs, muc=20.0, mum=20.0)

        # Merge parent and offspring
        pop_decs = np.vstack([pop_decs, off_decs])

        # Predict objectives using GP models
        pop_objs = mo_gp_predict(models, pop_decs, data_type, mse=False)

        # Environmental selection
        if strategy == 'ibea':
            index = _es_ibea(pop_objs, N)
        elif strategy == 'rvea':
            theta = (w / wmax) ** 2
            cons_zero = np.zeros((pop_decs.shape[0], 1))
            index = rvea_selection(pop_objs, cons_zero, V, theta)
        elif strategy == 'csdr':
            index = _es_csdr(pop_objs, N)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        pop_decs = pop_decs[index]
        pop_objs = pop_objs[index]

    return pop_decs, pop_objs


# =============================================================================
# IBEA Environmental Selection
# =============================================================================

def _es_ibea(pop_objs, N):
    """
    IBEA-based environmental selection.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objective values, shape (n, M)
    N : int
        Number to select

    Returns
    -------
    index : np.ndarray
        Selected indices
    """
    n = pop_objs.shape[0]
    if n <= N:
        return np.arange(n)

    fitness, I, C = ibea_fitness(pop_objs, kappa=0.05)
    choose = list(range(n))

    while len(choose) > N:
        fit_values = fitness[choose]
        min_idx = np.argmin(fit_values)
        to_remove = choose[min_idx]

        if C[to_remove] > 1e-10:
            fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

        choose.pop(min_idx)

    return np.array(choose)


# =============================================================================
# CSDR Environmental Selection (ES_CSDR from MATLAB)
# =============================================================================

def _es_csdr(pop_objs, N):
    """
    NSGA-II/CSDR-based environmental selection.

    Uses CSDR non-dominated sorting + crowding distance.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objective values, shape (n, M)
    N : int
        Number to select

    Returns
    -------
    index : np.ndarray
        Selected indices
    """
    n = pop_objs.shape[0]
    if n <= N:
        return np.arange(n)

    # Normalize for CSDR
    norm_objs = _normalize_csdr(pop_objs)

    # CSDR-based sorting
    front_no, max_fno = _nd_sort_csdr(norm_objs, N)

    # Select solutions from fronts < max_fno
    Next = front_no < max_fno
    remaining_needed = N - np.sum(Next)

    if remaining_needed > 0:
        # Use crowding distance for the last front
        last_front = np.where(front_no == max_fno)[0]
        cd = crowding_distance(norm_objs, front_no)
        sorted_last = last_front[np.argsort(-cd[last_front])]
        Next[sorted_last[:remaining_needed]] = True

    return np.where(Next)[0]


# =============================================================================
# Se_IBEA: Select infill points using IBEA fitness
# =============================================================================

def _se_ibea(pop_decs, pop_objs, A1Obj, mu):
    """
    Select mu infill points from predicted population using IBEA fitness,
    combining with archive data for normalization.

    Parameters
    ----------
    pop_decs : np.ndarray
        Population decisions, shape (NP, D)
    pop_objs : np.ndarray
        Population predicted objectives, shape (NP, M)
    A1Obj : np.ndarray
        Archive objectives, shape (NA, M)
    mu : int
        Number to select

    Returns
    -------
    PopNew : np.ndarray
        Selected decision variables, shape (mu, D)
    """
    # Normalize using combined range
    zmin = np.minimum(A1Obj.min(axis=0), pop_objs.min(axis=0))
    zmax = np.maximum(A1Obj.max(axis=0), pop_objs.max(axis=0))
    obj_range = zmax - zmin

    A1Obj_norm = A1Obj - zmin
    PopObj_norm = pop_objs - zmin
    if obj_range.max() > 0 and 0.05 * obj_range.max() < obj_range.min():
        A1Obj_norm = A1Obj_norm / np.maximum(obj_range, 1e-10)
        PopObj_norm = PopObj_norm / np.maximum(obj_range, 1e-10)

    # CSDR sort to get front-1 of archive and population separately
    FN_A1, _ = _nd_sort_csdr(A1Obj_norm, np.inf)
    FN_pop, _ = _nd_sort_csdr(PopObj_norm, np.inf)

    A1Obj_f1 = A1Obj_norm[FN_A1 == 1]
    PopObj_f1 = PopObj_norm[FN_pop == 1]
    PopDec_f1 = pop_decs[FN_pop == 1]

    if PopObj_f1.shape[0] == 0:
        # Fallback: select mu from population by IBEA
        idx = _es_ibea(PopObj_norm, min(mu, pop_decs.shape[0]))
        return pop_decs[idx]

    # Combine front-1 populations for IBEA fitness computation
    CObj = np.vstack([PopObj_f1, A1Obj_f1])
    n_pop = PopObj_f1.shape[0]
    n_a1 = A1Obj_f1.shape[0]

    if n_pop <= mu:
        return PopDec_f1

    kappa = 0.05
    fitness, I, C = ibea_fitness(CObj, kappa=kappa)

    # Remove worst solutions, preferring to remove from pop_f1
    next_pop = list(range(n_pop))
    next_a1 = list(range(n_pop, n_pop + n_a1))
    all_next = next_pop + next_a1

    while len(next_pop) > mu:
        fit_values = fitness[all_next]
        min_idx = np.argmin(fit_values)
        to_remove_global = all_next[min_idx]

        if C[to_remove_global] > 1e-10:
            fitness += np.exp(-I[to_remove_global, :] / C[to_remove_global] / kappa)

        if to_remove_global < n_pop:
            next_pop.remove(to_remove_global)
        else:
            next_a1.remove(to_remove_global)
        all_next = next_pop + next_a1

    return PopDec_f1[next_pop]
