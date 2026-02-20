"""
Performance Indicator-based Evolutionary Algorithm (PIEA)

This module implements PIEA for computationally expensive high-dimensional
multi/many-objective optimization. It adaptively selects among three performance
indicators (SDE, I_epsilon+, Minkowski distance) to train an SVR surrogate model,
then uses DE-based model optimization and hierarchical evaluation to guide the search.

References
----------
    [1] Y. Li, W. Li, S. Li, and Y. Zhao. A performance indicator-based
        evolutionary algorithm for expensive high-dimensional multi-/many-
        objective optimization. Information Sciences, 2024: 121045.

Notes
-----
Author: Jiangtao Shen (DDMTOLab implementation)
Date: 2026.02.16
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.STMO.NSGA_II_SDR import nd_sort_sdr
import warnings

warnings.filterwarnings("ignore")


class PIEA:
    """
    Performance Indicator-based Evolutionary Algorithm for expensive
    multi/many-objective optimization.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=None,
                 eta=5, r_max=20, tau=20,
                 save_data=True, save_path='./Data', name='PIEA', disable_tqdm=True):
        """
        Initialize PIEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: same as n)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (default: 100)
        eta : int, optional
            Number of pre-selected survivors (default: 5)
        r_max : int, optional
            Maximum repeat time of offspring generation (default: 20)
        tau : int, optional
            Window width for history list (default: 20)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'PIEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n if n is not None else 100
        self.eta = eta
        self.r_max = r_max
        self.tau = tau
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the PIEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        n_per_task = par_list(self.n, nt)

        # Set default initial samples: same as population size (NI = N in MATLAB)
        if self.n_initial is None:
            n_initial_per_task = n_per_task.copy()
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Archive A: all evaluated solutions (decs + objs)
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]

        # History tracking
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, interval=1)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, interval=1)

        # Initialize indicator history list for each task
        indicators_per_task = []
        for _ in range(nt):
            indicators_per_task.append(_init_indicators(self.tau))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                NI = n_per_task[i]
                indicators = indicators_per_task[i]

                # Estimate Pareto front shape
                Lp = _shape_estimate(arc_objs[i], NI)

                # Choose the working indicator based on adaptive probabilities
                rand_val = np.random.rand()
                if rand_val < indicators[0]['Pw']:
                    fitness = _cal_fitness_sde(arc_objs[i], Lp)
                    flag = 0
                elif rand_val < indicators[0]['Pw'] + indicators[1]['Pw']:
                    fitness = _cal_fitness_epsilon(arc_objs[i], 0.05)
                    flag = 1
                else:
                    fitness = _cal_fitness_md(arc_objs[i], Lp)
                    flag = 2

                # Train SVR surrogate model on fitness
                model, scaler = _train_svr(arc_decs[i], fitness)

                # Model-based optimization using DE
                Dec = arc_decs[i].copy()
                n_arc = Dec.shape[0]

                # Select NI solutions from Dec for initial Arc
                perm = np.random.permutation(n_arc)[:NI]
                Arc = Dec[perm].copy()

                for r in range(1, self.r_max + 1):
                    # Tournament selection (higher fitness is better)
                    mating_pool = tournament_selection(2, NI, fitness)
                    # DE generation: parents selected from Dec, mutants from Arc and Dec
                    offspring_dec = _de_operator(Dec[mating_pool], Arc, Dec[np.random.permutation(n_arc)[:NI]])
                    # Predict fitness with surrogate
                    offspring_fit = _predict_svr(model, scaler, offspring_dec)

                    if r == 1:
                        Arc = offspring_dec.copy()
                        ArcFit = offspring_fit.copy()
                    else:
                        # Keep better solutions
                        better = ArcFit < offspring_fit
                        Arc[better] = offspring_dec[better]
                        ArcFit[better] = offspring_fit[better]

                # Pre-selection: select top eta solutions by surrogate fitness
                order = np.argsort(ArcFit)[::-1]  # descending
                eta_actual = min(self.eta, len(order))
                Arc_selected = Arc[order[:eta_actual]]

                # Difference comparison: select most different from existing population
                norm_a_dec = arc_decs[i].copy()  # already in [0,1]
                norm_dec = Arc_selected.copy()    # already in [0,1]
                distance = np.min(cdist(norm_dec, norm_a_dec), axis=1)
                best_idx = np.argmax(distance)

                new_dec = Arc_selected[best_idx:best_idx + 1]

                # Remove duplicates
                new_dec = remove_duplicates(new_dec, decs[i])
                if new_dec.shape[0] == 0:
                    # If duplicate, try others
                    for k in range(eta_actual):
                        idx = order[k]
                        candidate = Arc[idx:idx + 1]
                        candidate = remove_duplicates(candidate, decs[i])
                        if candidate.shape[0] > 0:
                            new_dec = candidate
                            break
                if new_dec.shape[0] == 0:
                    # Generate random solution as fallback
                    new_dec = np.random.rand(1, dims[i])
                    new_dec = remove_duplicates(new_dec, decs[i])
                    if new_dec.shape[0] == 0:
                        continue

                # Expensive evaluation
                new_obj, _ = evaluation_single(problem, new_dec, i)

                # Update archive
                arc_decs[i] = np.vstack([arc_decs[i], new_dec])
                arc_objs[i] = np.vstack([arc_objs[i], new_obj])
                decs[i] = np.vstack([decs[i], new_dec])
                objs[i] = np.vstack([objs[i], new_obj])

                nfes_per_task[i] += new_dec.shape[0]
                pbar.update(new_dec.shape[0])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

                # Hierarchical evaluation
                score = _hierarchical_evaluate(arc_objs[i])

                # Update indicator information
                _update_information(flag, score, indicators)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results


# =============================================================================
# Indicator Initialization and Update
# =============================================================================

def _init_indicators(tau):
    """
    Initialize indicator history list.

    Parameters
    ----------
    tau : int
        Window width for history list

    Returns
    -------
    indicators : list of dict
        Three indicator records (SDE, I_epsilon+, Minkowski distance)
    """
    return [
        {'method': 'Shift-based density',
         'Choose_record': np.ones(tau),
         'Win_record': np.ones(tau),
         'Pw': 1.0 / 3},
        {'method': 'I_epsilon+',
         'Choose_record': np.ones(tau),
         'Win_record': np.ones(tau),
         'Pw': 1.0 / 3},
        {'method': 'Minkowski distance',
         'Choose_record': np.ones(tau),
         'Win_record': np.ones(tau),
         'Pw': 1.0 / 3},
    ]


def _update_information(flag, score, indicators):
    """
    Update indicator choice and win records, then recalculate probabilities.

    Parameters
    ----------
    flag : int
        Index of the chosen indicator (0, 1, or 2)
    score : int
        Score from hierarchical evaluation (0, 1, or 2)
    indicators : list of dict
        Three indicator records to update in place
    """
    eps = np.finfo(float).eps

    # Update Choose_record: shift window and append
    for k in range(3):
        indicators[k]['Choose_record'] = np.append(indicators[k]['Choose_record'][1:],
                                                    1.0 if k == flag else 0.0)

    # Update Win_record
    for k in range(3):
        if score == 0:
            indicators[k]['Win_record'] = np.append(indicators[k]['Win_record'][1:], 0.0)
        else:
            indicators[k]['Win_record'] = np.append(indicators[k]['Win_record'][1:],
                                                     score / 2.0 if k == flag else 0.0)

    # Recalculate probabilities
    p = np.array([
        (eps + np.sum(indicators[k]['Win_record'])) / (eps + np.sum(indicators[k]['Choose_record']))
        for k in range(3)
    ])
    p = p / np.sum(p)
    for k in range(3):
        indicators[k]['Pw'] = p[k]


# =============================================================================
# Pareto Front Shape Estimation
# =============================================================================

def _shape_estimate(pop_obj, N):
    """
    Estimate the shape parameter p of the Pareto front.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (n, M)
    N : int
        Population size for non-dominated sorting

    Returns
    -------
    p : float
        Estimated Minkowski distance exponent
    """
    # Non-dominated sorting, keep front 1
    n = pop_obj.shape[0]
    front_no, _ = nd_sort(pop_obj, min(N, n))
    nd_mask = front_no == 1
    nd_objs = pop_obj[nd_mask]

    if nd_objs.shape[0] < 20:
        return 1.0

    # Normalization: min-max to [0, 1]
    obj_min = np.min(nd_objs, axis=0)
    obj_max = np.max(nd_objs, axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    norm_objs = (nd_objs - obj_min) / obj_range

    k = 1.5

    CP = np.array([0.27, 0.36, 0.43, 0.5, 0.57, 0.66, 0.75, 0.86, 1,
                    1.15, 1.35, 1.6, 2, 2.4, 3.1, 4.2, 6.5])
    Vp = np.zeros(len(CP))

    n_nd = norm_objs.shape[0]
    for idx, cp in enumerate(CP):
        # Generalized norm: (sum(x^p))^(1/p)
        Gp = np.sum(norm_objs ** cp, axis=1) ** (1.0 / cp)
        temp = np.sort(Gp)
        Q1 = temp[max(int(n_nd * 0.25) - 1, 0)]
        Q3 = temp[max(int(n_nd * 0.75) - 1, 0)]
        Max_val = Q3 + k * (Q3 - Q1)
        # Denoise using box plot
        Gp_clean = Gp[Gp <= Max_val]
        if len(Gp_clean) > 0:
            max_gp = np.max(Gp_clean)
            if max_gp > 0:
                Vp[idx] = np.std(Gp_clean / max_gp)
            else:
                Vp[idx] = np.inf
        else:
            Vp[idx] = np.inf

    best_idx = np.argmin(Vp)
    return CP[best_idx]


# =============================================================================
# Fitness Functions (Three Performance Indicators)
# =============================================================================

def _cal_fitness_sde(pop_obj, Lp):
    """
    Calculate fitness using Shift-based Density Estimation (SDE).

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    Lp : float
        Minkowski distance exponent from shape estimation

    Returns
    -------
    fitness : np.ndarray
        Fitness values, shape (N,)
    """
    N = pop_obj.shape[0]
    fmax = np.max(pop_obj, axis=0)
    fmin = np.min(pop_obj, axis=0)
    f_range = fmax - fmin
    f_range[f_range == 0] = 1.0
    norm_obj = (pop_obj - fmin) / f_range

    # Shift-based density: for each pair (i, j), shift j's objective
    # to max(obj_j, obj_i) then compute distance
    Dis = np.full((N, N), np.inf)
    for idx_i in range(N):
        SPopObj = np.maximum(norm_obj, norm_obj[idx_i])
        for idx_j in range(N):
            if idx_j != idx_i:
                Dis[idx_i, idx_j] = np.linalg.norm(norm_obj[idx_i] - SPopObj[idx_j])

    fitness = np.min(Dis, axis=1)

    # Scale to [0, 3]
    f_max = np.max(fitness)
    f_min = np.min(fitness)
    denom = f_max + np.finfo(float).eps - f_min
    fitness = 3.0 / denom * (fitness - f_min)

    # Minkowski distance to ideal point
    ideal = np.min(norm_obj, axis=0).reshape(1, -1)
    dis = cdist(norm_obj, ideal, metric='minkowski', p=Lp).flatten()
    dis_max = np.max(dis)
    dis_min = np.min(dis)
    dis_denom = dis_max + np.finfo(float).eps - dis_min
    dis_scaled = -3.0 / dis_denom * (dis - dis_min)

    # Replace low-fitness solutions with distance-based fitness
    low_mask = fitness < 1e-4
    fitness[low_mask] = dis_scaled[low_mask]

    # Apply tansig (tanh)
    fitness = np.tanh(fitness)

    return fitness


def _cal_fitness_epsilon(pop_obj, kappa):
    """
    Calculate fitness using I_epsilon+ indicator.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    kappa : float
        Fitness scaling factor

    Returns
    -------
    fitness : np.ndarray
        Fitness values, shape (N,)
    """
    N = pop_obj.shape[0]
    fmin = np.min(pop_obj, axis=0)
    fmax = np.max(pop_obj, axis=0)
    f_range = fmax - fmin
    f_range[f_range == 0] = 1.0
    norm_obj = (pop_obj - fmin) / f_range

    # I(i,j) = max_k (obj_i_k - obj_j_k)
    I = np.max(norm_obj[:, np.newaxis, :] - norm_obj[np.newaxis, :, :], axis=2)

    C = np.max(np.abs(I), axis=0)
    C[C == 0] = 1e-6

    fitness = np.sum(-np.exp(-I / C / kappa), axis=0) + 1

    return fitness


def _cal_fitness_md(pop_obj, Lp):
    """
    Calculate fitness using Minkowski distance to ideal point.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    Lp : float
        Minkowski distance exponent

    Returns
    -------
    fitness : np.ndarray
        Fitness values, shape (N,) (negated distance, higher is better)
    """
    N = pop_obj.shape[0]
    fmax = np.max(pop_obj, axis=0)
    fmin = np.min(pop_obj, axis=0)
    f_range = fmax - fmin
    f_range[f_range == 0] = 1.0
    norm_obj = (pop_obj - fmin) / f_range

    ideal = np.min(norm_obj, axis=0).reshape(1, -1)
    dis = cdist(norm_obj, ideal, metric='minkowski', p=Lp).flatten()
    return -dis


# =============================================================================
# Surrogate Model (SVR)
# =============================================================================

def _train_svr(decs, fitness):
    """
    Train an SVR model on decision variables and fitness values.

    Parameters
    ----------
    decs : np.ndarray
        Decision variables, shape (N, D)
    fitness : np.ndarray
        Fitness values, shape (N,)

    Returns
    -------
    model : SVR
        Trained SVR model
    scaler : StandardScaler
        Feature scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(decs)
    model = SVR(kernel='rbf', gamma='auto')
    model.fit(X_scaled, fitness)
    return model, scaler


def _predict_svr(model, scaler, decs):
    """
    Predict fitness values using the trained SVR model.

    Parameters
    ----------
    model : SVR
        Trained SVR model
    scaler : StandardScaler
        Feature scaler
    decs : np.ndarray
        Decision variables, shape (N, D)

    Returns
    -------
    predictions : np.ndarray
        Predicted fitness values, shape (N,)
    """
    X_scaled = scaler.transform(decs)
    return model.predict(X_scaled)


# =============================================================================
# DE Operator (matching MATLAB OperatorDE)
# =============================================================================

def _de_operator(parents, donors1, donors2, CR=0.9, F=0.5):
    """
    Generate offspring using DE/current-to-rand/1 operator.

    Matches the behavior of MATLAB's OperatorDE(Problem, parents, donors1, donors2).

    Parameters
    ----------
    parents : np.ndarray
        Current population, shape (N, D)
    donors1 : np.ndarray
        First donor population, shape (N, D)
    donors2 : np.ndarray
        Second donor population, shape (N, D)
    CR : float
        Crossover rate
    F : float
        Differential weight

    Returns
    -------
    offspring : np.ndarray
        Offspring decision variables, shape (N, D)
    """
    N, D = parents.shape

    # DE/rand/1 mutation
    mutant = parents + F * (donors1 - donors2)

    # Binomial crossover
    mask = np.random.rand(N, D) < CR
    # Ensure at least one dimension is taken from mutant
    j_rand = np.random.randint(0, D, size=N)
    for idx in range(N):
        mask[idx, j_rand[idx]] = True

    offspring = np.where(mask, mutant, parents)

    # Clip to [0, 1]
    offspring = np.clip(offspring, 0.0, 1.0)

    return offspring


# =============================================================================
# Hierarchical Evaluation
# =============================================================================

def _hierarchical_evaluate(arc_objs):
    """
    Perform hierarchical evaluation of the newly added solution.

    The new solution is the last one in arc_objs. Check if it enters
    the non-dominated front (score=1), and if so, check if it survives
    SDR-based sorting within front 1 (score=2).

    Parameters
    ----------
    arc_objs : np.ndarray
        All objective values including the new solution, shape (N, M)

    Returns
    -------
    score : int
        0 if not in front 1, 1 if in front 1 but not surviving SDR,
        2 if surviving SDR within front 1
    """
    N = arc_objs.shape[0]

    # Standard non-dominated sorting
    front_no, _ = nd_sort(arc_objs, 1)

    score = 0
    if front_no[-1] == 1:
        # New solution is in front 1
        score = 1
        # Further check with SDR within front 1
        front1_mask = front_no == 1
        front1_objs = arc_objs[front1_mask]
        # Map the last solution's position in front1
        front1_sdr, _ = nd_sort_sdr(front1_objs, 1)
        # Check if the new solution (last in front1) is in SDR front 1
        if front1_sdr[-1] == 1:
            score = 2

    return score
