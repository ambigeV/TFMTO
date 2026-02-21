"""
Pairwise Comparison Surrogate-Assisted Evolutionary Algorithm (PC-SAEA)

This module implements PC-SAEA for computationally expensive multi-objective optimization.
It uses a Probabilistic Neural Network (PNN) based pairwise comparison surrogate that
classifies whether one solution is better than another, combined with adaptive selection
strategies based on surrogate reliability.

References
----------
    [1] H. Wang, Y. Jin, C. Sun, and J. Deng. A pairwise comparison based surrogate-
        assisted evolutionary algorithm for expensive multi-objective optimization.
        Swarm and Evolutionary Computation, 2023, 80: 101323.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class PCSAEA:
    """
    Pairwise Comparison Surrogate-Assisted Evolutionary Algorithm for expensive
    multi-objective optimization using PNN-based pairwise comparison surrogates.
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100,
                 delta=0.8, gmax=3000, spread=0.1925,
                 save_data=True, save_path='./Data', name='PC-SAEA', disable_tqdm=True):
        """
        Initialize PC-SAEA algorithm.

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
        delta : float, optional
            Reliability threshold for surrogate (default: 0.8)
        gmax : int, optional
            Number of surrogate evaluations per iteration (default: 3000)
        spread : float, optional
            PNN RBF spread parameter (default: 0.1925)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'PC-SAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.delta = delta
        self.gmax = gmax
        self.spread = spread
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the PC-SAEA algorithm.

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

        if self.n_initial is None:
            n_initial_per_task = [max(11 * dims[i] - 1, self.n) for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        mu_est = 5  # approximate candidates per iteration

        # Initialize populations
        pop_decs = []
        pop_objs = []
        for i in range(nt):
            N = n_per_task[i]
            if decs[i].shape[0] <= N:
                pop_decs.append(decs[i].copy())
                pop_objs.append(objs[i].copy())
            else:
                _, fitness = _env_selection(objs[i], N)
                idx = np.argsort(fitness)[:N]
                pop_decs.append(decs[i][idx].copy())
                pop_objs.append(objs[i][idx].copy())

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                D = dims[i]
                N = n_per_task[i]
                rate = nfes_per_task[i] / max_nfes_per_task[i]

                # ===== Step 1: Compute fitness and create training data =====
                train_input, train_output, Pa, Pmid = _cal_fitness_pc(
                    pop_objs[i], pop_decs[i], rate
                )

                if train_input.shape[0] < 4 or Pa.shape[0] < 1:
                    continue

                # ===== Step 2: Split train/test =====
                tr_in, tr_out, te_in, te_out = _data_process(
                    train_input, train_output
                )

                if tr_in.shape[0] < 2:
                    continue

                # ===== Step 3: Train PNN pairwise comparison model =====
                pnn = _PNN(self.spread)
                pnn.train(tr_in, tr_out, D)

                # ===== Step 4: Assess reliability =====
                error1, error2 = _assess_reliability(
                    pnn, te_in, te_out, D, Pa
                )

                # ===== Step 5: Surrogate-assisted selection =====
                candidates = _surrogate_assisted_selection(
                    pnn, error1, error2, Pmid, Pa, D, self.gmax, self.delta
                )

                if candidates is None or candidates.shape[0] == 0:
                    continue

                # Remove duplicates
                candidates = remove_duplicates(candidates, decs[i])
                if candidates.shape[0] == 0:
                    continue

                # ===== Step 6: Evaluate and update =====
                cand_objs, _ = evaluation_single(problem, candidates, i)

                decs[i] = np.vstack([decs[i], candidates])
                objs[i] = np.vstack([objs[i], cand_objs])

                # Environmental selection for working population
                merged_decs = np.vstack([pop_decs[i], candidates])
                merged_objs = np.vstack([pop_objs[i], cand_objs])
                sel_idx, _ = _env_selection(merged_objs, N)
                pop_decs[i] = merged_decs[sel_idx]
                pop_objs[i] = merged_objs[sel_idx]

                nfes_per_task[i] += candidates.shape[0]
                pbar.update(candidates.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=mu_est)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# Probabilistic Neural Network for Pairwise Comparison
# =============================================================================

class _PNN:
    """
    Probabilistic Neural Network for pairwise comparison classification.
    Uses RBF kernel density estimation per class.
    """

    def __init__(self, spread=0.1925):
        self.spread = spread
        self.train_X = None
        self.train_Y = None
        self.classes = None

    def train(self, train_input, train_output, D):
        """
        Train PNN on pairwise comparisons.

        Creates N*(N-1) pairwise training samples from N solutions,
        where input is [x_i, x_j] (2D dims) and output is 2 if i is
        better (lower class), 1 otherwise.

        Parameters
        ----------
        train_input : np.ndarray
            Training features, shape (N, D+1) where last col is fitness
        train_output : np.ndarray
            Class labels, shape (N,), values 1 or 2
        D : int
            Number of decision variables
        """
        N = train_input.shape[0]
        decs = train_input[:, :D]

        # Create all pairwise comparisons (excluding self)
        pairs_X = []
        pairs_Y = []

        for ii in range(N):
            for jj in range(N):
                if ii == jj:
                    continue
                pair_input = np.concatenate([decs[ii], decs[jj]])
                if train_output[ii] < train_output[jj]:
                    pairs_Y.append(2)  # i is better
                else:
                    pairs_Y.append(1)  # j is better or equal
                pairs_X.append(pair_input)

        self.train_X = np.array(pairs_X)
        self.train_Y = np.array(pairs_Y)
        self.classes = np.unique(self.train_Y)

    def _predict_raw(self, X):
        """
        Predict class using PNN (RBF kernel density estimation).

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (N, 2*D)

        Returns
        -------
        predictions : np.ndarray
            Predicted classes, shape (N,)
        """
        N = X.shape[0]
        predictions = np.zeros(N, dtype=int)

        # Precompute class masks
        masks = {}
        for c in self.classes:
            masks[c] = self.train_Y == c

        for ii in range(N):
            dists_sq = np.sum((self.train_X - X[ii]) ** 2, axis=1)
            rbf_vals = np.exp(-dists_sq / (2 * self.spread ** 2))

            best_class = self.classes[0]
            best_score = -1.0
            for c in self.classes:
                score = np.mean(rbf_vals[masks[c]])
                if score > best_score:
                    best_score = score
                    best_class = c

            predictions[ii] = best_class

        return predictions

    def predict(self, candidate_decs, D, Pa, flag=0):
        """
        Predict pairwise comparison with conflict detection.

        For each candidate, compare against preferred solutions (cycling).
        Forward (candidate vs ref) and reverse (ref vs candidate) predictions
        are made. Conflicts (disagreements) are detected.

        Parameters
        ----------
        candidate_decs : np.ndarray
            Candidate decisions, shape (N, D)
        D : int
            Number of decision variables
        Pa : np.ndarray
            Preferred solutions, shape (K, D)
        flag : int
            0 = return raw difference, 1 = mark conflicts as 1.5

        Returns
        -------
        labels : np.ndarray
            Prediction labels, shape (N,)
            flag=0: values in {-1, 0, 1}
            flag=1: values in {-1, 1, 1.5}
        """
        N = candidate_decs.shape[0]
        n_pref = Pa.shape[0]

        if n_pref == 0:
            return np.zeros(N)

        # Cycle through preferred solutions
        pref_idx = np.array([(ii + n_pref) % n_pref for ii in range(N)])
        pref_decs = Pa[pref_idx, :D]

        # Forward: [candidate, reference]
        forward_input = np.hstack([candidate_decs[:, :D], pref_decs])
        Y_forward = self._predict_raw(forward_input)

        # Reverse: [reference, candidate]
        reverse_input = np.hstack([pref_decs, candidate_decs[:, :D]])
        Y_reverse = self._predict_raw(reverse_input)

        result = Y_forward - Y_reverse  # {-1, 0, 1}

        if flag == 1:
            result = result.astype(float)
            result[result == 0] = 1.5

        return result.astype(float)


# =============================================================================
# Fitness Calculation with SDE Diversity
# =============================================================================

def _cal_fitness_pc(pop_objs, pop_decs, rate):
    """
    Compute convergence-diversity fitness and create training data.

    Uses dominance-based ranking (R) + SDE diversity (D) with
    adaptive weighting: Fitness = rate*R + (1-rate)*D.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (N, M)
    pop_decs : np.ndarray
        Decisions, shape (N, D)
    rate : float
        Progress ratio (FE/maxFE), 0=diversity focus, 1=convergence focus

    Returns
    -------
    train_input : np.ndarray
        Training features [decs, fitness], shape (N/2, D+1)
    train_output : np.ndarray
        Class labels (1=bad, 2=good), shape (N/2,)
    Pa : np.ndarray
        Preferred solutions (top 25%), shape (N/4, D)
    Pmid : np.ndarray
        Boundary solutions, shape (~N/4, D)
    """
    N, M = pop_objs.shape
    D = pop_decs.shape[1]

    if N < 4:
        return np.zeros((0, D + 1)), np.zeros(0), pop_decs, pop_decs

    # Normalize objectives
    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    obj_range = zmax - zmin
    obj_range[obj_range < 1e-10] = 1.0
    pop_obj_norm = (pop_objs - zmin) / obj_range

    # Dominance detection
    dominate = np.zeros((N, N), dtype=bool)
    for p in range(N):
        for q in range(p + 1, N):
            any_less = np.any(pop_objs[p] < pop_objs[q])
            any_greater = np.any(pop_objs[p] > pop_objs[q])
            if any_less and not any_greater:
                dominate[p, q] = True
            elif any_greater and not any_less:
                dominate[q, p] = True

    # Strength S(i) and Raw fitness R(i)
    S = np.sum(dominate, axis=1).astype(float)
    R = np.zeros(N)
    for p in range(N):
        dominators = np.where(dominate[:, p])[0]
        R[p] = np.sum(S[dominators])

    # Normalize R to [0,1]
    R_min, R_max = R.min(), R.max()
    if R_max - R_min > 1e-10:
        R = (R - R_min) / (R_max - R_min)
    else:
        R = np.zeros(N)

    # SDE diversity metric
    k = max(1, int(np.floor(np.sqrt(N))))
    # Compute SDE-modified distances
    sde_dist = np.full((N, N), np.inf)
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # SDE: shift objectives so that dominated dimensions use the dominator's value
            shifted = np.maximum(pop_obj_norm[q], pop_obj_norm[p])
            sde_dist[p, q] = np.sqrt(np.sum((pop_obj_norm[p] - shifted) ** 2))

    sorted_dists = np.sort(sde_dist, axis=1)
    k_idx = min(k, N - 1) - 1
    D_diversity = 1.0 / (sorted_dists[:, k_idx] + 2.0)

    # Normalize D to [0,1]
    D_min, D_max = D_diversity.min(), D_diversity.max()
    if D_max - D_min > 1e-10:
        D_diversity = (D_diversity - D_min) / (D_max - D_min)

    # Combined fitness
    fitness = rate * R + (1.0 - rate) * D_diversity

    # Sort and select training data
    sorted_idx = np.argsort(fitness)
    n_good = int(np.ceil(N / 4))
    n_bad = int(np.ceil(N / 4))

    good_idx = sorted_idx[:n_good]
    bad_idx = sorted_idx[-(n_bad):]

    # Training data: good (class 2) + bad (class 1)
    good_input = np.hstack([pop_decs[good_idx], fitness[good_idx].reshape(-1, 1)])
    bad_input = np.hstack([pop_decs[bad_idx], fitness[bad_idx].reshape(-1, 1)])

    train_input = np.vstack([good_input, bad_input])
    train_output = np.concatenate([
        np.full(n_good, 2),
        np.full(n_bad, 1)
    ])

    # Preferred solutions (Pa) and boundary solutions (Pmid)
    Pa = pop_decs[good_idx]
    mid_start = max(0, n_good - int(np.floor(N / 8)))
    mid_end = min(N, n_good + int(np.floor(N / 8)))
    Pmid = pop_decs[sorted_idx[mid_start:mid_end]]

    return train_input, train_output, Pa, Pmid


# =============================================================================
# Data Processing
# =============================================================================

def _data_process(train_input, train_output):
    """
    Stratified train/test split (75%/25%).

    Parameters
    ----------
    train_input : np.ndarray
        Features, shape (N, D+1)
    train_output : np.ndarray
        Labels, shape (N,)

    Returns
    -------
    tr_in, tr_out, te_in, te_out : np.ndarray
        Training and test splits
    """
    idx_good = np.where(train_output > 1)[0]
    idx_bad = np.where(train_output <= 1)[0]

    n_train_good = max(1, int(np.ceil(0.75 * len(idx_good))))
    n_train_bad = max(1, int(np.ceil(0.75 * len(idx_bad))))

    perm_good = np.random.permutation(len(idx_good))
    perm_bad = np.random.permutation(len(idx_bad))

    train_good = idx_good[perm_good[:n_train_good]]
    test_good = idx_good[perm_good[n_train_good:]]
    train_bad = idx_bad[perm_bad[:n_train_bad]]
    test_bad = idx_bad[perm_bad[n_train_bad:]]

    train_idx = np.concatenate([train_good, train_bad])
    test_idx = np.concatenate([test_good, test_bad])

    tr_in = train_input[train_idx]
    tr_out = train_output[train_idx]

    if len(test_idx) > 0:
        te_in = train_input[test_idx]
        te_out = train_output[test_idx]
    else:
        te_in = tr_in
        te_out = tr_out

    return tr_in, tr_out, te_in, te_out


# =============================================================================
# Reliability Assessment
# =============================================================================

def _assess_reliability(pnn, te_in, te_out, D, Pa):
    """
    Assess surrogate reliability on test set.

    Parameters
    ----------
    pnn : _PNN
        Trained PNN model
    te_in : np.ndarray
        Test features, shape (N_test, D+1)
    te_out : np.ndarray
        Test labels, shape (N_test,)
    D : int
        Number of decision variables
    Pa : np.ndarray
        Preferred solutions

    Returns
    -------
    error1 : float
        Error rate for good solutions (class 2)
    error2 : float
        Error rate for bad solutions (class 1)
    """
    if te_in.shape[0] == 0:
        return 1.0, 1.0

    # Predict on test set (flag=1 to mark conflicts)
    labels = pnn.predict(te_in[:, :D], D, Pa, flag=1)

    # error1: accuracy on good solutions (class 2)
    good_mask = te_out > 1
    if np.sum(good_mask) > 0:
        # Good solutions should have label=1 (candidate better than reference)
        # With flag=1: valid predictions have label in {-1, 1}, conflicts are 1.5
        valid_good = labels[good_mask] != 1.5
        if np.sum(valid_good) > 0:
            correct_good = labels[good_mask][valid_good] == 1
            error1 = 1.0 - np.mean(correct_good)
        else:
            error1 = 1.0
    else:
        error1 = 1.0

    # error2: accuracy on bad solutions (class 1)
    bad_mask = te_out <= 1
    if np.sum(bad_mask) > 0:
        valid_bad = labels[bad_mask] != 1.5
        if np.sum(valid_bad) > 0:
            correct_bad = labels[bad_mask][valid_bad] == -1
            error2 = 1.0 - np.mean(correct_bad)
        else:
            error2 = 1.0
    else:
        error2 = 1.0

    return error1, error2


# =============================================================================
# Surrogate-Assisted Selection
# =============================================================================

def _surrogate_assisted_selection(pnn, error1, error2, Pmid, Pa, D, gmax, delta):
    """
    Select candidates using surrogate with adaptive strategy.

    Path A (error1 < 1-delta): Exploit - select solutions model identifies as good
    Path B (error2 < 1-delta): Explore - select solutions model identifies as bad
    Path C: Random selection when model unreliable

    Parameters
    ----------
    pnn : _PNN
        Trained PNN model
    error1 : float
        Error rate for good class
    error2 : float
        Error rate for bad class
    Pmid : np.ndarray
        Boundary solutions for GA parents
    Pa : np.ndarray
        Preferred solutions
    D : int
        Decision dimension
    gmax : int
        Maximum surrogate evaluations
    delta : float
        Reliability threshold

    Returns
    -------
    candidates : np.ndarray or None
        Selected candidate solutions
    """
    n_pref = Pa.shape[0]
    if n_pref == 0:
        return None

    wmax = max(1, gmax // n_pref)

    # Generate initial candidates
    parents = np.vstack([Pmid[:, :D], Pa[:, :D]])
    np.random.shuffle(parents)
    candidates = ga_generation(parents, muc=15, mum=5)

    labels = pnn.predict(candidates[:, :D], D, Pa, flag=0)

    threshold = 1 - delta  # 0.2

    if error1 < threshold:
        # Path A: Exploit good solutions
        good_decs = np.zeros((wmax, D))
        good_labels = np.zeros(wmax)

        for it in range(wmax):
            sorted_idx = np.argsort(-labels)  # descending
            good_decs[it] = candidates[sorted_idx[0], :D]
            good_labels[it] = labels[sorted_idx[0]]

            # Select top as parents for next generation
            n_select = min(n_pref, len(sorted_idx))
            new_parents = np.vstack([candidates[sorted_idx[:n_select], :D], Pa[:, :D]])
            np.random.shuffle(new_parents)
            candidates = ga_generation(new_parents, muc=15, mum=5)
            labels = pnn.predict(candidates[:, :D], D, Pa, flag=0)

        # Post-processing: select high-confidence good solutions
        confident_mask = good_labels >= 0.95
        n_confident = np.sum(confident_mask)

        if n_confident == 0 or n_confident > max(1, n_pref // 2):
            # Fallback: select top n_pref/2
            n_select = max(1, n_pref // 2)
            top_idx = np.argsort(-good_labels)[:n_select]
            return good_decs[top_idx]
        else:
            return good_decs[confident_mask]

    elif error2 < threshold:
        # Path B: Explore bad solutions
        bad_decs = np.zeros((wmax, D))
        bad_labels = np.zeros(wmax)

        for it in range(wmax):
            sorted_idx = np.argsort(labels)  # ascending
            bad_decs[it] = candidates[sorted_idx[0], :D]
            bad_labels[it] = labels[sorted_idx[0]]

            n_select = min(n_pref, len(sorted_idx))
            new_parents = np.vstack([candidates[sorted_idx[:n_select], :D], Pa[:, :D]])
            np.random.shuffle(new_parents)
            candidates = ga_generation(new_parents, muc=15, mum=5)
            labels = pnn.predict(candidates[:, :D], D, Pa, flag=0)

        # Post-processing: select high-confidence bad solutions
        confident_mask = bad_labels <= -0.95
        n_confident = np.sum(confident_mask)

        if n_confident == 0 or n_confident > max(1, n_pref // 2):
            n_select = max(1, n_pref // 2)
            top_idx = np.argsort(bad_labels)[:n_select]
            return bad_decs[top_idx]
        else:
            return bad_decs[confident_mask]

    else:
        # Path C: Random selection (unreliable model)
        idx = np.random.randint(0, candidates.shape[0])
        return candidates[idx:idx + 1, :D]


# =============================================================================
# Environmental Selection (SPEA2-based)
# =============================================================================

def _env_selection(pop_objs, N):
    """
    SPEA2-based environmental selection.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (n_total, M)
    N : int
        Target size

    Returns
    -------
    selected : np.ndarray
        Indices of selected solutions
    fitness : np.ndarray
        Fitness values
    """
    n_total = pop_objs.shape[0]
    fitness = spea2_fitness(pop_objs)

    if n_total <= N:
        return np.arange(n_total), fitness

    # Select non-dominated (fitness < 1)
    next_mask = fitness < 1.0
    n_selected = np.sum(next_mask)

    if n_selected < N:
        sorted_idx = np.argsort(fitness)
        selected = sorted_idx[:N]
    elif n_selected > N:
        candidates = np.where(next_mask)[0]
        kept = spea2_truncation_fast(pop_objs[candidates], N)
        selected = candidates[kept]
    else:
        selected = np.where(next_mask)[0]

    return selected, fitness[selected]
