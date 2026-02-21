"""
Data-Driven Evolutionary Algorithm with Multi-Evolutionary Sampling Strategy (DDEA-MESS)

This module implements DDEA-MESS for expensive single-objective optimization problems.

References
----------
    [1] Yu, F., Gong, W., & Zhen, H. (2022). A data-driven evolutionary algorithm with
        multi-evolutionary sampling strategy for expensive optimization. Knowledge-Based
        Systems, 242, 108436.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.19
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class DDEA_MESS:
    """
    Data-Driven Evolutionary Algorithm with Multi-Evolutionary Sampling Strategy.

    Dynamically selects from three search strategies based on evaluation budget usage:
    1. Global search: DE/rand/1 prescreening on RBF model built from first min(N, 300) samples
    2. Local search: DE/best/1 on RBF model built from top tau samples by fitness
    3. Trust region search: Local optimization (L-BFGS-B) on RBF model around best solution
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, save_data=True,
                 save_path='./Data', name='DDEA-MESS', disable_tqdm=True):
        """
        Initialize DDEA-MESS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'DDEA-MESS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the DDEA-MESS algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Current working dataset
        current_decs = [decs[i].copy() for i in range(nt)]
        current_objs = [objs[i].copy() for i in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                X = current_decs[i]
                Y = current_objs[i]

                # Select strategy using MESS
                strategy_id = self._mess(nfes_per_task[i], 500)

                if strategy_id == 1:
                    candidate = self._strategy_global(X, Y, dim)
                elif strategy_id == 2:
                    candidate = self._strategy_local(X, Y, dim)
                else:
                    candidate = self._strategy_trust_region(X, Y, dim)

                # Ensure uniqueness
                candidate = self._ensure_uniqueness(candidate, X, dim)

                # Evaluate
                obj, _ = evaluation_single(problem, candidate, i)

                # Update dataset
                current_decs[i] = np.vstack([current_decs[i], candidate])
                current_objs[i] = np.vstack([current_objs[i], obj])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [current_decs[i].copy() for i in range(nt)]
        db_objs = [current_objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )

        return results

    def _mess(self, fes_used, fes_max):
        """
        Multi-Evolutionary Sampling Strategy selector.

        Dynamically computes probabilities for three strategies based on
        the ratio of used evaluations to total budget.

        Parameters
        ----------
        fes_used : int
            Number of function evaluations used (including initial samples)
        fes_max : int
            Maximum number of function evaluations (total budget)

        Returns
        -------
        int
            Strategy ID: 1 (global), 2 (local), or 3 (trust region)
        """
        ratio = fes_used / fes_max
        beta = (1 - ratio ** 3) ** 2
        alpha = abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))
        P3 = (1 - alpha) if alpha > 2 / 3 else 1 / 3
        P1 = (1 - P3) / 2
        r = np.random.rand()
        if r <= P1:
            return 1  # global search
        elif r <= 2 * P1:
            return 2  # local search
        else:
            return 3  # trust region search

    def _build_rbf_model(self, X, Y):
        """Build Gaussian RBF surrogate model."""
        Y_flat = Y.flatten()
        n_samples, dim = X.shape

        if n_samples > 1:
            dist_matrix = cdist(X, X, metric='euclidean')
            max_dist = dist_matrix.max()
            spread = max_dist / (dim * n_samples) ** (1.0 / dim)
        else:
            spread = 1.0

        try:
            rbf = RBFInterpolator(X, Y_flat, kernel='gaussian', epsilon=1.0 / spread)
        except Exception:
            rbf = RBFInterpolator(X, Y_flat, kernel='thin_plate_spline')

        def model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return rbf(x).reshape(-1, 1)

        return model

    def _strategy_global(self, X, Y, dim):
        """
        Strategy 1: Global search with DE/rand/1 prescreening.

        Builds RBF on first min(N, 300) samples (initial LHS provides good coverage).
        Runs 1 generation of DE/rand/1 with elite initialization from full database.
        """
        N = len(X)
        m = min(N, 300)

        # Build RBF on first m samples
        model = self._build_rbf_model(X[:m], Y[:m])

        # DE/rand/1 prescreening (1 generation)
        candidate = self._de_search(
            model, X, Y,
            lb=np.zeros(dim), ub=np.ones(dim),
            dim=dim, popsize=50, max_gen=1, mode='rand'
        )

        return candidate

    def _strategy_local(self, X, Y, dim):
        """
        Strategy 2: Local search with DE/best/1.

        Selects top tau = min(dim+25, N) solutions by fitness.
        Builds RBF on selected data within local bounds.
        Runs 10 generations of DE/best/1 with elite initialization from full database.
        """
        N = len(X)
        tau = min(dim + 25, N)

        # Select top tau solutions by fitness
        Y_flat = Y.flatten()
        idx = np.argsort(Y_flat)[:tau]
        X_local = X[idx]
        Y_local = Y[idx]

        lb_local = np.min(X_local, axis=0)
        ub_local = np.max(X_local, axis=0)

        # Handle degenerate bounds
        mask = (ub_local - lb_local) < 1e-10
        lb_local[mask] = np.maximum(0.0, lb_local[mask] - 0.05)
        ub_local[mask] = np.minimum(1.0, ub_local[mask] + 0.05)

        # Build RBF on local data
        model = self._build_rbf_model(X_local, Y_local)

        # DE/best/1 search (10 generations)
        candidate = self._de_search(
            model, X, Y,
            lb=lb_local, ub=ub_local,
            dim=dim, popsize=50, max_gen=10, mode='best'
        )

        return candidate

    def _strategy_trust_region(self, X, Y, dim):
        """
        Strategy 3: Trust region search with local optimization.

        Selects min(N, 5*dim) nearest neighbors to the best solution.
        Builds RBF on selected data and runs L-BFGS-B optimization.
        """
        N = len(X)
        m = min(N, 5 * dim)

        Y_flat = Y.flatten()
        idx_min = np.argmin(Y_flat)

        # Select m nearest neighbors to best
        dist = cdist(X, X[idx_min:idx_min + 1]).flatten()
        idx = np.argsort(dist)[:m]
        X_trs = X[idx]
        Y_trs = Y[idx]

        # Build RBF on trust region data
        model = self._build_rbf_model(X_trs, Y_trs)

        lb_trs = np.min(X_trs, axis=0)
        ub_trs = np.max(X_trs, axis=0)

        # Handle degenerate bounds
        mask = (ub_trs - lb_trs) < 1e-10
        lb_trs[mask] = np.maximum(0.0, lb_trs[mask] - 0.05)
        ub_trs[mask] = np.minimum(1.0, ub_trs[mask] + 0.05)

        # Local optimization starting from best solution
        x0 = X[idx_min]
        bounds = list(zip(lb_trs, ub_trs))

        def obj_func(x):
            return model(x.reshape(1, -1)).flatten()[0]

        try:
            result = minimize(obj_func, x0, method='trust-constr', bounds=bounds,
                              options={'maxiter': 20, 'disp': False})
            candidate = result.x.reshape(1, -1)
        except Exception:
            candidate = x0.reshape(1, -1)

        candidate = np.clip(candidate, 0.0, 1.0)
        return candidate

    def _de_search(self, surrogate_func, X_full, Y_full, lb, ub, dim,
                   popsize=50, max_gen=10, mode='rand'):
        """
        Run DE on surrogate model with elite initialization.

        Parameters
        ----------
        surrogate_func : callable
            Surrogate model, accepts (n, d) array, returns (n, 1) predictions
        X_full : np.ndarray
            Full database of decision variables (for elite initialization)
        Y_full : np.ndarray
            Full database of objectives (for elite initialization)
        lb, ub : np.ndarray
            Search bounds, shape (d,)
        dim : int
            Problem dimension
        popsize : int
            Population size
        max_gen : int
            Maximum DE generations
        mode : str
            'rand' for DE/rand/1/bin, 'best' for DE/best/1/bin
        """
        CR, F = 0.8, 0.5
        Y_flat = Y_full.flatten()
        N = len(Y_flat)

        # Elite initialization: top popsize from full database
        if N >= popsize:
            idx = np.argsort(Y_flat)[:popsize]
            pop = X_full[idx].copy()
            pop_objs = Y_flat[idx].copy()
        else:
            extra = lb + (ub - lb) * np.random.rand(popsize - N, dim)
            pop = np.vstack([X_full.copy(), extra])
            pop_objs = np.concatenate([Y_flat, surrogate_func(extra).flatten()])

        # Compute range for normalization
        range_vec = ub - lb
        range_vec = np.maximum(range_vec, 1e-30)

        for gen in range(max_gen):
            # Normalize to [0,1] within [lb, ub]
            pop_norm = np.clip((pop - lb) / range_vec, 0, 1)

            # Generate offspring in normalized space
            if mode == 'rand':
                off_norm = de_generation(pop_norm, F, CR)
            else:
                off_norm = self._de_best1(pop_norm, pop_objs, F, CR)

            # Denormalize to original space
            offspring = lb + np.clip(off_norm, 0, 1) * range_vec

            # Evaluate on surrogate
            off_objs = surrogate_func(offspring).flatten()

            # Comparison selection (standard DE)
            improved = off_objs < pop_objs
            pop[improved] = offspring[improved]
            pop_objs[improved] = off_objs[improved]

        best_idx = np.argmin(pop_objs)
        return pop[best_idx:best_idx + 1]

    def _de_best1(self, parents, objs, F=0.5, CR=0.8):
        """
        DE/best/1/bin offspring generation in [0,1] normalized space.

        Parameters
        ----------
        parents : np.ndarray
            Parent population, shape (n, d), in [0,1]
        objs : np.ndarray
            Objective values, shape (n,)
        F : float
            Differential weight
        CR : float
            Crossover rate

        Returns
        -------
        offspring : np.ndarray
            Offspring population, shape (n, d), clipped to [0,1]
        """
        popsize, dim = parents.shape
        best_idx = np.argmin(objs)
        best = parents[best_idx]

        offspring = parents.copy()
        for i in range(popsize):
            idxs = np.delete(np.arange(popsize), i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            mutant = best + F * (parents[r1] - parents[r2])
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) <= CR) | (np.arange(dim) == j_rand)
            offspring[i, mask] = mutant[mask]

        return np.clip(offspring, 0, 1)

    def _ensure_uniqueness(self, candidate, X, dim, epsilon=5e-3, max_trials=50):
        """Ensure candidate is not too close to existing samples."""
        scales = np.linspace(0.1, 1.0, max_trials)
        for t in range(max_trials):
            dist = cdist(candidate, X, metric='chebyshev').min()
            if dist >= epsilon:
                break
            perturbation = scales[t] * (np.random.rand(1, dim) - 0.5)
            candidate = np.clip(candidate + perturbation, 0.0, 1.0)
        return candidate
