"""
Multi-task Max-value Bayesian Optimization (MUMBO)

This module implements MUMBO for expensive multi-task optimization. The algorithm
uses an information-theoretic acquisition function based on mutual information
between candidate observations and the global optimum value g*. A multi-task
Gaussian process provides cross-task knowledge transfer, and the MUMBO acquisition
function exploits the bivariate predictive distribution to compute rho (predictive
correlation) between each task and the target task. The acquisition value is
divided by the evaluation cost of each task, enabling cost-aware task selection.

References
----------
    [1] Moss, Henry B., David S. Leslie, and Paul Rayson. "Mumbo: Multi-task
        max-value Bayesian optimization." Joint European Conference on Machine
        Learning and Knowledge Discovery in Databases. Springer, 2020.

    [2] Wang, Zi, and Stefanie Jegelka. "Max-value entropy search for efficient
        Bayesian optimization." ICML, 2017.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.18
Version: 2.0
"""
from tqdm import tqdm
import torch
import time
import numpy as np
from scipy.stats import norm, gumbel_r
from scipy.integrate import simpson
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mtgp_build
import warnings

warnings.filterwarnings("ignore")


class MUMBO:
    """
    Multi-task Max-value Bayesian Optimization (MUMBO).

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, task_cost=None,
                 n_gstar_samples=1, n_candidates=20, n_quad=10,
                 save_data=True, save_path='./Data', name='MUMBO',
                 disable_tqdm=True):
        """
        Initialize MUMBO.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance.
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50).
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100).
        task_cost : List[float] or None, optional
            Evaluation cost for each task. If None, all tasks have equal cost
            [1, 1, ..., 1].
        n_gstar_samples : int, optional
            Number of g* samples from Gumbel distribution (default: 1).
        n_candidates : int, optional
            Number of random candidates per task for acquisition (default: 20).
        n_quad : int, optional
            Number of quadrature points for Simpson integration (default: 10).
        save_data : bool, optional
            Whether to save optimization data (default: True).
        save_path : str, optional
            Path to save results (default: './Data').
        name : str, optional
            Name for the experiment (default: 'MUMBO').
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True).
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.task_cost = task_cost
        self.n_gstar_samples = n_gstar_samples
        self.n_candidates = n_candidates
        self.n_quad = n_quad
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute MUMBO.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            and runtime.
        """
        data_type = torch.double
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Task evaluation costs
        if self.task_cost is not None:
            task_cost = np.array(self.task_cost, dtype=float)
        else:
            task_cost = np.ones(nt, dtype=float)

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(total=sum(max_nfes_per_task),
                    initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt)
                           if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Build multi-task GP with normalized objectives
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)

            # Sample g* values from Gumbel extreme-value distribution
            g_samples = _sample_gstar(mtgp, decs, dims, nt, data_type,
                                      n_gstar_samples=self.n_gstar_samples)

            # Select (task, x) pair with highest acquisition/cost ratio
            best_task, best_x = _select_next_point(
                mtgp, g_samples, active_tasks, dims, nt, task_cost, data_type,
                n_candidates=self.n_candidates, n_quad=self.n_quad
            )

            # Evaluate on selected task
            candidate_np = best_x.reshape(1, -1)
            obj, _ = evaluation_single(problem, candidate_np, best_task)

            # Update data
            decs[best_task], objs[best_task] = vstack_groups(
                (decs[best_task], candidate_np),
                (objs[best_task], obj)
            )
            nfes_per_task[best_task] += 1
            pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )

        return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_task_range(nt, data_type=torch.double):
    """Get task index encoding vector consistent with mtgp_build."""
    return torch.linspace(0, 1, nt, dtype=data_type)



def _bivariate_stats_batch(mtgp, candidates_np, task_id, dims, nt,
                            data_type=torch.double):
    """
    Compute bivariate GP statistics for a batch of candidates on task z vs
    target task z0.

    For each candidate x_j, computes the joint posterior of (g, y) where
    g = f(x, z0) and y = f(x, z) + eps, then extracts mu_g, sigma_g,
    sigma_f_noisy, and rho.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Trained multi-task GP.
    candidates_np : np.ndarray
        Candidate points, shape (n, d).
    task_id : int
        Task index z for the candidate evaluation.
    dims : list[int]
        Dimensionality of each task.
    nt : int
        Number of tasks.
    data_type : torch.dtype
        Data type.

    Returns
    -------
    mu_g : np.ndarray, shape (n,)
    sigma_g : np.ndarray, shape (n,)
    sigma_f_noisy : np.ndarray, shape (n,)
    rho : np.ndarray, shape (n,)
    """
    max_dim = max(dims)
    task_range = _get_task_range(nt, data_type)
    n = candidates_np.shape[0]

    # Pad candidates to max_dim
    if candidates_np.shape[1] < max_dim:
        pad = np.zeros((n, max_dim - candidates_np.shape[1]))
        x_padded = np.hstack([candidates_np, pad])
    else:
        x_padded = candidates_np

    x_t = torch.tensor(x_padded, dtype=data_type)  # (n, max_dim)

    target_task_val = task_range[0].item()
    cand_task_val = task_range[task_id].item()

    # Build interleaved input: [x0_target, x0_cand, x1_target, x1_cand, ...]
    # Shape: (2*n, max_dim+1)
    x_target = torch.cat([x_t, torch.full((n, 1), target_task_val,
                                           dtype=data_type)], dim=1)
    x_cand = torch.cat([x_t, torch.full((n, 1), cand_task_val,
                                         dtype=data_type)], dim=1)
    # Interleave: row 0,1 = pair 0; row 2,3 = pair 1; ...
    x_joint = torch.stack([x_target, x_cand], dim=1).reshape(2 * n, -1)

    # Get noise variance
    noise_tensor = mtgp.likelihood.noise.detach().cpu()
    if noise_tensor.numel() > 1:
        noise_var = noise_tensor.mean().item()
    else:
        noise_var = noise_tensor.item()

    mtgp.eval()
    with torch.no_grad():
        post = mtgp.posterior(x_joint)
        mu_all = post.mean.squeeze(-1).cpu().numpy()          # (2*n,)
        cov_full = post.mvn.covariance_matrix.cpu().numpy()   # (2*n, 2*n)

    # Extract per-pair statistics from the block-diagonal-like structure
    mu_g_arr = np.empty(n)
    sigma_g_arr = np.empty(n)
    sigma_f_noisy_arr = np.empty(n)
    rho_arr = np.empty(n)

    for j in range(n):
        idx0 = 2 * j       # target
        idx1 = 2 * j + 1   # candidate

        mu_g_val = mu_all[idx0]
        sigma_g_sq = cov_full[idx0, idx0]
        sigma_f_sq = cov_full[idx1, idx1]
        sigma_gf = cov_full[idx0, idx1]

        sg = np.sqrt(max(sigma_g_sq, 1e-20))
        sf_noisy = np.sqrt(max(sigma_f_sq + noise_var, 1e-20))

        denom = sg * sf_noisy
        if denom < 1e-20:
            r = 0.0
        else:
            r = np.clip(sigma_gf / denom, -0.999, 0.999)

        mu_g_arr[j] = mu_g_val
        sigma_g_arr[j] = sg
        sigma_f_noisy_arr[j] = sf_noisy
        rho_arr[j] = r

    return mu_g_arr, sigma_g_arr, sigma_f_noisy_arr, rho_arr


def _sample_gstar(mtgp, decs, dims, nt, data_type, n_gstar_samples=1):
    """
    Sample g* values using Gumbel extreme-value distribution.

    Use mean-field approximation: evaluate GP posterior mean on a large random
    grid, then fit a Gumbel distribution to sample potential optimal values.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Trained multi-task GP.
    decs : list[np.ndarray]
        Current decision variables per task.
    dims : list[int]
        Dimensionality per task.
    nt : int
        Number of tasks.
    data_type : torch.dtype
        Data type.
    n_gstar_samples : int
        Number of g* samples.

    Returns
    -------
    g_samples : np.ndarray
        Array of g* samples, shape (n_gstar_samples,).
    """
    max_dim = max(dims)
    target_dim = dims[0]
    task_range = _get_task_range(nt, data_type)
    target_task_val = task_range[0].item()

    # Generate random grid: 1000*d points + existing evaluated points
    n_grid = min(1000 * target_dim, 5000)
    grid_np = np.random.rand(n_grid, max_dim)

    # Add existing target task points to grid
    existing = decs[0].copy()
    if existing.shape[1] < max_dim:
        pad = np.zeros((existing.shape[0], max_dim - existing.shape[1]))
        existing = np.hstack([existing, pad])
    grid_np = np.vstack([grid_np, existing])

    # Evaluate GP posterior mean on target task
    grid_t = torch.tensor(grid_np, dtype=data_type)
    task_col = torch.full((grid_t.shape[0], 1), target_task_val,
                          dtype=data_type)
    grid_with_task = torch.cat([grid_t, task_col], dim=1)

    mtgp.eval()
    with torch.no_grad():
        post = mtgp.posterior(grid_with_task)
        mu = post.mean.squeeze(-1).cpu().numpy()

    # Fit Gumbel distribution and sample
    y_max = mu.max()
    y_mean = mu.mean()
    y_std = mu.std()
    if y_std < 1e-10:
        y_std = 1e-10

    # Gumbel location/scale from extreme value theory
    loc = y_max
    scale = max(y_std * 0.1, 1e-6)
    g_samples = gumbel_r.rvs(loc=loc, scale=scale, size=n_gstar_samples)

    return g_samples


def _mumbo_acquisition_single(mu_g, sigma_g, mu_f, sigma_f_noisy, rho,
                               g_samples, n_quad=10):
    """
    Compute MUMBO acquisition value for a single (x, z) pair.

    Implements equation (11) from the tex:
        alpha = (1/N) sum_{g*} [
            rho^2 * gamma * phi(gamma) / (2 * Phi(gamma))
            - log Phi(gamma)
            + E_{theta ~ ESN}[ log Phi((gamma - rho*theta) / sqrt(1-rho^2)) ]
        ]

    Parameters
    ----------
    mu_g, sigma_g : float
        Target task posterior mean and std.
    mu_f, sigma_f_noisy : float
        Candidate task posterior mean and noisy std.
    rho : float
        Predictive correlation.
    g_samples : np.ndarray
        Samples of g*.
    n_quad : int
        Number of quadrature points for Simpson integration.

    Returns
    -------
    alpha : float
        MUMBO acquisition value.
    """
    if sigma_g < 1e-20:
        return 0.0

    rho_sq = rho ** 2
    sqrt_1_minus_rho_sq = np.sqrt(max(1.0 - rho_sq, 1e-20))

    total = 0.0
    for g_star in g_samples:
        gamma = (g_star - mu_g) / sigma_g

        phi_gamma = norm.pdf(gamma)
        cdf_gamma = norm.cdf(gamma)
        if cdf_gamma < 1e-30:
            cdf_gamma = 1e-30

        # First two analytic terms
        term1 = rho_sq * gamma * phi_gamma / (2.0 * cdf_gamma)
        term2 = -np.log(cdf_gamma)

        # Third term: expectation over ESN distribution via Simpson rule
        # ESN mean and variance (eq. 9-10)
        ratio = phi_gamma / cdf_gamma  # phi/Phi (inverse Mills ratio)
        esn_mean = rho * ratio
        esn_var = 1.0 - rho_sq * ratio * (gamma + ratio)
        esn_var = max(esn_var, 1e-20)
        esn_std = np.sqrt(esn_var)

        # Integration bounds: mean +/- 4*std
        lo = esn_mean - 4.0 * esn_std
        hi = esn_mean + 4.0 * esn_std
        if hi - lo < 1e-15:
            term3 = 0.0
        else:
            theta_grid = np.linspace(lo, hi, n_quad)

            # ESN density p(theta) (eq. 7)
            phi_theta = norm.pdf(theta_grid)
            arg_inner = (gamma - rho * theta_grid) / sqrt_1_minus_rho_sq
            Phi_inner = norm.cdf(arg_inner)
            esn_density = (1.0 / cdf_gamma) * phi_theta * Phi_inner

            # Integrand: p(theta) * log Phi(...)
            log_Phi_inner = np.log(np.maximum(Phi_inner, 1e-30))
            integrand = esn_density * log_Phi_inner

            term3 = simpson(integrand, x=theta_grid)

        total += term1 + term2 + term3

    alpha = total / len(g_samples)
    return max(alpha, 0.0)


def _select_next_point(mtgp, g_samples, active_tasks, dims, nt, task_cost,
                        data_type, n_candidates=20, n_quad=10):
    """
    Select the next (task, x) pair by maximizing alpha/cost.

    For each active task, generate random candidates in batch, compute MUMBO
    acquisition values, and select the (task, x) pair with highest ratio.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Trained multi-task GP.
    g_samples : np.ndarray
        Samples of g*.
    active_tasks : list[int]
        Indices of tasks with remaining budget.
    dims : list[int]
        Dimensionality per task.
    nt : int
        Number of tasks.
    task_cost : np.ndarray
        Evaluation cost per task.
    data_type : torch.dtype
        Data type.
    n_candidates : int
        Number of random candidates per task.
    n_quad : int
        Number of quadrature points for Simpson integration.

    Returns
    -------
    best_task : int
        Selected task index.
    best_x : np.ndarray
        Selected candidate point, shape (d,).
    """
    best_ratio = -np.inf
    best_task = active_tasks[0]
    best_x = np.random.rand(dims[active_tasks[0]])

    for task_id in active_tasks:
        dim = dims[task_id]
        candidates = np.random.rand(n_candidates, dim)

        # Batch bivariate stats computation
        mu_g, sigma_g, sigma_f_noisy, rho = _bivariate_stats_batch(
            mtgp, candidates, task_id, dims, nt, data_type
        )

        # Compute acquisition for each candidate
        for j in range(n_candidates):
            alpha = _mumbo_acquisition_single(
                mu_g[j], sigma_g[j], 0.0, sigma_f_noisy[j],
                rho[j], g_samples, n_quad=n_quad
            )
            ratio = alpha / task_cost[task_id]

            if ratio > best_ratio:
                best_ratio = ratio
                best_task = task_id
                best_x = candidates[j].copy()

    return best_task, best_x
