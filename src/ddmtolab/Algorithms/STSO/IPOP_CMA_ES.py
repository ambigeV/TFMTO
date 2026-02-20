"""
IPOP-CMA-ES (Increasing Population CMA-ES)

This module implements the IPOP-CMA-ES algorithm for single-objective optimization problems.

References
----------
    [1] Auger, A., & Hansen, N. (2005). A Restart CMA Evolution Strategy with Increasing Population Size. 2005 IEEE Congress on Evolutionary Computation, 2, 1769-1776. DOI: 10.1109/CEC.2005.1554902

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.27
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class IPOP_CMA_ES:
    """
    IPOP-CMA-ES for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=False,
                 save_data=True, save_path='./Data', name='IPOP-CMA-ES', disable_tqdm=True):
        """
        Initialize IPOP-CMA-ES Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Initial population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: False)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'IPOP_CMA_ES_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.use_n = use_n
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the IPOP-CMA-ES Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize parameters for each task
        params = []
        for t in range(nt):
            dim = problem.dims[t]

            # Determine initial population size
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dim))

            p = self._initialize_task_params(dim, lam)
            params.append(p)

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt
        objs = [None] * nt
        cons = [None] * nt
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]
        obj_hist = [[] for _ in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                p = params[i]

                # Generate offspring using cmaes_generation
                sample_decs = cmaes_generation(
                    m_dec=p['m_dec'],
                    sigma=p['sigma'],
                    B=p['B'],
                    D=p['D'],
                    lam=p['lam']
                )

                # Evaluate samples
                sample_objs, sample_cons = evaluation_single(problem, sample_decs, i)

                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = np.lexsort((sample_objs.flatten(), cvs))

                sample_decs = sample_decs[sort_indices]
                sample_objs = sample_objs[sort_indices]
                sample_cons = sample_cons[sort_indices]

                # Update current population
                decs[i] = sample_decs
                objs[i] = sample_objs
                cons[i] = sample_cons

                # Update objective history
                if len(obj_hist[i]) == 0:
                    obj_hist[i].append(sample_objs[0, 0])
                else:
                    obj_hist[i].append(min(obj_hist[i][-1], sample_objs[0, 0]))

                nfes_per_task[i] += p['lam']
                pbar.update(p['lam'])

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # Update mean decision variables
                old_dec = p['m_dec'].copy()
                p['m_dec'] = p['weights'] @ sample_decs[:p['mu']]

                # Update evolution paths
                diff = (p['m_dec'] - old_dec) / p['sigma']
                p['ps'] = (1 - p['cs']) * p['ps'] + \
                          np.sqrt(p['cs'] * (2 - p['cs']) * p['mueff']) * (p['invsqrtC'] @ diff)

                ps_norm = np.linalg.norm(p['ps'])
                hsig = ps_norm / np.sqrt(1 - (1 - p['cs']) ** (2 * nfes_per_task[i] / p['lam'])) / p['chiN'] < 1.4 + 2 / (p['dim'] + 1)

                p['pc'] = (1 - p['cc']) * p['pc'] + \
                          hsig * np.sqrt(p['cc'] * (2 - p['cc']) * p['mueff']) * diff

                # Update covariance matrix
                artmp = (sample_decs[:p['mu']] - old_dec) / p['sigma']
                delta = (1 - hsig) * p['cc'] * (2 - p['cc'])
                p['C'] = (1 - p['c1'] - p['cmu']) * p['C'] + \
                         p['c1'] * (np.outer(p['pc'], p['pc']) + delta * p['C']) + \
                         p['cmu'] * (artmp.T @ np.diag(p['weights']) @ artmp)

                # Update step size
                p['sigma'] = p['sigma'] * np.exp(p['cs'] / p['damps'] * (ps_norm / p['chiN'] - 1))

                # Update eigendecomposition periodically
                if nfes_per_task[i] - p['eigenFE'] > p['lam'] / (p['c1'] + p['cmu']) / p['dim'] / 10:
                    p['eigenFE'] = nfes_per_task[i]
                    p['C'] = np.triu(p['C']) + np.triu(p['C'], 1).T

                    eigvals, eigvecs = np.linalg.eigh(p['C'])

                    if np.min(eigvals) <= 0:
                        p['B'] = np.eye(p['dim'])
                        p['D'] = np.ones(p['dim'])
                        p['C'] = np.eye(p['dim'])
                    else:
                        p['B'] = eigvecs
                        p['D'] = np.sqrt(eigvals)

                    p['invsqrtC'] = p['B'] @ np.diag(1.0 / p['D']) @ p['B'].T

                # Restart strategy IPOP
                pre_gen = 10 + (30 * int(p['dim'] / p['lam']))
                obj_hist[i] = obj_hist[i][max(0, len(obj_hist[i]) - pre_gen):]
                obj_list = obj_hist[i] + sample_objs.flatten().tolist()

                # Check restart conditions
                if self._should_restart(p, obj_list):
                    # Double population size and reinitialize
                    new_lam = p['lam'] * 2
                    p = self._initialize_task_params(p['dim'], new_lam)
                    params[i] = p
                    obj_hist[i] = []

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(
            all_decs=all_decs,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            all_cons=all_cons,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        return results

    def _initialize_task_params(self, dim, lam):
        """
        Initialize CMA-ES parameters for a task.

        Parameters
        ----------
        dim : int
            Problem dimension
        lam : int
            Population size

        Returns
        -------
        dict
            Dictionary containing all CMA-ES parameters
        """

        mu = int(np.round(lam / 2))

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + cs + 2 * max(np.sqrt((mueff - 1) / (dim + 1)) - 1, 0)

        # Covariance update parameters
        cc = (4 + mueff / dim) / (4 + dim + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2 * mueff / 2))

        # Initialize
        m_dec = np.random.rand(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        invsqrtC = np.eye(dim)
        sigma = self.sigma0
        eigenFE = 0
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        return {
            'dim': dim, 'lam': lam, 'mu': mu, 'weights': weights, 'mueff': mueff,
            'cs': cs, 'damps': damps, 'cc': cc, 'c1': c1, 'cmu': cmu,
            'm_dec': m_dec, 'ps': ps, 'pc': pc, 'B': B, 'D': D, 'C': C,
            'invsqrtC': invsqrtC, 'sigma': sigma, 'eigenFE': eigenFE, 'chiN': chiN
        }

    def _should_restart(self, p, obj_list):
        """
        Check if restart conditions are met for IPOP strategy.

        Parameters
        ----------
        p : dict
            Current task parameters
        obj_list : list
            Combined objective history and current samples

        Returns
        -------
        bool
            True if restart is needed
        """
        if len(obj_list) == 0:
            return False

        # all(sigma{t} * (max(abs(pc{t}), sqrt(diag(C{t})))) < 1e-12 * Algo.sigma0)
        pc_c_max = np.maximum(np.abs(p['pc']), np.sqrt(np.diag(p['C'])))
        cond1 = np.all(p['sigma'] * pc_c_max < 1e-12 * self.sigma0)

        # any(sigma{t} * sqrt(diag(C{t})) > 1e8)
        cond2 = np.any(p['sigma'] * np.sqrt(np.diag(p['C'])) > 1e8)

        # sigma{t} * max(D{t}) == 0
        cond3 = p['sigma'] * np.max(p['D']) == 0

        # max(ObjList) - min(ObjList) < 1e-12
        cond4 = (np.max(obj_list) - np.min(obj_list)) < 1e-12

        return cond1 or cond2 or cond3 or cond4


def cmaes_generation(m_dec: np.ndarray, sigma: float, B: np.ndarray, D: np.ndarray, lam: int = None) -> np.ndarray:
    """
    Generate offspring population using CMA-ES sampling strategy.

    Parameters
    ----------
    m_dec : np.ndarray
        Mean decision vector, shape (d,)
    sigma : float
        Step size (global scaling factor)
    B : np.ndarray
        Eigenvector matrix from covariance matrix decomposition, shape (d, d)
    D : np.ndarray
        Square root of eigenvalues (standard deviations), shape (d,)
    lam : int, optional
        Number of offspring to generate (default: None)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (lam, d)
    """
    d = len(m_dec)

    # If lam is None, generate a default number
    if lam is None:
        lam = int(4 + 3 * np.log(d))

    offdecs = np.zeros((lam, d))

    for i in range(lam):
        z = np.random.randn(d)
        offdec = m_dec + sigma * (B @ (D * z))
        offdec = np.clip(offdec, 0, 1)
        offdecs[i] = offdec

    return offdecs