"""
EBS (Evolutionary Biocoenosis-based Symbiosis)

This module implements the EBS algorithm for evolutionary many-tasking optimization.

References
----------
    [1] Liaw, R. T., & Ting, C. K. (2019). Evolutionary many-tasking based on biocoenosis through symbiosis: A framework and benchmark problems. In 2019 IEEE Congress on Evolutionary Computation (CEC) (pp. 2266-2273). IEEE.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.09
Version: 1.0
"""
from tqdm import tqdm
import copy
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class EBS:
    """
    Evolutionary Biocoenosis-based Symbiosis for many-task optimization.

    EBS uses multiple CMA-ES instances with adaptive information exchange among tasks.
    Each task maintains two CMA-ES distributions:
    - One updated when knowledge transfer occurs
    - One updated when no knowledge transfer occurs

    The information exchange probability is controlled adaptively based on the
    improvement ratio from self-generated offspring versus offspring from other tasks.

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
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=True,
                 gen_init=10, save_data=True, save_path='./Data',
                 name='EBS', disable_tqdm=True):
        """
        Initialize EBS Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size for CMA-ES (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: True)
        gen_init : int, optional
            Number of initial generations for alternating CMA-ES before using gamma (default: 10)
            During this phase, two CMA-ES alternate (one without transfer, one with transfer)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EBS_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.use_n = use_n
        self.gen_init = gen_init
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EBS Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = np.max(dims)  # Unified dimension
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize CMA-ES parameters for each task (using unified dimension)
        params = []
        for t in range(nt):
            # Determine population size based on original task dimension
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dims[t]))

            # Distribution for self-generated offspring (no knowledge transfer)
            params_s = cmaes_init_params(d_max, lam=lam, sigma0=self.sigma0)
            # Distribution for knowledge transfer offspring (same starting point)
            params_o = copy.deepcopy(params_s)
            params_o['m_dec'] = params_s['m_dec'].copy()

            params.append({
                'real_dim': dims[t],  # Real dimension for this task
                'params_s': params_s,  # Self distribution (no transfer)
                'params_o': params_o,  # Other distribution (with transfer)
            })

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt  # In unified space
        objs = [None] * nt
        cons = [None] * nt
        best_objs = [np.inf] * nt  # Best-so-far objective for each task
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Initialize information exchange statistics
        improvements_s = [0] * nt
        evals_s = [0] * nt
        improvements_o = [0] * nt
        evals_o = [0] * nt
        gamma = [0.0] * nt  # Probability of information exchange (will be computed after init phase)

        # Initialize generation counter for each task (for alternating during init phase)
        gen_count = [0] * nt

        pbar = tqdm(total=sum(max_nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Step 1: Determine transfer flags and generate offspring accordingly
            transfer_flags = []
            offspring_list = []

            for i in active_tasks:
                p = params[i]

                # Decide whether to perform knowledge transfer
                if gen_count[i] < self.gen_init:
                    # During init phase: alternate between no transfer and transfer
                    # Even generations: no transfer (is_transfer=False)
                    # Odd generations: transfer (is_transfer=True)
                    is_transfer = (gen_count[i] % 2 == 1)
                else:
                    # After init phase: use gamma probability
                    is_transfer = np.random.rand() < gamma[i]

                transfer_flags.append(is_transfer)

                # Select distribution based on transfer decision
                if is_transfer:
                    ps = p['params_o']  # Use knowledge transfer distribution
                else:
                    ps = p['params_s']  # Use self distribution

                # Generate offspring using selected CMA-ES distribution
                sample_decs = cmaes_sample(
                    ps['m_dec'], ps['sigma'], ps['B'], ps['D'], ps['lam']
                )
                offspring_list.append(sample_decs)

            # Concatenate all offspring (all in unified space d_max)
            concat_offspring = np.vstack(offspring_list)

            # Step 2: For each task, select candidates based on transfer decision
            candidate_list = []

            for idx, i in enumerate(active_tasks):
                p = params[i]
                is_transfer = transfer_flags[idx]

                if is_transfer:
                    # Perform knowledge transfer: sample from concatenate offspring
                    # Randomly select lambda candidates from concatenate offspring
                    n_candidates = min(p['params_s']['lam'], concat_offspring.shape[0])
                    candidate_indices = np.random.choice(
                        concat_offspring.shape[0],
                        size=n_candidates,
                        replace=False
                    )
                    candidate_decs = concat_offspring[candidate_indices].copy()

                    # If we need more candidates, duplicate some
                    while candidate_decs.shape[0] < p['params_s']['lam']:
                        extra_idx = np.random.choice(concat_offspring.shape[0])
                        candidate_decs = np.vstack([candidate_decs, concat_offspring[extra_idx:extra_idx + 1]])

                    candidate_list.append(candidate_decs)
                else:
                    # No knowledge transfer: use self-generated offspring
                    candidate_list.append(offspring_list[idx])

            # Step 3: Evaluate, update population and CMA-ES parameters
            for idx, i in enumerate(active_tasks):
                p = params[i]
                candidate_decs = candidate_list[idx]  # In unified space
                is_transfer = transfer_flags[idx]

                # Convert to real space for evaluation (truncate to real dimension)
                candidate_decs_real = candidate_decs[:, :dims[i]]
                candidate_decs_real = np.clip(candidate_decs_real, 0, 1)  # Ensure bounds

                # Evaluate candidates (in real space)
                sample_objs, sample_cons = evaluation_single(problem, candidate_decs_real, i)

                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = constrained_sort(sample_objs, cvs)

                # Sort in unified space
                sorted_decs = candidate_decs[sort_indices]
                sorted_objs = sample_objs[sort_indices]
                sorted_cons = sample_cons[sort_indices]

                # Update evaluation counts
                if is_transfer:
                    evals_o[i] += p['params_s']['lam']
                else:
                    evals_s[i] += p['params_s']['lam']

                # Check if best-so-far is improved
                best_candidate_obj = sorted_objs[0, 0]
                if best_candidate_obj < best_objs[i]:
                    best_objs[i] = best_candidate_obj
                    if is_transfer:
                        improvements_o[i] += 1
                    else:
                        improvements_s[i] += 1

                # Increment generation counter
                gen_count[i] += 1

                # Update gamma after init phase is complete
                if gen_count[i] == self.gen_init:
                    # Compute gamma based on accumulated statistics
                    if evals_s[i] > 0 and evals_o[i] > 0:
                        R_s = improvements_s[i] / evals_s[i]
                        R_o = improvements_o[i] / evals_o[i]
                        if (R_s + R_o) > 0:
                            gamma[i] = R_o / (R_s + R_o)
                        else:
                            gamma[i] = 0.0  # Default if no improvements
                    else:
                        gamma[i] = 0.0  # Default
                elif gen_count[i] > self.gen_init:
                    # Continue updating gamma based on accumulated statistics
                    if evals_s[i] > 0 and evals_o[i] > 0:
                        R_s = improvements_s[i] / evals_s[i]
                        R_o = improvements_o[i] / evals_o[i]
                        if (R_s + R_o) > 0:
                            gamma[i] = R_o / (R_s + R_o)

                # Update current population (store in unified space)
                decs[i] = sorted_decs
                objs[i] = sorted_objs
                cons[i] = sorted_cons

                nfes_per_task[i] += p['params_s']['lam']
                pbar.update(p['params_s']['lam'])

                # Convert to real space for history (truncate to real dimension)
                decs_real = sorted_decs[:, :dims[i]]
                append_history(all_decs[i], decs_real, all_objs[i], sorted_objs, all_cons[i], sorted_cons)

                # Update the appropriate CMA-ES distribution
                if is_transfer:
                    # Update knowledge transfer CMA-ES
                    cmaes_update(p['params_o'], sorted_decs, nfes_per_task[i])
                else:
                    # Update self CMA-ES
                    cmaes_update(p['params_s'], sorted_decs, nfes_per_task[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results (all_decs are already in real space)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name, save_data=self.save_data
        )

        return results

