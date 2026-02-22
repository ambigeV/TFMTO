"""
Multi-Objective Symbiosis-Based Optimization (MO-SBO)

This module implements the MO-SBO algorithm for multi-objective many-task optimization
based on symbiotic relationships in biocoenosis. The algorithm adaptively controls
knowledge transfer rates by tracking six types of symbiotic interactions: mutualism,
commensalism, parasitism, competition, amensalism, and neutralism.

References
----------
    [1] R.-T. Liaw and C.-K. Ting. "Evolutionary Manytasking Optimization Based on Symbiosis in Biocoenosis." Proceedings of the AAAI Conference on Artificial Intelligence, 33(01): 4295-4303, 2019.

Notes
-----
The code is developed in accordance with the MATLAB-based MTO-platform framework.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.21
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MO_SBO:
    """
    Multi-Objective Symbiosis-Based Optimization for many-task multi-objective optimization.

    The algorithm uses symbiotic relationships between tasks to adaptively control
    knowledge transfer rates. Six types of symbiotic interactions are tracked:

    - Mutualism (MIJ): Both tasks benefit (transferred solution ranks high in both)
    - Commensalism (OIJ): One benefits, other neutral
    - Parasitism (PIJ): One benefits, other harmed
    - Competition (CIJ): Both harmed
    - Amensalism (AIJ): One harmed, other neutral
    - Neutralism (NIJ): Both neutral

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, benefit=0.25, harm=0.5,
                 mu_c=20, mu_m=15, save_data=True, save_path='./Data',
                 name='MO-SBO', disable_tqdm=True):
        """
        Initialize MO-SBO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        benefit : float, optional
            Beneficial factor threshold for symbiosis categorization (default: 0.25)
        harm : float, optional
            Harmful factor threshold for symbiosis categorization (default: 0.5)
        mu_c : float, optional
            Distribution index for simulated binary crossover (default: 20)
        mu_m : float, optional
            Distribution index for polynomial mutation (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-SBO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.benefit = benefit
        self.harm = harm
        self.mu_c = mu_c
        self.mu_m = mu_m
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MO-SBO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_dim = max(dims)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population in native space and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt

        # Convert to unified space (pad with random values for unused dimensions)
        uni_decs = []
        for t in range(nt):
            padded = np.random.rand(n, max_dim)
            padded[:, :dims[t]] = decs[t]
            uni_decs.append(padded)

        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Symbiosis interaction counters (initialized to 1 to avoid division by zero)
        RIJ = 0.5 * np.ones((nt, nt))  # Transfer rates
        MIJ = np.ones((nt, nt))  # Mutualism
        NIJ = np.ones((nt, nt))  # Neutralism
        CIJ = np.ones((nt, nt))  # Competition
        OIJ = np.ones((nt, nt))  # Commensalism
        PIJ = np.ones((nt, nt))  # Parasitism
        AIJ = np.ones((nt, nt))  # Amensalism

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            # === Step 1: Generate offspring for each task (SBX + PM in unified space) ===
            off_uni_decs = []
            off_rank_o = []
            off_belong_t = []
            for t in range(nt):
                off_t = ga_generation(uni_decs[t], self.mu_c, self.mu_m)
                off_uni_decs.append(off_t)
                # Offspring inherits positional rank from parent (1-based)
                off_rank_o.append(np.arange(1, n + 1))
                off_belong_t.append(np.full(n, t))

            # === Step 2: Knowledge transfer based on symbiosis ===
            for t in range(nt):
                # Find task with highest transfer rate (exclude self)
                rij_row = RIJ[t].copy()
                rij_row[t] = -np.inf
                transfer_task = int(np.argmax(rij_row))

                if np.random.rand() < RIJ[t, transfer_task]:
                    si = int(np.floor(n * RIJ[t, transfer_task]))
                    if si > 0:
                        ind1 = np.random.permutation(n)[:si]
                        ind2 = np.random.permutation(n)[:si]
                        # Replace offspring of task t with solutions from transfer_task
                        off_uni_decs[t][ind1] = off_uni_decs[transfer_task][ind2]
                        off_belong_t[t][ind1] = transfer_task

            # === Step 3: Evaluate, compute rankC, and selection for each task ===
            all_rank_c = [None] * nt
            for t in range(nt):
                # Evaluate offspring (trim to native dimensions)
                off_native = off_uni_decs[t][:, :dims[t]]
                off_objs_t, off_cons_t = evaluation_single(problem, off_native, t)
                nfes += n
                pbar.update(n)

                # NSGA-II sort offspring to get rankC (1-based)
                all_rank_c[t] = _nsga2_rank(off_objs_t, off_cons_t)

                # Merge parents and offspring
                merged_uni = np.vstack([uni_decs[t], off_uni_decs[t]])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t])

                # NSGA-II selection: keep top n
                merged_rank = _nsga2_rank(merged_objs, merged_cons)
                select_idx = np.argsort(merged_rank)[:n]

                uni_decs[t] = merged_uni[select_idx]
                objs[t] = merged_objs[select_idx]
                cons[t] = merged_cons[select_idx]
                decs[t] = uni_decs[t][:, :dims[t]]

            # === Step 4: Update symbiosis counters ===
            for t in range(nt):
                transferred_idx = np.where(off_belong_t[t] != t)[0]
                for k in transferred_idx:
                    rc = all_rank_c[t][k]  # Rank after evaluation on task t
                    ro = off_rank_o[t][k]  # Positional rank from parent population
                    src = int(off_belong_t[t][k])

                    if rc < n * self.benefit:
                        # Transferred solution performs well on task t
                        if ro < n * self.benefit:
                            MIJ[t, src] += 1  # Both benefit
                        elif ro > n * (1 - self.harm):
                            PIJ[t, src] += 1  # One benefits, other harmed
                        else:
                            OIJ[t, src] += 1  # One benefits, other neutral
                    elif rc > n * (1 - self.harm):
                        # Transferred solution performs poorly on task t
                        if ro > n * (1 - self.harm):
                            CIJ[t, src] += 1  # Both harmed
                    else:
                        # Transferred solution is neutral on task t
                        if ro > n * (1 - self.harm):
                            AIJ[t, src] += 1  # One harmed, other neutral
                        elif n * self.benefit <= ro <= n * (1 - self.harm):
                            NIJ[t, src] += 1  # Both neutral

            # === Step 5: Update transfer rates ===
            RIJ = (MIJ + OIJ + PIJ) / (MIJ + OIJ + PIJ + AIJ + CIJ + NIJ)

            append_history(all_decs, decs, all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data
        )

        return results


def _nsga2_rank(objs, cons):
    """
    Compute NSGA-II rank for each individual.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (pop_size, n_obj)
    cons : np.ndarray
        Constraint values, shape (pop_size, n_con)

    Returns
    -------
    rank : np.ndarray
        1-based rank for each individual (lower is better), shape (pop_size,)
    """
    pop_size = objs.shape[0]
    if cons is not None and cons.size > 0:
        front_no, _ = nd_sort(objs, cons, pop_size)
    else:
        front_no, _ = nd_sort(objs, pop_size)
    crowd_dis = crowding_distance(objs, front_no)
    sorted_indices = np.lexsort((-crowd_dis, front_no))
    rank = np.empty(pop_size, dtype=int)
    rank[sorted_indices] = np.arange(1, pop_size + 1)
    return rank
