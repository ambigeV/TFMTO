"""
Multifactorial Evolutionary Algorithm Based on Diffusion Gradient Descent (MFEA-DGD)

This module implements MFEA-DGD for multi-task optimization using gradient
estimation via finite differences with random orthogonal directions.

References
----------
    [1] Liu, Zhaobo, et al. "Multifactorial Evolutionary Algorithm Based on
        Diffusion Gradient Descent." IEEE Transactions on Cybernetics,
        1-13, 2023.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MFEA_DGD:
    """
    Multifactorial Evolutionary Algorithm Based on Diffusion Gradient Descent.

    Uses gradient estimation via random finite differences to guide crossover
    and mutation operators:
    - Random perturbation direction from Gaussian distribution
    - Finite-difference gradient estimation per parent pair
    - Gradient-guided blend crossover with opposition-based learning (OBL)
    - Gradient descent mutation for non-transfer offspring
    - Adaptive sigma randomly selected each generation

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
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.7, gamma=0.1,
                 save_data=True, save_path='./Data', name='MFEA-DGD',
                 disable_tqdm=True):
        """
        Initialize MFEA-DGD algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.7)
        gamma : float, optional
            Smoothing factor for gradient norm tracking (default: 0.1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-DGD')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.gamma = gamma
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MFEA-DGD algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt
        pop_size = n * nt

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified space
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons,
                                            type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        L = 0.0  # smoothed gradient norm

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        gen = 1
        while nfes < max_nfes:
            # Random sigma from {10^-1, ..., 10^-5}
            sigma = 10.0 ** (-np.random.randint(1, 6))

            # Compute per-task decision variable bounds
            max_dec = []
            min_dec = []
            for t in range(nt):
                max_dec.append(np.max(pop_decs[t], axis=0))
                min_dec.append(np.min(pop_decs[t], axis=0))

            # Merge populations
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)

            maxD = m_decs.shape[1]
            maxC = m_cons.shape[1]  # unified constraint dimension
            n_pairs = pop_size // 2

            # Preallocate main offspring and probe arrays
            main_off_decs = np.zeros((pop_size, maxD))
            main_off_sfs = np.zeros((pop_size, 1), dtype=int)

            probe_decs_list = []
            probe_objs_list = []
            probe_cons_list = []
            probe_sfs_list = []

            shuffled = np.random.permutation(pop_size)

            k = 0.7 + 0.6 * np.random.rand()

            for pair_idx in range(n_pairs):
                p1 = shuffled[pair_idx]
                p2 = shuffled[pair_idx + n_pairs]
                sf1 = m_sfs[p1].item()
                sf2 = m_sfs[p2].item()

                # Random Gaussian direction (matching RandOrthMat(D, 1))
                sd = np.random.randn(maxD)

                # --- Gradient estimation via finite differences ---
                QWE = np.zeros((2, maxD))
                parents = [p1, p2]
                factors = [sf1, sf2]

                for x in range(2):
                    pidx = parents[x]
                    ft = factors[x]

                    # Positive probe
                    probe_pos = m_decs[pidx] + sd * sigma
                    # Negative probe
                    probe_neg = m_decs[pidx] - sd * sigma

                    # Evaluate probes on parent's task
                    probe_pos_trimmed = np.clip(probe_pos[:dims[ft]], 0, 1)
                    probe_neg_trimmed = np.clip(probe_neg[:dims[ft]], 0, 1)
                    obj_pos, con_pos = evaluation_single(
                        problem, probe_pos_trimmed, ft)
                    obj_neg, con_neg = evaluation_single(
                        problem, probe_neg_trimmed, ft)
                    nfes += 2
                    pbar.update(2)

                    # Finite-difference gradient estimate
                    L1 = obj_pos[0, 0] - obj_neg[0, 0]
                    QWE[x, :] = sd * L1 / sigma

                    # Store probes as offspring for selection
                    # Pad constraints to unified dimension
                    con_pos_pad = np.zeros(maxC)
                    con_neg_pad = np.zeros(maxC)
                    c_pos = con_pos.flatten()
                    c_neg = con_neg.flatten()
                    if len(c_pos) > 0:
                        con_pos_pad[:len(c_pos)] = c_pos
                    if len(c_neg) > 0:
                        con_neg_pad[:len(c_neg)] = c_neg

                    # Positive probe
                    probe_dec_pos_full = m_decs[pidx].copy()
                    probe_dec_pos_full[:dims[ft]] = probe_pos_trimmed
                    probe_decs_list.append(probe_dec_pos_full)
                    probe_objs_list.append(obj_pos.flatten())
                    probe_cons_list.append(con_pos_pad)
                    probe_sfs_list.append(ft)

                    # Negative probe
                    probe_dec_neg_full = m_decs[pidx].copy()
                    probe_dec_neg_full[:dims[ft]] = probe_neg_trimmed
                    probe_decs_list.append(probe_dec_neg_full)
                    probe_objs_list.append(obj_neg.flatten())
                    probe_cons_list.append(con_neg_pad)
                    probe_sfs_list.append(ft)

                # Update smoothed gradient norm L
                qwe_norm = np.linalg.norm(QWE)
                if qwe_norm > L:
                    L = (1 - self.gamma) * qwe_norm + self.gamma * L

                # Avoid division by zero
                L_safe = max(L, 1e-15)

                idx1 = pair_idx * 2
                idx2 = pair_idx * 2 + 1

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    # --- Transfer: gradient-guided crossover + OBL ---
                    r1 = np.random.randint(2)
                    r2 = 1 - r1
                    factor = factors[np.random.randint(2)]

                    # Gradient-guided blend crossover
                    off_dec1 = _dgd_crossover(
                        m_decs[parents[r1]], m_decs[parents[r2]],
                        QWE, L_safe, sigma, maxD)

                    main_off_decs[idx1] = off_dec1

                    # OBL: alternate between full OBL and bounds-based
                    if gen % 2 == 0:
                        # Full OBL (even gen → mod(gen,2)=0, rand()>0 always true)
                        main_off_decs[idx2] = 1.0 - off_dec1
                    else:
                        # Bounds-based opposition (odd gen)
                        main_off_decs[idx2] = (
                            k * (max_dec[factor] + min_dec[factor]) - off_dec1)

                    # Task imitation: random parent's MFFactor
                    main_off_sfs[idx1] = factors[np.random.randint(2)]
                    main_off_sfs[idx2] = factors[np.random.randint(2)]
                else:
                    # --- No transfer: gradient descent mutation ---
                    main_off_decs[idx1] = m_decs[p1] - QWE[0, :] * sigma / L_safe
                    main_off_decs[idx2] = m_decs[p2] - QWE[1, :] * sigma / L_safe

                    main_off_sfs[idx1] = sf1
                    main_off_sfs[idx2] = sf2

                # Clip to [0, 1]
                main_off_decs[idx1] = np.clip(main_off_decs[idx1], 0, 1)
                main_off_decs[idx2] = np.clip(main_off_decs[idx2], 0, 1)

            # --- Evaluate main offspring ---
            main_off_objs = np.full((pop_size, m_objs.shape[1]), np.inf)
            main_off_cons = np.zeros((pop_size, m_cons.shape[1]))
            for idx in range(pop_size):
                t = main_off_sfs[idx].item()
                main_off_objs[idx], main_off_cons[idx] = evaluation_single(
                    problem, main_off_decs[idx, :dims[t]], t)
            nfes += pop_size
            pbar.update(pop_size)

            # --- Build probe arrays ---
            n_probes = len(probe_decs_list)
            probe_decs = np.array(probe_decs_list)
            probe_objs = np.array(probe_objs_list).reshape(n_probes, -1)
            probe_cons = np.array(probe_cons_list).reshape(n_probes, -1)
            probe_sfs = np.array(probe_sfs_list).reshape(n_probes, 1)

            # --- Selection: merge parents + main offspring + probes ---
            all_off_decs = np.vstack([main_off_decs, probe_decs])
            all_off_objs = np.vstack([main_off_objs, probe_objs])
            all_off_cons = np.vstack([main_off_cons, probe_cons])
            all_off_sfs = np.vstack([
                main_off_sfs, probe_sfs])

            merged_decs = np.vstack([m_decs, all_off_decs])
            merged_objs = np.vstack([m_objs, all_off_objs])
            merged_cons = np.vstack([m_cons, all_off_cons])
            merged_sfs = np.vstack([m_sfs, all_off_sfs])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))

            # Record history
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs, all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


# ============================================================
# Operators
# ============================================================

def _dgd_crossover(par_dec1, par_dec2, QWE, L, sigma, D):
    """
    Gradient-guided blend crossover.

    Applies gradient descent step to both parents, then performs
    a random blend crossover.

    Parameters
    ----------
    par_dec1 : np.ndarray
        First parent (shape (D,)), already selected by random r1
    par_dec2 : np.ndarray
        Second parent (shape (D,)), already selected by random r2
    QWE : np.ndarray
        Gradient estimates, shape (2, D)
    L : float
        Smoothed gradient norm (> 0)
    sigma : float
        Perturbation step size
    D : int
        Decision vector dimensionality

    Returns
    -------
    off_dec : np.ndarray
        Offspring decision vector, shape (D,)
    """
    u = np.random.rand(D)
    cf = np.zeros(D)
    r1 = 0.6 * np.random.rand()
    r2 = -0.6 * np.random.rand()
    cf[u <= 0.5] = r1
    cf[u > 0.5] = r2

    # Gradient descent step on both parents
    p1 = par_dec1 - QWE[0, :] * sigma / L
    p2 = par_dec2 - QWE[1, :] * sigma / L

    # Blend crossover (only first offspring is used in MATLAB)
    off_dec = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    return off_dec
