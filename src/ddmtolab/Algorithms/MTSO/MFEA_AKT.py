"""
Multifactorial Evolutionary Algorithm with Adaptive Knowledge Transfer (MFEA-AKT)

This module implements MFEA-AKT for multi-task optimization with adaptive crossover
operator selection for inter-task knowledge transfer.

References
----------
    [1] Zhou, Lei, et al. "Toward Adaptive Knowledge Transfer in Multifactorial
        Evolutionary Computation." IEEE Transactions on Cybernetics, 51(5):
        2563-2576, 2021.

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


class MFEA_AKT:
    """
    Multifactorial Evolutionary Algorithm with Adaptive Knowledge Transfer.

    Extends MFEA with 6 crossover operators for inter-task transfer and an
    adaptive mechanism to select the best operator based on improvement tracking.

    The 6 crossover operators are:
        0: Two-point crossover
        1: Uniform crossover
        2: Arithmetical crossover (r=0.25)
        3: Geometric crossover (r=0.2)
        4: BLX-alpha crossover (a=0.3)
        5: SBX crossover

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

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, gap=20, muc=2, mum=5,
                 save_data=True, save_path='./Data', name='MFEA-AKT', disable_tqdm=True):
        """
        Initialize MFEA-AKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.3)
        gap : int, optional
            History window size for operator selection fallback (default: 20)
        muc : float, optional
            Distribution index for SBX crossover (default: 2)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-AKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.gap = gap
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MFEA-AKT algorithm.

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
        pop_size = n * nt  # total population size

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

        # Per-individual metadata (as lists of arrays, one per task)
        pop_cxf = [np.random.randint(0, 6, size=(n,)) for _ in range(nt)]

        # Record of best CX factor per generation (for fallback selection)
        cfb_record = []

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        gen = 1
        while nfes < max_nfes:
            # Merge populations from all tasks into single arrays
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)
            m_cxf = np.concatenate(pop_cxf)  # (pop_size,)

            # --- Generation ---
            off_decs = np.zeros_like(m_decs)
            off_objs = np.full_like(m_objs, np.inf)
            off_cons = np.zeros_like(m_cons)
            off_sfs = np.zeros_like(m_sfs)
            off_cxf = np.zeros(pop_size, dtype=int)
            off_istran = np.zeros(pop_size, dtype=bool)
            off_parnum = np.full(pop_size, -1, dtype=int)  # -1 = no parent tracked

            shuffled_index = np.random.permutation(pop_size)

            for i in range(0, pop_size, 2):
                p1 = shuffled_index[i]
                p2 = shuffled_index[i + pop_size // 2] if i + pop_size // 2 < pop_size else shuffled_index[(i + 1) % pop_size]
                sf1 = m_sfs[p1].item()
                sf2 = m_sfs[p2].item()

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    p = [p1, p2]
                    if sf1 == sf2:
                        # Same task: SBX crossover
                        off_decs[i], off_decs[i + 1] = crossover(
                            m_decs[p1], m_decs[p2], mu=self.muc)
                        off_cxf[i] = m_cxf[p1]
                        off_cxf[i + 1] = m_cxf[p2]
                        off_istran[i] = False
                        off_istran[i + 1] = False
                    else:
                        # Different tasks: hyberCX with adaptive operator
                        alpha = m_cxf[p[np.random.randint(2)]]
                        off_decs[i], off_decs[i + 1] = _hyber_crossover(
                            m_decs[p1], m_decs[p2], alpha, self.muc)
                        off_cxf[i] = alpha
                        off_cxf[i + 1] = alpha
                        off_istran[i] = True
                        off_istran[i + 1] = True

                    # Task imitation (random parent's task)
                    for k in [i, i + 1]:
                        rand_p = p[np.random.randint(2)]
                        off_sfs[k] = m_sfs[rand_p]
                        if off_istran[k]:
                            off_parnum[k] = rand_p
                else:
                    # No transfer: mutation
                    off_decs[i] = mutation(m_decs[p1], mu=self.mum)
                    off_decs[i + 1] = mutation(m_decs[p2], mu=self.mum)
                    off_sfs[i] = sf1
                    off_sfs[i + 1] = sf2
                    off_cxf[i] = m_cxf[p1]
                    off_cxf[i + 1] = m_cxf[p2]

                # Clip to [0, 1]
                off_decs[i] = np.clip(off_decs[i], 0, 1)
                off_decs[i + 1] = np.clip(off_decs[i + 1], 0, 1)

            # --- Evaluation ---
            for idx in range(pop_size):
                t = off_sfs[idx].item()
                dec_trimmed = off_decs[idx, :dims[t]]
                off_objs[idx], off_cons[idx] = evaluation_single(
                    problem, dec_trimmed, t)

            nfes += pop_size
            pbar.update(pop_size)

            # --- Calculate best CXFactor ---
            imp_num = np.full(6, -np.inf)  # max improvement per operator
            has_any_improvement = False
            for idx in range(pop_size):
                if off_parnum[idx] >= 0:
                    cfc = off_objs[idx, 0]
                    par_idx = off_parnum[idx]
                    pfc = m_objs[par_idx, 0]
                    if pfc != 0:
                        imp = (pfc - cfc) / abs(pfc)
                    else:
                        imp = -cfc if cfc > 0 else 0.0
                    cx = off_cxf[idx]
                    if imp > imp_num[cx]:
                        imp_num[cx] = imp
                        if imp > 0:
                            has_any_improvement = True

            if has_any_improvement:
                # Best operator is the one with highest max improvement
                max_idx = np.argmax(imp_num)
            else:
                # Fallback: most frequent best operator in recent history
                if len(cfb_record) > 0:
                    start = max(0, len(cfb_record) - self.gap)
                    recent = cfb_record[start:]
                    counts = np.bincount(recent, minlength=6)
                    max_idx = np.argmax(counts)
                else:
                    max_idx = np.random.randint(0, 6)

            cfb_record.append(max_idx)

            # --- Adaptive CXFactor update ---
            for idx in range(pop_size):
                if off_parnum[idx] >= 0:
                    cfc = off_objs[idx, 0]
                    par_idx = off_parnum[idx]
                    pfc = m_objs[par_idx, 0]
                    imp = (pfc - cfc) / abs(pfc) if pfc != 0 else 0.0
                    if imp < 0:
                        # Offspring worsened → adopt best operator
                        off_cxf[idx] = max_idx
                else:
                    # Non-transfer offspring: 50% best, 50% random
                    if np.random.rand() < 0.5:
                        off_cxf[idx] = max_idx
                    else:
                        off_cxf[idx] = np.random.randint(0, 6)

            # --- Selection: merge parents + offspring, keep best n per task ---
            merged_decs = np.vstack([m_decs, off_decs])
            merged_objs = np.vstack([m_objs, off_objs])
            merged_cons = np.vstack([m_cons, off_cons])
            merged_sfs = np.vstack([m_sfs, off_sfs])
            merged_cxf = np.concatenate([m_cxf, off_cxf])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            pop_cxf = []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                t_cxf = merged_cxf[indices]

                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))
                pop_cxf.append(t_cxf[sel])

            # Record history (transform back to real space for storage)
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
# Crossover operators for knowledge transfer
# ============================================================

def _hyber_crossover(par1, par2, alpha, muc=2):
    """
    Apply one of 6 crossover operators based on alpha.

    Parameters
    ----------
    par1, par2 : np.ndarray
        Parent decision vectors, shape (d,)
    alpha : int
        Crossover operator index (0-5)
    muc : float
        Distribution index for SBX (operator 5)

    Returns
    -------
    off1, off2 : np.ndarray
        Offspring decision vectors, shape (d,)
    """
    if alpha == 0:
        off1 = _tp_crossover(par1, par2)
        off2 = _tp_crossover(par2, par1)
    elif alpha == 1:
        off1 = _uf_crossover(par1, par2)
        off2 = _uf_crossover(par2, par1)
    elif alpha == 2:
        off1 = _ari_crossover(par1, par2)
        off2 = _ari_crossover(par2, par1)
    elif alpha == 3:
        off1 = _geo_crossover(par1, par2)
        off2 = _geo_crossover(par2, par1)
    elif alpha == 4:
        off1 = _blxa_crossover(par1, par2, a=0.3)
        off2 = _blxa_crossover(par2, par1, a=0.3)
    else:  # alpha == 5
        off1, off2 = crossover(par1, par2, mu=muc)
    return off1, off2


def _tp_crossover(par1, par2):
    """Two-point crossover."""
    d = len(par1)
    i, j = sorted(np.random.randint(0, d, size=2))
    off = par1.copy()
    off[i:j + 1] = par2[i:j + 1]
    return off


def _uf_crossover(par1, par2):
    """Uniform crossover."""
    mask = np.random.randint(0, 2, size=len(par1)).astype(bool)
    off = par1.copy()
    off[mask] = par2[mask]
    return off


def _ari_crossover(par1, par2, r=0.25):
    """Arithmetical crossover with ratio r."""
    return r * par1 + (1 - r) * par2


def _geo_crossover(par1, par2, r=0.2):
    """Geometric crossover with ratio r."""
    p1 = np.maximum(par1, 1e-15)
    p2 = np.maximum(par2, 1e-15)
    return np.power(p1, r) * np.power(p2, 1 - r)


def _blxa_crossover(par1, par2, a=0.3):
    """BLX-alpha crossover."""
    cmin = np.minimum(par1, par2)
    cmax = np.maximum(par1, par2)
    interval = cmax - cmin
    low = cmin - interval * a
    high = cmax + interval * a
    return low + (high - low) * np.random.rand(len(par1))
