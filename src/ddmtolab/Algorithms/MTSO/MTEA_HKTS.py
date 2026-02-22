"""
Multi-Task Evolutionary Algorithm with Hierarchical Knowledge Transfer Strategy (MTEA-HKTS)

This module implements MTEA-HKTS for multi-task optimization using KLD-based
variable ordering, adaptive knowledge transfer with hierarchical strategy
selection, and alternating GA/DE operators.

References
----------
    [1] Zhao, Ben, et al. "A Multi-Task Evolutionary Algorithm for Solving
        the Problem of Transfer Targets." Information Sciences, 681: 121214,
        2024.

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


class MTEA_HKTS:
    """
    Multi-Task EA with Hierarchical Knowledge Transfer Strategy.

    Uses KLD-based decision variable alignment across tasks, adaptive
    transfer probability control via a task selection table, and
    alternating GA (SBX+PM) / DE (rand/1/bin) operators.

    Three operation modes per generation:
    - sign=0 (10%): Separate transferred population evaluated independently
    - sign=1 (9%): Transferred individuals replace worst, standard GA/DE
    - sign=2 (81%): Transferred individuals in temp pop, cross-population GA/DE

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

    def __init__(self, problem, n=None, max_nfes=None, pTransfer=0.5,
                 mu=2, mum=5, F=0.5, CR=0.5, minx=0.1, Lb=0.1, Ub=0.7,
                 save_data=True, save_path='./Data', name='MTEA-HKTS',
                 disable_tqdm=True):
        """
        Initialize MTEA-HKTS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        pTransfer : float, optional
            Initial transfer portion (default: 0.5)
        mu : float, optional
            SBX crossover distribution index (default: 2)
        mum : float, optional
            Polynomial mutation distribution index (default: 5)
        F : float, optional
            DE mutation factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.5)
        minx : float, optional
            Minimum scale boundary (default: 0.1)
        Lb : float, optional
            Lower bound for transfer probability (default: 0.1)
        Ub : float, optional
            Upper bound for transfer probability (default: 0.7)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-HKTS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.pTransfer = pTransfer
        self.mu = mu
        self.mum = mum
        self.F = F
        self.CR = CR
        self.minx = minx
        self.Lb = Lb
        self.Ub = Ub
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-HKTS algorithm.

        Returns
        -------
        Results
            Optimization results
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Convert to unified space
        pop_decs, pop_cons = space_transfer(
            problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = objs
        maxD = pop_decs[0].shape[1]
        maxC = pop_cons[0].shape[1]

        # Sort each task by objective
        for t in range(nt):
            si = np.argsort(pop_objs[t][:, 0])
            pop_decs[t] = pop_decs[t][si]
            pop_objs[t] = pop_objs[t][si]
            pop_cons[t] = pop_cons[t][si]

        # Transfer probability table (diagonal = 0)
        scale = np.full((nt, nt), self.pTransfer)
        table = np.full((nt, nt), 0.5)
        np.fill_diagonal(table, 0.0)

        # Archive: 3N individuals per task (decs + objs)
        arch_decs = []
        arch_objs = []
        for t in range(nt):
            idx = np.random.randint(n, size=3 * n)
            arch_decs.append(pop_decs[t][idx].copy())
            arch_objs.append(pop_objs[t][idx].copy())

        gen = 1
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            for t in range(nt):
                n_t = len(pop_decs[t])

                # --- Operation mode ---
                if np.random.rand() < 0.9:
                    sign = 1 if np.random.rand() < 0.1 else 2
                else:
                    sign = 0

                # --- Source task selection (roulette wheel) ---
                m_pt = _select_task(table[t])

                # --- Transfer option ---
                if np.random.rand() < table[t, m_pt]:
                    option = 1  # KLD-aligned transfer
                else:
                    m_pt = t
                    option = 2  # Self with exclusion

                if np.random.rand() > 0.9:
                    option = 0  # Random mapping

                # --- Variable ordering (using archives) ---
                order = _var_order(arch_decs[m_pt], arch_decs[t],
                                   dims[m_pt], dims[t], option)

                # --- Population preparation ---
                transpop_decs = None
                if sign == 1 or sign == 2:
                    nTransfer = max(round(scale[t, m_pt] * n_t), 1)
                    nTransfer = min(nTransfer, n_t)
                    temp_decs = pop_decs[t][::-1].copy()  # worst first
                    trans = _m_transfer(
                        pop_decs[m_pt], pop_decs[t],
                        dims[m_pt], dims[t], nTransfer, order, option)
                    temp_decs[:nTransfer] = trans
                else:  # sign == 0
                    nTransfer = max(round(0.1 * n_t), 1)
                    temp_decs = pop_decs[t].copy()
                    transpop_decs = _m_transfer(
                        pop_decs[m_pt], pop_decs[t],
                        dims[m_pt], dims[t], nTransfer, order, option)

                # --- Generation ---
                op = 'GA' if t % 2 == 0 else 'DE'
                if sign == 1 or sign == 0:
                    if op == 'GA':
                        off_decs = _gen_ga(temp_decs, self.mu, self.mum)
                    else:
                        off_decs = _gen_de(temp_decs, self.F, self.CR)
                else:  # sign == 2
                    if op == 'GA':
                        off_decs = _gen_ga1(
                            pop_decs[t], temp_decs, self.mu, self.mum)
                    else:
                        off_decs = _gen_de1(
                            pop_decs[t], temp_decs, self.F, self.CR)

                # --- Evaluate offspring ---
                o_objs, o_cons_r = evaluation_single(
                    problem, off_decs[:, :dims[t]], t)
                o_cons = np.zeros((len(off_decs), maxC))
                if maxC > 0 and o_cons_r.shape[1] > 0:
                    o_cons[:, :o_cons_r.shape[1]] = o_cons_r
                nfes += len(off_decs)
                pbar.update(len(off_decs))

                # --- Merge and select ---
                if sign == 0 and transpop_decs is not None:
                    tp_objs, tp_cons_r = evaluation_single(
                        problem, transpop_decs[:, :dims[t]], t)
                    tp_cons = np.zeros((len(transpop_decs), maxC))
                    if maxC > 0 and tp_cons_r.shape[1] > 0:
                        tp_cons[:, :tp_cons_r.shape[1]] = tp_cons_r
                    nfes += len(transpop_decs)
                    pbar.update(len(transpop_decs))
                    m_decs = np.vstack([pop_decs[t], off_decs, transpop_decs])
                    m_objs = np.vstack([pop_objs[t], o_objs, tp_objs])
                    m_cons = np.vstack([pop_cons[t], o_cons, tp_cons])
                else:
                    m_decs = np.vstack([pop_decs[t], off_decs])
                    m_objs = np.vstack([pop_objs[t], o_objs])
                    m_cons = np.vstack([pop_cons[t], o_cons])
                    tp_objs = None

                # Sort by objective, select top N
                si = np.argsort(m_objs[:, 0])
                pop_decs[t] = m_decs[si[:n_t]]
                pop_objs[t] = m_objs[si[:n_t]]
                pop_cons[t] = m_cons[si[:n_t]]

                # --- Update archive ---
                seg = gen % 3
                s_idx = seg * n_t
                for i in range(n_t):
                    if pop_objs[t][i, 0] < arch_objs[t][s_idx + i, 0]:
                        arch_decs[t][s_idx + i] = pop_decs[t][i]
                        arch_objs[t][s_idx + i] = pop_objs[t][i]

                # --- Transfer quality tracking ---
                rev_pop = pop_decs[t][::-1]  # worst first
                quality = transpop_decs if sign == 0 else off_decs

                ia_sum = 0
                for r_idx in range(n_t):
                    for q_idx in range(len(quality)):
                        if np.array_equal(rev_pop[r_idx], quality[q_idx]):
                            ia_sum += (r_idx + 1)
                            break

                norm = n_t / 2.0 * (n_t + 1)
                ratio = ia_sum / norm if norm > 0 else 0

                if sign == 0:
                    scale[t, m_pt] = 0.1 + ratio * 0.4
                else:
                    scale[t, m_pt] = self.minx + ratio * (0.5 - self.minx)

                # --- Transfer probability table update ---
                if m_pt != t and option != 0:
                    denom = scale[t, m_pt] + scale[t, t]
                    if denom > 0:
                        temp_val = (scale[t, m_pt] - scale[t, t]) / denom
                    else:
                        temp_val = 0
                    w = 0.1 + np.random.rand() * 0.8
                    table[t, m_pt] = (
                        self.Lb + w * (table[t, m_pt] - self.Lb) +
                        (1 - w) * temp_val * (self.Ub - self.Lb))
                    if np.isnan(table[t, m_pt]) or table[t, m_pt] < self.Lb:
                        table[t, m_pt] = self.Lb
                    if table[t, m_pt] > self.Ub:
                        table[t, m_pt] = self.Ub

            # Record history
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, real_cons)
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
# Helper functions
# ============================================================

def _select_task(table_row):
    """Roulette wheel selection based on transfer probability row."""
    s = np.sum(table_row)
    if s <= 0:
        return np.random.randint(len(table_row))
    probs = table_row / s
    return np.random.choice(len(table_row), p=probs)


def _var_order(prev_decs, this_decs, prev_dims, this_dims, option):
    """
    KLD-based decision variable ordering between two populations.

    Maps each target dimension to the source dimension with minimum KLD.
    """
    if option == 0:
        return np.random.randint(prev_dims, size=this_dims)

    m_prev = np.mean(prev_decs[:, :prev_dims], axis=0)
    m_this = np.mean(this_decs[:, :this_dims], axis=0)

    prev_max = np.max(prev_decs[:, :prev_dims], axis=0)
    prev_min = np.min(prev_decs[:, :prev_dims], axis=0)
    this_max = np.max(this_decs[:, :this_dims], axis=0)
    this_min = np.min(this_decs[:, :this_dims], axis=0)
    prev_range = np.maximum(prev_max - prev_min, 1e-15)
    this_range = np.maximum(this_max - this_min, 1e-15)

    var_prev = np.var(prev_decs[:, :prev_dims], axis=0, ddof=1)
    var_this = np.var(this_decs[:, :this_dims], axis=0, ddof=1)
    var_prev = np.maximum(var_prev, 1e-15)
    var_this = np.maximum(var_this, 1e-15)

    order = np.zeros(this_dims, dtype=int)
    for i in range(this_dims):
        # Scale source variance by range ratio
        scaled_var = var_prev * (this_range[i] / prev_range) ** 2
        scaled_var = np.maximum(scaled_var, 1e-15)

        # KLD(source_j || target_i)
        KLD = (np.log2(np.sqrt(scaled_var) / np.sqrt(var_this[i])) +
               (var_this[i] + (m_prev - m_this[i]) ** 2) / (2 * scaled_var)
               - 0.5)

        if option == 2 and i < prev_dims:
            KLD[i] = np.max(KLD) + 1  # exclude same-index match

        order[i] = np.argmin(KLD)
    return order


def _m_transfer(prev_decs, this_decs, prev_dims, this_dims,
                n_transfer, order, option):
    """
    Transfer and transform decision variables from source to target.

    Scales variables by range ratio and shifts by mean difference
    when source/target distributions don't overlap sufficiently.
    """
    prev_max = np.max(prev_decs[:, :prev_dims], axis=0).copy()
    prev_min = np.min(prev_decs[:, :prev_dims], axis=0).copy()
    this_max = np.max(this_decs[:, :this_dims], axis=0)
    this_min = np.min(this_decs[:, :this_dims], axis=0)
    prev_range = np.maximum(
        np.max(prev_decs[:, :prev_dims], axis=0) -
        np.min(prev_decs[:, :prev_dims], axis=0), 1e-15)
    this_range = np.maximum(this_max - this_min, 1e-15)
    m_prev = np.mean(prev_decs[:, :prev_dims], axis=0)
    m_this = np.mean(this_decs[:, :this_dims], axis=0)

    # Scale prev bounds (in-place, cumulative per MATLAB code)
    if option != 0:
        for i in range(this_dims):
            j = order[i]
            sc = this_range[i] / prev_range[j]
            prev_max[j] = (prev_max[j] - m_prev[j]) * sc + m_prev[j]
            prev_min[j] = (prev_min[j] - m_prev[j]) * sc + m_prev[j]

    n_transfer = min(n_transfer, len(prev_decs))
    new_decs = this_decs[:n_transfer].copy()

    for nn in range(n_transfer):
        for i in range(this_dims):
            j = order[i]
            if option != 0 and prev_range[j] > 1e-15:
                new_decs[nn, i] = ((prev_decs[nn, j] - m_prev[j]) *
                                   (this_range[i] / prev_range[j]) +
                                   m_prev[j])
            else:
                new_decs[nn, i] = prev_decs[nn, j]

            # Check overlap for mean shift
            need_shift = True
            if option != 0:
                if (prev_min[j] <= this_max[i] and
                        prev_max[j] >= this_max[i] and
                        m_prev[j] <= this_max[i]):
                    need_shift = False
                if (prev_max[j] >= this_min[i] and
                        prev_min[j] <= this_min[i] and
                        m_prev[j] >= this_min[i]):
                    need_shift = False

            if need_shift:
                new_decs[nn, i] += m_this[i] - m_prev[j]

            new_decs[nn, i] = np.clip(new_decs[nn, i], 0, 1)

    return new_decs


def _gen_ga(pop_decs, mu, mum):
    """Standard GA generation: SBX crossover + polynomial mutation."""
    n_pop, D = pop_decs.shape
    perm = np.random.permutation(n_pop)
    n_pairs = n_pop // 2
    off = np.zeros((n_pairs * 2, D))

    for i in range(n_pairs):
        p1 = perm[i]
        p2 = perm[i + n_pop // 2]
        c1, c2 = crossover(pop_decs[p1], pop_decs[p2], mu=mu)
        off[2 * i] = np.clip(mutation(c1, mu=mum), 0, 1)
        off[2 * i + 1] = np.clip(mutation(c2, mu=mum), 0, 1)
    return off


def _gen_de(pop_decs, F, CR):
    """Standard DE/rand/1/bin with random boundary repair."""
    n_pop, D = pop_decs.shape
    off = np.zeros_like(pop_decs)

    for i in range(n_pop):
        indices = np.arange(n_pop)
        indices = indices[indices != i]
        a, b, c = np.random.choice(indices, 3, replace=False)

        v = pop_decs[a] + F * (pop_decs[b] - pop_decs[c])
        # Binomial crossover
        u = pop_decs[i].copy()
        j_rand = np.random.randint(D)
        mask = np.random.rand(D) < CR
        mask[j_rand] = True
        u[mask] = v[mask]

        # Random boundary repair
        rand_dec = np.random.rand(D)
        u[u > 1] = rand_dec[u > 1]
        u[u < 0] = rand_dec[u < 0]
        off[i] = u
    return off


def _gen_ga1(pop_decs, temp_decs, mu, mum):
    """Cross-population GA: SBX between temp and original, then mutation."""
    n_pop, D = pop_decs.shape
    perm = np.random.permutation(n_pop)
    off = np.zeros((n_pop, D))

    for i in range(n_pop):
        p1 = perm[i]
        c1, _ = crossover(temp_decs[i], pop_decs[p1], mu=mu)
        off[i] = np.clip(mutation(c1, mu=mum), 0, 1)
    return off


def _gen_de1(pop_decs, temp_decs, F, CR):
    """Cross-population DE: mutation from original, crossover with temp."""
    n_pop, D = pop_decs.shape
    off = np.zeros_like(pop_decs)

    for i in range(n_pop):
        indices = np.arange(n_pop)
        indices = indices[indices != i]
        a, b, c = np.random.choice(indices, 3, replace=False)

        v = pop_decs[a] + F * (pop_decs[b] - pop_decs[c])
        # Binomial crossover with temp (not original)
        u = temp_decs[i].copy()
        j_rand = np.random.randint(D)
        mask = np.random.rand(D) < CR
        mask[j_rand] = True
        u[mask] = v[mask]

        rand_dec = np.random.rand(D)
        u[u > 1] = rand_dec[u > 1]
        u[u < 0] = rand_dec[u < 0]
        off[i] = u
    return off
