"""
Multi-objective Multi-task Evolutionary Algorithm with Progressive Auto-Encoding (MO-MTEA-PAE)

This module implements MO-MTEA-PAE for multi-task multi-objective optimization problems.

References
----------
    [1] Q. Gu, Y. Li, W. Gong, Z. Yuan, B. Ning, C. Hu, and J. Wu, "Progressive Auto-Encoding for Domain Adaptation in Evolutionary Multi-Task Optimization," Applied Soft Computing, vol. 175, p. 113916, 2025.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MO_MTEA_PAE:
    """
    Multi-objective Multi-task Evolutionary Algorithm with Progressive Auto-Encoding.

    This algorithm features:
    - Kernel-based NFC (Nonlinear Feature Coupling) for cross-task knowledge transfer
    - Two transfer strategies: segment transfer (historical distribution) and stochastic transfer (current distribution)
    - Adaptive selection between DE and GA offspring generation
    - Adaptive selection between transfer types based on success rates
    - SPEA2 environmental selection per task
    - Elite solution transfer across tasks

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
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None,
                 Seg=10, TNum=20, TGap=5,
                 F=0.5, CR=0.9, MuC=20, MuM=15,
                 save_data=True, save_path='./Data',
                 name='MO-MTEA-PAE', disable_tqdm=True):
        """
        Initialize MO-MTEA-PAE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        Seg : int, optional
            Number of segments for DisPop update schedule (default: 10)
        TNum : int, optional
            Number of transfer solutions per transfer event (default: 20)
        TGap : int, optional
            Transfer gap in generations (default: 5)
        F : float, optional
            DE mutation factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        MuC : float, optional
            SBX crossover distribution index (default: 20)
        MuM : float, optional
            PM mutation distribution index (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-MTEA-PAE')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.Seg = Seg
        self.TNum = TNum
        self.TGap = TGap
        self.F = F
        self.CR = CR
        self.MuC = MuC
        self.MuM = MuM
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MO-MTEA-PAE algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        dims = problem.dims
        d_max = max(dims)

        # Clamp TNum
        TNum = min(self.TNum, min(n_per_task) // 2)

        # SegGap: how often to update DisPop (in generations)
        # total_gen ≈ max_nfes / N, SegGap = total_gen / Seg
        SegGap = max(1, max_nfes_per_task[0] // (n_per_task[0] * self.Seg))

        # Initialize populations in unified space (d_max)
        decs = []
        objs = []
        cons = []
        for t in range(nt):
            decs_t = np.random.rand(n_per_task[t], d_max)
            objs_t, cons_t = evaluation_single(problem, decs_t[:, :dims[t]], t)
            decs.append(decs_t)
            objs.append(objs_t)
            cons.append(cons_t)
        nfes_per_task = n_per_task.copy()

        # SPEA2 selection for initial population
        fit = []
        for t in range(nt):
            sel_idx, fit_t = self._spea2_select(objs[t], cons[t], n_per_task[t])
            decs[t] = decs[t][sel_idx]
            objs[t] = objs[t][sel_idx]
            cons[t] = cons[t][sel_idx]
            fit.append(fit_t[sel_idx])

        # Save native-space history
        all_decs = []
        all_objs = []
        all_cons = []
        for t in range(nt):
            all_decs.append([decs[t][:, :dims[t]].copy()])
            all_objs.append([objs[t].copy()])
            all_cons.append([cons[t].copy()])

        # Archive and DisPop (deep copies)
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]
        arc_cons = [c.copy() for c in cons]
        dis_decs = [d.copy() for d in decs]
        dis_objs = [o.copy() for o in objs]
        dis_cons = [c.copy() for c in cons]

        # KT/OP flags for current population (reset each gen)
        kt_flags = [np.zeros(n_per_task[t], dtype=int) for t in range(nt)]
        op_flags = [np.zeros(n_per_task[t], dtype=int) for t in range(nt)]

        # Success tracking (cumulative)
        succ_t = np.full((nt, 2), float(TNum))   # successful transfer count [seg, sto]
        sum_t = succ_t.copy()                     # total transfer count [seg, sto]
        succ_g = np.array([[float(n_per_task[t])] * 2 for t in range(nt)])  # successful gen [DE, GA]
        sum_g = succ_g.copy()                     # total gen [DE, GA]

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        gen_count = 0
        while sum(nfes_per_task) < total_nfes:
            gen_count += 1

            # === Step 1: Generate Offspring ===
            off_decs_list = []
            off_objs_list = []
            off_cons_list = []
            off_kt_list = []
            off_op_list = []

            for t in range(nt):
                Nt = n_per_task[t]
                pG = succ_g[t] / sum_g[t]

                off_decs_t, off_op_t = self._generation(
                    decs[t], Nt, pG, d_max)
                off_kt_t = np.zeros(Nt, dtype=int)

                off_decs_list.append(off_decs_t)
                off_kt_list.append(off_kt_t)
                off_op_list.append(off_op_t)

            # === Step 2: Knowledge Transfer ===
            if TNum > 0 and gen_count % self.TGap == 0:
                # Backup archive
                arc_back_decs = [d.copy() for d in arc_decs]
                arc_back_objs = [o.copy() for o in arc_objs]
                arc_back_cons = [c.copy() for c in arc_cons]

                # Re-sort DisPop and prepare sorted current pop (temporarily in arc)
                sorted_pop_decs = []
                sorted_pop_objs = []
                sorted_pop_cons = []
                for t in range(nt):
                    # SPEA2 re-sort DisPop
                    sel_idx, _ = self._spea2_select(dis_objs[t], dis_cons[t], n_per_task[t])
                    dis_decs[t] = dis_decs[t][sel_idx]
                    dis_objs[t] = dis_objs[t][sel_idx]
                    dis_cons[t] = dis_cons[t][sel_idx]

                    # Sorted current pop (used as "Arc" for stochastic transfer)
                    sel_idx2, _ = self._spea2_select(objs[t], cons[t], n_per_task[t])
                    sorted_pop_decs.append(decs[t][sel_idx2])
                    sorted_pop_objs.append(objs[t][sel_idx2])
                    sorted_pop_cons.append(cons[t][sel_idx2])

                for t in range(nt):
                    # Pick random source task
                    k = np.random.randint(nt)
                    while k == t:
                        k = np.random.randint(nt)

                    Nt = n_per_task[t]
                    Nk = n_per_task[k]

                    # --- Segment Transfer ---
                    # Select TNum best from Pop{k}
                    nd_idx = np.where(fit[k] < 1)[0]
                    if len(nd_idx) < TNum:
                        s_best_idx = np.arange(min(TNum, Nk))
                    else:
                        s_best_idx = nd_idx[np.random.permutation(len(nd_idx))[:TNum]]
                    s_best_decs = decs[k][s_best_idx, :dims[k]]

                    # NFC with DisPop distributions
                    t_dis = dis_decs[t][:, :dims[t]]
                    k_dis = dis_decs[k][:, :dims[k]]
                    seg_dec = self._nfc(t_dis, k_dis, s_best_decs)
                    # Pad to d_max
                    if dims[t] < d_max:
                        seg_dec = np.hstack([seg_dec,
                                             np.random.rand(seg_dec.shape[0], d_max - dims[t])])

                    # --- Stochastic Replacement Transfer ---
                    s_best_decs2 = sorted_pop_decs[k][:TNum, :dims[k]]

                    t_dis2 = sorted_pop_decs[t][:, :dims[t]]
                    k_dis2 = sorted_pop_decs[k][:, :dims[k]]
                    sto_dec = self._nfc(t_dis2, k_dis2, s_best_decs2)
                    if dims[t] < d_max:
                        sto_dec = np.hstack([sto_dec,
                                             np.random.rand(sto_dec.shape[0], d_max - dims[t])])

                    # Select between segment and stochastic
                    pT = succ_t[t] / sum_t[t]
                    tr_decs = np.zeros((TNum, d_max))
                    tr_kt = np.zeros(TNum, dtype=int)
                    for i in range(TNum):
                        if np.random.rand() < pT[0] / (pT[0] + pT[1]):
                            tr_decs[i] = seg_dec[i]
                            tr_kt[i] = 1  # segment
                        else:
                            tr_decs[i] = sto_dec[i]
                            tr_kt[i] = 2  # stochastic
                    tr_decs = np.clip(tr_decs, 0, 1)

                    # Replace random offspring with transferred solutions
                    replace_idx = np.random.permutation(len(off_decs_list[t]))[:TNum]
                    off_decs_list[t][replace_idx] = tr_decs
                    off_kt_list[t][replace_idx] = tr_kt
                    off_op_list[t][replace_idx] = 0

                # Restore archive
                arc_decs = arc_back_decs
                arc_objs = arc_back_objs
                arc_cons = arc_back_cons

            # Update DisPop every SegGap generations
            if gen_count % SegGap == 0:
                dis_decs = [d.copy() for d in decs]
                dis_objs = [o.copy() for o in objs]
                dis_cons = [c.copy() for c in cons]

            # === Step 3: Environmental Selection ===
            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                Nt = n_per_task[t]

                # Reset parent flags
                kt_flags[t] = np.zeros(Nt, dtype=int)
                op_flags[t] = np.zeros(Nt, dtype=int)

                # Elite solution transfer: replace one random offspring with best from another task
                k = np.random.randint(nt)
                while k == t:
                    k = np.random.randint(nt)
                rnd_idx = np.random.randint(Nt)
                off_decs_list[t][rnd_idx] = decs[k][0].copy()  # best from task k (index 0)
                off_kt_list[t][rnd_idx] = 3  # elite transfer
                off_op_list[t][rnd_idx] = 0

                # Track total counts (before evaluation)
                sum_t[t, 0] += np.sum(off_kt_list[t] == 1)
                sum_t[t, 1] += np.sum(off_kt_list[t] == 2)
                sum_g[t, 0] += np.sum(off_op_list[t] == 1)
                sum_g[t, 1] += np.sum(off_op_list[t] == 2)

                # Evaluate offspring
                off_objs_t, off_cons_t = evaluation_single(
                    problem, off_decs_list[t][:, :dims[t]], t)
                nfes_per_task[t] += len(off_decs_list[t])
                pbar.update(len(off_decs_list[t]))

                # Merge parent + offspring
                merged_decs = np.vstack([decs[t], off_decs_list[t]])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t])
                merged_kt = np.concatenate([kt_flags[t], off_kt_list[t]])
                merged_op = np.concatenate([op_flags[t], off_op_list[t]])

                # SPEA2 selection
                sel_idx, fit_all = self._spea2_select(
                    merged_objs, merged_cons, Nt)

                # Identify failed solutions
                all_idx_set = set(range(len(merged_decs)))
                sel_idx_set = set(sel_idx.tolist())
                failed_idx = np.array(sorted(all_idx_set - sel_idx_set))

                # Update success counts
                succ_t[t, 0] += np.sum(merged_kt[sel_idx] == 1)
                succ_t[t, 1] += np.sum(merged_kt[sel_idx] == 2)
                succ_g[t, 0] += np.sum(merged_op[sel_idx] == 1)
                succ_g[t, 1] += np.sum(merged_op[sel_idx] == 2)

                # Update population
                decs[t] = merged_decs[sel_idx]
                objs[t] = merged_objs[sel_idx]
                cons[t] = merged_cons[sel_idx]
                fit[t] = fit_all[sel_idx]
                kt_flags[t] = merged_kt[sel_idx]
                op_flags[t] = merged_op[sel_idx]

                # Update archive with failed solutions
                if len(failed_idx) > 0:
                    failed_decs = merged_decs[failed_idx]
                    failed_objs = merged_objs[failed_idx]
                    failed_cons = merged_cons[failed_idx]
                    arc_decs[t] = np.vstack([arc_decs[t], failed_decs])
                    arc_objs[t] = np.vstack([arc_objs[t], failed_objs])
                    arc_cons[t] = np.vstack([arc_cons[t], failed_cons])

                # Trim archive to N by random selection
                if len(arc_decs[t]) > Nt:
                    perm = np.random.permutation(len(arc_decs[t]))[:Nt]
                    arc_decs[t] = arc_decs[t][perm]
                    arc_objs[t] = arc_objs[t][perm]
                    arc_cons[t] = arc_cons[t][perm]

                # Append native-space history
                append_history(all_decs[t], decs[t][:, :dims[t]],
                               all_objs[t], objs[t],
                               all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, pop_decs, N, pG, d_max):
        """
        Generate offspring using adaptive DE or GA.

        Parameters
        ----------
        pop_decs : np.ndarray, shape (N, d_max)
            Parent population (sorted by SPEA2 fitness, index 0 = best)
        N : int
            Population size
        pG : np.ndarray, shape (2,)
            Success rates for [DE, GA]
        d_max : int
            Unified dimension

        Returns
        -------
        off_decs : np.ndarray, shape (N, d_max)
        op_flags : np.ndarray, shape (N,)
            1=DE, 2=GA
        """
        off_decs = np.zeros((N, d_max))
        op_flags = np.zeros(N, dtype=int)

        # Tournament selection indices for GA
        rank_vals = np.arange(1, N + 1)
        ind_order = tournament_selection(2, 2 * N, rank_vals)

        for i in range(N):
            p_de = pG[0] / (pG[0] + pG[1]) if (pG[0] + pG[1]) > 0 else 0.5

            if np.random.rand() < p_de:
                # DE/rand/1 with rank-based selection
                x1 = self._rank_selection(N, exclude=[i])
                x2 = self._rank_selection(N, exclude=[i, x1])
                x3 = self._random_selection(N, exclude=[i, x1, x2])

                mutant = pop_decs[x1] + self.F * (pop_decs[x2] - pop_decs[x3])
                # DE binomial crossover
                j_rand = np.random.randint(d_max)
                mask = np.random.rand(d_max) < self.CR
                mask[j_rand] = True
                off_decs[i] = np.where(mask, mutant, pop_decs[i])
                op_flags[i] = 1
            else:
                # GA: SBX crossover + PM mutation
                p1 = ind_order[i]
                p2 = ind_order[i + N // 2] if (i + N // 2) < len(ind_order) else ind_order[-1]
                off1, _ = crossover(pop_decs[p1], pop_decs[p2], mu=self.MuC)
                off1 = mutation(off1, mu=self.MuM)
                off_decs[i] = off1
                op_flags[i] = 2

            off_decs[i] = np.clip(off_decs[i], 0, 1)

        return off_decs, op_flags

    @staticmethod
    def _rank_selection(N, exclude=None):
        """Rank-based DE parent selection. Lower index = better rank."""
        if exclude is None:
            exclude = []
        for _ in range(1000):
            x = np.random.randint(N)
            # rank = x+1 (1-indexed), acceptance prob = (N - rank) / N
            if x not in exclude and np.random.rand() < (N - (x + 1)) / N:
                return x
        candidates = [i for i in range(N) if i not in exclude]
        return np.random.choice(candidates) if candidates else 0

    @staticmethod
    def _random_selection(N, exclude=None):
        """Random selection excluding specified indices."""
        if exclude is None:
            exclude = []
        candidates = [i for i in range(N) if i not in exclude]
        return np.random.choice(candidates) if candidates else 0

    @staticmethod
    def _spea2_select(objs, cons, N):
        """
        SPEA2 environmental selection.

        Returns
        -------
        sel_idx : np.ndarray
            Indices of selected solutions (sorted by fitness)
        fitness : np.ndarray
            SPEA2 fitness of all input solutions
        """
        pop_size = objs.shape[0]
        N = min(N, pop_size)

        fitness = spea2_fitness(objs, cons)

        next_mask = fitness < 1
        n_selected = np.sum(next_mask)

        if n_selected < N:
            sorted_idx = np.argsort(fitness)
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[sorted_idx[:N]] = True
        elif n_selected > N:
            selected_idx = np.where(next_mask)[0]
            keep_idx = spea2_truncation(objs[selected_idx], N)
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[selected_idx[keep_idx]] = True

        # Sort selected by fitness
        sel_idx = np.where(next_mask)[0]
        sorted_by_fit = np.argsort(fitness[sel_idx])
        sel_idx = sel_idx[sorted_by_fit]

        return sel_idx, fitness

    @staticmethod
    def _nfc(target_pop, source_pop, source_best, kernel='poly'):
        """
        Nonlinear Feature Coupling (NFC) for cross-task knowledge transfer.

        Maps source_best solutions from source task to target task space
        using kernel autoencoding.

        Parameters
        ----------
        target_pop : np.ndarray, shape (N, D_target)
            Target task population distribution
        source_pop : np.ndarray, shape (N, D_source)
            Source task population distribution
        source_best : np.ndarray, shape (TNum, D_source)
            Solutions to transfer from source task

        Returns
        -------
        mapped : np.ndarray, shape (TNum, D_target)
        """
        D_target = target_pop.shape[1]
        D_source = source_pop.shape[1]

        # Pad to same dimension
        T_H = target_pop.copy()
        S_H = source_pop.copy()
        if D_target < D_source:
            T_H = np.hstack([T_H, np.zeros((T_H.shape[0], D_source - D_target))])
        elif D_target > D_source:
            S_H = np.hstack([S_H, np.zeros((S_H.shape[0], D_target - D_source))])

        # Transpose to (dim, N) - columns are samples
        S_H_T = S_H.T.astype(np.float64)
        T_H_T = T_H.T.astype(np.float64)

        # Kernel matrix
        kk = MO_MTEA_PAE._kernelmatrix(kernel, S_H_T, S_H_T)

        d = kk.shape[0]
        Q0 = kk @ kk.T
        P = T_H_T @ kk.T
        reg = 1e-5 * np.eye(d)
        W = P @ np.linalg.pinv(Q0 + reg)

        # Map source_best
        S_Best = source_best.copy().astype(np.float64)
        if D_target <= D_source:
            K_map = MO_MTEA_PAE._kernelmatrix(kernel, S_H_T, S_Best.T)
            mapped = (W @ K_map).T
            mapped = mapped[:, :D_target]
        else:
            S_Best = np.hstack([S_Best, np.zeros((S_Best.shape[0], D_target - D_source))])
            K_map = MO_MTEA_PAE._kernelmatrix(kernel, S_H_T, S_Best.T)
            mapped = (W @ K_map).T

        return mapped

    @staticmethod
    def _kernelmatrix(kernel, X, X2):
        """
        Compute kernel matrix between column-format data.

        Parameters
        ----------
        kernel : str
            Kernel type ('poly', 'lin', 'rbf')
        X : np.ndarray, shape (dim, N1)
        X2 : np.ndarray, shape (dim, N2)

        Returns
        -------
        K : np.ndarray, shape (N1, N2)
        """
        # Dimension padding
        d1 = X.shape[0]
        d2 = X2.shape[0]
        if d1 < d2:
            X = np.vstack([X, np.zeros((d2 - d1, X.shape[1]))])
        elif d1 > d2:
            X2 = np.vstack([X2, np.zeros((d1 - d2, X2.shape[1]))])

        if kernel == 'poly':
            b, d = 0.1, 5
            return (X.T @ X2 + b) ** d
        elif kernel == 'lin':
            return X.T @ X2
        elif kernel == 'rbf':
            n1sq = np.sum(X ** 2, axis=0)
            n2sq = np.sum(X2 ** 2, axis=0)
            D = n1sq[:, None] + n2sq[None, :] - 2 * X.T @ X2
            return np.exp(-D / (2 * 0.1 ** 2))
        else:
            raise ValueError(f'Unsupported kernel: {kernel}')
