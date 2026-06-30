"""
MFEA-Uni-II: MFEA-II under a *unified shared-LHS initialization* protocol.

Motivation
----------
The vanilla MFEA-II / ICL-MFEA-II comparison is budget-unfair: ICL-MFEA-II spends
an extra `lhs_archive` evaluations per task to seed its similarity prompt, so under a
fixed evaluation budget it runs *fewer* generations than MFEA-II.  MFEA-Uni-II and
ICL-MFEA-Uni-II remove that asymmetry by sharing one initialization protocol:

  1. Draw a single space-filling LHS of `lhs_init` points per task, evaluated once.
     This is the *only* up-front cost, charged identically to both methods.
  2. The best `n` of those `lhs_init` points become the initial population.
  3. MFEA-Uni-II then proceeds exactly like MFEA-II (online-MLE RMP), discarding the
     remaining LHS points.  ICL-MFEA-Uni-II keeps the full LHS as its ICL prompt.

Because both pay `lhs_init * n_tasks` up front and `n * n_tasks` per generation, they
run the **same number of generations**:  (max_nfes*nt - lhs_init*nt) / (n*nt).

The LHS draw is seeded deterministically from the experiment name's (problem, run)
suffix (see `derive_shared_seed`), so the matching MFEA-Uni-II and ICL-MFEA-Uni-II
runs start from the *identical* sample — the only difference between them is the RMP
estimator, which is the whole point of the comparison.

References
----------
    [1] Bali et al. "Multifactorial evolutionary algorithm with online transfer
        parameter estimation: MFEA-II." IEEE TEVC 24.1 (2019): 69-83.
"""
import time
import hashlib

import numpy as np
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.MTSO.MFEA import mfea_selection
from ddmtolab.Algorithms.MTSO.MFEA_II import learnRMP


def derive_shared_seed(name):
    """
    Derive a deterministic RNG seed from an experiment name's (problem, run) suffix.

    BatchExperiment names each run ``{algo_name}_{problem_name}_{run_id}``.  Stripping
    the leading algorithm token and hashing the remaining ``{problem}_{run}`` key gives
    a seed that is identical for MFEA-Uni-II and ICL-MFEA-Uni-II on the same problem and
    run — so both draw the same shared LHS — yet varies across problems and runs.

    Falls back to hashing the whole name if it has no ``_`` separators.
    """
    parts = name.split('_')
    key = '_'.join(parts[-2:]) if len(parts) >= 2 else name
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


def select_best_init(decs, objs, cons, n):
    """
    Select the best `n` individuals per task from an evaluated LHS sample.

    Single-objective minimization: keep the `n` lowest-objective points per task.

    Parameters
    ----------
    decs, objs, cons : list[np.ndarray]
        Per-task LHS decision variables, objectives, constraints.
    n : int
        Initial population size per task.

    Returns
    -------
    sel_decs, sel_objs, sel_cons : list[np.ndarray]
        The best-`n` subset per task.
    """
    sel_decs, sel_objs, sel_cons = [], [], []
    for i in range(len(decs)):
        order = np.argsort(objs[i].ravel())[:n]
        sel_decs.append(decs[i][order])
        sel_objs.append(objs[i][order])
        sel_cons.append(cons[i][order])
    return sel_decs, sel_objs, sel_cons


class MFEA_Uni_II:
    """
    MFEA-II with the unified shared-LHS initialization protocol.

    Identical to `MFEA_II` except the initial population is the best `n` of a shared
    space-filling LHS (`lhs_init` points per task) rather than a fresh random sample.
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
        'max_nfes': 'equal',
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, lhs_init=200,
                 save_data=True, save_path='./Data',
                 name='MFEA-Uni-II', disable_tqdm=True):
        """
        Parameters
        ----------
        n : int
            Initial population size per task (default 100).
        max_nfes : int
            Max function evaluations per task (default 10000).
        lhs_init : int
            Shared space-filling LHS size per task, evaluated once at start; the best
            `n` become the initial population (default 200).  Must match the value
            used by the paired ICL-MFEA-Uni-II run for the budgets to align.
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.lhs_init = lhs_init
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        L = self.lhs_init
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # --- Shared LHS initialization ---
        # Seed from the (problem, run) suffix so this draw is identical to the paired
        # ICL-MFEA-Uni-II run; the best n of the LHS become the initial population.
        np.random.seed(derive_shared_seed(self.name))
        lhs_decs = initialization(problem, L, method='lhs')
        lhs_objs, lhs_cons = evaluation(problem, lhs_decs)
        nfes = L * nt

        decs, objs, cons = select_best_init(lhs_decs, lhs_objs, lhs_cons, n)
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform populations to unified search space for knowledge transfer
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni')
        pop_objs = objs

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # Learn RMP matrix online (pure MLE)
            rmpMatrix = learnRMP(pop_decs, dims)

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)

            off_decs = np.zeros_like(pop_decs)
            off_objs = np.zeros_like(pop_objs)
            off_cons = np.zeros_like(pop_cons)
            off_sfs = np.zeros_like(pop_sfs)

            # Randomly pair individuals for assortative mating
            shuffled_index = np.random.permutation(pop_decs.shape[0])

            for i in range(0, len(shuffled_index), 2):
                p1 = shuffled_index[i]
                p2 = shuffled_index[i + 1]
                sf1 = pop_sfs[p1].item()
                sf2 = pop_sfs[p2].item()
                rmp_value = rmpMatrix[sf1, sf2]

                # Cross-task transfer: crossover if same task or rmp condition met
                if sf1 == sf2 or np.random.rand() < rmp_value:
                    off_dec1, off_dec2 = crossover(pop_decs[p1, :], pop_decs[p2, :], mu=2)
                    off_decs[i, :] = off_dec1
                    off_decs[i + 1, :] = off_dec2
                    off_sfs[i] = np.random.choice([sf1, sf2])
                    off_sfs[i + 1] = sf1 if off_sfs[i] == sf2 else sf2
                else:
                    # No transfer: crossover within the same task
                    for x, p in enumerate([p1, p2]):
                        sf = pop_sfs[p].item()
                        same_sf_indices = np.where(pop_sfs.flatten() == sf)[0]
                        same_sf_indices = same_sf_indices[same_sf_indices != p]
                        idx = np.random.choice(same_sf_indices)

                        off_dec_curr, _ = crossover(pop_decs[p, :], pop_decs[idx, :], mu=2)
                        off_dec_curr = mutation(off_dec_curr, mu=5)
                        off_decs[i + x, :] = off_dec_curr
                        off_sfs[i + x] = sf

                # Trim to task dimensionality and evaluate offspring
                task_idx1 = off_sfs[i].item()
                task_idx2 = off_sfs[i + 1].item()

                off_dec1_trimmed = off_decs[i, :dims[task_idx1]]
                off_dec2_trimmed = off_decs[i + 1, :dims[task_idx2]]
                off_objs[i, :], off_cons[i, :] = evaluation_single(problem, off_dec1_trimmed, task_idx1)
                off_objs[i + 1, :], off_cons[i + 1, :] = evaluation_single(problem, off_dec2_trimmed, task_idx2)

            # Merge parents and offspring populations
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(
                (pop_decs, off_decs), (pop_objs, off_objs), (pop_cons, off_cons), (pop_sfs, off_sfs)
            )

            # Environmental selection: keep best n individuals per task
            pop_decs, pop_objs, pop_cons, pop_sfs = mfea_selection(pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            # Transform back to native search space
            decs, cons = space_transfer(problem=problem, decs=pop_decs, cons=pop_cons, type='real')

            nfes += n * nt
            pbar.update(n * nt)

            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=max_nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name, save_data=self.save_data)

        return results
