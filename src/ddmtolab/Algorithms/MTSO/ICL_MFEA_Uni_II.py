"""
ICL-MFEA-Uni-II: ICL-MFEA-II under the *unified shared-LHS initialization* protocol.

This is the ICL counterpart to `MFEA_Uni_II`.  Both methods share one initialization:

  1. Draw a single space-filling LHS of `lhs_init` points per task, evaluated once
     (the only up-front cost, charged identically to MFEA-Uni-II).
  2. The best `n` of those points become the initial population.
  3. The *full* `lhs_init` LHS is retained as the ICL prompt / similarity archive that
     feeds the zero-shot inter-task relatedness prior ρ from generation 0.

Crucially, the archive is *not* an extra evaluation cost here — it is exactly the same
LHS that produced the initial population.  So ICL-MFEA-Uni-II and MFEA-Uni-II both pay
`lhs_init * n_tasks` up front and `n * n_tasks` per generation, and therefore run the
**same number of generations**.  The matching runs even share the identical LHS sample
(seeded from the (problem, run) suffix via `derive_shared_seed`); the only difference
between the two methods is the RMP estimator:

    MFEA-Uni-II      : rmp*_ij = argmin_rmp  loglik(rmp, popdata)              (MLE)
    ICL-MFEA-Uni-II  : rmp*_ij = argmin_rmp  loglik(rmp, popdata) + λ(g)·(rmp − ρ_ij)²

with ρ the symmetrised TabPFN cross-predictive task similarity and λ(g) decaying over
generations.  As before, λ → 0 hands estimation back to the pure MFEA-II MLE.

References
----------
    [1] Bali et al. "Multifactorial evolutionary algorithm with online transfer
        parameter estimation: MFEA-II." IEEE TEVC 24.1 (2019): 69-83.
"""
import time

import numpy as np
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.MTSO.MFEA import mfea_selection
from ddmtolab.Algorithms.MTSO.ICL_MFEA_II import learnRMP_MAP
from ddmtolab.Algorithms.MTSO.MFEA_Uni_II import derive_shared_seed, select_best_init
from ddmtolab.Methods.Algo_Methods.tfm_task_covar_utils import (
    compute_task_similarity_matrix_directed_classification,
)


class ICL_MFEA_Uni_II:
    """
    ICL-MFEA-II with the unified shared-LHS initialization protocol.

    The initial population is the best `n` of a shared LHS (`lhs_init` points per task);
    that same LHS is kept as the ICL similarity prompt.  No evaluations are spent beyond
    the shared LHS, so the generation count matches `MFEA_Uni_II` exactly.
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

    def __init__(self, problem, n=None, max_nfes=None,
                 lambda_0=1.0, lambda_decay=0.05, tau=1.0,
                 n_classes=2, n_estimators=1, device='cpu',
                 lhs_init=200, rho_archive_cap=300,
                 save_data=True, save_path='./Data',
                 name='ICL-MFEA-Uni-II', disable_tqdm=True):
        """
        Parameters
        ----------
        n : int
            Initial population size per task (default 100).
        max_nfes : int
            Max function evaluations per task (default 10000).
        lambda_0 : float
            Initial MAP regularisation weight (default 1.0 — locked value).
        lambda_decay : float
            Exponential decay rate per generation (λ(g) = lambda_0 * exp(-g*decay)).
        tau : float
            Sharpness of the CE→similarity map passed to the TabPFN util.
        n_classes : int
            Quantile bins for the classification CE (default 2).
        n_estimators : int
            TabPFN ensemble size (default 1).
        device : str
            'cpu' or 'cuda' for TabPFN inference.
        lhs_init : int
            Shared space-filling LHS size per task, evaluated once at start.  The best
            `n` become the initial population and the *full* LHS becomes the ICL prompt
            archive — no extra evaluations beyond this (default 200).  Must match the
            paired MFEA-Uni-II run for the budgets to align.
        rho_archive_cap : int
            Max archive points per task fed to TabPFN per generation; the archive is
            randomly subsampled to this cap to bound inference cost (default 300).
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.lambda_0 = lambda_0
        self.lambda_decay = lambda_decay
        self.tau = tau
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.device = device
        self.lhs_init = lhs_init
        self.rho_archive_cap = rho_archive_cap
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.rmp_history = []      # off-diagonal rmp per generation
        self.rho_history = []      # symmetrised ICL prior ρ per generation
        self.lambda_history = []   # λ per generation

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
        # Seeded identically to the paired MFEA-Uni-II run, so both draw the same LHS.
        # The best n become the initial population; the FULL LHS is the ICL prompt.
        np.random.seed(derive_shared_seed(self.name))
        lhs_decs = initialization(problem, L, method='lhs')
        lhs_objs, lhs_cons = evaluation(problem, lhs_decs)
        nfes = L * nt

        decs, objs, cons = select_best_init(lhs_decs, lhs_objs, lhs_cons, n)
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified search space for knowledge transfer
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni')
        pop_objs = objs

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        # --- ICL similarity archive (unified space) ---
        # Seeded with the FULL shared LHS (already evaluated above — no extra cost), so
        # the cross-task similarity ρ has signal from generation 0.  The archive then
        # accumulates all offspring each generation (mirrors BO's growing history).
        lhs_decs_u, _ = space_transfer(problem=problem, decs=lhs_decs, cons=lhs_cons, type='uni')
        arch_decs = [d.copy() for d in lhs_decs_u]
        arch_objs = [o.copy() for o in lhs_objs]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 0
        while nfes < max_nfes:

            # --- ICL prior: zero-shot symmetric inter-task relatedness ρ ---
            rho_decs, rho_objs = [], []
            for i in range(nt):
                m = arch_decs[i].shape[0]
                if m > self.rho_archive_cap:
                    sel = np.random.choice(m, self.rho_archive_cap, replace=False)
                    rho_decs.append(arch_decs[i][sel])
                    rho_objs.append(arch_objs[i].ravel()[sel])
                else:
                    rho_decs.append(arch_decs[i])
                    rho_objs.append(arch_objs[i].ravel())
            S = compute_task_similarity_matrix_directed_classification(
                rho_decs, rho_objs,
                n_classes=self.n_classes,
                n_estimators=self.n_estimators,
                device=self.device,
                tau=self.tau,
            )
            rho_prior = 0.5 * (S + S.T)   # symmetrise to match RMP's symmetry

            # --- λ schedule (generation-decaying) ---
            lambda_reg = self.lambda_0 * np.exp(-gen * self.lambda_decay)

            # --- MAP-regularised RMP estimation ---
            rmpMatrix = learnRMP_MAP(pop_decs, dims, rho_prior, lambda_reg)

            self.rho_history.append(rho_prior.copy())
            self.rmp_history.append(rmpMatrix.copy())
            self.lambda_history.append(lambda_reg)

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

            # Accumulate this generation's offspring into the similarity archive
            # (unified space), keyed by skill factor — these exploratory points keep
            # the archive informative as the population converges.
            off_sf_flat = off_sfs.ravel()
            for i in range(nt):
                mask = off_sf_flat == i
                if mask.any():
                    arch_decs[i] = np.vstack([arch_decs[i], off_decs[mask]])
                    arch_objs[i] = np.vstack([arch_objs[i], off_objs[mask]])

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

            if nt >= 2:
                pbar.set_postfix_str(
                    f"gen={gen} rmp(0,1)={rmpMatrix[0, 1]:.3f} "
                    f"ρ(0,1)={rho_prior[0, 1]:.3f} λ={lambda_reg:.3f}"
                )

            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)
            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=max_nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name, save_data=self.save_data)

        return results
