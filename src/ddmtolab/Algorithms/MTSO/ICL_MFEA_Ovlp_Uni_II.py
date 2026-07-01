"""
ICL-MFEA-Ovlp-Uni-II: MFEA-II with an ICL MAP prior on the RMP whose zero-shot
similarity is a decision-space ELITE-OVERLAP estimate (classifier two-sample test),
under the unified shared-LHS initialization protocol.

Why this exists (vs ICL-MFEA-Uni-II)
------------------------------------
ICL-MFEA-Uni-II derives its RMP prior ρ from a *fitness-class* cross-entropy — "can a
classifier of good-vs-bad solutions trained on task i predict task j?".  That is the
correct intuition for an MTBO IndexKernel (fitness-landscape / output correlation),
but it is the WRONG inductive bias for the MFEA-II RMP.

MFEA-II's RMP is the mixing coefficient of a 2-Gaussian model over the two tasks'
ELITE populations *in decision space* (`learnRMP` never touches objective values).
It measures how much the promising regions OVERLAP, so that cross-task crossover of
decision vectors produces viable offspring.  Two tasks can have highly correlated
landscapes yet shifted optima — high fitness-class similarity but disjoint elite
regions — in which case a fitness-class prior pushes RMP up and drives crossover into
the useless gap between the two basins.

ICL-MFEA-Ovlp-Uni-II therefore replaces the prior with a zero-shot classifier
two-sample test (C2ST) on the elite decision vectors:

    label = task membership (i vs j),  features = elite decision vectors (X only)
    CE → log2 (classifier confused ⇒ elites overlap)  ⇒ ρ → 1
    CE → 0    (classifier separates ⇒ elites disjoint) ⇒ ρ → 0

This is the honest ICL analog of `learnRMP`'s own Gaussian-overlap test: a pre-trained,
distribution-free, elite-focused overlap estimate.  Everything else — the MAP penalty
λ(g)·(rmp − ρ)², the generation-decaying λ, the shared-LHS init, the budget — is
identical to ICL-MFEA-Uni-II, so the generation count still matches MFEA-Uni-II.

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
    compute_task_overlap_matrix_membership_c2st,
)


class ICL_MFEA_Ovlp_Uni_II:
    """
    ICL-MFEA-II with a decision-space elite-overlap (C2ST) RMP prior, shared-LHS init.

    Identical to `ICL_MFEA_Uni_II` except the ρ prior is a zero-shot classifier
    two-sample test on elite decision vectors (X-space overlap) rather than a
    fitness-class cross-entropy — the currency that actually matches the RMP.
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
                 elite_frac=0.5, n_estimators=1, device='cpu',
                 lhs_init=200, rho_archive_cap=300,
                 save_data=True, save_path='./Data',
                 name='ICL-MFEA-Ovlp-Uni-II', disable_tqdm=True):
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
            Sharpness of the CE→overlap map (τ=1 linear; τ<1 toward 1; τ>1 toward 0).
        elite_frac : float
            Fraction of each task's archive (lowest objective) treated as elites for
            the C2ST overlap prior (default 0.5).  This is the inductive-bias knob:
            it defines "which promising region" the RMP prior measures overlap of.
        n_estimators : int
            TabPFN ensemble size (default 1).
        device : str
            'cpu' or 'cuda' for TabPFN inference.
        lhs_init : int
            Shared space-filling LHS size per task, evaluated once at start; best n →
            init pop, full LHS → overlap-prior archive (default 200).  Must match the
            paired MFEA-Uni-II run for the budgets to align.
        rho_archive_cap : int
            Max archive points per task fed to TabPFN per generation (default 300).
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.lambda_0 = lambda_0
        self.lambda_decay = lambda_decay
        self.tau = tau
        self.elite_frac = elite_frac
        self.n_estimators = n_estimators
        self.device = device
        self.lhs_init = lhs_init
        self.rho_archive_cap = rho_archive_cap
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.rmp_history = []      # off-diagonal rmp per generation
        self.rho_history = []      # symmetric elite-overlap prior ρ per generation
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

        # --- Shared LHS initialization (identical to the paired MFEA-Uni-II run) ---
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

        # --- Overlap-prior archive (unified space) = full shared LHS, grows with offspring ---
        lhs_decs_u, _ = space_transfer(problem=problem, decs=lhs_decs, cons=lhs_cons, type='uni')
        arch_decs = [d.copy() for d in lhs_decs_u]
        arch_objs = [o.copy() for o in lhs_objs]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 0
        while nfes < max_nfes:

            # --- ICL prior: zero-shot elite decision-space OVERLAP ρ (C2ST) ---
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
            S = compute_task_overlap_matrix_membership_c2st(
                rho_decs, rho_objs,
                elite_frac=self.elite_frac,
                n_estimators=self.n_estimators,
                device=self.device,
                tau=self.tau,
                random_state=gen,
            )
            rho_prior = 0.5 * (S + S.T)   # already symmetric; kept for safety

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

            # Accumulate this generation's offspring into the overlap archive.
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
