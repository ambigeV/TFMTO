"""
ICL-MFEA-II: MFEA-II with an in-context-learning (ICL) MAP prior on the online RMP.

Motivation
----------
MFEA-II's only learned transfer coefficient is the off-diagonal of the random-mating
probability matrix ``rmp[i, j]``.  Vanilla MFEA-II estimates it every generation by
*maximum likelihood* — `minimize_scalar(loglik(...))` — fitting a 2-component Gaussian
mixture to the two task populations (see `MFEA_II.learnRMP` / `loglik`).  This is the
exact EC analogue of the MTBO IndexKernel off-diagonal ρ, and shares its weakness:
at small / early populations the Gaussian models are crude, so the MLE rmp is
unreliable precisely when transfer matters most.

ICL-MFEA-II places a **MAP prior** on rmp, mirroring MTBO-TFM-MAP-Sym:

    rmp*_ij = argmin_rmp  loglik(rmp, popdata)  +  λ(g) · (rmp − ρ_ij)²

where
  ρ_ij   = symmetrised TabPFN cross-predictive similarity between tasks i and j,
           a zero-shot inter-task relatedness estimate (no extra evaluations), and
  λ(g)   = λ₀ · exp(−g · decay)  decays over generations g.

At small g the ICL prior dominates (reliable relatedness before MLE is trustworthy);
as the population grows λ → 0 and rmp reverts to the pure MFEA-II MLE solution —
graceful fallback to standard MFEA-II.

This is the EC instantiation of the same paradigm as ICL-MTBO: a pre-trained
in-context learner supplies a zero-shot inter-task prior, MAP regularisation hands
off to the native estimator as data accrues.

References
----------
    [1] Bali et al. "Multifactorial evolutionary algorithm with online transfer
        parameter estimation: MFEA-II." IEEE TEVC 24.1 (2019): 69-83.
"""
import time

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.MTSO.MFEA import mfea_selection
from ddmtolab.Algorithms.MTSO.MFEA_II import loglik
from ddmtolab.Methods.Algo_Methods.tfm_task_covar_utils import (
    compute_task_similarity_matrix_directed_classification,
)


class ICL_MFEA_II:
    """
    MFEA-II with an ICL MAP prior on the online-learned RMP.

    The only departure from MFEA-II is `learnRMP`: instead of a pure-MLE rmp per
    task pair, the rmp is fitted under a MAP objective regularised toward a
    symmetrised TabPFN cross-predictive similarity ρ, with a generation-decaying λ.
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
                 lhs_archive=200, rho_archive_cap=300,
                 save_data=True, save_path='./Data',
                 name='ICL-MFEA-II', disable_tqdm=True):
        """
        Parameters
        ----------
        n : int
            Population size per task (default 100).
        max_nfes : int
            Max function evaluations per task (default 10000).
        lambda_0 : float
            Initial MAP regularisation weight (default 1.0 — locked value).
        lambda_decay : float
            Exponential decay rate per generation (default 0.05 — locked value).
            λ(g) = lambda_0 * exp(-g * lambda_decay).
        tau : float
            Sharpness of the CE→similarity map passed to the TabPFN util.
        n_classes : int
            Quantile bins for the classification CE (default 2).
        n_estimators : int
            TabPFN ensemble size (default 1).
        device : str
            'cpu' or 'cuda' for TabPFN inference.
        lhs_archive : int
            Extra space-filling LHS points per task, evaluated once at start to
            seed the ICL similarity archive (default 200).  These give ρ signal
            from generation 0 — the converged EC population alone cannot.  They
            count toward the evaluation budget (initial evals = n + lhs_archive).
        rho_archive_cap : int
            Max archive points per task fed to TabPFN per generation; the archive
            is randomly subsampled to this cap to bound inference cost (default 300).
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
        self.lhs_archive = lhs_archive
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
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population and evaluate for each task
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified search space for knowledge transfer
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni')
        pop_objs = objs

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        # --- ICL similarity archive (unified space) ---
        # Seed with the initial population plus an extra space-filling LHS, so the
        # cross-task similarity ρ has signal from generation 0.  The archive then
        # accumulates all offspring each generation (mirrors BO's growing history).
        arch_decs = [pd.copy() for pd in pop_decs]
        arch_objs = [o.copy() for o in pop_objs]
        if self.lhs_archive > 0:
            extra_decs = initialization(problem, self.lhs_archive)
            extra_objs, extra_cons = evaluation(problem, extra_decs)
            extra_decs_u, _ = space_transfer(problem=problem, decs=extra_decs, cons=extra_cons, type='uni')
            arch_decs = [np.vstack([arch_decs[i], extra_decs_u[i]]) for i in range(nt)]
            arch_objs = [np.vstack([arch_objs[i], extra_objs[i]]) for i in range(nt)]
            nfes += self.lhs_archive * nt

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 0
        while nfes < max_nfes:

            # --- ICL prior: zero-shot symmetric inter-task relatedness ρ ---
            # Computed on the unified-space similarity archive (space-filling LHS
            # seed + accumulated offspring), subsampled to rho_archive_cap.
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
            # (unified space), keyed by skill factor — these exploratory points
            # keep the archive informative as the population converges.
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


def learnRMP_MAP(subpops, vars, rho_prior, lambda_reg):
    """
    Learn the RMP matrix under a MAP objective regularised toward an ICL prior ρ.

    Identical to `MFEA_II.learnRMP` except the per-pair objective minimised is

        loglik(rmp, popdata)  +  λ · (rmp − ρ_ij)²

    instead of the pure NLL `loglik(rmp, popdata)`.  When λ = 0 this reduces
    exactly to MFEA-II.

    Parameters
    ----------
    subpops : list[np.ndarray] | list[dict]
        Per-task subpopulations (unified-space decision variables).
    vars : list[int]
        Dimensionality per task.
    rho_prior : np.ndarray
        Symmetric (numtasks × numtasks) ICL relatedness prior, entries in [0, 1].
    lambda_reg : float
        MAP regularisation weight for this generation.

    Returns
    -------
    rmpMatrix : np.ndarray
        Symmetric (numtasks × numtasks) RMP matrix, diagonal 1.0.
    """
    if isinstance(subpops, list) and isinstance(subpops[0], np.ndarray):
        subpops = [{'data': pop} for pop in subpops]

    numtasks = len(subpops)
    maxDim = max(vars)
    rmpMatrix = np.eye(numtasks)

    # Build per-task probabilistic (Gaussian) models with a 10% random-sample floor
    probmodel = []
    for i in range(numtasks):
        model = {}
        model['nsamples'] = subpops[i]['data'].shape[0]
        nrandsamples = int(np.floor(0.1 * model['nsamples']))
        randMat = np.random.rand(nrandsamples, maxDim)
        combined_data = np.vstack([subpops[i]['data'], randMat])
        model['mean'] = np.mean(combined_data, axis=0)
        model['stdev'] = np.std(combined_data, axis=0, ddof=1)
        probmodel.append(model)

    # Pairwise MAP-regularised RMP
    for i in range(numtasks):
        for j in range(i + 1, numtasks):
            popdata = [
                {'probmatrix': np.ones((probmodel[i]['nsamples'], 2))},
                {'probmatrix': np.ones((probmodel[j]['nsamples'], 2))}
            ]

            Dim = min(vars[i], vars[j])

            for k in range(probmodel[i]['nsamples']):
                for l in range(Dim):
                    popdata[0]['probmatrix'][k, 0] *= norm.pdf(
                        subpops[i]['data'][k, l], probmodel[i]['mean'][l], probmodel[i]['stdev'][l])
                    popdata[0]['probmatrix'][k, 1] *= norm.pdf(
                        subpops[i]['data'][k, l], probmodel[j]['mean'][l], probmodel[j]['stdev'][l])

            for k in range(probmodel[j]['nsamples']):
                for l in range(Dim):
                    popdata[1]['probmatrix'][k, 0] *= norm.pdf(
                        subpops[j]['data'][k, l], probmodel[i]['mean'][l], probmodel[i]['stdev'][l])
                    popdata[1]['probmatrix'][k, 1] *= norm.pdf(
                        subpops[j]['data'][k, l], probmodel[j]['mean'][l], probmodel[j]['stdev'][l])

            rho_ij = float(rho_prior[i, j])

            # Normalise the NLL to a per-sample mean so the prior penalty (bounded
            # in [0, λ]) competes on equal footing regardless of sample count —
            # this keeps λ₀ scale-invariant and transferable from the BO setting.
            n_pts = popdata[0]['probmatrix'].shape[0] + popdata[1]['probmatrix'].shape[0]

            # MAP objective: mean-NLL + λ · (rmp − ρ)²
            def map_obj(x, _popdata=popdata, _rho=rho_ij, _n=n_pts):
                return loglik(x, _popdata, numtasks) / _n + lambda_reg * (x - _rho) ** 2

            result = minimize_scalar(map_obj, bounds=(0, 1), method='bounded')

            rmp_value = max(0, result.x + np.random.normal(0, 0.01))
            rmp_value = min(rmp_value, 1)

            rmpMatrix[i, j] = rmp_value
            rmpMatrix[j, i] = rmp_value

    return rmpMatrix
