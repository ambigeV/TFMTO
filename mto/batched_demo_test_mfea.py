"""
Batched Demo: three-way RMP-prior ablation on CEC17-MTSO 30D.

    MFEA-Uni-II          : no prior — pure online-MLE RMP.
    ICL-MFEA-Uni-II      : MAP prior from a FITNESS-CLASS cross-entropy (the
                           IndexKernel / fitness-landscape currency — the wrong
                           inductive bias for RMP; kept as an ablation).
    ICL-MFEA-Ovlp-Uni-II : MAP prior from an ELITE DECISION-SPACE OVERLAP estimate
                           (classifier two-sample test) — the currency that actually
                           matches RMP (mixing of two Gaussian elite populations in X).

All three use the *unified shared-LHS initialization* protocol so the comparison is
budget-fair: one space-filling LHS of LHS_INIT points per task is evaluated once, the
best N_POP become the initial population, and that same LHS seeds the ICL prompt.
Because all three pay LHS_INIT*nt up front and N_POP*nt per generation, they run the
SAME number of generations:  (MAX_NFES*nt - LHS_INIT*nt) / (N_POP*nt).

With the defaults below (MAX_NFES=1200, N_POP=20, LHS_INIT=200, nt=2):
    (1200*2 - 200*2) / (20*2) = 2000 / 40 = 50 generations each.

Matching runs share the identical LHS sample (seeded from the problem/run suffix); the
only thing that differs across the three methods is the RMP estimator / prior.

Data layout (auto-managed by BatchExperiment):
    ./Data_MFEA_30D/{algo_name}/{algo_name}_{problem_name}_{run_id}.pkl
Results (mean ± 0.5*std convergence curves) → ./Results_MFEA_30D/.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso_30d import CEC17MTSO_30D
from ddmtolab.Algorithms.MTSO.MFEA_Uni_II import MFEA_Uni_II
from ddmtolab.Algorithms.MTSO.ICL_MFEA_Ovlp_Uni_II import ICL_MFEA_Ovlp_Uni_II
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS   = 5
N_POP    = 20          # initial population size per task (best N_POP of the shared LHS)
MAX_NFES = 1200        # evaluation budget per task
MAX_WORKERS = 4        # parallel processes — reduce if memory is tight

# --- Shared-LHS init protocol (identical for both methods) ---
# One space-filling LHS of LHS_INIT points/task, evaluated once; best N_POP → init pop;
# full LHS → ICL prompt.  Both methods pay LHS_INIT*nt up front, so both run
# (MAX_NFES*nt - LHS_INIT*nt) / (N_POP*nt) generations  (= 50 with these defaults).
LHS_INIT        = 200
RHO_ARCHIVE_CAP = 300

# --- ICL MAP config (locked defaults from the BO ablation) ---
LAMBDA_0     = 1.0
LAMBDA_DECAY = 0.05
TAU          = 1.0
N_ESTIMATORS = 1
DEVICE       = 'cpu'

# Elite-fraction sweep for the decision-overlap (C2ST) prior.  This is the key
# inductive-bias knob: it sets how tight the "promising region" is whose cross-task
# overlap defines ρ.  0.5 = top-half elites, 0.1 = top-10% (sharpest elites).
ELITE_FRACS = [0.5, 0.3, 0.1]

def ovlp_name(frac):
    return f'ICL-MFEA-Ovlp{int(round(frac * 100)):02d}-Uni-II'

DATA_PATH    = './Data_MFEA_30D'
RESULTS_PATH = './Results_MFEA_30D'

# Ablation: no-prior baseline vs the correct decision-overlap prior at three elite
# fractions (tighter elites = sharper "promising region" for the overlap test):
#   MFEA-Uni-II            : no prior (pure online MLE RMP)
#   ICL-MFEA-Ovlp50-Uni-II : overlap prior, top-50% elites
#   ICL-MFEA-Ovlp30-Uni-II : overlap prior, top-30% elites
#   ICL-MFEA-Ovlp10-Uni-II : overlap prior, top-10% elites
ALGO_ORDER = ['MFEA-Uni-II'] + [ovlp_name(f) for f in ELITE_FRACS]

# =============================================================================
# Entry point — required on macOS/Windows (spawn-based multiprocessing)
# =============================================================================

if __name__ == '__main__':
    print('\n===== CEC17-MTSO 30D : MFEA-Uni-II vs ICL-MFEA-Ovlp {50,30,10}% elite =====')

    batch_exp = BatchExperiment(base_path=DATA_PATH, clear_folder=False)

    # --- Problems: CEC17-MTSO 30D P1–P9 ---
    benchmark = CEC17MTSO_30D()
    for prob_name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']:
        batch_exp.add_problem(getattr(benchmark, prob_name), prob_name)

    # --- Algorithms ---
    # Baseline: no prior (pure online-MLE RMP).
    batch_exp.add_algorithm(MFEA_Uni_II, 'MFEA-Uni-II',
        n=N_POP, max_nfes=MAX_NFES, lhs_init=LHS_INIT, disable_tqdm=True)

    # Decision-overlap (C2ST) prior at each elite fraction in the sweep.
    for frac in ELITE_FRACS:
        batch_exp.add_algorithm(ICL_MFEA_Ovlp_Uni_II, ovlp_name(frac),
            n=N_POP, max_nfes=MAX_NFES,
            lambda_0=LAMBDA_0, lambda_decay=LAMBDA_DECAY, tau=TAU,
            elite_frac=frac, n_estimators=N_ESTIMATORS, device=DEVICE,
            lhs_init=LHS_INIT, rho_archive_cap=RHO_ARCHIVE_CAP,
            disable_tqdm=True)

    # --- Run (parallel across workers) ---
    batch_exp.run(n_runs=N_RUNS, verbose=True, max_workers=MAX_WORKERS)

    # --- Results Analysis ---
    analyzer = DataAnalyzer(
        data_path=DATA_PATH,
        save_path=RESULTS_PATH,
        algorithm_order=ALGO_ORDER,
        figure_format='png',
        log_scale=False,
        show_std_band=True,
        std_scale=0.5,
        best_so_far=True,
        clear_results=True,
    )
    analyzer.run()
