"""
Batched Demo: MFEA-Uni-II vs ICL-MFEA-Uni-II on CEC17-MTSO 30D.

Both methods use the *unified shared-LHS initialization* protocol so the comparison
is budget-fair: one space-filling LHS of LHS_INIT points per task is evaluated once,
the best N_POP become the initial population, and that same LHS is the ICL prompt.
Because both pay LHS_INIT*nt up front and N_POP*nt per generation, they run the
SAME number of generations:  (MAX_NFES*nt - LHS_INIT*nt) / (N_POP*nt).

With the defaults below (MAX_NFES=600, N_POP=20, LHS_INIT=200, nt=2):
    (600*2 - 200*2) / (20*2) = 800 / 40 = 20 generations each.

The matching MFEA-Uni-II / ICL-MFEA-Uni-II runs share the identical LHS sample
(seeded from the problem/run suffix); the only difference between the two methods is
the RMP estimator — MLE in MFEA-Uni-II, MAP-toward-ICL-prior in ICL-MFEA-Uni-II.

Data layout (auto-managed by BatchExperiment):
    ./Data_MFEA_30D/{algo_name}/{algo_name}_{problem_name}_{run_id}.pkl
Results (mean ± 0.5*std convergence curves) → ./Results_MFEA_30D/.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso_30d import CEC17MTSO_30D
from ddmtolab.Algorithms.MTSO.MFEA_Uni_II import MFEA_Uni_II
from ddmtolab.Algorithms.MTSO.ICL_MFEA_Uni_II import ICL_MFEA_Uni_II
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS   = 5
N_POP    = 20          # initial population size per task (best N_POP of the shared LHS)
MAX_NFES = 600         # evaluation budget per task
MAX_WORKERS = 4        # parallel processes — reduce if memory is tight

# --- Shared-LHS init protocol (identical for both methods) ---
# One space-filling LHS of LHS_INIT points/task, evaluated once; best N_POP → init pop;
# full LHS → ICL prompt.  Both methods pay LHS_INIT*nt up front, so both run
# (MAX_NFES*nt - LHS_INIT*nt) / (N_POP*nt) generations  (= 20 with these defaults).
LHS_INIT        = 200
RHO_ARCHIVE_CAP = 300

# --- ICL MAP config (locked defaults from the BO ablation) ---
LAMBDA_0     = 1.0
LAMBDA_DECAY = 0.05
TAU          = 1.0
N_CLASSES    = 2
N_ESTIMATORS = 1
DEVICE       = 'cpu'

DATA_PATH    = './Data_MFEA_30D'
RESULTS_PATH = './Results_MFEA_30D'

ALGO_ORDER = ['MFEA-Uni-II', 'ICL-MFEA-Uni-II']

# =============================================================================
# Entry point — required on macOS/Windows (spawn-based multiprocessing)
# =============================================================================

if __name__ == '__main__':
    print('\n========== CEC17-MTSO 30D : MFEA-Uni-II vs ICL-MFEA-Uni-II ==========')

    batch_exp = BatchExperiment(base_path=DATA_PATH, clear_folder=False)

    # --- Problems: CEC17-MTSO 30D P1–P9 ---
    benchmark = CEC17MTSO_30D()
    for prob_name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']:
        batch_exp.add_problem(getattr(benchmark, prob_name), prob_name)

    # --- Algorithms ---
    batch_exp.add_algorithm(MFEA_Uni_II, 'MFEA-Uni-II',
        n=N_POP, max_nfes=MAX_NFES, lhs_init=LHS_INIT, disable_tqdm=True)

    batch_exp.add_algorithm(ICL_MFEA_Uni_II, 'ICL-MFEA-Uni-II',
        n=N_POP, max_nfes=MAX_NFES,
        lambda_0=LAMBDA_0, lambda_decay=LAMBDA_DECAY, tau=TAU,
        n_classes=N_CLASSES, n_estimators=N_ESTIMATORS, device=DEVICE,
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
